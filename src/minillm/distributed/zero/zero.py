"""
minillm.distributed.zero.zero
========================
ZeRO: Zero Redundancy Optimizer（零冗余优化器）——DeepSpeed 的核心并行技术
--------------------------------------------------------------------------
背景: DDP 的显存冗余
    每张卡都存一份完整的 参数 + 梯度 + 优化器状态。
    Adam 的优化器状态 = 一阶矩 m + 二阶矩 v = 2 × 参数量！
    所以 DDP 每卡显存 ≈ 参数(1N) + 梯度(1N) + 状态(2N) = 4N 个元素。
    对 7B 模型 fp32: 4 × 7e9 × 4B ≈ 112 GB —— 一张卡根本装不下。

ZeRO 的思路: 不复制，切开分到各卡，用到时再拼回来（通信）。
    三档递进（每档多切一样东西）:
        ZeRO-1: 切优化器状态 m/v   → 每卡只存 2N/P
        ZeRO-2: + 切梯度 g         → 梯度只存 N/P
        ZeRO-3: + 切参数 θ         → 参数也只存 N/P（前向时按需 all-gather）

    显存公式（每卡元素数）:
        DDP:    N + N + 2N            = 4N
        ZeRO-1: N + N + 2N/P          = 2N + 2N/P
        ZeRO-2: N + N/P + 2N/P        = N + 3N/P
        ZeRO-3: N/P + N/P + 2N/P      = 4N/P

    ★ ZeRO 的根基（本文件的核心验证）:
        "每卡只更新自己的分片" ≡ "全量更新"
        因为优化器状态逐元素独立（m_i 只影响 θ_i），把参数/梯度/状态
        按同样方式切块，每卡算自己那块，拼接起来 == 全量更新结果。

通信代价:
    ZeRO-1/2 每步通信量 ≈ DDP（1 次梯度归约 + 1 次参数 all-gather）
    ZeRO-3 前向每层要 all-gather 参数（通信更多, 换显存）

本文件:
    1. memory_breakdown(): 显存公式对比表（教学核心）
    2. adamw_step(): 参考的全量 AdamW 更新（对照目标）
    3. ZeRO-1/2/3 分片训练模拟: 每步"分片更新 + 拼接"
       ≡ DDP 全量更新（误差 0 验证）
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# 1. 显存公式
# ---------------------------------------------------------------------------
def memory_breakdown(N: float, P: int, bytes_per_elem: float = 4.0):
    """N: 参数量（元素个数）; P: 卡数。返回每卡显存（GB）。"""
    gb = lambda elems: elems * bytes_per_elem / 1e9
    return {
        "DDP": gb(4 * N),
        "ZeRO-1": gb(2 * N + 2 * N / P),
        "ZeRO-2": gb(N + 3 * N / P),
        "ZeRO-3": gb(4 * N / P),
    }


# ---------------------------------------------------------------------------
# 2. 参考实现: DDP 全量 AdamW 更新
# ---------------------------------------------------------------------------
def adamw_step(theta, g, m, v, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0):
    """一步全量 AdamW 更新（参考目标）。返回 (theta_new, m_new, v_new)。"""
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g ** 2
    m_hat = m / (1 - beta1 ** 2)   # 简化: bias correction 用固定系数
    v_hat = v / (1 - beta2 ** 2)
    theta_new = theta * (1 - lr * wd) - lr * m_hat / (v_hat.sqrt() + eps)
    return theta_new, m, v


# ---------------------------------------------------------------------------
# 3. ZeRO 分片训练模拟
# ---------------------------------------------------------------------------
def _chunk_r(x, r: int, P: int):
    """取第 r 卡的 1/P 连续分片（模拟 reduce-scatter 后本卡拿到的块）。"""
    c = x.numel() // P
    return x.flatten()[r * c:(r + 1) * c]


def _gather(parts, P: int):
    """把 P 张卡的分片拼回全量（模拟 all-gather）。"""
    return torch.cat([parts[r].flatten() for r in range(P)])


def run_zero(stage: str, P: int, theta0, steps: int = 5, seed: int = 0):
    """模拟 ZeRO-1/2/3 训练 steps 步, 返回每步"拼接出的全量参数"。

    与参考 DDP 全量训练对比: 拼接结果必须逐元素相等（误差 0）。
    """
    torch.manual_seed(seed)
    N = theta0.numel()
    c = N // P
    assert N % P == 0, "演示用: 参数需能被卡数整除"

    # m/v 永远按块切（ZeRO 的核心: 优化器状态不复制）
    m = {r: torch.zeros(c) for r in range(P)}
    v = {r: torch.zeros(c) for r in range(P)}
    if stage == "zero3":
        theta = {r: _chunk_r(theta0, r, P).clone() for r in range(P)}   # 参数分片
    else:
        theta = {r: theta0.clone() for r in range(P)}                   # 参数全量冗余

    all_thetas = []
    for _ in range(steps):
        # 当前全量参数（zero3 需拼, zero1/2 上轮 all-gather 后各卡相同）
        cur = _gather(theta, P) if stage == "zero3" else theta[0]
        g_global = 2.0 * cur - 1.0          # 模拟一步梯度（来源不影响分片数学）

        for r in range(P):
            g_r = _chunk_r(g_global, r, P)  # 本卡梯度分片（reduce-scatter 的结果）
            # 用本卡分片 m/v 更新: m_i 只影响 θ_i —— 逐元素独立, 分片可行
            m[r] = 0.9 * m[r] + 0.1 * g_r
            v[r] = 0.999 * v[r] + 0.001 * g_r ** 2
            m_hat = m[r] / (1 - 0.9 ** 2)
            v_hat = v[r] / (1 - 0.999 ** 2)
            delta = 0.01 * m_hat / (v_hat.sqrt() + 1e-8)

            if stage == "zero3":
                theta[r] = theta[r] - delta          # 分片整体替换
            else:
                # 全量参数里只改自己的第 r 块（其他块不动, 是旧值）
                flat = theta[r].flatten().clone()
                flat[r * c:(r + 1) * c] -= delta
                theta[r] = flat.view_as(theta[r])

        if stage == "zero3":
            all_thetas.append(_gather(theta, P))     # 前向时才拼
        else:
            # all-gather 参数: 每卡把自己"更新过的第 r 块"广播出去
            full = torch.cat([_chunk_r(theta[r], r, P) for r in range(P)])
            theta = {r: full.clone() for r in range(P)}
            all_thetas.append(full)
    return all_thetas


def run_ddp(theta0, steps: int = 5, seed: int = 0):
    """参考: DDP 全量训练（每卡持有全量参数+梯度+状态）。"""
    torch.manual_seed(seed)
    theta = theta0.clone()
    m = torch.zeros_like(theta)
    v = torch.zeros_like(theta)
    all_thetas = []
    for _ in range(steps):
        g = 2.0 * theta - 1.0
        theta, m, v = adamw_step(theta, g, m, v)
        all_thetas.append(theta)
    return all_thetas


# ---------------------------------------------------------------------------
# 自校验 python -m minillm.distributed.zero.zero
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 66)
    print("显存对比: 7B 参数模型 (fp32), 每卡占用")
    print("=" * 66)
    N = 7e9
    for P in [1, 8, 64]:
        mem = memory_breakdown(N, P)
        print(f"P={P:2d} 卡 | DDP {mem['DDP']:8.1f} GB | ZeRO-1 {mem['ZeRO-1']:8.1f} GB | "
              f"ZeRO-2 {mem['ZeRO-2']:8.1f} GB | ZeRO-3 {mem['ZeRO-3']:8.1f} GB")
    print(f"\n  ← DDP 112 GB 单卡装不下 7B; ZeRO-3 在 64 卡时每卡仅需 "
          f"{memory_breakdown(N,64)['ZeRO-3']:.1f} GB")

    print("\n" + "=" * 66)
    print("核心验证: ZeRO 分片更新 ≡ DDP 全量更新")
    print("=" * 66)
    torch.manual_seed(0)
    theta0 = torch.randn(1024)
    ref = run_ddp(theta0)

    for stage in ["zero1", "zero2", "zero3"]:
        for P in [1, 2, 4, 8]:
            got = run_zero(stage, P, theta0)
            err = max((a - b).abs().max().item() for a, b in zip(got, ref))
            print(f"  {stage.upper()} P={P:2d}: 与 DDP 全量训练 {len(ref)} 步误差 {err:.2e} "
                  f"{'✅' if err < 1e-6 else '❌'}")
            assert err < 1e-6

    print("\n结论: 无论切多少份, 每卡只算自己分片的更新, 拼起来和全量更新一模一样")
    print("  → ZeRO 只是'把存储切开 + 通信拼接', 训练数学不变")
