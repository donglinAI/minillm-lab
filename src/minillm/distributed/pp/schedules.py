"""
minillm.distributed.pp.schedules
================================
流水线并行（Pipeline Parallelism）调度：GPipe / 1F1B
----------------------------------------------------
为什么需要流水线并行（PP）?
    模型按"层"切到多张卡（层是天然边界）:
        卡0: 层0-1    卡1: 层2-3    卡2: 层4-5    卡3: 层6-7
    数据必须依次流过所有卡（前向: 0→1→2→3, 后向反向流回）。
    但一次只喂一个 batch → 只有一张卡在干活，其他全闲着。

关键概念: micro-batch（微批）
    把一个大 batch 切成 M 个小 batch，逐个流过流水线。
    卡0 算 micro-batch 1 的同时……等卡1 开始算 micro-batch 1 后，
    卡0 立刻接着算 micro-batch 2 —— 像工厂流水线，多卡同时忙。

两种调度（核心区别: 后向何时开始）:
    GPipe:  所有 micro-batch 的前向全部做完，才开始后向
            → 前向阶段要把 M 个 micro-batch 的激活全存着等后向
            → 激活显存峰值 = M 个 micro-batch（O(M)）
   1F1B:   前向和后向交错（One Forward One Backward）
            → 后向一旦能开始就立即开始，激活随用随释放
            → 激活显存峰值 ≈ 1 个 micro-batch（O(1)）

★ 重要（容易误解的点）:
    在"同步、通信不与计算重叠"的简单模型里，两种调度的总时间和
    bubble 率相同（都受流水线填充+排水限制）。1F1B 的真正优势是:
    1. 激活显存从 O(M) 降到 O(1) —— 同样显存能跑更大的 batch
       （GPipe 要同时存 M 份激活等后向, 1F1B 算完一个后向立刻释放）
    2. 后向提前 → 梯度通信可以与计算重叠（真实 GPU 上省时间）

1F1B 的每卡动作序列（warmup → 交替 → cooldown）:
    卡 i 先做 (P-1-i) 个前向（warmup），然后每做一个前向就跟一个后向
    （稳态），最后只剩后向（cooldown）。
    例 (P=4, M=4):
        卡0: f0 f1 f2 f3 | b0 b1 b2 b3      （warmup 3 个, 后向全靠等）
        卡1: f0 f1 f2 b0 f3 b1 b2 b3
        卡2: f0 f1 b0 f2 b1 f3 b2 b3
        卡3: f0 b0 f1 b1 f2 b2 f3 b3       （warmup 0, 立刻交替）

bubble（气泡）: 每卡在流水线中"无事可做"的时间占比。
    bubble 率 = (总时间×卡数 − 每卡实际工作量×卡数) / (总时间×卡数)
    micro-batch 数 M 越大、stage 数 P 越小，bubble 越小。

本文件:
    1. simulate(): 虚拟时间线模拟器（教学核心）——对比 GPipe / 1F1B 的 bubble
    2. 真实训练等价验证: GPipe 切层训练与单卡全模型梯度完全一致
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. 时间线模拟器（虚拟层, 不跑真实计算）
# ---------------------------------------------------------------------------
def _make_sequences(schedule: str, P: int, M: int):
    """生成每卡的动作序列（每个动作 = (前向/后向, micro-batch 号)）。"""
    seqs = []
    for i in range(P):
        if schedule == "gpipe":
            # 所有前向做完, 再所有后向
            seqs.append([("f", j) for j in range(M)] + [("b", j) for j in range(M)])
        else:  # 1f1b
            W = P - 1 - i          # warmup: 前向 P-1-i 次（把流水线填满）
            seq = [("f", j) for j in range(W)]
            fj, bj = W, 0
            while fj < M and bj < M:      # 稳态: 一前向一后向交替
                seq.append(("f", fj)); fj += 1
                seq.append(("b", bj)); bj += 1
            while bj < M:                 # cooldown: 只剩后向
                seq.append(("b", bj)); bj += 1
            seqs.append(seq)
    return seqs


def _run(seqs, P: int, M: int, f: float, b: float, global_barrier: bool):
    """事件驱动执行：每卡按自己的动作序列，跨卡依赖满足才能开始。

    global_barrier=True（GPipe）: 后向必须等所有前向完成（两阶段同步）。
    global_barrier=False（1F1B）: 后向依赖满足就立即开始。
    """
    f_end = [[None] * M for _ in range(P)]
    b_end = [[None] * M for _ in range(P)]
    clock = [0.0] * P
    timeline = []
    k = [0] * P
    total = sum(len(s) for s in seqs)
    done = 0
    # 激活显存跟踪: 每个 f 产生 1 个 micro-batch 的激活, 每个 b 释放 1 个
    live = [0] * P
    peak = [0] * P

    all_f_done = lambda: all(f_end[i][j] is not None for i in range(P) for j in range(M))

    while done < total:
        best = None
        for i in range(P):
            if k[i] >= len(seqs[i]):
                continue
            kind, j = seqs[i][k[i]]
            if kind == "f":
                if i > 0 and f_end[i - 1][j] is None:
                    continue                      # 前一卡前向未完成
                if j > 0 and f_end[i][j - 1] is None:
                    continue                      # 本卡上一 micro-batch 前向未完成
                start = max([clock[i]] + ([f_end[i - 1][j]] if i > 0 else []) + ([f_end[i][j - 1]] if j > 0 else []))
            else:  # "b"
                if f_end[i][j] is None:
                    continue                      # 本 micro-batch 前向没做完
                if i < P - 1 and b_end[i + 1][j] is None:
                    continue                      # 下一卡梯度还没传回来
                if j > 0 and b_end[i][j - 1] is None:
                    continue                      # 本卡后向顺序
                if global_barrier and not all_f_done():
                    continue                      # GPipe: 所有前向完成才开始后向
                start = max([clock[i], f_end[i][j]] + ([b_end[i + 1][j]] if i < P - 1 else []) + ([b_end[i][j - 1]] if j > 0 else []))
            if best is None or start < best[0]:
                best = (start, i)
        assert best is not None, "调度死锁"
        _, i = best
        kind, j = seqs[i][k[i]]
        if kind == "f":
            start = max([clock[i]] + ([f_end[i - 1][j]] if i > 0 else []) + ([f_end[i][j - 1]] if j > 0 else []))
            end = start + f
            f_end[i][j] = end
            live[i] += 1
            peak[i] = max(peak[i], live[i])
        else:
            start = max([clock[i], f_end[i][j]] + ([b_end[i + 1][j]] if i < P - 1 else []) + ([b_end[i][j - 1]] if j > 0 else []))
            end = start + b
            b_end[i][j] = end
            live[i] -= 1
        clock[i] = end
        timeline.append((i, kind, j, start, end))
        k[i] += 1
        done += 1
    return timeline, peak


def simulate(schedule: str, P: int, M: int, f: float = 1.0, b: float = 2.0):
    """模拟 P 卡 × M 个 micro-batch 的流水线调度，返回时间线 + bubble 统计。

    f : 单个 micro-batch 单 stage 前向耗时
    b : 后向耗时（约 2×f 是经验值: 后向要存激活+算两次）
    """
    seqs = _make_sequences(schedule, P, M)
    timeline, peak = _run(seqs, P, M, f, b, global_barrier=(schedule == "gpipe"))
    total = max(e[4] for e in timeline)
    busy_per_card = M * (f + b)               # 每卡实际工作量: M 个前向 + M 个后向
    idle_per_card = total - busy_per_card
    return timeline, {
        "total_time": total,
        "busy_per_card": busy_per_card,
        "idle_per_card": idle_per_card,
        "bubble_ratio": idle_per_card / total,
        "peak_activations": max(peak),        # 每卡同时存活的 micro-batch 激活峰值
    }


def print_timeline(timeline, P: int) -> None:
    for i in range(P):
        evs = sorted([e for e in timeline if e[0] == i], key=lambda e: e[3])
        line = " ".join(f"{'f' if k == 'f' else 'b'}{mb}" for _, k, mb, _, _ in evs)
        print(f"  卡{i}: {line}")


# ---------------------------------------------------------------------------
# 2. 真实训练等价验证（切层 + micro-batch 不改变训练数学）
# ---------------------------------------------------------------------------
def _verify_training_equivalence() -> None:
    torch.manual_seed(0)
    P, M, D = 4, 2, 16
    xs = [torch.randn(3, D) for _ in range(M)]
    ys = [torch.randn(3, D) for _ in range(M)]

    def make_model():
        return nn.Sequential(
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D),
        )

    # 参考: 单卡全模型
    ref = make_model()
    ref_loss = sum(((ref(x) - y) ** 2).mean() for x, y in zip(xs, ys))
    ref_loss.backward()
    ref_grads = [p.grad.clone() for p in ref.parameters()]

    # GPipe: 切成 4 个 stage, 逐 micro-batch 前向（计算图不变, 数学等价）
    stages = [make_model()[0:2], make_model()[2:4], make_model()[4:6], make_model()[6:8]]
    state = ref.state_dict()
    for s in stages:
        s.load_state_dict({k: state[k] for k in s.state_dict()})

    total_loss = torch.zeros(())
    for x, y in zip(xs, ys):
        h = x
        for s in stages:
            h = s(h)
        total_loss = total_loss + ((h - y) ** 2).mean()
    total_loss.backward()

    pp_grads = [p.grad for s in stages for p in s.parameters()]
    max_err = max((a - b).abs().max().item() for a, b in zip(pp_grads, ref_grads))
    print(f"GPipe 切层训练 vs 单卡全模型: 梯度最大误差 {max_err:.2e} (应=0)")
    assert max_err < 1e-6
    print("✅ 训练等价: 切层 + micro-batch 不改变训练数学")


# ---------------------------------------------------------------------------
# 自校验
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    P, M = 4, 4
    print(f"时间线对比 (P={P} 卡, M={M} 个 micro-batch, 前向 f=1, 后向 b=2):\n")

    tl_g, st_g = simulate("gpipe", P, M)
    tl_1, st_1 = simulate("1f1b", P, M)

    print("【GPipe】(所有前向做完才开始后向):")
    print_timeline(tl_g, P)
    print(f"  总时间 {st_g['total_time']:.0f} | bubble {st_g['bubble_ratio']*100:.1f}% | "
          f"激活峰值 {st_g['peak_activations']} 个 micro-batch\n")

    print("【1F1B】(后向一旦可行立即执行):")
    print_timeline(tl_1, P)
    print(f"  总时间 {st_1['total_time']:.0f} | bubble {st_1['bubble_ratio']*100:.1f}% | "
          f"激活峰值 {st_1['peak_activations']} 个 micro-batch\n")

    print("对比结论:")
    print(f"  · 总时间/bubble 相同（同步模型）: 都是 {st_g['total_time']:.0f} / "
          f"{st_g['bubble_ratio']*100:.1f}%")
    print("  · 1F1B 后向提前, 梯度通信可与计算重叠（真实 GPU 上省时间）")

    # 激活峰值对比: 用 M 远大于 P 的场景（M=16, P=4）
    _, sg = simulate("gpipe", 4, 16)
    _, s1 = simulate("1f1b", 4, 16)
    print(f"\n激活显存峰值 (P=4, M=16): GPipe={sg['peak_activations']} 个 micro-batch vs "
          f"1F1B={s1['peak_activations']} 个")
    print(f"  ← 1F1B 省 {sg['peak_activations']/s1['peak_activations']:.0f}x 显存"
          f"（峰值≈卡数 P, 与 M 无关; GPipe 峰值=M, M 越大越吃显存）")
    print("  · 这就是 1F1B 让同样显存能跑更大 batch 的原因")

    print("\nmicro-batch 数 M 对 bubble 的影响 (P=4, 1F1B):")
    for m in [4, 8, 16]:
        _, s1 = simulate("1f1b", 4, m)
        print(f"  M={m:2d}: bubble = {s1['bubble_ratio']*100:.1f}%  ← M 越大, 流水线越满, 气泡越小")

    print("\n" + "=" * 50)
    _verify_training_equivalence()
