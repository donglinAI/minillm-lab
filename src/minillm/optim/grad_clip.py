"""
minillm.optim.grad_clip
=======================
Gradient Clipping（梯度裁剪）: clip_grad_norm_
-----------------------------------------------
为什么需要:
    loss 偶尔出现异常大的梯度（坏 batch / 数值抖动）→ 一步把参数冲飞。
    裁剪让"总梯度范数"不超过 max_norm, 训练稳定。

怎么做:
    1. 把所有参数的梯度拼成一个"范数池":
       total_norm = √( Σ||g_i||₂² )    (norm_type=2 时 = 全局 L2 范数)
    2. 若 total_norm > max_norm, 所有梯度统一缩放:
       g_i ← g_i × (max_norm / total_norm)
    3. 返回裁剪前的 total_norm（供日志观察: 多大才算异常）

★ 关键: 统一缩放而非截断每个梯度
    · 保留梯度的"方向", 只改"长度" → 更新方向不变, 只是步长变小
    · 逐个梯度 clip（像 clamp）会扭曲方向, 不是标准做法
"""

from __future__ import annotations

import torch


def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
    """与 torch.nn.utils.clip_grad_norm_ 对齐的全局梯度裁剪。

    Parameters
    ----------
    parameters : 参数迭代器（含 .grad 的叶子张量）
    max_norm : 允许的最大总范数
    norm_type : 范数阶数（默认 2 = L2）

    Returns
    -------
    total_norm : 裁剪前的总范数
    """
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.tensor(0.0)

    # 1) 全局总范数: √(Σ ||g_i||^p^p)  — 注意 torch 的实现对 p≠∞ 是
    #    (Σ ||g_i||_p^p)^(1/p), 即把所有梯度的 p 范数的 p 次方相加再开 p 次方
    total_norm = torch.zeros((), dtype=params[0].dtype, device=params[0].device)
    for p in params:
        g = p.grad.detach()
        total_norm = total_norm + g.norm(norm_type) ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)

    # 2) 超限才缩放（clip_coef < 1 → 统一缩放; ≥1 → 不动）
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            p.grad.detach().mul_(clip_coef)
    return total_norm


# ---------------------------------------------------------------------------
# 自校验
# ---------------------------------------------------------------------------
def _verify_against_torch() -> None:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(), torch.nn.Linear(8, 4))
    x, y = torch.randn(5, 6), torch.randn(5, 4)
    ((model(x) - y) ** 2).mean().backward()

    # 我们的实现
    norm_mine = clip_grad_norm_(model.parameters(), max_norm=0.1)
    grads_mine = [p.grad.clone() for p in model.parameters()]

    # torch 参考（重新算一遍梯度）
    model.zero_grad()
    ((model(x) - y) ** 2).mean().backward()
    norm_ref = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    grads_ref = [p.grad.clone() for p in model.parameters()]

    err_norm = abs(norm_mine.item() - norm_ref.item())
    err_grad = max((a - b).abs().max().item() for a, b in zip(grads_mine, grads_ref))
    print(f"  裁剪前总范数: ours={norm_mine.item():.4f}  torch={norm_ref.item():.4f} 差 {err_norm:.2e}")
    print(f"  裁剪后梯度最大误差 {err_grad:.2e}")
    assert err_norm < 1e-6 and err_grad < 1e-6, "应与 torch 完全一致"
    print("  ✅ 与 torch.nn.utils.clip_grad_norm_ 完全一致")


if __name__ == "__main__":
    print("=" * 60)
    print("梯度裁剪演示")
    print("=" * 60)

    # 构造一个梯度范数很大的场景
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(), torch.nn.Linear(8, 4))
    x, y = torch.randn(5, 6), torch.randn(5, 4)
    ((model(x) - y) ** 2).mean().backward()

    before = sum(g.grad.norm().item() ** 2 for g in model.parameters()) ** 0.5
    total_norm = clip_grad_norm_(model.parameters(), max_norm=0.1)
    after = sum(g.grad.norm().item() ** 2 for g in model.parameters()) ** 0.5
    print(f"  裁剪前总范数: {before:.4f}")
    print(f"  裁剪后总范数: {after:.4f} (≤ max_norm=0.1)")
    print(f"  缩放系数:    {0.1 / before:.4f}  ← 所有梯度统一 × 该系数, 方向不变")

    print("\n" + "=" * 60)
    print("数值对齐验证")
    print("=" * 60)
    _verify_against_torch()
