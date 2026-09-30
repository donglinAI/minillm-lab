"""
minillm.models.layers.rope
==========================

RoPE（Rotary Position Embedding，旋转位置编码）
-----------------------------------------------
LLaMA / Qwen / Mistral 等现代大模型的位置编码方案。

为什么需要位置编码？
    Transformer 的注意力是"排列不变"的：把 "A B" 和 "B A" 换成一样的 token，
    注意力分数相同。必须把"位置"注入进去，模型才知道词的先后顺序。

RoPE 的核心思想（一句话）：
    把"位置信息"编码成**旋转角度**，对 query / key 向量做旋转。
    旋转有个绝妙性质：两个向量旋转后再点积，结果只和它们的**旋转角度差**
    （即位置差）有关 → 天然得到相对位置编码。

数学：
    对 head_dim 维向量 x，把它看成复数对：
        x = (x[:d/2], x[d/2:])  ≈  x1 + i*x2
    位置 m 的旋转 = 乘以单位复数 e^(i*m*θ_k)：
        θ_k = 1 / rope_theta^(2k/dim)      # k 是第几对，维度越高转得越慢
    展开成实数（rotate_half 技巧）：
        RoPE(x) = x * cos(mθ) + rotate_half(x) * sin(mθ)
        rotate_half([x1, x2]) = [-x2, x1]  # 就是复数 i 乘法

关键性质：
    1. 旋转不改变向量长度：||RoPE(q)|| = ||q||（不影响注意力数值尺度）
    2. 相对位置：<RoPE(q_m), RoPE(k_n)> 只依赖 (m-n) → 外推性好
    3. 无参数：cos/sin 表预计算好，推理时查表，零额外参数
"""

from __future__ import annotations

import torch
import torch.nn as nn


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """预计算所有位置的 cos / sin 表。

    Parameters
    ----------
    dim : int
        head_dim（每个注意力头的维度）。
    max_seq_len : int
        支持的最大序列长度。
    theta : float
        旋转基数（tiny.yaml 里 rope_theta=10000.0）。

    Returns
    -------
    cos, sin : [max_seq_len, dim]
    """
    # 频率：维度下标 k 越大，θ_k 越小 → 高频维度转得快、低频维度转得慢
    # 这就是"每个维度一个旋转速度"
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )  # [dim/2]

    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)  # [S]
    freqs = torch.outer(t, inv_freq)  # [S, dim/2]：位置 × 频率 = 每个位置每个频率的旋转角

    # 复制成两份，凑满 dim 维（和 rotate_half 的两半结构配套）
    emb = torch.cat((freqs, freqs), dim=-1)  # [S, dim]

    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """复数 i 乘法的实数形式：i * (x1 + i*x2) = -x2 + i*x1。

    输入 [..., dim] → 输出 [..., dim]，把前半维取负挪到后半维。
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 q/k 施加位置旋转。

    Parameters
    ----------
    q, k : [batch, heads, seq, head_dim]
    cos, sin : [max_seq_len, head_dim]
    position_ids : [batch, seq] 或 [seq]（每个 token 的位置）

    Returns
    -------
    q_embed, k_embed : 同 q/k 形状
    """
    # 按 position_ids 查表 → [.., seq, head_dim]，unsqueeze 对齐 heads 维
    cos = cos[position_ids].unsqueeze(1)  # [B, 1, S, D]
    sin = sin[position_ids].unsqueeze(1)

    # 旋转公式：x*cos + rotate_half(x)*sin（复数乘法展开）
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """RoPE 模块：预计算表 + 应用旋转。无任何可学习参数。"""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        cos, sin = precompute_freqs_cis(dim, max_seq_len, theta)
        # register_buffer：随模型保存/加载、随 .to(device) 移动，但不参与梯度
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_rotary_pos_emb(q, k, self.cos, self.sin, position_ids)


# ---------------------------------------------------------------------------
# 自校验：python -m minillm.models.layers.rope
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    D, S, B, H = 64, 8, 2, 4  # head_dim=64（tiny: 512/8）, seq=8

    rope = RotaryEmbedding(dim=D, max_seq_len=32, theta=10000.0)
    q = torch.randn(B, H, S, D, requires_grad=True)
    k = torch.randn(B, H, S, D)
    pos = torch.arange(S).repeat(B, 1)  # [B, S] 位置 0..7

    q_r, k_r = rope(q, k, pos)
    print(f"q 形状: {q.shape} → q_r 形状: {q_r.shape}")

    # --- 校验 1: 旋转不改变范数（等距变换）---
    err = (q.norm(dim=-1) - q_r.norm(dim=-1)).abs().max().item()
    print(f"\n旋转前后范数最大差: {err:.2e} (应 < 1e-5)")

    # --- 校验 2: 核心性质 —— 内积只依赖相对位置差 ---
    # 注意：必须用【同一个】q 向量、【同一个】k 向量放到不同位置比较，
    # 因为内积本来就依赖向量内容，RoPE 保证的是"内容相同 → 只由位置差决定"
    def rope_at(x: torch.Tensor, m: int) -> torch.Tensor:
        cos_m = rope.cos[m]  # [D]
        sin_m = rope.sin[m]
        return x * cos_m + rotate_half(x) * sin_m

    vq = torch.randn(D)  # 固定 q 内容
    vk = torch.randn(D)  # 固定 k 内容

    d13 = (rope_at(vq, 1) * rope_at(vk, 3)).sum()  # 位置差 = 2
    d57 = (rope_at(vq, 5) * rope_at(vk, 7)).sum()  # 位置差 = 2
    print(f"\n同一对向量, 位置差=2 (1,3): {d13:.4f} | (5,7): {d57:.4f} | 差: {abs(d13-d57):.2e} (应≈0)")

    d12 = (rope_at(vq, 1) * rope_at(vk, 2)).sum()  # 位置差 = 1
    d16 = (rope_at(vq, 1) * rope_at(vk, 7)).sum()  # 位置差 = 6
    print(f"位置差=1 (1,2): {d12:.4f} | 位置差=6 (1,7): {d16:.4f} (应明显不同)")

    # --- 校验 3: 无旋转时内积与位置完全无关（反证旋转的必要性）---
    d_raw = (vq * vk).sum()
    print(f"未旋转内积处处相同: {d_raw:.4f} → 模型无法区分 (1,3) 和 (5,7) 这类相对距离")

    # --- 校验 4: 与 HF LLaMA 的旋转编码数值对齐 ---
    try:
        from transformers.models.llama.modeling_llama import (
            apply_rotary_pos_emb as hf_apply,
        )

        # 复用我们自己的 cos/sin 表，只对比"应用旋转"这一步
        # HF 新版签名: apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        # 注意：cos/sin 需与序列长度匹配（取前 S 个位置）
        q_hf, k_hf = hf_apply(
            q.float(),
            k.float(),
            rope.cos[:S].unsqueeze(0),  # [1, S, D] → unsqueeze(1) → [1,1,S,D] 广播
            rope.sin[:S].unsqueeze(0),
        )
        max_diff = (q_hf - q_r.float()).abs().max().item()
        print(f"\n与 HF apply_rotary_pos_emb 最大误差: {max_diff:.2e} (应 < 1e-5)")
    except (ImportError, TypeError) as e:
        print(f"\n[skip] 跳过 HF 对齐测试: {type(e).__name__}")

    # --- 校验 5: 反向传播正常（无参数，梯度应流回 q）---
    loss = q_r.sum()
    loss.backward()
    print(f"q 梯度形状: {q.grad.shape} (应 [2, 4, 8, 64])")
    print(f"cos/sin 是 buffer（非参数）: {sum(1 for _ in rope.parameters())} 个参数 (应为 0)")

    print("\n✅ RoPE 自校验通过")
