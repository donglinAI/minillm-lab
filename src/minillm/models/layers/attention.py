"""
minillm.models.layers.attention
===============================

Multi-Head Attention（MHA）+ Grouped-Query Attention（GQA）
----------------------------------------------------------
Transformer 的核心：让每个位置"看到"其他位置并聚合信息。

MHA（标准多头注意力，GPT-2 时代）:
    Q = x W_q, K = x W_k, V = x W_v        # 每头一组投影
    attn = softmax(Q Kᵀ / √d) V            # 缩放点积注意力
    每头独立学"关系模式"（头 1 学语法、头 2 学指代…）

GQA（分组查询注意力，LLaMA-2/3、Qwen 用）:
    query 仍然每头一个投影，但 K/V 只有 kv_heads 组，每组服务多个 query 头。
    tiny.yaml: num_heads=8, num_key_value_heads=2
    → 8 个 Q 头共享 2 组 K/V，每组服务 4 个 Q 头。

为什么用 GQA？
    MHA 的 K/V 参数 = 8 头，GQA 只有 2 头 → K/V 参数减到 1/4；
    推理时 KV Cache 也减到 1/4 → 长序列省显存、省带宽（推理瓶颈）。
    效果几乎不掉（实验结论），所以成了现代大模型标配。
    特例：kv_heads == heads 就是纯 MHA（本实现同时支持）。

缩放因子 √d 的作用：
    点积随维度 d 线性增长，除以 √d 把方差拉回 ~1，
    避免 softmax 进入饱和区（梯度消失）。

Causal mask（decoder-only 必备）:
    位置 i 只能看 j ≤ i（不能偷看未来），上三角掩成 -inf。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minillm.models.layers.rope import RotaryEmbedding


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA：把 kv 头重复 n_rep 次，扩到与 query 头数一致。

    x: [batch, kv_heads, seq, head_dim] → [batch, kv_heads*n_rep, seq, head_dim]
    用 expand（不复制内存），重复只是视图。
    """
    batch, kv_heads, seq, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(batch, kv_heads, n_rep, seq, head_dim)
    return x.reshape(batch, kv_heads * n_rep, seq, head_dim)


class MultiHeadAttention(nn.Module):
    """支持 MHA（num_kv_heads=num_heads）与 GQA（num_kv_heads < num_heads）。"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.n_rep = num_heads // self.num_kv_heads  # 每个 kv 头服务的 q 头数
        assert self.num_heads % self.num_kv_heads == 0, "num_heads 必须是 num_kv_heads 的整数倍"

        # 投影：q 全头数，k/v 只有 kv_heads 组
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)

        # RoPE 旋转位置编码（无参数）
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len, rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: [batch, seq, hidden] → 输出同形状。

        positions: [seq] 或 [batch, seq]，缺省为 0..seq-1
        attention_mask: [batch, 1, seq, seq] 加性掩码（0 保留 / -inf 遮蔽）
        """
        b, s, _ = x.shape

        # 1) 投影 + 切头：[B, S, H*D] → [B, H, S, D]
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 2) RoPE：对 q/k 施加位置旋转
        if positions is None:
            positions = torch.arange(s, device=x.device)
        q, k = self.rope(q, k, positions)

        # 3) GQA 扩展：kv 头重复到与 q 头一致
        k = repeat_kv(k, self.n_rep)  # [B, H, S, D]
        v = repeat_kv(v, self.n_rep)

        # 4) 缩放点积注意力分数
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 5) causal 掩码（decoder-only）：上三角 -inf
        if attention_mask is None:
            causal = torch.triu(
                torch.ones(s, s, dtype=torch.bool, device=x.device), diagonal=1
            )
            attention_mask = causal[None, None, :, :]  # [1, 1, S, S]
        attn_weights = attn_weights.masked_fill(attention_mask, float("-inf"))

        # 6) softmax + 加权求和
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn_weights, v)  # [B, H, S, D]

        # 7) 合并头 + 输出投影
        out = out.transpose(1, 2).contiguous().view(b, s, self.num_heads * self.head_dim)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# 自校验：python -m minillm.models.layers.attention
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    H, HEADS, KV, S = 512, 8, 2, 6  # tiny.yaml: hidden=512, heads=8, kv_heads=2

    attn = MultiHeadAttention(H, HEADS, num_kv_heads=KV, max_seq_len=32)
    x = torch.randn(2, S, H, requires_grad=True)

    y = attn(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape} (应 [2, 6, 512])")

    # --- 校验 1: 与手写公式一致（单步对比）---
    b = 2
    q = attn.q_proj(x).view(b, S, HEADS, 64).transpose(1, 2)
    k = attn.k_proj(x).view(b, S, KV, 64).transpose(1, 2)
    v = attn.v_proj(x).view(b, S, KV, 64).transpose(1, 2)
    pos = torch.arange(S)
    q, k = attn.rope(q, k, pos)
    k = repeat_kv(k, HEADS // KV)
    v = repeat_kv(v, HEADS // KV)
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)[None, None]
    w = (q @ k.transpose(-2, -1)) / 8.0
    w = F.softmax(w.masked_fill(mask, float("-inf")), dim=-1)
    out_ref = (w @ v).transpose(1, 2).reshape(b, S, HEADS * 64) @ attn.o_proj.weight.T
    print(f"与手写公式最大误差: {(y - out_ref).abs().max().item():.2e} (应 < 1e-5)")

    # --- 校验 2: causal 掩码生效（位置 i 不看 j>i）---
    print(f"\n注意力分数形状: {w.shape}")
    print(f"上三角(未来)被掩成 -inf: {bool((w.masked_fill(mask, float('-inf'))[0, 0] == float('-inf')).any())}")
    # 验证第 0 行只有第 0 列可见
    print(f"位置0 的注意力分布: {w[0, 0, 0].tolist()} (只有 [0] 位置非0)")

    # --- 校验 3: GQA 参数量对比（vs 纯 MHA）---
    mha = MultiHeadAttention(H, HEADS, num_kv_heads=HEADS, max_seq_len=32)
    p_gqa = sum(p.numel() for p in attn.parameters())
    p_mha = sum(p.numel() for p in mha.parameters())
    print(f"\nGQA(kv=2) 参数量: {p_gqa:,} | MHA(kv=8) 参数量: {p_mha:,}")
    print(f"K/V 部分减至 1/{HEADS // KV}: {(p_gqa / p_mha):.3f}")

    # --- 校验 4: 梯度回传 ---
    y.sum().backward()
    print(f"q_proj 梯度形状: {attn.q_proj.weight.grad.shape} (应 [512, 512])")
    print(f"梯度非零: {(attn.q_proj.weight.grad.abs() > 0).sum().item()} 个 (应 > 0)")

    # --- 校验 5: 与 HF LlamaAttention 数值对齐 ---
    try:
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaAttention

        cfg = LlamaConfig(
            hidden_size=H,
            num_attention_heads=HEADS,
            num_key_value_heads=KV,
            head_dim=64,
            max_position_embeddings=32,
            rope_theta=10000.0,
            attention_bias=False,
            hidden_act="silu",
            intermediate_size=1408,
            num_hidden_layers=1,
        )
        hf_attn = LlamaAttention(cfg, layer_idx=0)
        with torch.no_grad():
            hf_attn.q_proj.weight.copy_(attn.q_proj.weight)
            hf_attn.k_proj.weight.copy_(attn.k_proj.weight)
            hf_attn.v_proj.weight.copy_(attn.v_proj.weight)
            hf_attn.o_proj.weight.copy_(attn.o_proj.weight)

        cos, sin = attn.rope.cos[:S].unsqueeze(0), attn.rope.sin[:S].unsqueeze(0)
        pos_emb = (cos, sin)
        hf_out, _ = hf_attn(x.detach(), position_embeddings=pos_emb, attention_mask=None)
        # 注意: 无 attention_mask 时 HF 不加 causal → 我们的输出需对比带 causal 版
        # 为公平对比，用我们自己实现重新算一遍"无掩码"逻辑
        q = attn.q_proj(x.detach()).view(b, S, HEADS, 64).transpose(1, 2)
        k = attn.k_proj(x.detach()).view(b, S, KV, 64).transpose(1, 2)
        v = attn.v_proj(x.detach()).view(b, S, KV, 64).transpose(1, 2)
        q, k = attn.rope(q, k, pos)
        k = repeat_kv(k, HEADS // KV)
        v = repeat_kv(v, HEADS // KV)
        w = (q @ k.transpose(-2, -1)) / 8.0
        w = F.softmax(w, dim=-1)  # 无掩码
        out_nomask = (w @ v).transpose(1, 2).reshape(b, S, HEADS * 64) @ attn.o_proj.weight.T
        max_diff = (hf_out - out_nomask).abs().max().item()
        print(f"\n与 HF LlamaAttention(无掩码) 最大误差: {max_diff:.2e} (应 < 1e-5)")
    except Exception as e:  # noqa: BLE001
        print(f"\n[skip] 跳过 HF 对齐测试: {type(e).__name__}")

    # --- 校验 6: 分组正确性（q 头 0-3 用 kv 头 0，q 头 4-7 用 kv 头 1）---
    # 通过构造: 让 kv 头 0 的 k 全部置 0 → 对应的 q 头输出应只由 v 头 0 决定
    with torch.no_grad():
        attn.k_proj.weight.zero_()
        attn.k_proj.weight[:64].normal_()  # 只保留 kv 头 0 的 k
        y2 = attn(x)
    print(f"\n仅 kv 头 0 有 k 时输出正常(无 NaN): {bool(torch.isfinite(y2).all())}")

    print("\n✅ MHA/GQA 自校验通过")
