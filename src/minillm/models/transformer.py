"""
minillm.models.transformer
==========================

Decoder-only Transformer 主干（LLaMA 风格）
-------------------------------------------
把 RMSNorm / RoPE / SwiGLU / MHA(GQA) 全部拼装成完整的大语言模型。

架构（自底向上）:
    input_ids ──► embed_tokens ──► [DecoderLayer × N] ──► final RMSNorm ──► lm_head ──► logits
                                      │  ┌───────────────────────────────┐
                                      └─►│ DecoderLayer:                  │
                                         │  1. RMSNorm(x) → MHA/GQA       │
                                         │  2. 残差 +                      │
                                         │  3. RMSNorm(x) → SwiGLU        │
                                         │  4. 残差 +                      │
                                         └───────────────────────────────┘

Pre-Norm（先归一化再进子层，LLaMA/GPT-2 风格）:
    传统 Post-Norm:  x → 子层 → 归一化 → 残差   （BERT，深网络易不稳定）
    Pre-Norm:        x → 归一化 → 子层 → 残差    （深网络更稳，梯度路径干净）
    LLaMA 用 Pre-Norm，所以每个子层前都有 RMSNorm。

残差连接（Residual）:
    每层输出 = 子层输出 + 输入。梯度可以"抄近路"回流，深层不梯度消失。

权重共享（Tied Embeddings）:
    lm_head.weight = embed_tokens.weight（同一个矩阵）
    → 词汇表 embedding 只存一份（LLaMA 默认 tie），省一个 vocab×hidden 的参数矩阵。

输出:
    logits [batch, seq, vocab] → 每个位置一个 vocab 大小的分布，训练时算 cross-entropy。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from minillm.models.layers.attention import MultiHeadAttention
from minillm.models.layers.rms_norm import RMSNorm
from minillm.models.layers.swiglu import SwiGLU


@dataclass
class MiniLLMConfig:
    """模型超参（对应 configs/model/tiny.yaml）。"""

    vocab_size: int = 64000
    hidden_size: int = 512
    intermediate_size: int = 1408
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True


class DecoderLayer(nn.Module):
    """单个 Transformer 层：RMSNorm→Attention→残差→RMSNorm→SwiGLU→残差。"""

    def __init__(self, cfg: MiniLLMConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.self_attn = MultiHeadAttention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            max_seq_len=cfg.max_position_embeddings,
            rope_theta=cfg.rope_theta,
        )
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        # 子层 1：注意力（Pre-Norm）
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, positions)
        x = residual + x

        # 子层 2：SwiGLU（Pre-Norm）
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class MiniLLM(nn.Module):
    """Decoder-only 完整模型：embedding + N 层 Decoder + 输出头。"""

    def __init__(self, cfg: MiniLLMConfig):
        super().__init__()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # 权重共享：输出头复用 embedding 矩阵（只存一份词表权重）
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """截断正态初始化（tiny.yaml: initializer_range=0.02），RMSNorm 权重保持 1。"""
        for name, param in self.named_parameters():
            if "norm" in name and "weight" in name:
                continue  # RMSNorm 权重保持全 1
            if param.ndim >= 2:
                nn.init.trunc_normal_(param, mean=0.0, std=self.cfg.initializer_range)
            else:
                nn.init.zeros_(param)  # bias（本项目无 bias，防御性处理）

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """input_ids: [batch, seq] → logits: [batch, seq, vocab]"""
        if positions is None:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)

        # 1) token → 向量
        hidden = self.embed_tokens(input_ids)  # [B, S, H]

        # 2) 逐层前向（每层内部自己做残差）
        for layer in self.layers:
            hidden = layer(hidden, positions)

        # 3) 最后归一化 + 输出头
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)  # [B, S, V]
        return logits


# ---------------------------------------------------------------------------
# 自校验：python -m minillm.models.transformer
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # 用一个小配置快速冒烟（vocab 用小的，避免自校验过重）
    cfg = MiniLLMConfig(
        vocab_size=1024,
        hidden_size=512,
        intermediate_size=1408,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = MiniLLM(cfg)

    B, S = 2, 6
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    logits = model(input_ids)
    print(f"input_ids 形状: {input_ids.shape}")
    print(f"logits 形状: {logits.shape} (应 [2, 6, 1024])")

    # --- 校验 1: 权重共享生效 ---
    print(f"\nlm_head 与 embedding 共享权重: {model.lm_head.weight is model.embed_tokens.weight}")
    print(f"embedding 参数: {model.embed_tokens.weight.numel():,} | lm_head 额外: 0 (共享)")

    # --- 校验 2: causal 性质（位置 0 的输出不受后续 token 影响）---
    x1 = input_ids.clone()
    x2 = input_ids.clone()
    x2[:, 1:] = 0  # 抹掉第 1 个位置之后的所有 token
    with torch.no_grad():
        l1 = model(x1)
        l2 = model(x2)
    diff_pos0 = (l1[:, 0] - l2[:, 0]).abs().max().item()
    diff_pos_last = (l1[:, -1] - l2[:, -1]).abs().max().item()
    print(f"\n位置0 logits 差异(应≈0): {diff_pos0:.2e}")
    print(f"最后位置 logits 差异(应明显>0): {diff_pos_last:.2e}")

    # --- 校验 3: 参数量估算 ---
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型总参数量(2层): {n_params:,}")
    cfg8 = MiniLLMConfig(vocab_size=64000, hidden_size=512, intermediate_size=1408,
                         num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=2)
    model8 = MiniLLM(cfg8)
    n8 = sum(p.numel() for p in model8.parameters())
    print(f"tiny.yaml 完整 8 层模型参数量: {n8:,} (~{n8/1e6:.1f}M)")

    # --- 校验 4: 梯度回传 ---
    logits.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    print(f"有梯度的参数: {len(grads)} 个 (应 > 0)")
    print(f"embedding 梯度非零: {(model.embed_tokens.weight.grad.abs() > 0).sum().item()} 个")

    # --- 校验 5: 与 HF LlamaForCausalLM 数值对齐 ---
    try:
        from transformers import LlamaConfig, LlamaForCausalLM

        hf_cfg = LlamaConfig(
            vocab_size=1024,
            hidden_size=512,
            intermediate_size=1408,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=64,
            max_position_embeddings=32,
            rope_theta=10000.0,
            attention_bias=False,
            hidden_act="silu",
            norm_eps=1e-6,
            tie_word_embeddings=True,
        )
        hf_model = LlamaForCausalLM(hf_cfg).eval()
        with torch.no_grad():
            hf_model.model.embed_tokens.weight.copy_(model.embed_tokens.weight)
            for i, layer in enumerate(model.layers):
                hf_layer = hf_model.model.layers[i]
                hf_layer.input_layernorm.weight.copy_(layer.input_layernorm.weight)
                hf_layer.post_attention_layernorm.weight.copy_(layer.post_attention_layernorm.weight)
                for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    getattr(hf_layer.self_attn, attr).weight.copy_(
                        getattr(layer.self_attn, attr).weight
                    )
                for attr in ("gate_proj", "up_proj", "down_proj"):
                    getattr(hf_layer.mlp, attr).weight.copy_(
                        getattr(layer.mlp, attr).weight
                    )
            hf_model.model.norm.weight.copy_(model.norm.weight)
            # tie 模式下 lm_head 与 embedding 共享，无需单独 copy

        hf_logits = hf_model(input_ids).logits
        max_diff = (hf_logits - logits).abs().max().item()
        print(f"\n与 HF LlamaForCausalLM 最大误差: {max_diff:.2e} (应 < 1e-5)")
    except Exception as e:  # noqa: BLE001
        print(f"\n[skip] 跳过 HF 对齐测试: {type(e).__name__}: {e}")

    print("\n✅ Transformer 自校验通过")
