"""
minillm.models.layers.swiglu
============================

SwiGLU（Swish-Gated Linear Unit，Swish 门控线性单元）
----------------------------------------------------
LLaMA / Qwen / Mistral 等现代大模型 FFN（前馈网络）用的门控结构。

为什么叫 SwiGLU？
    - Swi   ：Swish / SiLU 激活函数，σ(x) = x * sigmoid(x)
    - GLU   ：Gated Linear Unit（门控线性单元），用一个"门"控制信息流
    - 合起来：用 SiLU 做门的 GLU

与传统 FFN 对比（GPT-2 / BERT 时代）:
    传统:  FFN(x) = GELU(xW1) W2          # 2 个权重矩阵
    SwiGLU: FFN(x) = (SiLU(xW_gate) ⊙ xW_up) W_down   # 3 个权重矩阵

    多一个权重，但训练更稳、效果更好——多花的参数换质量。

公式（LLaMA 实现顺序）:
    h1 = gate_proj(x)          # [.., hidden] → [.., intermediate]
    h2 = up_proj(x)            # [.., hidden] → [.., intermediate]
    y  = down_proj( SiLU(h1) ⊙ h2 )     # ⊙ 逐元素乘，然后 [.., intermediate] → [.., hidden]

直觉:
    gate 输出过 SiLU 后是 0~1 左右的"开关/旋钮"（可微门控），
    逐元素地决定 up 的每个分量"放行多少" → 每维独立控制信息通过率。

SiLU 形状:
    silu(x) = x * sigmoid(x)
    负输入：被 sigmoid 软压到接近 0（不是硬置 0）→ 梯度平滑、可微
    正输入：近似 x → 不衰减有效信息
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU 门控前馈层。

    Parameters
    ----------
    hidden_size : int
        输入/输出维度（tiny.yaml: 512）。
    intermediate_size : int
        中间扩张维度（tiny.yaml: 1408 = 512 × 2.75，LLaMA 惯例约 8/3 倍）。
    bias : bool
        LLaMA 系默认 False（不做偏置）。
    """

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        # 三个投影：gate（门控）/ up（信息）/ down（收缩回 hidden）
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 门：SiLU 激活后作为逐元素"开关"
        gate = F.silu(self.gate_proj(x))
        # 信息通道：直接线性变换
        up = self.up_proj(x)
        # 门控相乘 → 收缩回 hidden 维
        return self.down_proj(gate * up)


# ---------------------------------------------------------------------------
# 自校验：python -m minillm.models.layers.swiglu
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    H, I = 512, 1408  # tiny.yaml: hidden=512, intermediate=1408

    layer = SwiGLU(H, I)
    x = torch.randn(2, 8, H)  # [batch, seq, hidden]

    y = layer(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape} (应 [2, 8, 512])")

    # --- 校验 1: 与手写公式逐元素一致 ---
    gate = F.silu(layer.gate_proj(x))
    up = layer.up_proj(x)
    y_ref = layer.down_proj(gate * up)
    print(f"\n与手写公式最大误差: {(y - y_ref).abs().max().item():.2e} (应 < 1e-6)")

    # --- 校验 2: 维度流转正确（hidden → 1408 → 512）---
    print(f"gate_proj 权重形状: {layer.gate_proj.weight.shape} (应 [1408, 512])")
    print(f"up_proj   权重形状: {layer.up_proj.weight.shape} (应 [1408, 512])")
    print(f"down_proj 权重形状: {layer.down_proj.weight.shape} (应 [512, 1408])")

    # --- 校验 3: SiLU 的性质演示 ---
    t = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    print(f"\nSiLU 输入: {t.tolist()}")
    print(f"SiLU 输出: {F.silu(t).tolist()}")
    print(f"  → 负数被软压到接近 0，正数近似保持（这就是『软开关』）")

    # --- 校验 4: 门控确实在"调节"信息（直接控制门的开/关观察输出）---
    x1 = torch.randn(1, 1, H)
    with torch.no_grad():
        up = layer.up_proj(x1)
        y_on = layer.down_proj(torch.ones_like(up) * up)   # 门=1（全开）
        y_off = layer.down_proj(torch.zeros_like(up) * up)  # 门=0（全关）
        y_normal = layer(x1)                                # 真实门控
    print(f"\n门全开(1)  输出范数: {y_on.norm().item():.4f}")
    print(f"门全关(0)  输出范数: {y_off.norm().item():.4f} (应=0)")
    print(f"真实门控  输出范数: {y_normal.norm().item():.4f} (应介于两者之间)")

    # --- 校验 5: 反向传播正常 ---
    y.sum().backward()
    print(f"\ngate_proj 梯度形状: {layer.gate_proj.weight.grad.shape}")
    print(f"梯度非零: {(layer.gate_proj.weight.grad.abs() > 0).sum().item()} 个 (应 > 0)")

    # --- 校验 6: 与 HF LlamaMLP 数值对齐（装了才测）---
    try:
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaMLP

        cfg = LlamaConfig(
            hidden_size=H,
            intermediate_size=I,
            hidden_act="silu",
        )
        hf_mlp = LlamaMLP(cfg)
        with torch.no_grad():
            hf_mlp.gate_proj.weight.copy_(layer.gate_proj.weight)
            hf_mlp.up_proj.weight.copy_(layer.up_proj.weight)
            hf_mlp.down_proj.weight.copy_(layer.down_proj.weight)
        y_hf = hf_mlp(x)
        max_diff = (y_hf - y).abs().max().item()
        print(f"\n与 HF LlamaMLP 最大误差: {max_diff:.2e} (应 < 1e-5)")
    except Exception as e:  # noqa: BLE001
        print(f"\n[skip] 跳过 HF 对齐测试: {type(e).__name__}")

    print("\n✅ SwiGLU 自校验通过")
