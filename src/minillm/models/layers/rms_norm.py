"""
minillm.models.layers.rms_norm
==============================

RMSNorm（Root Mean Square Layer Normalization）
-----------------------------------------------
LLaMA / Qwen / Mistral 等现代大模型用的归一化层，取代传统 LayerNorm。

为什么用 RMSNorm 而不是 LayerNorm？
- LayerNorm 做两件事：减均值（center）+ 除标准差（scale）
- RMSNorm 只做第二件：除以均方根，不减均值
- 好处：省一次均值计算（少一次全局归约，分布式下更省通信），训练更稳定
- 实验表明：对 Transformer 而言，减均值那步贡献很小，去掉不影响效果

公式（对最后一维 dim 计算）:
    RMS(x) = sqrt( mean(x^2) + eps )
    y      = x / RMS(x) * weight          # weight 是逐维可学习缩放，初始全 1

形状:
    x: [..., dim]  →  y: [..., dim]       # 完全按元素操作，不改变形状

对比 LayerNorm:
    LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias     # 有 bias
    RMSNorm:   y = x / sqrt(mean(x^2) + eps) * weight               # 无 bias
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMS 归一化层。

    Parameters
    ----------
    dim : int
        归一化的维度大小（通常是 hidden_size）。
    eps : float
        分母保护项，防止除零（tiny.yaml 里 norm_eps=1e-6）。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 可学习的逐维缩放，初始化为 1（等价于"什么都不做"的起点）
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]，在最后一维做归一化

        # 1) 逐元素平方 → 最后一维求均值 → 加 eps → 开方 = 均方根
        #    keepdim=True 保住维度，方便广播相除
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()

        # 2) 归一化：除以均方根（scale 到单位 RMS）
        x_norm = x / rms

        # 3) 乘上可学习缩放（affine 变换）
        return x_norm * self.weight


# ---------------------------------------------------------------------------
# 自校验：python -m minillm.models.layers.rms_norm
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    dim = 512

    norm = RMSNorm(dim)
    x = torch.randn(2, 8, dim)          # [batch, seq, hidden]

    y = norm(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")

    # --- 校验 1: 输出行的 RMS 应该 ≈ 1（weight=1 时归一化到单位 RMS）---
    rms_out = y.pow(2).mean(dim=-1).sqrt()
    print(f"\n输出 RMS 均值: {rms_out.mean().item():.6f} (应接近 1.0)")

    # --- 校验 2: 与手写公式逐元素一致 ---
    rms_ref = x.pow(2).mean(dim=-1, keepdim=True).add(norm.eps).sqrt()
    y_ref = x / rms_ref * norm.weight
    max_diff = (y - y_ref).abs().max().item()
    print(f"与手写公式最大误差: {max_diff:.2e} (应 < 1e-6)")

    # --- 校验 3: 与 LayerNorm 的关系（只差均值项）---
    # LayerNorm 输出 = (x - mean)/sqrt(var) * w；RMSNorm 无均值项，
    # 用 F.normalize 对比：F.normalize 除 L2 范数，RMSNorm 除 RMS = L2/sqrt(dim)
    import torch.nn.functional as F

    y_l2 = F.normalize(x, p=2, dim=-1)  # 每行 L2 范数为 1
    # RMSNorm 等价于 L2 归一化后乘 sqrt(dim)（再乘 weight=1）
    y_equiv = y_l2 * (dim ** 0.5)
    max_diff2 = (y_equiv - y).abs().max().item()
    print(f"与 L2 归一化等价式最大误差: {max_diff2:.2e} (应 < 1e-5)")

    # --- 校验 4: weight 可学习（改动后输出随之缩放）---
    with torch.no_grad():
        norm.weight.fill_(2.0)
    y2 = norm(x)
    print(f"\nweight=1 时输出均值: {y.mean().item():.4f}")
    print(f"weight=2 时输出均值: {y2.mean().item():.4f} (约 2 倍)")
    with torch.no_grad():
        norm.weight.fill_(1.0)

    # --- 校验 5: 梯度能回传（backward 不报错）---
    loss = y.sum()
    loss.backward()
    print(f"\nweight 梯度形状: {norm.weight.grad.shape} (应为 [512])")
    print(f"weight 梯度非零: {(norm.weight.grad.abs() > 0).sum().item()} 个 (应 > 0)")

    # --- 校验 6: 与 HF transformers LlamaRMSNorm 数值对齐（装了才测）---
    try:
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        hf_norm = LlamaRMSNorm(dim, eps=1e-6).to(x.dtype)
        with torch.no_grad():
            hf_norm.weight.copy_(norm.weight)
        y_hf = hf_norm(x)
        max_diff3 = (y_hf - y).abs().max().item()
        print(f"\n与 HF LlamaRMSNorm 最大误差: {max_diff3:.2e} (应 < 1e-5)")
    except ImportError:
        print("\n[skip] 未安装 transformers，跳过 HF 对齐测试")

    print("\n✅ RMSNorm 自校验通过")
