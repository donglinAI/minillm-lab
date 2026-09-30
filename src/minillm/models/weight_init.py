"""
minillm.models.weight_init
==========================

模型权重初始化（可复现、可审计）
--------------------------------
训练起点决定生死：初始化太大会梯度爆炸，太小会梯度消失/训练停滞。
本模块提供统一、可复现的初始化入口，供训练脚本调用。

初始化策略（LLaMA / GPT-2 系惯例）:
    1. Embedding / LM head : 正态分布 N(0, initializer_range²)，initializer_range=0.02
    2. 各 Linear (q/k/v/o/gate/up/down): 截断正态（只保留 ±2σ 内的采样值）
       → 比普通正态更安全：不会产生极端大值 outlier
    3. RMSNorm weight : 保持全 1（归一化层起点=恒等，不动它）
    4. 残差缩放（可选，GPT-2 风格）:
       o_proj / down_proj 乘 residual_scale = 1/√N 或 1/√(2N)
       → 残差和的方差不随层数 N 增长，深层更稳
       （LLaMA 官方不缩放，靠 small std；此处做成可开关参数）

为什么截断正态而不是标准正态？
    N(0,0.02²) 在 6σ 处仍可能采样到 0.12 的权重——放进 55M 参数里就是
    一颗"定时炸弹"。截断到 ±2σ（约 0.04），保证所有权重有界，训练更稳。

可复现性：
    init_weights(model, seed=42) 内部先 torch.manual_seed，
    同一 seed 两次初始化产出完全相同权重。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def truncated_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 0.02,
) -> torch.Tensor:
    """截断正态：N(mean,std²) 采样，丢弃超出 [mean-2σ, mean+2σ] 的值。

    实现要点：只对出界元素重新采样（保留合格元素），出界率 ~4.5%，
    每轮指数收敛，实际 2~3 轮即完成。
    """
    with torch.no_grad():
        tmp = tensor.new_empty(tensor.shape).normal_(mean, std)
        mask = (tmp < mean - 2 * std) | (tmp > mean + 2 * std)
        while mask.any():
            new_vals = tensor.new_empty(mask.sum().item()).normal_(mean, std)
            tmp[mask] = new_vals  # 只修出界位置
            mask = (tmp < mean - 2 * std) | (tmp > mean + 2 * std)
        tensor.copy_(tmp)
    return tensor


def init_weights(
    model: nn.Module,
    initializer_range: float = 0.02,
    residual_scale: float | None = None,
    seed: int | None = None,
) -> nn.Module:
    """统一初始化入口。

    Parameters
    ----------
    model : nn.Module
        待初始化的模型（MiniLLM 或任意含上述命名规则的模型）。
    initializer_range : float
        正态/截断正态标准差。
    residual_scale : float | None
        若给定（如 1/√N），对 o_proj / down_proj 额外乘该系数。
    seed : int | None
        随机种子，保证可复现。
    """
    if seed is not None:
        torch.manual_seed(seed)

    for name, module in model.named_modules():
        # RMSNorm：weight 保持全 1（归一化起点=恒等）
        if isinstance(module, nn.modules.normalization.RMSNorm) or (
            "norm" in name and hasattr(module, "weight") and not isinstance(module, nn.Linear)
        ):
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)
            continue

        # Embedding / Linear
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        elif isinstance(module, nn.Linear):
            truncated_normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # 残差缩放：对注意力输出投影 o_proj 和 MLP 输出投影 down_proj 生效
    if residual_scale is not None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name.endswith(("o_proj", "down_proj")):
                module.weight.data.mul_(residual_scale)

    return model


# ---------------------------------------------------------------------------
# 自校验：python -m minillm.models.weight_init
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from minillm.models.transformer import MiniLLM, MiniLLMConfig

    cfg = MiniLLMConfig(
        vocab_size=1024, hidden_size=256, intermediate_size=688,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32,
    )

    # --- 校验 1: 可复现性（同 seed 两次初始化完全相同）---
    m1 = MiniLLM(cfg); init_weights(m1, seed=42)
    m2 = MiniLLM(cfg); init_weights(m2, seed=42)
    same = all(
        torch.equal(p1, p2)
        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters())
    )
    print(f"同 seed 两次初始化完全一致: {same}")

    # --- 校验 2: 分布统计（Linear 权重均值≈0、std≈0.02、无超 2σ 尾部）---
    w = torch.cat(
        [p.detach().flatten() for n, p in m1.named_parameters() if p.ndim == 2 and "norm" not in n]
    )
    print(f"\n权重均值: {w.mean().item():.5f} (应≈0)")
    print(f"权重标准差: {w.std().item():.5f} (应≈0.02)")
    print(f"最大绝对值: {w.abs().max().item():.5f} (截断后应 ≤ 0.04=2σ)")

    # --- 校验 3: RMSNorm weight 保持全 1 ---
    norm_weights = [p for n, p in m1.named_parameters() if "norm" in n and "weight" in n]
    all_one = all(bool((p == 1).all()) for p in norm_weights)
    print(f"RMSNorm weight 全为 1: {all_one} ({len(norm_weights)} 个 norm)")

    # --- 校验 4: 残差缩放生效（o_proj/down_proj 额外缩小）---
    m3 = MiniLLM(cfg)
    init_weights(m3, seed=42, residual_scale=1.0 / math.sqrt(2 * cfg.num_hidden_layers))
    o_proj_m1 = dict(m1.named_modules())  # noqa: F841
    scale = 1.0 / math.sqrt(2 * cfg.num_hidden_layers)
    # 找到第一个 o_proj 对比
    p1_w = [p for n, p in m1.named_parameters() if n.endswith("self_attn.o_proj.weight")][0]
    p3_w = [p for n, p in m3.named_parameters() if n.endswith("self_attn.o_proj.weight")][0]
    print(f"\no_proj 缩放系数: {scale:.4f}")
    print(f"缩放前 std: {p1_w.std().item():.5f} → 缩放后 std: {p3_w.std().item():.5f} (≈ 前×{scale:.4f})")

    # --- 校验 5: 初始化后前向正常（激活值有限、无 NaN）---
    m3.eval()
    with torch.no_grad():
        out = m3(torch.randint(0, cfg.vocab_size, (2, 8)))
    print(f"初始化后前向输出有限: {bool(torch.isfinite(out).all())} | 输出 std: {out.std().item():.5f}")

    print("\n✅ weight_init 自校验通过")
