"""
minillm.optim.scheduler
=======================
WarmupCosine 学习率调度（预训练默认: 线性预热 + 余弦退火）
----------------------------------------------------------
为什么需要学习率调度:
    · 训练初期梯度方向乱, 大 lr 会让参数乱跳 → 用 warmup 从小 lr 线性爬升,
      让模型先"站稳"
    · 训练后期需要 lr 变小才能收敛到精细区域 → 余弦退火平滑降到 min_lr
    · 大模型预训练标配: 0.1% 到 1% 步数的 warmup + cosine decay

公式:
    warmup 阶段 (step < warmup_steps):
        lr = lr_max × step / warmup_steps          (线性 0 → lr_max)
    cosine 阶段 (warmup ≤ step < total_steps):
        p  = (step - warmup) / (total - warmup)    (0→1)
        lr = min_lr + 0.5·(lr_max - min_lr)·(1 + cos(π·p))   (lr_max → min_lr)
    step ≥ total_steps: lr = min_lr

验证: 与手算公式逐点一致 + 打印关键点曲线。
"""

from __future__ import annotations

import math


class WarmupCosineScheduler:
    def __init__(self, lr_max: float = 1e-3, warmup_steps: int = 1000,
                 total_steps: int = 10000, min_lr: float = 0.0):
        self.lr_max = lr_max
        self.warmup = warmup_steps
        self.total = total_steps
        self.min_lr = min_lr

    def get_lr(self, step: int) -> float:
        """任意步的学习率（step 从 0 计）。"""
        if step < self.warmup:
            # 线性预热: 0 → lr_max
            return self.lr_max * step / self.warmup
        if step >= self.total:
            return self.min_lr
        # 余弦退火: lr_max → min_lr
        p = (step - self.warmup) / (self.total - self.warmup)
        return self.min_lr + 0.5 * (self.lr_max - self.min_lr) * (1 + math.cos(math.pi * p))

    def lr_curve(self, steps: int) -> list[float]:
        return [self.get_lr(s) for s in range(steps)]


# ---------------------------------------------------------------------------
# 自校验: 关键点与手算公式一致
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sch = WarmupCosineScheduler(lr_max=1e-3, warmup_steps=1000, total_steps=10000, min_lr=0.0)

    print("=" * 60)
    print("WarmupCosine 学习率曲线 (lr_max=1e-3, warmup=1000, total=10000)")
    print("=" * 60)
    points = [0, 250, 500, 999, 1000, 2000, 5000, 7500, 9999, 10000, 20000]
    for s in points:
        lr = sch.get_lr(s)
        bar = "#" * int(lr / 1e-3 * 30)
        print(f"  step {s:5d}: lr = {lr:.2e} {bar}")

    # 关键点手算验证
    assert abs(sch.get_lr(0) - 0.0) < 1e-12, "step 0 应为 0"
    assert abs(sch.get_lr(500) - 5e-4) < 1e-12, "warmup 中点应为 lr_max/2"
    assert abs(sch.get_lr(1000) - 1e-3) < 1e-12, "warmup 结束应达 lr_max"
    assert abs(sch.get_lr(5500) - 5e-4) < 1e-12, "余弦中点应为 lr_max/2"
    assert abs(sch.get_lr(10000) - 0.0) < 1e-12, "total 处应为 min_lr"
    print("\n  ✅ 关键点全部与公式一致 (step0=0, warmup 中点=lr_max/2, 结束=lr_max, 余弦中点=lr_max/2, 结束=0)")
