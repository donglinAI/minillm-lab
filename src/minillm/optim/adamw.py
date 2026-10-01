"""
minillm.optim.adamw
===================
手撸 AdamW（Decoupled Weight Decay，解耦权重衰减）
--------------------------------------------------
Adam 为什么是"自适应学习率":
    每个参数有自己的有效步长 = lr / √v̂, 由梯度历史决定。
    梯度大的维度 v 大 → 步长小; 梯度小的维度 v 小 → 步长大。
    所以 Adam 对不同维度"一视同仁地快速", 不需要手工调各维度 lr。

Adam 的四件套:
    m = β1·m + (1-β1)·g          一阶矩: 梯度的动量（方向）
    v = β2·v + (1-β2)·g²         二阶矩: 梯度平方的滑动平均（幅度）
    m̂ = m/(1-β1^t)               偏置修正: m/v 初始为 0, 早期被严重低估
    v̂ = v/(1-β2^t)
    更新: θ ← θ − lr·m̂/(√v̂+ε) − lr·wd·θ

Adam vs AdamW（关键区别）:
    Adam : weight decay 作为 L2 正则加进梯度 g += wd·θ
           → 衰减项混进 m/v, 被"自适应"地缩放, 不再是真正衰减
    AdamW: weight decay 直接作用在权重上（不进 m/v）
           → 每个维度被均匀地衰减 lr·wd·θ, 与自适应步长解耦
    大模型预训练默认用 AdamW（泛化更好、超参更稳）。

手算案例（θ=1.0, g=0.1, β1=0.9, β2=0.999, lr=0.1, wd=0.01, t=1）:
    ① 解耦衰减: θ = 1.0×(1−0.1×0.01)         = 0.999
    ② m  = 0.1×0.1                            = 0.01
       v  = 0.001×0.01                        = 1e-5
    ③ step_size = lr/(1−β1¹) = 0.1/0.1        = 1.0
       v̂ = 1e-5/(1−0.999) = 0.01 → √v̂        = 0.1
       θ′ = 0.999 − 1.0×(0.01/0.1)            = 0.899
"""

from __future__ import annotations

import torch


class AdamW:
    """手撸 AdamW, 更新公式与 torch.optim.AdamW 完全一致（含 bias correction 的
    step_size 合并方式）, 保证数值对齐。"""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = weight_decay
        # 每个参数独立的动量/方差/步数
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = [0] * len(self.params)

    def zero_grad(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.t[i] += 1
            t = self.t[i]

            # 1) 解耦的权重衰减: 直接缩放参数（不进 m/v, 系数是 lr 不是 step_size）
            #    ← AdamW 与 Adam 的关键区别, 必须和 torch 一样在更新前做
            p.data.mul_(1 - self.lr * self.wd)

            # 2) 一阶矩 / 二阶矩（指数滑动平均）
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g

            # 3) Adam 更新: step_size 合并了一阶矩 bias correction
            #    step_size = lr/(1-β1^t); 二阶矩对 v 单独修正
            step_size = self.lr / (1 - self.beta1 ** t)
            denom = (self.v[i] / (1 - self.beta2 ** t)).sqrt() + self.eps
            p.data.addcdiv_(self.m[i], denom, value=-step_size)

    # ---- checkpoint 支持: 优化器状态（m/v/t/lr）可保存/恢复 ----
    def state_dict(self) -> dict:
        return {
            "lr": self.lr, "wd": self.wd, "eps": self.eps,
            "beta1": self.beta1, "beta2": self.beta2,
            "m": self.m, "v": self.v, "t": self.t,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.lr, self.wd, self.eps = sd["lr"], sd["wd"], sd["eps"]
        self.beta1, self.beta2 = sd["beta1"], sd["beta2"]
        # m/v 可能来自 CPU checkpoint, 要搬到参数所在设备(GPU 训练恢复时)
        self.m = [m.to(p.device) for m, p in zip(sd["m"], self.params)]
        self.v = [v.to(p.device) for v, p in zip(sd["v"], self.params)]
        self.t = sd["t"]


# ---------------------------------------------------------------------------
# 自校验: 与 torch.optim.AdamW 数值对齐
# ---------------------------------------------------------------------------
def _verify_against_torch() -> None:
    torch.manual_seed(0)

    # 同一模型, 同一梯度序列, 训练 10 步
    def make_model():
        return torch.nn.Sequential(
            torch.nn.Linear(6, 8), torch.nn.Tanh(), torch.nn.Linear(8, 4),
        )

    torch.manual_seed(0)
    mine = make_model()
    torch.manual_seed(0)
    ref = make_model()

    opt_mine = AdamW(mine.parameters(), lr=1e-2, weight_decay=0.01)
    opt_ref = torch.optim.AdamW(ref.parameters(), lr=1e-2, weight_decay=0.01)

    xs = [torch.randn(5, 6) for _ in range(10)]
    ys = [torch.randn(5, 4) for _ in range(10)]
    max_err = 0.0
    for x, y in zip(xs, ys):
        l1 = ((mine(x) - y) ** 2).mean()
        opt_mine.zero_grad(); l1.backward(); opt_mine.step()

        l2 = ((ref(x) - y) ** 2).mean()
        opt_ref.zero_grad(); l2.backward(); opt_ref.step()

        for a, b in zip(mine.parameters(), ref.parameters()):
            max_err = max(max_err, (a - b).abs().max().item())

    print(f"手撸 AdamW vs torch.optim.AdamW: 10 步训练后参数最大误差 {max_err:.2e} "
          f"{'✅' if max_err < 1e-6 else '❌'}")
    assert max_err < 1e-6


if __name__ == "__main__":
    print("=" * 60)
    print("手算演示: 单参数 θ=1.0, 固定梯度 g=0.1, 3 步")
    print("=" * 60)
    p = torch.tensor([1.0], requires_grad=True)
    opt = AdamW([p], lr=0.1, weight_decay=0.01)
    for step in range(3):
        p.grad = torch.tensor([0.1])
        before = p.item()
        opt.step()
        print(f"  step {step+1}: θ {before:.4f} → {p.item():.4f} | "
              f"m={opt.m[0].item():.4f} v={opt.v[0].item():.2e} t={opt.t[0]}")
        if step == 0:
            assert abs(p.item() - 0.899) < 1e-4, "第一步应与手算 0.899 一致"
    print("  ✓ 第一步与手算 0.899 一致")

    print("\n" + "=" * 60)
    print("数值对齐验证")
    print("=" * 60)
    _verify_against_torch()
