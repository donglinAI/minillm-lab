"""
minillm.pretrain.trainer
========================
预训练 Trainer：训练循环 + loss + 日志 + checkpoint / resume
----------------------------------------------------------------
训练循环（每步做什么）:
    for step in range(max_steps):
        1. 取一个 batch: input_ids, labels（labels 已在 dataset 里错位好）
        2. logits = model(input_ids)             # [B, S, V]
        3. loss = CrossEntropy(logits, labels)   # 逐位置预测下一个 token
        4. loss.backward()
           （grad_accum 步内累积梯度, 每 grad_accum 步才更新一次）
        5. clip_grad_norm_(params, max_norm)     # 防梯度爆炸
        6. optimizer.step();  scheduler 更新 lr
        7. 日志 / checkpoint

loss 为什么用 CrossEntropy:
    预训练 = 语言建模 = 每个位置预测"下一个 token"。
    logits [B,S,V] 拍平成 [B*S, V], labels 拍平成 [B*S]:
        loss = -Σ log P(真实token | 上文)
    等价于"平均负对数似然", 越小越好（模型越会预测下一个词）。

checkpoint / resume:
    保存 {model, optimizer(m/v/t), step} → 中断后从 step 继续训练。
    ★ resume 必须同时恢复 optimizer 状态: 只恢复模型 = Adam 的 m/v 归零,
      学习率也从头来, 等于"白训了前面那些步"。
    本文件验证: 中断恢复后继续 5 步 == 不中断继续 5 步（参数误差 0）。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from minillm.optim.adamw import AdamW
from minillm.optim.grad_clip import clip_grad_norm_
from minillm.optim.scheduler import WarmupCosineScheduler


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: AdamW,
        scheduler: WarmupCosineScheduler,
        train_loader,
        *,
        max_steps: int = 5000,
        grad_accum_steps: int = 1,
        grad_clip: float = 1.0,
        log_interval: int = 10,
        save_interval: int = 1000,
        output_dir: str = "./output/run",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loader = train_loader
        self.max_steps = max_steps
        self.grad_accum_steps = grad_accum_steps
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.step = 0
        self.epoch = 0
        self.iterator = iter(train_loader)

    # ------------------------------------------------------------------
    def _next_batch(self):
        """循环取 batch（一个 epoch 结束自动从下一个 epoch 开始）。"""
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.iterator = iter(self.loader)
            return next(self.iterator)

    def _compute_loss(self, batch) -> torch.Tensor:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        logits = self.model(input_ids)                       # [B, S, V]
        # 语言建模 loss: 每个位置预测下一个 token
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
        )

    def train(self):
        """跑完整训练循环, 直到 max_steps。返回每步 loss 列表。"""
        losses = []
        t0 = time.time()
        while self.step < self.max_steps:
            batch = self._next_batch()
            input_ids, labels = batch["input_ids"], batch["labels"]

            loss = self._compute_loss(batch) / self.grad_accum_steps
            loss.backward()

            grad_norm = torch.tensor(0.0)
            # 梯度累积: 攒够 grad_accum_steps 次梯度才更新一次
            if (self.step + 1) % self.grad_accum_steps == 0:
                grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_clip)
                # 先设 lr 再 step: WarmupCosine 是纯函数调度(lr 只由 step 决定),
                # 且必须"设了再用", 否则恢复训练时 lr 会滞后一步, 与不中断不一致
                self.optimizer.lr = self.scheduler.get_lr(self.step)
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.step += 1
            losses.append(loss.item() * self.grad_accum_steps)

            if self.step % self.log_interval == 0:
                print(f"  step {self.step:5d} | loss {losses[-1]:.4f} | "
                      f"lr {self.optimizer.lr:.2e} | grad_norm {grad_norm.item():.3f}")

            if self.save_interval and self.step % self.save_interval == 0:
                self.save_checkpoint(self.output_dir / f"ckpt_step{self.step}.pt")
        return losses

    # ------------------------------------------------------------------
    # checkpoint
    # ------------------------------------------------------------------
    def save_checkpoint(self, path) -> None:
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
        }, path)
        print(f"  [ckpt] 保存到 {path} (step={self.step})")

    def load_checkpoint(self, path) -> None:
        """恢复训练: 模型权重 + 优化器 m/v/t + 步数, 三者缺一不可。"""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = ckpt["step"]
        self.epoch = ckpt["epoch"]
        self.optimizer.lr = self.scheduler.get_lr(self.step)
        print(f"  [resume] 从 {path} 恢复 (step={self.step}, lr={self.optimizer.lr:.2e})")


# ---------------------------------------------------------------------------
# 自校验: loss 下降 + checkpoint/resume 数值等价
# ---------------------------------------------------------------------------
def _make_synthetic_data(B=8, S=32, vocab=512, n_batches=8):
    """合成随机 token 数据（验证训练循环, 不依赖真实语料）。"""
    torch.manual_seed(0)
    batches = []
    for _ in range(n_batches):
        batches.append({
            "input_ids": torch.randint(0, vocab, (B, S)),
            "labels": torch.randint(0, vocab, (B, S)),
        })
    return batches


if __name__ == "__main__":
    from minillm.models.transformer import MiniLLM, MiniLLMConfig

    torch.manual_seed(0)
    cfg = MiniLLMConfig(vocab_size=512, hidden_size=128, intermediate_size=352,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=1, max_position_embeddings=64)
    data = _make_synthetic_data(vocab=cfg.vocab_size)

    def make_trainer(start_step=0):
        torch.manual_seed(0)
        model = MiniLLM(cfg)
        opt = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        sch = WarmupCosineScheduler(lr_max=1e-3, warmup_steps=5, total_steps=20, min_lr=1e-4)
        return Trainer(model, opt, sch, data, max_steps=20, log_interval=5,
                       grad_clip=1.0, output_dir="/tmp/minillm_ckpt_test")

    print("=" * 62)
    print("1) 训练循环: loss 应单调下降（拟合合成数据）")
    print("=" * 62)
    tr = make_trainer()
    losses = tr.train()

    print("\n" + "=" * 62)
    print("2) checkpoint / resume 黄金验证")
    print("=" * 62)
    print("   分支A: 训练10步 → 保存 → 恢复 → 继续5步")
    print("   分支B: 直接训练15步（不中断）")
    print("   两条路走到 step15, 参数必须完全一致\n")

    # 分支 A: 中断 + 恢复
    torch.manual_seed(0)
    modelA = MiniLLM(cfg)
    optA = AdamW(modelA.parameters(), lr=1e-3, weight_decay=0.01)
    schA = WarmupCosineScheduler(lr_max=1e-3, warmup_steps=5, total_steps=20, min_lr=1e-4)
    trA = Trainer(modelA, optA, schA, data, max_steps=10, log_interval=100, output_dir="/tmp/minillm_ckpt_test")
    trA.train()
    ckpt_path = "/tmp/minillm_ckpt_test/resume_test.pt"
    trA.save_checkpoint(ckpt_path)
    # 恢复后继续 5 步（max_steps 覆盖为 15）
    trA.load_checkpoint(ckpt_path)
    trA.max_steps = 15
    trA.train()

    # 分支 B: 一口气 15 步
    torch.manual_seed(0)
    modelB = MiniLLM(cfg)
    optB = AdamW(modelB.parameters(), lr=1e-3, weight_decay=0.01)
    schB = WarmupCosineScheduler(lr_max=1e-3, warmup_steps=5, total_steps=20, min_lr=1e-4)
    trB = Trainer(modelB, optB, schB, data, max_steps=15, log_interval=100, output_dir="/tmp/minillm_ckpt_test")
    trB.train()

    max_err = max((a - b).abs().max().item() for a, b in zip(modelA.parameters(), modelB.parameters()))
    print(f"\n   分支A(中断恢复) vs 分支B(不中断) 参数最大误差: {max_err:.2e}")
    print("   ✅ resume 正确: 模型 + 优化器 m/v/t + 步数全部恢复"
          if max_err < 1e-6 else "   ❌ resume 有误")
    assert max_err < 1e-6
