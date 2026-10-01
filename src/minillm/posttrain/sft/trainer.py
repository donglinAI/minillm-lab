"""
minillm.posttrain.sft.trainer
=============================
SFT 训练器：复用预训练 Trainer + SFTDataset，只换数据、调小 lr
----------------------------------------------------------------
为什么 SFT 不需要新的训练循环:
    Trainer 的 loss 是 F.cross_entropy(flat_logits, flat_labels)，
    labels 里的 -100 会被默认的 ignore_index=-100 自动跳过
    → 天然支持 mask 指令部分！SFT 只需：
      1. 用 SFTDataset 产出 (input_ids, labels[含-100])
      2. 学习率调小（微调，防止遗忘预训练知识）
      3. weight_decay 设 0（SFT 只做适配，不需要正则）

★ 为什么 SFT 要用小学习率（灾难性遗忘）:
    预训练模型已经学会了"语言"；SFT 只是"教它听指令回答"。
    学习率大 → 每步权重改动大 → 把预训练学到的语言能力冲掉
                 （灾难性遗忘：会答新题但不会说人话了）
    学习率小 → 权重只小幅调整 → 保留语言能力的同时学会回答
    验证2 会实测: 同样 SFT 30 步, 高 lr 的"预训练能力"损失
    远大于低 lr——这就是 SFT 用 1e-5 级 lr 的原因。
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from minillm.optim.adamw import AdamW
from minillm.optim.scheduler import WarmupCosineScheduler
from minillm.pretrain.trainer import Trainer
from minillm.posttrain.sft.dataset import SFTDataset


class SFTTrainer:
    """SFT 训练器：组合 SFTDataset + 预训练 Trainer。

    和预训练的唯一差别在超参:
        lr        1e-5 ~ 1e-4（预训练通常 1e-3 级）
        wd        0（微调不加正则）
        max_steps 少（SFT 数据量小, 通常几个 epoch 就够）
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_data,
        *,
        seq_len: int = 512,
        mode: str = "pad",
        batch_size: int = 1,
        lr: float = 1e-5,
        weight_decay: float = 0.0,
        warmup_steps: int = 10,
        total_steps: int | None = None,
        min_lr: float = 1e-6,
        max_steps: int = 1000,
        grad_accum_steps: int = 1,
        grad_clip: float = 1.0,
        log_interval: int = 10,
        save_interval: int = 1000,
        output_dir: str = "./output/sft",
        device: str = "cpu",
    ):
        ds = SFTDataset(train_data, tokenizer, seq_len=seq_len, mode=mode)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        if total_steps is None:
            total_steps = max_steps
        opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sch = WarmupCosineScheduler(lr_max=lr, warmup_steps=warmup_steps,
                                    total_steps=total_steps, min_lr=min_lr)
        self.trainer = Trainer(
            model, opt, sch, loader,
            max_steps=max_steps,
            grad_accum_steps=grad_accum_steps,
            grad_clip=grad_clip,
            log_interval=log_interval,
            save_interval=save_interval,
            output_dir=output_dir,
            device=device,
        )

    def train(self):
        return self.trainer.train()

    def save_checkpoint(self, path):
        self.trainer.save_checkpoint(path)

    def load_checkpoint(self, path):
        self.trainer.load_checkpoint(path)


# ---------------------------------------------------------------------------
# 自校验
# ---------------------------------------------------------------------------
class SmallTok:
    """自校验用的小词表 tokenizer（ASCII 字符 ↔ id, vocab=256）。

    生产环境用项目的真实 Tokenizer（vocab 151644）；这里只是为了
    CPU 上跑得快：每个字符一个 id, 保证无碰撞、可解码。
    """

    vocab_size = 256
    eos_token_id = 1
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True):
        return [ord(c) % 256 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        out = "".join(chr(i) for i in ids)
        if skip_special_tokens:
            out = out.replace(chr(0), "").replace(chr(1), "")
        return out


def _demo_sft_data():
    return [
        {"instruction": "What is the capital of France?", "output": "Paris."},
        {"instruction": "What is 2+3?", "output": "5."},
        {"instruction": "Is the sun a star?", "output": "Yes."},
    ]


def _greedy(model, prompt_ids, tok, max_new=10):
    """极简贪心解码（无 KV cache）：逐 token 生成, 直到 eos。"""
    gen = list(prompt_ids)
    import torch
    for _ in range(max_new):
        logits = model(torch.tensor([gen]))[0, -1]      # [V]
        nxt = logits.argmax().item()
        if nxt == tok.eos_token_id:
            break
        gen.append(nxt)
    return gen[len(prompt_ids):]


if __name__ == "__main__":
    import torch
    from minillm.models.transformer import MiniLLM, MiniLLMConfig
    from minillm.posttrain.sft.dataset import build_chat

    cfg = MiniLLMConfig(vocab_size=256, hidden_size=64,
                        intermediate_size=176, num_hidden_layers=1,
                        num_attention_heads=4, num_key_value_heads=1,
                        max_position_embeddings=64)
    tok = SmallTok()
    sft_data = _demo_sft_data()

    print("=" * 66)
    print("1) 端到端 SFT：loss 下降 + 学会回答（生成验证）")
    print("=" * 66)
    torch.manual_seed(0)
    model = MiniLLM(cfg)
    tr = SFTTrainer(model, tok, sft_data, seq_len=64, lr=3e-3,
                    warmup_steps=5, max_steps=40, log_interval=10,
                    output_dir="/tmp/minillm_sft_test")
    losses = tr.train()

    def answer_prob(model, item):
        prompt_txt, ans_txt = build_chat(item)
        pid = tok.encode(prompt_txt, add_special_tokens=False)
        with torch.no_grad():
            logits = model(torch.tensor([pid]))
            probs = logits[0, -1].softmax(-1)
        return probs[tok.encode(ans_txt)[0]].item()

    # 训练前的概率（用另一个同样初始化的模型测）
    torch.manual_seed(0)
    model0 = MiniLLM(cfg)
    p0 = answer_prob(model0, sft_data[0])
    p1 = answer_prob(model, sft_data[0])
    print(f"   最终 loss {losses[-1]:.4f}（初始 {losses[0]:.4f}）")
    print(f"   answer 首 token 概率: {p0:.3f} → {p1:.3f}")
    gen = tok.decode(_greedy(model, tok.encode(build_chat(sft_data[0])[0]), tok))
    print(f"   greedy 生成: \"{gen}\"（真实 answer: \"{sft_data[0]['output']}\"）")

    print("\n" + "=" * 66)
    print("2) 灾难性遗忘: 高 lr vs 低 lr, 谁更忘预训练")
    print("=" * 66)
    # 预训练阶段: 学会一个固定文本模式 "abcde..." 循环
    pre_text = "abcdefgh" * 4                      # 32 字符, 模式可学
    pre_ids = tok.encode(pre_text, add_special_tokens=False)
    # 自回归错位: 位置 i 预测 i+1（最后一个位置预测开头的 a）
    pre_labels = pre_ids[1:] + [pre_ids[0]]
    pre_batch = {"input_ids": torch.tensor([pre_ids]),
                 "labels": torch.tensor([pre_labels])}

    def pretrain_loss(model):
        logits = model(pre_batch["input_ids"])
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            pre_batch["labels"].reshape(-1)).item()

    torch.manual_seed(0)
    base = MiniLLM(cfg)
    opt = AdamW(base.parameters(), lr=3e-3)
    for _ in range(60):
        opt.zero_grad()
        logits = base(pre_batch["input_ids"])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), pre_batch["labels"].reshape(-1))
        loss.backward()
        opt.step()
    pre0 = pretrain_loss(base)
    print(f"   预训练 60 步后, 文本模式 loss = {pre0:.4f}（已记住模式）")

    import copy
    results = []
    for lr, tag in ((3e-3, "高 lr=3e-3"), (3e-4, "低 lr=3e-4")):
        model_s = copy.deepcopy(base)
        tr_s = SFTTrainer(model_s, tok, sft_data, seq_len=64, lr=lr,
                          warmup_steps=5, max_steps=30, log_interval=100,
                          output_dir="/tmp/minillm_sft_test")
        tr_s.train()
        pre_after = pretrain_loss(model_s)
        ap = answer_prob(model_s, sft_data[0])
        results.append((tag, pre_after - pre0, ap))
        print(f"   {tag:12s}: 预训练 loss {pre0:.4f}→{pre_after:.4f}"
              f"(遗忘 {pre_after-pre0:+.4f}) | answer 概率 {ap:.3f}")

    low_forget = results[1][1]
    high_forget = results[0][1]
    print(f"   → 高 lr 遗忘 {high_forget:+.4f} vs 低 lr 遗忘 {low_forget:+.4f}"
          f" = {'✅ 低 lr 保留预训练能力, SFT 用低 lr' if low_forget < high_forget else '❌'}")
