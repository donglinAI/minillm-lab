"""
minillm.posttrain.sft.dataset
==============================
SFT（指令微调）数据构造：prompt + answer + mask 指令部分
----------------------------------------------------------------
预训练学会"预测下一个 token"；SFT 学会"听指令回答问题"。

对话模板（简化 LLaMA-2 Chat）:
    prompt = "[INST] 中国的首都是哪里? [/INST]"
    answer = "北京<eos>"
    input_ids = encode(prompt) + encode(answer)
    labels    = [-100]*(len(prompt)-1) + answer_ids + [-100]   # 自回归错位版

★ 自回归错位（关键约定）:
    位置 i 的模型输出要预测 input_ids[i+1]。
    Trainer 的 loss 不做 shift, 所以 dataset 负责错位:
      - prompt 内部位置(前 len(prompt)-1 个): 预测 prompt 自己 → -100 屏蔽
      - prompt 最后一个位置: 预测 answer 首 token → 保留 (这是关键一跳!)
      - answer 位置: 预测 a1..eos → 保留
      - 序列末尾(预测 eos 之后): 无意义 → -100

★ mask 指令部分:
    labels 里 prompt 内部位置全部标 -100，只有 answer 相关的
    位置是真实 token id。CrossEntropyLoss(ignore_index=-100) 会跳过 -100:
        → 模型只在 answer 上算 loss、只回传 answer 位置的梯度
        → 指令部分只是"上文", 模型学的是"如何回答"
    若不 mask: 模型还要花容量去预测指令本身(背模板),
      目标被稀释, 甚至学会"把指令原样背出来"。

两种模式:
    mode="pad" : 每条样本独立 padding 到 seq_len（padding 位置也标 -100）
    mode="pack": 多条样本连续拼进 seq_len 的序列, mask 跟着移动
                 不浪费 padding, 大厂预训练/SFT 都用 packing
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

# 对话模板的定界符（简化版；真实工程里用 tokenizer 的 chat_template）
INST_START = "[INST] "
INST_END = " [/INST]"


def build_chat(item: dict) -> tuple[str, str]:
    """把一条 {instruction, output} 拆成 (prompt_text, answer_text)。"""
    prompt = f"{INST_START}{item['instruction']}{INST_END}"
    return prompt, item["output"]


class SFTDataset(Dataset):
    """SFT 数据集：输出可直接喂 Trainer 的 batch（input_ids + labels）。

    Parameters
    ----------
    data : list[dict]
        每条 {"instruction": str, "output": str}
    tokenizer :
        项目的 Tokenizer（encode/decode/eos_token_id/pad_token_id）
    seq_len : int
        序列长度（pad 模式: 每条样本的目标长度；pack 模式: 打包后每条长度）
    mode : str
        "pad" | "pack"
    """

    def __init__(self, data, tokenizer, seq_len: int = 512, mode: str = "pad"):
        assert mode in ("pad", "pack"), f"mode 只支持 pad/pack, 收到 {mode}"
        self.tok = tokenizer
        self.seq_len = seq_len
        self.pad_id = tokenizer.pad_token_id
        self.mode = mode

        # 每条样本先独立构造: (input_ids, labels)
        #   input_ids = prompt + answer
        #   ★ 自回归错位（Trainer 的 loss 不做 shift, 由 dataset 负责错位）:
        #     约定: 位置 i 的模型输出预测 input_ids[i+1]
        #     - prompt 内部位置(0..k-2): 预测 prompt 自己 → -100 屏蔽
        #     - 位置 k-1 (prompt 末尾): 预测 answer 首 token → 保留!
        #     - answer 位置: 预测 a1..eos → 保留
        #     - 末尾(预测 eos 之后): 无意义 → -100
        samples = []
        for item in data:
            prompt_text, answer_text = build_chat(item)
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
            answer_ids = answer_ids + [tokenizer.eos_token_id]  # 回答以 eos 结尾
            k = len(prompt_ids)
            ids = prompt_ids + answer_ids
            labels = [-100] * (k - 1) + answer_ids + [-100]
            assert len(labels) == len(ids)
            samples.append((ids, labels))

        if mode == "pad":
            self.examples = [self._pad_one(p, a) for p, a in samples]
        else:
            self.examples = self._pack_all(samples)

    # ------------------------------------------------------------------
    def _pad_one(self, ids, labels):
        """单条样本 → (input_ids, labels)，定长 seq_len。"""
        # 超长: 截掉头部（保留尾部 answer），mask 跟着平移
        if len(ids) > self.seq_len:
            ids = ids[-self.seq_len:]
            labels = labels[-self.seq_len:]
        # 不足: 尾部补 pad, mask 也补 -100
        n = len(ids)
        ids = ids + [self.pad_id] * (self.seq_len - n)
        labels = labels + [-100] * (self.seq_len - n)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def _pack_all(self, samples):
        """packing: 把多条样本连续拼进 seq_len 的序列（mask 跟着移动）。"""
        examples = []
        cur_ids, cur_labels = [], []
        for ids, labels in samples:
            # 单条就超长 → 独立截断成一条（尾部保留 answer）
            if len(ids) > self.seq_len:
                if cur_ids:  # 先把当前序列归档
                    examples.append(self._finalize(cur_ids, cur_labels))
                    cur_ids, cur_labels = [], []
                ids, labels = ids[-self.seq_len:], labels[-self.seq_len:]
                examples.append((torch.tensor(ids), torch.tensor(labels)))
                continue

            # 装得下就拼; 装不下就归档旧序列、开新序列
            if len(cur_ids) + len(ids) > self.seq_len:
                examples.append(self._finalize(cur_ids, cur_labels))
                cur_ids, cur_labels = [], []
            cur_ids += ids
            cur_labels += labels

        if cur_ids:
            examples.append(self._finalize(cur_ids, cur_labels))
        return examples

    def _finalize(self, ids, labels):
        """把一条序列补齐到 seq_len 并转 tensor。"""
        n = len(ids)
        ids = ids + [self.pad_id] * (self.seq_len - n)
        labels = labels + [-100] * (self.seq_len - n)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids, labels = self.examples[idx]
        return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# 自校验 python -m minillm.posttrain.sft.dataset
# ---------------------------------------------------------------------------
def _demo_data():
    return [
        {"instruction": "中国的首都是哪里？", "output": "中国的首都是北京。"},
        {"instruction": "1+1 等于几？", "output": "等于 2。"},
        {"instruction": "太阳从哪边升起？", "output": "太阳从东方升起。"},
    ]


if __name__ == "__main__":
    from minillm.tokenizer.tokenizer import Tokenizer
    from minillm.models.transformer import MiniLLM, MiniLLMConfig

    tok = Tokenizer()
    data = _demo_data()

    print("=" * 66)
    print("1) mask 指令部分 + 自回归错位")
    print("=" * 66)
    ds = SFTDataset(data, tok, seq_len=64, mode="pad")
    ex0 = ds[0]
    ids, labels = ex0["input_ids"], ex0["labels"]
    prompt_txt = f"{INST_START}{data[0]['instruction']}{INST_END}"
    n_prompt = len(tok.encode(prompt_txt, add_special_tokens=False))
    ans_ids = tok.encode(data[0]["output"], add_special_tokens=False) + [tok.eos_token_id]
    print(f"   样本0: prompt {n_prompt} token, answer {len(ans_ids)} token(含eos)")
    print(f"   input_ids[:{n_prompt}] = {ids[:n_prompt].tolist()}")
    print(f"   labels  [:{n_prompt-1}] = {labels[:n_prompt-1].tolist()}")
    print(f"   → prompt 内部位置全为 -100 = "
          f"{bool((labels[:n_prompt-1] == -100).all())} ✅")
    print(f"   labels[{n_prompt-1}] (prompt 末位→answer首token) = {labels[n_prompt-1].item()}"
          f" 应为 {ans_ids[0]} = "
          f"{'✅ 错位正确' if labels[n_prompt-1].item() == ans_ids[0] else '❌'}")
    print(f"   labels[{n_prompt}:{n_prompt+4}] = {labels[n_prompt:n_prompt+4].tolist()}… "
          f"(answer 内部: a1..eos 保留)")

    print("\n" + "=" * 66)
    print("2) mask 的数学等价: masked loss 梯度 == 只看 answer 的 loss 梯度")
    print("=" * 66)
    # ★ vocab_size 必须覆盖所有出现的 id: tokenizer 的 eos=151643 排在词表末尾,
    #   合法下标是 0..151642, 所以模型 vocab 要取 max(vocab_size, eos_id)+1
    #   （tiny.yaml 里 64000 会导致 embedding 越界, 真实训练必须改）
    model_vocab = max(tok.vocab_size, tok.eos_token_id) + 1

    def _small(ids, labs):
        """自校验提速: 把真实 token id 重映射到 0..255 的小词表。
        mask(-100) 保持不变, 只影响演示速度, 不影响 mask 的数学性质。"""
        ids_s = (ids % 256).long()
        labs_s = torch.where(labs >= 0, labs % 256, labs).long()
        return ids_s, labs_s

    # 验证 2/3/4 用映射后的小词表模型（vocab=256, CPU 快）
    cfg = MiniLLMConfig(vocab_size=256, hidden_size=64,
                        intermediate_size=176, num_hidden_layers=1,
                        num_attention_heads=4, num_key_value_heads=1,
                        max_position_embeddings=64)
    torch.manual_seed(0)
    model = MiniLLM(cfg)
    import torch.nn.functional as F

    ids = ids.unsqueeze(0)      # [1, S]
    labs = labels.unsqueeze(0)  # [1, S]
    ids, labs = _small(ids, labs)   # 重映射到小词表（仅自校验提速）
    logits = model(ids)         # [1, S, V]

    # 分支 M: masked（labels 含 -100, CE 自动忽略）
    lm = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), labs.reshape(-1))
    lm.backward()
    g_masked = [p.grad.detach().clone() for p in model.parameters()]
    model.zero_grad()

    # 分支 R: 只取 answer 位置（等价参考）——重新 forward 建新图
    model.zero_grad()
    logits = model(ids)
    ans_mask = labs.reshape(-1) != -100
    lr = F.cross_entropy(logits.reshape(-1, cfg.vocab_size)[ans_mask],
                         labs.reshape(-1)[ans_mask])
    lr.backward()
    g_ref = [p.grad.detach().clone() for p in model.parameters()]
    max_err = max((a - b).abs().max().item() for a, b in zip(g_masked, g_ref))
    print(f"   masked loss = {lm.item():.4f} | answer-only loss = {lr.item():.4f}")
    print(f"   两条路的梯度最大误差: {max_err:.2e} "
          f"{'✅ mask == 只看 answer' if max_err < 1e-5 else '❌'}")

    print("\n" + "=" * 66)
    print("3) mask vs 不 mask: 梯度来源不同")
    print("=" * 66)
    model.zero_grad()
    logits = model(ids)
    unmasked_labs = ids.clone()          # 不 mask: 全位置都预测真实 token
    lu = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), unmasked_labs.reshape(-1))
    lu.backward()
    g_unmasked = [p.grad.detach().clone() for p in model.parameters()]
    model.zero_grad()
    logits = model(ids)
    lm2 = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), labs.reshape(-1))
    lm2.backward()
    g_masked2 = [p.grad.detach().clone() for p in model.parameters()]
    # 对比: 两路梯度差的来源 = prompt 位置
    diff = max((a - b).abs().max().item() for a, b in zip(g_unmasked, g_masked2))
    print(f"   masked loss   = {lm.item():.4f}  (只含 answer 位置的项)")
    print(f"   unmasked loss = {lu.item():.4f}  (还含 prompt 位置的项: 预测指令本身)")
    print(f"   两路梯度之差 = {diff:.2e} → unmasked 把梯度花在'复述指令'上, "
          f"SFT 不需要这个")
    print(f"   (masked 的梯度 == 只看 answer 的梯度, 已在验证2证明)")

    print("\n" + "=" * 66)
    print("4) SFT 真训练几步: 回答首 token 概率应显著上升")
    print("=" * 66)
    torch.manual_seed(0)
    model = MiniLLM(cfg)
    from minillm.optim.adamw import AdamW
    opt = AdamW(model.parameters(), lr=3e-3)
    ds = SFTDataset(data, tok, seq_len=64, mode="pad")

    def answer_first_tok_prob(model, item):
        """输入 prompt, 测 answer 第一个 token 的预测概率（小词表映射）。"""
        prompt_txt, ans_txt = build_chat(item)
        pid = torch.tensor([tok.encode(prompt_txt, add_special_tokens=False)])
        with torch.no_grad():
            logits = model(pid % 256)               # [1, S, V]（id 映射到小词表）
            probs = logits[0, -1].softmax(-1)      # 下一个 token 分布
        aid = tok.encode(ans_txt, add_special_tokens=False)[0] % 256
        return probs[aid].item()

    p_before = answer_first_tok_prob(model, data[0])
    for step in range(40):
        opt.zero_grad()
        total = 0.0
        for b in ds:
            ids_b, labs_b = _small(b["input_ids"], b["labels"])
            ids_b = ids_b.unsqueeze(0)
            labs_b = labs_b.unsqueeze(0)
            logits = model(ids_b)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), labs_b.reshape(-1))
            loss.backward()
            total += loss.item()
        opt.step()
        if step in (0, 19, 39):
            print(f"   step {step+1:2d}: loss {total/len(ds):.4f}")
    p_after = answer_first_tok_prob(model, data[0])
    print(f"   '回答首 token'概率: 训练前 {p_before:.2e} → 训练后 {p_after:.2e} "
          f"({'✅ 显著上升, 模型学会回答' if p_after > max(p_before * 3, 1e-4) else '⚠ 变化不大'})")
