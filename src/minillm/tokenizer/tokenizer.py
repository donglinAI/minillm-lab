"""
minillm.tokenizer.tokenizer
============================

统一的 Tokenizer 封装。

设计选择
--------
1. **不手撸 BPE 训练**（那是可选的 ``train_bpe.py`` 的事），这里直接加载
   HuggingFace 上已经训好的 BPE tokenizer。理由：
   - BPE 训练本身不影响我们要学的 Transformer / 并行 / 推理算法；
   - 用现成 tokenizer 能立刻拿到中文支持好的词表，免去自己清洗语料；
   - 但我们把所有调用收束到这一个类里，后面想换实现只改这一处。

2. **默认选 Qwen2.5 tokenizer**：中文词表覆盖好，自带 ``<|endoftext|>`` 作为
   pad/eos，和 LLaMA 式架构兼容。

3. **懒加载**：``transformers.AutoTokenizer`` 在 import 时不触发，只在
   ``Tokenizer()`` 实例化时才下载/加载，避免单元测试时网络依赖。

对外接口
--------
- ``encode(text) -> list[int]``
- ``decode(ids) -> str``
- ``batch_encode(texts, max_length, padding, truncation) -> dict``
  返回 ``{"input_ids": LongTensor, "attention_mask": LongTensor}``
- 特殊 id 直接通过属性访问：``pad_token_id / eos_token_id / bos_token_id``
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch


# 默认 tokenizer：Qwen2.5 系列，中文友好，词表 ~152k
DEFAULT_TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B"


class Tokenizer:
    """对 HuggingFace tokenizer 的薄封装。"""

    def __init__(self, name_or_path: str = DEFAULT_TOKENIZER_NAME):
        # 延迟 import，避免在只跑配置/单测时强制联网下载
        from transformers import AutoTokenizer

        self.name_or_path = name_or_path
        self._tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

        # 兜底：有些 tokenizer 没有显式 pad_token（如 GPT-2），用 eos 顶替
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token

    # ------------------------------------------------------------------
    # 基本属性
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """真实词表大小。建模型时用这个，不要写死 yaml 里的数。"""
        return self._tok.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self._tok.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self._tok.eos_token_id

    @property
    def bos_token_id(self) -> Optional[int]:
        return self._tok.bos_token_id

    # ------------------------------------------------------------------
    # 单条编码 / 解码
    # ------------------------------------------------------------------
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """把一段文本转成 token id 列表。"""
        return self._tok.encode(text, add_special_tokens=add_special_tokens)

    def decode(
        self,
        ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """把 token id 列表还原成文本。"""
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------
    # batch 编码（DataLoader collate 会用到）
    # ------------------------------------------------------------------
    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: Optional[str] = "pt",
    ) -> Dict[str, torch.Tensor]:
        """批量编码。

        Returns
        -------
        dict with:
            input_ids: LongTensor, shape [batch, seq_len]
            attention_mask: LongTensor, shape [batch, seq_len]
                （1 = 真实 token，0 = padding）
        """
        encoded = self._tok(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        return {
            "input_ids": encoded["input_ids"].long(),
            "attention_mask": encoded["attention_mask"].long(),
        }

    # ------------------------------------------------------------------
    # 友好打印
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"Tokenizer(name_or_path={self.name_or_path!r}, "
            f"vocab_size={self.vocab_size}, "
            f"pad_id={self.pad_token_id}, eos_id={self.eos_token_id})"
        )


# ---------------------------------------------------------------------------
# 自校验：python -m minillm.tokenizer.tokenizer
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tok = Tokenizer()
    print(tok)
    print()

    samples = [
        "今天天气真好，我们一起去公园散步吧。",
        "HuggingFace's tokenizer makes NLP easy.",
        "大模型算法工程师需要掌握预训练、后训练和推理加速。",
    ]

    print("=== 单条 encode/decode 往返 ===")
    for s in samples:
        ids = tok.encode(s)
        recovered = tok.decode(ids)
        print(f"原文: {s}")
        print(f"token 数: {len(ids)}")
        print(f"往返: {recovered}")
        print(f"往返一致: {recovered == s}")
        print("-" * 60)

    print("=== batch_encode ===")
    batch = tok.batch_encode(samples, max_length=32, padding=True, truncation=True)
    print("input_ids shape:", batch["input_ids"].shape)
    print("attention_mask shape:", batch["attention_mask"].shape)
    print("input_ids[0]:", batch["input_ids"][0].tolist())
    print("attention_mask[0]:", batch["attention_mask"][0].tolist())
