"""
minillm.data.dataset
=====================

预训练 Dataset：把 jsonl 文本 tokenize 后打包成长序列。

核心思路（和 GPT-2 / LLaMA 预训练一致）
--------------------------------------
1. 把所有数据源的文本读出来，按 manifest 里的 weight 决定每个源读多少条。
2. 每条文本 tokenize 后在末尾加一个 <eos>，作为文档边界。
3. 把所有 token id 拼成一个超长的一维数组。
4. 按 ``seq_len + 1`` 切分：
   - input_ids = tokens[i : i+seq_len]
   - labels    = tokens[i+1 : i+seq_len+1]   （错位一位，预测下一个 token）

为什么要 packing？
------------------
如果每条文本单独成一个样本，短文本会浪费大量 padding。
packing 把多段文本拼到同一个 seq_len 里，GPU 利用率从 30% 提到 90%+。
代价是不同文档之间没有显式边界 mask，但对预训练影响很小。

缓存
----
第一次跑会把所有文本 tokenize 成一个大数组，存到 ``.bin`` 文件。
第二次直接 mmap 读，不用重新 tokenize。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class PackedPretrainDataset(Dataset):
    """预训练打包数据集。

    Parameters
    ----------
    manifest_path : str
        data_cache/manifest.json 路径。
    tokenizer : Tokenizer
        已经初始化的 tokenizer 实例。
    seq_len : int
        序列长度（1024 或 2048）。
    cache_dir : str
        tokenize 结果缓存目录，默认和 jsonl 同目录。
    """

    def __init__(
        self,
        manifest_path: str,
        tokenizer,
        seq_len: int = 1024,
        cache_dir: str = "./data_cache",
    ):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        # 读 manifest
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # 缓存文件路径（一个项目共用一个大 bin）
        cache_bin = Path(cache_dir) / f"tokenized_packed_seq{seq_len}.bin"

        if cache_bin.exists():
            # 直接 mmap 读缓存
            print(f"[dataset] 命中 tokenize 缓存: {cache_bin}")
            self.tokens = np.memmap(cache_bin, dtype=np.int32, mode="r")
        else:
            # 第一次：逐源读取、tokenize、拼接
            all_tokens: List[int] = []
            eos_id = tokenizer.eos_token_id

            for source_name, info in manifest.items():
                jsonl_path = Path(info["path"])
                weight = info["weight"]
                num_lines = info["num_lines"]

                # 按 weight 决定读多少条（简单按比例采样）
                # 所有 weight 之和是 1，这里直接全读，后面 DataLoader 再 shuffle
                # 但为了演示 weight 机制，这里按比例抽样
                # 教学项目：先全读，weight 留给 DataLoader 层面做混合
                print(f"[dataset] 读取 {source_name}: {num_lines} 条, weight={weight}")

                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in tqdm(f, total=num_lines, desc=f"  tokenizing {source_name}"):
                        item = json.loads(line)
                        text = item["text"]
                        ids = tokenizer.encode(text, add_special_tokens=True)
                        # 文档边界加 eos
                        ids.append(eos_id)
                        all_tokens.extend(ids)

            # 存成 numpy memmap，下次直接读
            arr = np.array(all_tokens, dtype=np.int32)
            arr.tofile(cache_bin)
            self.tokens = np.memmap(cache_bin, dtype=np.int32, mode="r+")
            print(f"[dataset] 总 token 数: {len(self.tokens):,}")

        # 样本数：每个样本需要 seq_len + 1 个 token（input + label 错开）
        self.num_samples = len(self.tokens) // (self.seq_len + 1)
        print(f"[dataset] 样本数: {self.num_samples:,}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 取 seq_len + 1 个连续 token
        start = idx * (self.seq_len + 1)
        chunk = np.array(self.tokens[start : start + self.seq_len + 1], dtype=np.int64)

        # input_ids = 前 seq_len 个
        # labels    = 后 seq_len 个（错开一位）
        input_ids = torch.from_numpy(chunk[:-1])
        labels = torch.from_numpy(chunk[1:])

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# 自校验：python -m minillm.data.dataset
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from minillm.tokenizer.tokenizer import Tokenizer

    tok = Tokenizer()
    ds = PackedPretrainDataset(
        manifest_path="./data_cache/manifest.json",
        tokenizer=tok,
        seq_len=128,  # 自校验用小 seq_len
    )

    print(f"\n数据集长度: {len(ds)}")
    sample = ds[0]
    print(f"input_ids shape: {sample['input_ids'].shape}")
    print(f"labels shape: {sample['labels'].shape}")
    print(f"input_ids[:20]: {sample['input_ids'][:20].tolist()}")
    print(f"labels[:20]:    {sample['labels'][:20].tolist()}")

    # 验证错位关系：labels[i] 应该等于 input_ids[i+1]
    mismatch = (sample["input_ids"][1:] != sample["labels"][:-1]).sum().item()
    print(f"\n错位验证: mismatch = {mismatch} (应为 0)")

    # 解码第一条样本看看
    print(f"\n解码 input_ids 前 50 token:")
    print(tok.decode(sample["input_ids"][:50].tolist()))
