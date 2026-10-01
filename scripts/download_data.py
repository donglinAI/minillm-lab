"""
scripts/download_data.py
========================
数据下载与预处理（SFT 部分）

用法:
    python scripts/download_data.py --dataset sft --limit 50000
        # 下载中文指令微调数据(默认 shibing624/alpaca-zh, 51k 条)
        # → data_cache/sft_zh.jsonl  每行 {"instruction": ..., "output": ...}

    python scripts/download_data.py --dataset pretrain
        # 预训练数据下载走 src/minillm/data/download.py（已有实现）

SFT 数据字段说明（alpaca 类数据集）:
    原始每条: {instruction, input, output}
    input 非空时是任务的补充输入, 拼进 instruction（Alpaca 原始做法）,
    预处理后 jsonl 只保留 {instruction, output}（SFTDataset 的输入格式）。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_SFT_DATASET = "shibing624/alpaca-zh"


def download_sft(dataset: str, limit: int, out_path: Path) -> None:
    """用 HF streaming 模式下载前 limit 条指令数据 → jsonl。

    streaming 的好处: 只拉取需要的条数, 不下载整个数据集。
    """
    from datasets import load_dataset

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download_data] 加载 {dataset} (streaming, 前 {limit} 条)...")
    ds = load_dataset(dataset, split="train", streaming=True)

    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            instruction = (row.get("instruction") or "").strip()
            inp = (row.get("input") or "").strip()
            output = (row.get("output") or "").strip()
            if not instruction or not output:
                continue
            # Alpaca 约定: 有 input 时拼进 instruction（任务 + 任务输入）
            if inp:
                instruction = f"{instruction}\n{inp}"
            f.write(json.dumps({"instruction": instruction,
                                "output": output}, ensure_ascii=False) + "\n")
            n += 1
            if n >= limit:
                break

    print(f"[download_data] 完成: {n} 条 → {out_path}")
    print(f"  示例: {json.dumps({'instruction': instruction, 'output': output[:30]}, ensure_ascii=False)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="数据下载与预处理")
    ap.add_argument("--dataset", choices=["sft", "pretrain"], default="sft")
    ap.add_argument("--limit", type=int, default=50000,
                    help="SFT 数据下载条数上限")
    ap.add_argument("--out", default=None, help="输出 jsonl 路径")
    args = ap.parse_args()

    if args.dataset == "sft":
        out = Path(args.out or "data_cache/sft_zh.jsonl")
        download_sft(DEFAULT_SFT_DATASET, args.limit, out)
    else:
        print("[download_data] 预训练数据请使用: python -m minillm.data.download（或参照其配置）")


if __name__ == "__main__":
    main()
