"""
minillm.data.download
=====================

按 DataConfig 下载所有数据源，统一转成 jsonl 格式落到 data_cache_dir。

输出格式
--------
每个数据源生成一个 ``<source_name>.jsonl``，每行一条 JSON：

.. code-block:: json

    {"text": "这是一段中文文本..."}

设计要点
--------
1. **幂等**：如果目标 jsonl 已存在且非空，直接跳过，不重复下载。
2. **不依赖网络做训练**：下载完后训练阶段只读本地 jsonl，不连 HF。
3. **保留权重信息**：同时写一个 ``manifest.json``，记录每个文件的路径、条数、
   采样权重，后面 DataLoader 直接读这个 manifest 做混合采样。
4. **lazy import datasets**：只在真正下载时才 import，避免 import 本模块就触发网络。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from minillm.config.train_config import DataConfig, DataSourceConfig


def _download_one_source(source: DataSourceConfig, output_dir: Path) -> Path:
    """下载单个数据源，转成 jsonl。

    Returns
    -------
    Path : 生成的 jsonl 文件路径
    """
    out_path = output_dir / f"{source.name}.jsonl"

    # 幂等：已存在且非空就跳过
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] {source.name} 已存在: {out_path}")
        return out_path

    print(f"[download] {source.name}: path={source.path}, subset={source.subset}, split={source.split}")

    # 延迟 import，避免 import 本模块就拉网络
    from datasets import load_dataset

    # load_dataset 支持路径 + subset + split 的组合
    # 例：load_dataset("wikimedia/wikipedia", "20231101.zh", split="train[:5%]")
    if source.subset:
        ds = load_dataset(source.path, source.subset, split=source.split)
    else:
        ds = load_dataset(source.path, split=source.split)

    # 写出 jsonl
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for example in tqdm(ds, desc=f"  writing {source.name}"):
            text = example.get(source.text_field, "")
            if not text or not text.strip():
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n += 1

    print(f"[done] {source.name}: {n} 条 -> {out_path}")
    return out_path


def download_all(data_config: DataConfig) -> Dict[str, dict]:
    """下载 DataConfig 里配置的所有数据源。

    Returns
    -------
    manifest : dict
        {source_name: {"path": ..., "num_lines": ..., "weight": ...}}
    """
    output_dir = Path(data_config.data_cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, dict] = {}

    for source in data_config.sources:
        jsonl_path = _download_one_source(source, output_dir)

        # 数一下行数（已经写好了，快速 wc 一下）
        with open(jsonl_path, "r", encoding="utf-8") as f:
            num_lines = sum(1 for _ in f)

        manifest[source.name] = {
            "path": str(jsonl_path),
            "num_lines": num_lines,
            "weight": source.weight,
        }

    # 写 manifest.json，后面 DataLoader 直接读这个文件
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n[manifest] 已写入: {manifest_path}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    return manifest


# ---------------------------------------------------------------------------
# 自校验 / 命令行入口
#   python -m minillm.data.download configs/data/pretrain_zh.yaml
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    
    from minillm.config.train_config import load_data_config

    if len(sys.argv) > 1:
        data_yaml = sys.argv[1]
    else:
        data_yaml = "configs/data/pretrain_zh.yaml"

    print(f"[main] 读取数据配置: {data_yaml}")
    cfg = load_data_config(data_yaml)
    download_all(cfg)
