"""
minillm.posttrain.sft.train
============================
SFT 真数据训练入口
----------------------------------------------------------------
用法:
    python -m minillm.posttrain.sft.train \
        --config configs/train/sft_single_gpu.yaml              # 正式训练
    python -m minillm.posttrain.sft.train \
        --config configs/train/sft_single_gpu.yaml --limit 300 \
        --steps 15 --mini                                        # CPU 冒烟
    python -m minillm.posttrain.sft.train \
        --config ... --resume ./output/sft_tiny/ckpt_step100.pt # 断点续训

流程:
    读 yaml 配置 → 构建模型(真实 tokenizer 词表) → 读 jsonl 指令数据
    → SFTDataset → SFTTrainer(复用预训练循环) → 训练 → checkpoint
    → 生成验证(抽 3 条指令, greedy 生成, 看是否学会回答)
"""

from __future__ import annotations

import argparse
import json
import os
import random

import torch
import yaml

from minillm.config.train_config import load_train_config
from minillm.models.transformer import MiniLLM, MiniLLMConfig
from minillm.tokenizer.tokenizer import Tokenizer
from minillm.posttrain.sft.dataset import build_chat
from minillm.posttrain.sft.trainer import SFTTrainer


def load_jsonl(path: str, limit: int | None = None) -> list[dict]:
    """读 sft jsonl → list[{"instruction","output"}]。"""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("instruction") and row.get("output"):
                items.append(row)
            if limit and len(items) >= limit:
                break
    return items


def build_model(model_cfg, tokenizer) -> MiniLLM:
    """从 ModelConfig 构建 MiniLLM。

    ★ vocab_size 修正: tiny.yaml 默认 64000, 但真实 tokenizer
      (Qwen2.5-0.5B) 词表 151643, eos=151643 排在词表末尾 ——
      模型 vocab 必须覆盖所有会出现的 id, 否则 embedding 越界。
    """
    real_vocab = max(tokenizer.vocab_size, tokenizer.eos_token_id) + 1
    if model_cfg.vocab_size != real_vocab:
        print(f"  [warn] tiny.yaml vocab_size={model_cfg.vocab_size} "
              f"与 tokenizer 实际 {real_vocab} 不一致, 已自动修正为 {real_vocab}")
    return MiniLLM(MiniLLMConfig(
        vocab_size=real_vocab,
        hidden_size=model_cfg.hidden_size,
        intermediate_size=model_cfg.intermediate_size,
        num_hidden_layers=model_cfg.num_hidden_layers,
        num_attention_heads=model_cfg.num_attention_heads,
        num_key_value_heads=model_cfg.num_key_value_heads,
        max_position_embeddings=model_cfg.max_position_embeddings,
        rope_theta=model_cfg.rope_theta,
        norm_eps=model_cfg.norm_eps,
        initializer_range=model_cfg.initializer_range,
        tie_word_embeddings=True,
    ))


def verify_generation(model, tokenizer, items, k=3, max_new=24, device="cpu"):
    """抽 k 条指令, greedy 生成回答, 打印对比（学会回答了吗）。

    注意: 训练步数很少/模型很小(lr 还没走完 warmup)时, 生成乱码是正常的,
    正式训练(max_steps=500, GPU)后才会像样。
    """
    print("\n" + "-" * 66)
    print("生成验证（训练后的模型回答效果；训练不足时乱码属正常现象）")
    print("-" * 66)
    for item in random.sample(items, min(k, len(items))):
        prompt_txt, ans_txt = build_chat(item)
        pid = tokenizer.encode(prompt_txt, add_special_tokens=False)
        gen = list(pid)
        for _ in range(max_new):
            logits = model(torch.tensor([gen], device=device))[0, -1]
            nxt = logits.argmax().item()
            if nxt == tokenizer.eos_token_id:
                break
            gen.append(nxt)
        pred = tokenizer.decode(gen[len(pid):])
        print(f"  指令: {item['instruction'][:40]}")
        print(f"  模型: {pred[:60]}")
        print(f"  真实: {item['output'][:60]}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT 真数据训练")
    ap.add_argument("--config", default="configs/train/sft_single_gpu.yaml")
    ap.add_argument("--limit", type=int, default=None,
                    help="只读前 N 条指令数据（冒烟用）")
    ap.add_argument("--steps", type=int, default=None, help="覆盖 max_steps")
    ap.add_argument("--seq_len", type=int, default=None, help="覆盖 seq_len(冒烟用)")
    ap.add_argument("--warmup", type=int, default=None, help="覆盖 warmup_steps(冒烟用)")
    ap.add_argument("--mini", action="store_true",
                    help="CPU 冒烟: 用小模型(hidden=96, layers=2)")
    ap.add_argument("--resume", default=None, help="从 checkpoint 恢复")
    ap.add_argument("--device", default="auto",
                    help="auto(有 GPU 用 GPU, 否则 CPU) | cuda | cpu")
    args = ap.parse_args()

    # 0) 设备选择: 训练必须真的在 GPU 上跑, 否则 GPU 利用率 0
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[sft.train] 指定了 cuda 但环境没有 GPU, 请用 --device auto 或 cpu")
    print(f"[sft.train] 训练设备: {device}")

    # 1) 配置
    cfg = load_train_config(args.config)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # 数据配置路径: load_train_config 已把 data_config 替换成 DataConfig 对象,
    # 但 SFT 的自定义字段(jsonl_path 等)不在 DataConfig 里, 这里直接读原始 yaml
    raw_cfg = yaml.safe_load(open(args.config, encoding="utf-8")) or {}

    # 2) 模型 + tokenizer
    print("[sft.train] 构建模型 + tokenizer...")
    tok = Tokenizer()
    if args.mini:
        print("  [warn] --mini: 使用小模型冒烟（hidden=96, layers=2）")
        cfg.model.hidden_size = 96
        cfg.model.intermediate_size = 256
        cfg.model.num_hidden_layers = 2
        cfg.model.num_attention_heads = 4
        cfg.model.num_key_value_heads = 1
    model = build_model(cfg.model, tok)
    model = model.to(device)          # ★ 关键: 模型搬到训练设备(GPU/CPU)
    n_params = sum(p.numel() for p in model.parameters())
    real_vocab = model.lm_head.out_features
    print(f"  模型参数量: {n_params/1e6:.1f}M | vocab={real_vocab}")

    # 3) 数据
    with open(_resolve_path(args.config, raw_cfg["data_config"]), encoding="utf-8") as f:
        dcfg = yaml.safe_load(f) or {}
    jsonl_path = dcfg.get("jsonl_path", "data_cache/sft_zh.jsonl")
    seq_len = args.seq_len or dcfg.get("seq_len", 512)
    mode = dcfg.get("mode", "pad")
    max_train_samples = args.limit or dcfg.get("max_train_samples", 10000)

    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"找不到 SFT 数据 {jsonl_path}。请先运行:\n"
            f"  python scripts/download_data.py --dataset sft --limit 50000")
    items = load_jsonl(jsonl_path, max_train_samples)
    if not items:
        raise SystemExit(
            f"[sft.train] 错误: {jsonl_path} 中没有有效数据(0 条)。\n"
            f"  可能原因: 文件是空的/下载中断/字段格式不对。\n"
            f"  修复: rm -f {jsonl_path} && bash scripts/sft.sh（会重新下载）")
    print(f"[sft.train] 加载 {len(items)} 条指令数据 ← {jsonl_path}")

    # 4) 训练
    trainer = SFTTrainer(
        model, tok, items,
        seq_len=seq_len, mode=mode,
        batch_size=cfg.batch_size,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        warmup_steps=args.warmup if args.warmup is not None else cfg.lr_scheduler.warmup_steps,
        total_steps=cfg.lr_scheduler.max_steps,
        min_lr=cfg.optimizer.lr * cfg.lr_scheduler.min_lr_ratio,
        max_steps=args.steps or cfg.max_steps,
        grad_accum_steps=cfg.grad_accum_steps,
        grad_clip=cfg.grad_clip,
        log_interval=cfg.log_interval,
        save_interval=cfg.save_interval,
        output_dir=cfg.output_dir,
        device=device,
    )
    if args.resume:
        trainer.load_checkpoint(args.resume)

    print("[sft.train] 开始训练...")
    trainer.train()

    # 5) 生成验证
    verify_generation(model, tok, items, device=device)


def _resolve_path(config_path: str, ref: str) -> str:
    """把配置里相对路径解析为绝对路径（相对 config 所在目录）。
    用 normpath 归一化 ../ 和 ./，不能盲目 replace("./", "")。
    """
    base = os.path.dirname(os.path.abspath(config_path))
    return os.path.normpath(os.path.join(base, ref))


if __name__ == "__main__":
    main()
