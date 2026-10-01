"""
minillm.pretrain.train
========================
预训练真实训练入口（单卡 / 多卡 DDP）
----------------------------------------------------------------
用法:
    单卡: python -m minillm.pretrain.train \\
          --config configs/train/pretrain_single_gpu.yaml
    多卡: torchrun --nproc_per_node=4 -m minillm.pretrain.train \\
          --config configs/train/pretrain_ddp.yaml
    冒烟: python -m minillm.pretrain.train --config ... \\
          --mini --steps 5 --split "train[:200]" --seq_len 64

流程:
    DDP 初始化(多卡) → 下载数据(HF→jsonl→manifest, 幂等) →
    tokenize 缓存 → PackedPretrainDataset → 模型(vocab 自动修正) →
    DDP 包装 + 数据分片 → Trainer(手写 AdamW + WarmupCosine) →
    checkpoint(resume 黄金语义: 模型+优化器+步数)

多卡要点(真实 DDP):
    - torchrun 启动 N 个进程, 每进程一张卡(NCCL)
    - DistributedSampler 数据分片: 每卡看不同的 batch
    - DDP backward 自动 all-reduce 平均梯度(不是求和!)
    - checkpoint 只在 rank 0 写盘
"""

from __future__ import annotations

import argparse
import os
import random

import torch

from minillm.config.train_config import DataConfig, load_train_config
from minillm.data.download import download_all
from minillm.data.dataset import PackedPretrainDataset
from minillm.distributed.ddp import init_distributed, is_main, make_dataloader, wrap_ddp
from minillm.models.transformer import MiniLLM, MiniLLMConfig
from minillm.tokenizer.tokenizer import Tokenizer
from minillm.optim.adamw import AdamW
from minillm.optim.scheduler import WarmupCosineScheduler
from minillm.pretrain.trainer import Trainer


def build_model(model_cfg, tokenizer, mini: bool) -> MiniLLM:
    """从 ModelConfig 构建 MiniLLM；vocab 按真实 tokenizer 修正。"""
    real_vocab = max(tokenizer.vocab_size, tokenizer.eos_token_id) + 1
    if model_cfg.vocab_size != real_vocab:
        print(f"  [warn] 配置 vocab_size={model_cfg.vocab_size} 与 tokenizer 实际 "
              f"{real_vocab} 不一致, 已自动修正为 {real_vocab}")
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


def main() -> None:
    ap = argparse.ArgumentParser(description="预训练真实训练(单卡/多卡 DDP)")
    ap.add_argument("--config", default="configs/train/pretrain_single_gpu.yaml")
    ap.add_argument("--steps", type=int, default=None, help="覆盖 max_steps")
    ap.add_argument("--seq_len", type=int, default=None, help="覆盖序列长度(冒烟用)")
    ap.add_argument("--split", default=None,
                    help="覆盖数据 split(冒烟用, 如 train[:200])")
    ap.add_argument("--mini", action="store_true",
                    help="CPU 冒烟: 小模型(hidden=96, layers=2)")
    ap.add_argument("--resume", default=None, help="从 checkpoint 恢复")
    ap.add_argument("--device", default="auto",
                    help="单卡时: auto | cuda | cpu(多卡由 torchrun 决定)")
    args = ap.parse_args()

    # 0) DDP 初始化(多卡) / 设备选择(单卡) —— 必须在构建模型前
    rank, world, device = init_distributed()
    if world == 1 and args.device != "auto":
        device = args.device
    print(f"  [pretrain] rank={rank}/{world}, device={device}")

    # 1) 配置
    cfg = load_train_config(args.config)
    torch.manual_seed(cfg.seed + rank)      # 每 rank 不同种子(数据分片后互补)
    random.seed(cfg.seed + rank)

    # 2) tokenizer + 模型
    tok = Tokenizer()
    if args.mini:
        if is_main(rank):
            print("  [warn] --mini: 小模型冒烟(hidden=96, layers=2)")
        cfg.model.hidden_size = 96
        cfg.model.intermediate_size = 256
        cfg.model.num_hidden_layers = 2
        cfg.model.num_attention_heads = 4
        cfg.model.num_key_value_heads = 1
    model = build_model(cfg.model, tok, args.mini)
    model = wrap_ddp(model, device, world)
    if is_main(rank):
        n = sum(p.numel() for p in model.parameters())
        print(f"  模型参数量: {n/1e6:.1f}M | vocab={model.lm_head.out_features}")

    # 3) 数据: 下载(幂等) → tokenize 缓存 → 打包数据集
    if is_main(rank):
        print("[pretrain] 下载数据(已缓存则跳过)...")
    data_cfg = _load_data_config(args.config)
    if args.split:
        for s in data_cfg.sources:
            s.split = args.split
    manifest = download_all(data_cfg)      # 幂等: jsonl 存在则跳过
    if is_main(rank):
        print(f"[pretrain] manifest: {len(manifest)} 个数据源")

    seq_len = args.seq_len or data_cfg.seq_len
    cache_dir = data_cfg.data_cache_dir
    ds = PackedPretrainDataset(
        os.path.join(data_cfg.data_cache_dir, "manifest.json"),
        tok, seq_len=seq_len, cache_dir=cache_dir)
    if is_main(rank):
        print(f"[pretrain] 序列长度 {seq_len}, 样本数 {len(ds)}")

    # 4) DataLoader(DDP 数据分片) + 优化器 + Trainer
    loader = make_dataloader(ds, cfg.batch_size, rank, world, shuffle=True)
    opt = AdamW(model.parameters(), lr=cfg.optimizer.lr,
                weight_decay=cfg.optimizer.weight_decay)
    sch = WarmupCosineScheduler(
        lr_max=cfg.optimizer.lr,
        warmup_steps=cfg.lr_scheduler.warmup_steps,
        total_steps=args.steps or cfg.lr_scheduler.max_steps,
        min_lr=cfg.optimizer.lr * cfg.lr_scheduler.min_lr_ratio,
    )
    trainer = Trainer(
        model, opt, sch, loader,
        max_steps=args.steps or cfg.max_steps,
        grad_accum_steps=cfg.grad_accum_steps,
        grad_clip=cfg.grad_clip,
        log_interval=cfg.log_interval,
        save_interval=cfg.save_interval,
        output_dir=cfg.output_dir,
        device=device,
    )
    if args.resume:
        if is_main(rank) or world == 1:
            trainer.load_checkpoint(args.resume)
        if world > 1:
            # 恢复后各卡步数/权重一致; 后续每步 DDP 自动同步梯度
            pass

    # 5) 训练(多卡: 每步梯度 all-reduce 平均, 各卡步调一致)
    if is_main(rank):
        print("[pretrain] 开始训练...")
    trainer.train()

    # 6) 最终 checkpoint(rank 0 保存一次即可)
    if is_main(rank) or world == 1:
        trainer.save_checkpoint(os.path.join(cfg.output_dir, "final.pt"))
        print(f"[pretrain] 完成。最终权重: {os.path.join(cfg.output_dir, 'final.pt')}")


def _load_data_config(config_path: str) -> DataConfig:
    """从训练配置里 data_config 指向的 yaml 加载 DataConfig。"""
    import yaml
    raw = yaml.safe_load(open(config_path, encoding="utf-8")) or {}
    base = os.path.dirname(os.path.abspath(config_path))
    data_path = os.path.normpath(os.path.join(base, raw["data_config"]))
    with open(data_path, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    return DataConfig.from_dict(d)


if __name__ == "__main__":
    main()
