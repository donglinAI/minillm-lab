"""
minillm.config.train_config
============================

项目统一的配置入口。

设计原则
--------
1. 用 ``dataclass`` 定义配置，字段即文档，IDE 自动补全。
2. 不引入 hydra / omegaconf，只靠 ``PyYAML`` + 一个手工 ``from_dict``，
   这样每一行加载逻辑都能读、能改、能调试。
3. 训练配置里通过路径引用 model / data 的 yaml，加载时递归展开，
   最终得到一棵完整的配置树。

使用方式
--------
>>> from minillm.config.train_config import load_train_config
>>> cfg = load_train_config("configs/train/pretrain_single_gpu.yaml")
>>> print(cfg.model.hidden_size, cfg.optimizer.lr)
512 0.0003
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# 模型配置
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Decoder-only Transformer 的结构超参（LLaMA 式）。"""

    model_type: str = "llama-like"

    # ---- 基础维度 ----
    vocab_size: int = 64000
    hidden_size: int = 512
    intermediate_size: int = 1408          # SwiGLU 的中间层维度
    num_hidden_layers: int = 8

    # ---- 注意力 ----
    num_attention_heads: int = 8         # query 头数
    num_key_value_heads: int = 2          # GQA：KV 头数；=num_attention_heads 即为 MHA

    # ---- 归一化 / 激活 ----
    hidden_act: str = "silu"              # SwiGLU
    norm_eps: float = 1e-6
    rms_norm: bool = True                  # True=RMSNorm，False=LayerNorm
    rope_theta: float = 10000.0          # RoPE 的 base 频率

    # ---- 序列长度与初始化 ----
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02       # 截断正态的标准差
    residual_scale: float = 1.0            # 残差支路乘子（GPT-2 式 N^-0.5 可后续调）

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        # 只取 dataclass 里声明过的字段，忽略 yaml 里多余的 key，避免 typo 静默生效
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ---------------------------------------------------------------------------
# 数据配置
# ---------------------------------------------------------------------------
@dataclass
class DataSourceConfig:
    """单个数据集源。"""

    name: str
    path: str                              # HF datasets 上的 repo id
    subset: Optional[str] = None           # 数据集的 config 名（如 "20231101.zh"）
    split: str = "train"
    text_field: str = "text"               # 文本字段名
    weight: float = 1.0                    # 混合采样权重

    @classmethod
    def from_dict(cls, d: dict) -> "DataSourceConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class DataConfig:
    data_cache_dir: str = "./data_cache"
    # 数据源列表（预训练 / SFT / DPO 各自的 yaml 都填这里，按 name 区分用途）
    sources: List[DataSourceConfig] = field(default_factory=list)

    # ---- 序列处理 ----
    seq_len: int = 1024                    # 打包后的定长序列
    packing: bool = True                   # 短文本拼接成定长，提升吞吐
    shuffle_buffer: int = 10000

    @classmethod
    def from_dict(cls, d: dict) -> "DataConfig":
        # sources 是一个列表，需要逐项转成 DataSourceConfig
        sources = [DataSourceConfig.from_dict(x) for x in d.get("sources", [])]
        valid_scalar = {
            "data_cache_dir", "seq_len", "packing", "shuffle_buffer",
        }
        return cls(
            sources=sources,
            **{k: v for k, v in d.items() if k in valid_scalar},
        )


# ---------------------------------------------------------------------------
# 优化器 / 学习率调度
# ---------------------------------------------------------------------------
@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    eps: float = 1e-8

    @classmethod
    def from_dict(cls, d: dict) -> "OptimizerConfig":
        betas = tuple(d.get("betas", (0.9, 0.95)))
        return cls(
            name=d.get("name", "adamw"),
            lr=float(d.get("lr", 3e-4)),
            betas=(float(betas[0]), float(betas[1])),
            weight_decay=float(d.get("weight_decay", 0.1)),
            eps=float(d.get("eps", 1e-8)),
        )


@dataclass
class LRSchedulerConfig:
    name: str = "cosine"
    warmup_steps: int = 200
    max_steps: int = 5000
    min_lr_ratio: float = 0.1              # 最低 lr = base_lr * min_lr_ratio

    @classmethod
    def from_dict(cls, d: dict) -> "LRSchedulerConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ---------------------------------------------------------------------------
# 分布式配置
# ---------------------------------------------------------------------------
@dataclass
class DistributedConfig:
    """对应 Megatron + DeepSpeed 的并行组合。

    zero_stage:
        0 = 不切分
        1 = 切优化器状态（DeepSpeed ZeRO-1）
        2 = 切梯度（ZeRO-2）
        3 = 切参数（ZeRO-3）
    """

    backend: str = "nccl"
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1         # Megatron 张量并行
    pipeline_parallel_size: int = 1        # 流水线并行
    zero_stage: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "DistributedConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ---------------------------------------------------------------------------
# 训练总配置
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # ---- 通过路径引用的子配置（加载时会被替换成真正的对象）----
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    output_dir: str = "./output/run"

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    # ---- 训练循环 ----
    batch_size: int = 32                    # 每个 micro-batch
    grad_accum_steps: int = 4               # 梯度累积步数
    grad_clip: float = 1.0
    max_steps: int = 5000
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000

    tensorboard_dir: str = "./output/run/tb"
    seed: int = 42


# ---------------------------------------------------------------------------
# 加载入口
# ---------------------------------------------------------------------------
def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_model_config(path: str) -> ModelConfig:
    """从 yaml 文件加载 ModelConfig。"""
    return ModelConfig.from_dict(_read_yaml(path))


def load_data_config(path: str) -> DataConfig:
    """从 yaml 文件加载 DataConfig。"""
    return DataConfig.from_dict(_read_yaml(path))


def load_train_config(path: str) -> TrainConfig:
    """从训练配置 yaml 加载完整 TrainConfig。

    训练配置里的 ``model_config`` / ``data_config`` 字段是另外两个 yaml 的路径，
    这里递归把它们读成真正的 ModelConfig / DataConfig 对象。
    """
    raw = _read_yaml(path)
    base_dir = os.path.dirname(os.path.abspath(path))

    # 解析 model / data 子配置（路径相对于当前 yaml 所在目录）
    model_path = os.path.join(base_dir, raw["model_config"])
    data_path = os.path.join(base_dir, raw["data_config"])

    cfg = TrainConfig(
        model=load_model_config(model_path),
        data=load_data_config(data_path),
        output_dir=raw.get("output_dir", "./output/run"),
        optimizer=OptimizerConfig.from_dict(raw.get("optimizer", {})),
        lr_scheduler=LRSchedulerConfig.from_dict(raw.get("lr_scheduler", {})),
        distributed=DistributedConfig.from_dict(raw.get("distributed", {})),
        batch_size=int(raw.get("batch_size", 32)),
        grad_accum_steps=int(raw.get("grad_accum_steps", 4)),
        grad_clip=float(raw.get("grad_clip", 1.0)),
        max_steps=int(raw.get("max_steps", 5000)),
        log_interval=int(raw.get("log_interval", 10)),
        eval_interval=int(raw.get("eval_interval", 500)),
        save_interval=int(raw.get("save_interval", 1000)),
        tensorboard_dir=raw.get("tensorboard_dir", "./output/run/tb"),
        seed=int(raw.get("seed", 42)),
    )
    return cfg


# ---------------------------------------------------------------------------
# 直接运行此文件时做一次自校验：python -m minillm.config.train_config
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        p = sys.argv[1]
    else:
        # 默认指向项目里的示例配置（相对仓库根目录）
        p = os.path.join(
            os.path.dirname(__file__), "../../../configs/train/pretrain_single_gpu.yaml"
        )
    cfg = load_train_config(p)
    print("=== ModelConfig ===")
    print(cfg.model)
    print("=== DataConfig ===")
    print(cfg.data)
    print("=== Optimizer ===")
    print(cfg.optimizer)
    print("=== LR Scheduler ===")
    print(cfg.lr_scheduler)
    print("=== Distributed ===")
    print(cfg.distributed)
    print(f"batch_size={cfg.batch_size}, grad_accum={cfg.grad_accum_steps}, "
          f"effective_bs={cfg.batch_size * cfg.grad_accum_steps}")
