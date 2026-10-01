"""
minillm.distributed.ddp
========================
真实 DDP（Data Parallel）封装 —— 对应 PyTorch DistributedDataParallel + NCCL

★ 与 engine.py 的区别:
    engine.py 是"虚拟世界模拟器": 用 Python 对象模拟 8 个进程,
    通信用 cat/sum 模拟, 不能真实跑多卡。
    本文件是可以真实在单机多卡 GPU 上跑的版本:
    torchrun --nproc_per_node=N 会启动 N 个真实进程,
    每个进程管一张卡, 进程间用 NCCL 通信。

真实 DDP 的四个要点:
  1. 进程网格: 环境变量 RANK(全局编号) / LOCAL_RANK(本机编号) /
     WORLD_SIZE(总进程数), 由 torchrun 注入
  2. 数据分片: DistributedSampler 把数据集切成 N 份, 每个 rank 拿不同部分
  3. 梯度平均: backward 时 DDP 自动做 all-reduce(求和 ÷ N)
     ★ 为什么必须平均: 每 rank 的 loss 是各自 batch 的 mean,
        若求和, 梯度会差 N 倍, 参数更新就错了
        (engine.py 里修过的 bug, 真实版由 torch DDP 保证)
  4. checkpoint: 只允许 rank 0 写盘(各卡权重一致, 写 N 份是浪费)

用法(配合 Trainer):
    rank, world, device = init_distributed()      # 必须在构建模型前调用
    model = wrap_ddp(model, device, world)
    loader = make_dataloader(ds, batch_size, rank, world, shuffle=True)
    trainer = Trainer(model, opt, sch, loader, device=device)
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def init_distributed(backend: str = "nccl") -> tuple[int, int, str]:
    """初始化进程组（torchrun 环境）。返回 (rank, world_size, device_str)。

    - 多卡(torchrun): 用注入的 RANK/LOCAL_RANK/WORLD_SIZE, NCCL 后端,
      每进程绑定一张卡 (cuda:LOCAL_RANK)
    - 单卡(直接 python): 不初始化进程组, 自动选 cuda/cpu
    """
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, rank=rank, world_size=world)
        device = f"cuda:{local_rank}"
        print(f"  [ddp] rank {rank}/{world} 已初始化进程组, device=cuda:{local_rank}")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return rank, world, device


def is_main(rank: int) -> bool:
    """只有 rank 0 做日志/保存等主进程工作。"""
    return rank == 0


def wrap_ddp(model: torch.nn.Module, device: str, world_size: int) -> torch.nn.Module:
    """模型搬到 device；多卡时包 DDP（backward 自动 all-reduce 平均梯度）。"""
    model = model.to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[int(os.environ["LOCAL_RANK"])])
    return model


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """去掉 DDP 包装, 拿原始模型（保存/加载权重时用, 避免 module. 前缀）。"""
    return model.module if hasattr(model, "module") else model


def make_dataloader(
    ds: Dataset,
    batch_size: int,
    rank: int,
    world_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """构造 DataLoader；多卡时用 DistributedSampler 给每个 rank 分不同的数据。

    注意: DistributedSampler 自带 shuffle, 此时 DataLoader 不能再 shuffle,
    否则每 epoch 每卡拿到的子集会被打乱成"随机取全集"（破坏分片语义）。
    """
    if world_size > 1:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                     shuffle=shuffle)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          shuffle=False, drop_last=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def sync_barrier(rank: int) -> None:
    """多卡时同步所有进程（DDP 训练步调必须一致）。"""
    if rank is not None and dist.is_initialized():
        dist.barrier()


def broadcast_model(model: torch.nn.Module, src: int = 0) -> None:
    """把 rank0 的权重广播到所有 rank（保证起点一致；torch DDP 内部也会做）。

    教学说明: 单机多卡初始权重来自同一份随机种子时本可省略,
    但显式广播是分布式训练的通用姿势(多机/混合初始化时必需)。
    """
    if dist.is_initialized() and dist.get_world_size() > 1:
        for p in model.parameters():
            dist.broadcast(p.data, src=src)
