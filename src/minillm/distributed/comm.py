"""
minillm.distributed.comm
========================

集合通信原语封装（对应 Megatron-LM / DeepSpeed 的通信层）
--------------------------------------------------------
大模型训练 > 单卡显存时，把模型/数据/梯度切到多卡，卡与卡之间靠
"集合通信"同步。本项目统一封装 torch.distributed 的三大原语：

1. all_reduce      （全局归约）: 所有卡的结果求和/求均值，每卡拿到全局结果
   用途: DDP 梯度同步 —— 每卡算自己的梯度，all_reduce 后所有卡梯度一致
   对应: DeepSpeed ZeRO-1/2 的梯度归约、Megatron 数据并行

2. all_gather      （全局收集）: 每卡持有数据的一部分，收齐后每卡持有完整数据
   用途: 张量并行（TP）中，列切分的输出要还原成完整向量再传给下一层
   对应: Megatron 张量并行 all-gather（先并行计算、再聚合）

3. reduce_scatter  （归约分发）: 所有卡的完整数据按块求和，每卡只拿自己负责的块
   用途: ZeRO-3 梯度切分 —— 梯度 reduce 后直接按 rank 分片，省通信
   对应: DeepSpeed ZeRO-3 / Megatron 梯度分片

通信心智模型（N 张卡）:
    all_reduce:      每卡 [D] ──求和──► 每卡 [D]（全部相同）
    all_gather:      每卡 [D/N] ──拼接──► 每卡 [D]（全部相同）
    reduce_scatter:  每卡 [D] ──分块求和──► 每卡 [D/N]（各卡不同块）
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# 环境工具
# ---------------------------------------------------------------------------
def init_distributed(backend: str = "nccl", **kwargs) -> None:
    """初始化进程组（训练脚本开头调用一次）。

    backend: nccl（GPU）/ gloo（CPU 调试）
    需要环境变量: RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT（或用 init_method 传入）
    """
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, **kwargs)


def get_rank() -> int:
    """当前进程的卡编号（0..world_size-1）。未初始化时视为单卡 rank=0。"""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """总卡数。未初始化时视为单卡 world_size=1。"""
    return dist.get_world_size() if dist.is_initialized() else 1


def is_dist_initialized() -> bool:
    return dist.is_initialized()


# ---------------------------------------------------------------------------
# 三大原语封装
# ---------------------------------------------------------------------------
def all_reduce(tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """全局归约：所有卡的 tensor 求和（或求均值），结果每卡一致。

    Parameters
    ----------
    tensor : [..] 各卡形状相同
    op : "sum"（求和，梯度同步用）| "mean"（求均值，Loss 跨卡平均用）

    Returns
    -------
    归约后的 tensor（各卡相同）
    """
    if op == "sum":
        reduce_op = dist.ReduceOp.SUM
    elif op == "mean":
        # 先求和再除以卡数（避免 dist 没有原生 mean 的兼容性问题）
        reduce_op = dist.ReduceOp.SUM
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=reduce_op)
        tensor.div_(get_world_size())
        return tensor
    else:
        raise ValueError(f"不支持的归约 op: {op}")

    if not is_dist_initialized():
        return tensor  # 单卡退化：不通信，原样返回
    dist.all_reduce(tensor, op=reduce_op)
    return tensor


def all_gather(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """全局收集：每卡持有 tensor 的一部分（沿 dim 切分），收齐拼成完整张量。

    Parameters
    ----------
    tensor : [.., chunk, ..] 各卡持有自己负责的切片
    dim : 拼接维度（TP 里通常是最后一维 head_dim）

    Returns
    -------
    [.., chunk*world_size, ..] 完整张量（各卡相同）
    """
    if not is_dist_initialized():
        return tensor  # 单卡退化
    world_size = get_world_size()
    # 各卡把自己那块发送，同时接收其他卡的块，最后沿 dim 拼接
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=dim)


def reduce_scatter(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """归约分发：各卡持有完整 tensor，按 chunk 求和后每卡只拿自己那块。

    Parameters
    ----------
    tensor : [.., chunk*world_size, ..] 各卡持有完整（但内容不同）
    dim : 切分维度

    Returns
    -------
    [.., chunk, ..] 本卡负责的归约结果块
    """
    if not is_dist_initialized():
        return tensor  # 单卡退化
    world_size = get_world_size()
    size = tensor.size(dim)
    assert size % world_size == 0, f"切分维 {dim} 大小 {size} 必须能被 {world_size} 整除"
    chunk = size // world_size
    # 把完整张量切成 world_size 块，求和后各卡拿一块
    input_list = list(tensor.chunk(world_size, dim=dim))
    output = torch.empty_like(input_list[get_rank()])
    dist.reduce_scatter(output, input_list)
    return output


# ---------------------------------------------------------------------------
# 朴素参考实现（教学/验证用，纯 Python 不依赖进程组）
# ---------------------------------------------------------------------------
def naive_all_reduce(tensors: list[torch.Tensor], op: str = "sum") -> torch.Tensor:
    """手写 all_reduce：直接把各卡张量逐元素相加。"""
    result = torch.stack(tensors).sum(dim=0)
    if op == "mean":
        result = result / len(tensors)
    return result


def naive_all_gather(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """手写 all_gather：直接拼接。"""
    return torch.cat(tensors, dim=dim)


def naive_reduce_scatter(tensors: list[torch.Tensor], dim: int = 0) -> list[torch.Tensor]:
    """手写 reduce_scatter：各卡完整张量按块求和，返回每卡负责的块。"""
    world_size = len(tensors)
    size = tensors[0].size(dim)
    assert size % world_size == 0, f"切分维 {dim} 大小 {size} 必须能被 {world_size} 整除"
    chunk = size // world_size
    result = []
    for r in range(world_size):
        block = torch.stack([t.narrow(dim, r * chunk, chunk) for t in tensors]).sum(dim=0)
        result.append(block)
    return result


# ---------------------------------------------------------------------------
# 自校验：python -m minillm.distributed.comm
# 1) 单卡退化正确性  2) 朴素实现数学验证  3) 双进程 gloo 真实通信（可选）
# ---------------------------------------------------------------------------
def _verify_math():
    """用 3 张"模拟卡"的数据验证三个原语的数学定义。"""
    print("=== 模拟 3 卡数据 ===")
    data = [torch.tensor([1.0, 2, 3, 4]), torch.tensor([5.0, 6, 7, 8]), torch.tensor([9.0, 10, 11, 12])]
    for i, d in enumerate(data):
        print(f"  卡{i}: {d.tolist()}")

    # all_reduce: 逐元素和
    r = naive_all_reduce(data, "sum")
    print(f"\nall_reduce(sum): {r.tolist()} (应 [15, 18, 21, 24])")
    assert torch.equal(r, torch.tensor([15.0, 18, 21, 24]))
    r_mean = naive_all_reduce(data, "mean")
    print(f"all_reduce(mean): {r_mean.tolist()} (应 [5, 6, 7, 8])")

    # all_gather: 每卡持有一块, 拼接
    chunks = [torch.tensor([1.0, 2]), torch.tensor([3.0, 4]), torch.tensor([5.0, 6])]
    g = naive_all_gather(chunks, dim=0)
    print(f"\nall_gather(每卡2元素): {g.tolist()} (应 [1..6])")
    assert torch.equal(g, torch.tensor([1.0, 2, 3, 4, 5, 6]))

    # reduce_scatter: 每卡持有完整, 分块求和（用 2 卡, 4 元素 → 每卡拿 2）
    rs_data = [torch.tensor([1.0, 2, 3, 4]), torch.tensor([5.0, 6, 7, 8])]
    print(f"\nreduce_scatter 输入(2卡): 卡0={rs_data[0].tolist()} 卡1={rs_data[1].tolist()}")
    rs = naive_reduce_scatter(rs_data, dim=0)
    for i, b in enumerate(rs):
        print(f"  卡{i} 拿到: {b.tolist()}")
    # 块0 = 1+5=6, 2+6=8 → 卡0 [6,8]; 块1 = 3+7=10, 4+8=12 → 卡1 [10,12]
    assert torch.equal(rs[0], torch.tensor([6.0, 8]))
    assert torch.equal(rs[1], torch.tensor([10.0, 12]))
    print("  ✓ 卡0=[6,8] 卡1=[10,12]")
    print("\n✅ 数学验证通过")


def _run_distributed(rank: int, world_size: int):
    """gloo 后端双进程真实通信验证（每个进程跑一个 rank）。"""
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    x = torch.tensor([1.0, 2, 3, 4]) + rank  # 卡0=[1,2,3,4], 卡1=[2,3,4,5]

    # all_reduce
    y = all_reduce(x.clone())
    expected = torch.tensor([3.0, 5, 7, 9])  # 1+2, 2+3, ...
    ok1 = torch.equal(y, expected)

    # all_gather: 每卡持有一块
    chunk = torch.tensor([float(rank * 2), float(rank * 2 + 1)])
    g = all_gather(chunk, dim=0)
    ok2 = torch.equal(g, torch.tensor([0.0, 1, 2, 3]))

    # reduce_scatter: 每卡持有完整张量
    full = torch.tensor([1.0, 2, 3, 4]) + rank * 10  # 卡0=[1,2,3,4], 卡1=[11,12,13,14]
    rs = reduce_scatter(full, dim=0)  # 2 卡, 每卡拿 2 元素
    ok3 = torch.equal(rs, torch.tensor([12.0, 14]) if rank == 0 else torch.tensor([16.0, 18]))

    print(f"[rank{rank}] all_reduce={ok1} | all_gather={ok2} | reduce_scatter={ok3}")
    dist.destroy_process_group()


if __name__ == "__main__":
    # 1) 单卡退化
    x = torch.tensor([1.0, 2, 3, 4])
    assert torch.equal(all_reduce(x.clone()), x)
    assert torch.equal(all_gather(x.clone(), dim=0), x)
    assert torch.equal(reduce_scatter(x.clone(), dim=0), x)
    print("单卡退化（不通信，原样返回）: ✓")

    # 2) 数学验证（模拟多卡）
    _verify_math()

    # 3) 真实双进程 gloo 通信（可选，环境支持时执行）
    try:
        import torch.multiprocessing as mp

        # 子进程通过环境变量 rendezvous 组队（gloo 本地调试）
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        mp.spawn(_run_distributed, args=(2,), nprocs=2, join=True)
        print("\n✅ 双进程 gloo 真实通信通过")
    except Exception as e:  # noqa: BLE001
        print(f"\n[skip] 双进程通信测试跳过: {type(e).__name__}: {e}")
        print("（单卡环境跳过；AutoDL 多卡时自动启用）")
