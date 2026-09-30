"""
scripts/dist_smoke.py
=====================
单机多卡 GPU 分布式冒烟测试：验证 comm.py 的三大原语在多卡真实环境跑通。

用法（在 AutoDL / 任意单机多卡 Linux 上）:
    torchrun --nproc_per_node=2 scripts/dist_smoke.py     # 2 卡
    torchrun --nproc_per_node=4 scripts/dist_smoke.py     # 4 卡
    torchrun --nproc_per_node=8 scripts/dist_smoke.py     # 8 卡

torchrun 会自动设置 RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT，
我们的 init_distributed(backend="nccl") 直接用默认 env:// rendezvous 即可。
"""

import torch

from minillm.distributed.comm import (
    all_gather,
    all_reduce,
    get_rank,
    get_world_size,
    init_distributed,
    reduce_scatter,
)


def main() -> None:
    # 1) 初始化进程组（nccl 后端，GPU 通信）
    init_distributed(backend="nccl")
    rank = get_rank()
    world = get_world_size()

    # 2) 每进程绑定一块 GPU（torchrun 第 rank 个进程用第 rank 块卡）
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    name = torch.cuda.get_device_name(rank)
    mem = torch.cuda.get_device_properties(rank).total_memory / 1e9
    print(f"[rank{rank}/{world}] GPU: {name} | 显存: {mem:.1f}GB")

    # 3) 构造每卡不同的数据，验证三大原语
    base = torch.arange(4, dtype=torch.float32, device=device)
    x = base + rank * 10.0  # 卡0=[0,1,2,3] 卡1=[10,11,12,13] ...
    print(f"[rank{rank}] 本地张量: {x.tolist()}")

    # --- all_reduce: 所有卡求和, 每卡应得到 sum(卡0..卡N-1) ---
    y = all_reduce(x.clone())
    expected_sum = torch.zeros_like(base, device=device)
    for r in range(world):
        expected_sum += base + r * 10.0
    exp = expected_sum
    ok1 = torch.equal(y, exp)
    print(f"[rank{rank}] all_reduce → {y.tolist()} (期望{exp.tolist()}) {'✅' if ok1 else '❌'}")

    # --- all_gather: 每卡持有一块, 拼成完整 ---
    chunk = torch.tensor([rank * 2.0, rank * 2 + 1.0], device=device)
    g = all_gather(chunk, dim=0)
    exp_g = torch.arange(world * 2, dtype=torch.float32, device=device)
    ok2 = torch.equal(g, exp_g)
    print(f"[rank{rank}] all_gather → {g.tolist()} (期望{exp_g.tolist()}) {'✅' if ok2 else '❌'}")

    # --- reduce_scatter: 每卡持有完整, 分块求和后各拿一块 ---
    full = torch.arange(world * 2, dtype=torch.float32, device=device) + rank * 10.0
    rs = reduce_scatter(full, dim=0)  # 每卡拿 world*2/world = 2 元素
    # 期望: 第 r 块 = 各卡该块之和 = sum_{r'}(块 r 的 base + r'*10) = 块base*world + 10*sum(r')
    blocks = torch.stack(
        [torch.arange(world * 2, dtype=torch.float32) + r * 10.0 for r in range(world)]
    )
    exp_rs = blocks[:, rank * 2 : rank * 2 + 2].sum(dim=0).to(device)
    ok3 = torch.equal(rs, exp_rs)
    print(f"[rank{rank}] reduce_scatter → {rs.tolist()} (期望{exp_rs.tolist()}) {'✅' if ok3 else '❌'}")

    # 4) 全局汇总结果
    torch.distributed.barrier()
    results = torch.tensor([float(ok1), float(ok2), float(ok3)], device=device)
    all_reduce(results)
    if rank == 0:
        print(f"\n{'='*50}")
        print(f"3 个原语全部通过: {bool((results == world).all())}")
        print(f"总卡数: {world} | 通信验证完成 ✅")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
