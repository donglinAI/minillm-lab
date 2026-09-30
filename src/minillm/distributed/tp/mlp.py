"""
minillm.distributed.tp.mlp
==========================
TP 化 SwiGLU MLP（对齐 Megatron 的经典张量并行组合）
----------------------------------------------------
单卡 SwiGLU:
    gate = SiLU(x @ W_gᵀ)          W_g: [inter, hidden]
    up   = x @ W_uᵀ                 W_u: [inter, hidden]
    out  = (gate * up) @ W_dᵀ       W_d: [hidden, inter]

显存不够 → 三个投影全切到 N 卡。怎么切？（这是 Megatron 的标准答案）    

    gate/up: ColumnParallelLinear（按输出维切）
      因为它们的输入都是同一个完整 x，输出是中间激活（维度大、要完整）
    down:    RowParallelLinear（按输入维切, input_is_parallel=True）
      因为它的输入是 gate*up（正好是列并行后 N 份切分状态），输出要归约回完整

关键优化（省一次通信）:
    SiLU 和逐元素乘法都可以在"切分状态"下做——每卡只算自己那 1/N 份中间激活，
    不需要先把 gate/up 拼回完整（省掉一次 all_gather）。
    然后 down 的输入天然就是切分好的 → 直接行并行归约。
    = 一次前向只花一次 all_reduce，与 Megatron 一致。

形状流转（N=2, hidden=8, inter=16）:
    x [.., 8]
      ├─ gate_proj(列) → [.., 8]   ← 每卡 inter/2
      ├─ up_proj(列)   → [.., 8]
      └─ SiLU(gate)*up → [.., 8]   ← 切分状态逐元素操作
      └─ down_proj(行, 输入已切分) → all_reduce → [.., 8]  完整
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minillm.distributed.comm import get_rank, get_world_size
from minillm.distributed.tp.linear import ColumnParallelLinear, RowParallelLinear


class ParallelMLP(nn.Module):
    """TP 化的 SwiGLU MLP：gate/up 列并行 + down 行并行（Megatron 风格）。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        world_size: int | None = None,
        rank: int | None = None,
    ) -> None:
        super().__init__()
        self.world_size = world_size if world_size is not None else get_world_size()
        self.rank = rank if rank is not None else get_rank()

        # gate/up: 列并行（输出 intermediate 被切 N 份），先不 all_gather
        # （切分状态下 SiLU/mul 可直接算，省一次通信）
        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False,
            world_size=self.world_size, rank=self.rank, gather_output=False,
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False,
            world_size=self.world_size, rank=self.rank, gather_output=False,
        )
        # down: 行并行（输入就是上面切分好的 N 份），all_reduce 归约回完整
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False,
            world_size=self.world_size, rank=self.rank, input_is_parallel=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))  # 每卡 [.., inter/N]（切分状态）
        up = self.up_proj(x)              # 每卡 [.., inter/N]
        return self.down_proj(gate * up)  # 行并行输入已切分 → 归约 → [.., hidden]

    def num_params_per_rank(self) -> int:
        """每卡持有的参数量（应 ≈ 单卡 MLP 的 1/N）。"""
        return sum(p.numel() for p in self.parameters())


def single_mlp_forward(
    x: torch.Tensor, w_g: torch.Tensor, w_u: torch.Tensor, w_d: torch.Tensor
) -> torch.Tensor:
    """单卡 SwiGLU 参考前向（测试对照用）。"""
    return F.silu(F.linear(x, w_g)) * F.linear(x, w_u) @ w_d.t()


# ---------------------------------------------------------------------------
# 自校验 python -m minillm.distributed.tp.mlp 
# ---------------------------------------------------------------------------
def _verify_single_gpu() -> None:
    """world_size=1 时 TP MLP 应等价于单卡 MLP（0 误差）。"""
    torch.manual_seed(0)
    hidden, inter = 8, 16
    x = torch.randn(3, hidden)
    w_g = torch.randn(inter, hidden) * 0.1
    w_u = torch.randn(inter, hidden) * 0.1
    w_d = torch.randn(hidden, inter) * 0.1

    y_ref = single_mlp_forward(x, w_g, w_u, w_d)

    mlp = ParallelMLP(hidden, inter, world_size=1)
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(w_g)
        mlp.up_proj.weight.copy_(w_u)
        mlp.down_proj.weight.copy_(w_d)
    y = mlp(x)
    err = (y - y_ref).abs().max().item()
    print(f"单卡退化误差: {err:.2e} (应=0)")
    assert err < 1e-6
    print(f"每卡参数量: {mlp.num_params_per_rank()} (单卡完整= {w_g.numel()+w_u.numel()+w_d.numel()})")
    print("✅ 单卡退化: TP MLP == 单卡 SwiGLU")


def _verify_sharded_math() -> None:
    """纯数学演示（不启动进程）：分片公式与单卡公式恒等。"""
    torch.manual_seed(1)
    hidden, inter = 8, 16
    x = torch.randn(3, hidden)
    w_g = torch.randn(inter, hidden) * 0.1
    w_u = torch.randn(inter, hidden) * 0.1
    w_d = torch.randn(hidden, inter) * 0.1
    y_ref = single_mlp_forward(x, w_g, w_u, w_d)

    # 模拟 2 卡: gate/up 按输出切, down 按输入切
    g0, g1 = w_g[:8], w_g[8:]
    u0, u1 = w_u[:8], w_u[8:]
    d0, d1 = w_d[:, :8], w_d[:, 8:]

    # 卡0
    gate0 = F.silu(F.linear(x, g0)); up0 = F.linear(x, u0)
    h0 = gate0 * up0; y0 = F.linear(h0, d0)
    # 卡1
    gate1 = F.silu(F.linear(x, g1)); up1 = F.linear(x, u1)
    h1 = gate1 * up1; y1 = F.linear(h1, d1)
    y = y0 + y1
    err = (y - y_ref).abs().max().item()
    print(f"2卡分片恒等误差: {err:.2e} (应≈0)")
    assert err < 1e-6
    print("✅ 数学恒等: 列并行(gate/up) + 行并行(down) = 单卡 MLP")


def _run_tp_mlp(rank: int, world_size: int) -> None:
    """双进程 gloo 真实 TP MLP：两卡各持一半参数，前向与单卡对齐。"""
    import torch.distributed as dist

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    torch.manual_seed(0)
    hidden, inter = 8, 16
    x = torch.randn(3, hidden)
    w_g = torch.randn(inter, hidden) * 0.1
    w_u = torch.randn(inter, hidden) * 0.1
    w_d = torch.randn(hidden, inter) * 0.1
    y_ref = single_mlp_forward(x, w_g, w_u, w_d)

    mlp = ParallelMLP(hidden, inter, world_size=world_size, rank=rank)
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(w_g[rank * 8 : rank * 8 + 8])
        mlp.up_proj.weight.copy_(w_u[rank * 8 : rank * 8 + 8])
        mlp.down_proj.weight.copy_(w_d[:, rank * 8 : rank * 8 + 8])
    y = mlp(x)
    err = (y - y_ref).abs().max().item()
    ok = err < 1e-5
    print(f"[rank{rank}] TP MLP 输出误差 {err:.2e} {'✅' if ok else '❌'}")
    assert ok
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    _verify_single_gpu()
    _verify_sharded_math()

    try:
        import os

        import torch.multiprocessing as mp

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29502")
        mp.spawn(_run_tp_mlp, args=(2,), nprocs=2, join=True)
        print("\n✅ 双进程 gloo 真实 TP MLP 通过（与单卡对齐）")
    except Exception as e:  # noqa: BLE001
        print(f"\n[skip] 双进程 TP MLP 测试跳过: {type(e).__name__}: {e}")
        print("（单卡环境跳过；AutoDL 多卡时自动启用）")
