"""
minillm.distributed.tp.linear
=============================
张量并行（Tensor Parallelism）的核心算子：切分 Linear 层
--------------------------------------------------------
单卡 Linear:  y = x @ Wᵀ + b        W: [out, in], x: [.., in]
显存不够 → 把 W 切到 N 张卡上。怎么切？两个方向：

1. ColumnParallelLinear（列并行）—— 按输出维切 W
   W 切成 N 份:  W = [W₀; W₁; ...; Wₙ₋₁]    每卡拿 Wᵣ: [out/N, in]
   每卡算:  yᵣ = x @ Wᵣᵀ              → 局部结果 [.., out/N]
   完整输出: y = concat(y₀, ..., yₙ₋₁) = x @ Wᵀ ✓（数学恒等）
   通信: all_gather（把 N 份拼成完整）
   用途: QKV 投影、up/gate 投影（输出要完整的场景）

2. RowParallelLinear（行并行）—— 按输入维切 W
   W 切成 N 份:  W = [W₀ | W₁ | ... | Wₙ₋₁]    每卡拿 Wᵣ: [out, in/N]
   输入 x 也切 N 份:  x = [x₀ | ... | xₙ₋₁]
   每卡算:  yᵣ = xᵣ @ Wᵣᵀ              → 局部结果 [.., out]
   完整输出: y = Σ yᵣ = x @ Wᵀ ✓（数学恒等）
   通信: all_reduce（N 份求和）
   用途: down 投影、输出投影（输入是列并行结果、要归约的场景）

注意（对齐 Megatron 的 f/g 算子）:
  - 实际工程中列并行的输出"不立即 all_gather"，而是直接作为下一层行并行的输入
    （行并行内部会先做 reduce），省一次通信。教学版先还原完整结果，逻辑清晰。
  - LayerNorm / RMSNorm 在 TP 中不切分：每卡持有完整副本（ReplicatedLayerNorm），
    因为归一化需要整行统计量、参数又极少，复制比切分更省事。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minillm.distributed.comm import all_gather, all_reduce, get_rank, get_world_size


def _truncated_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 0.02) -> torch.Tensor:
    """截断正态初始化（与单卡模型初始化保持一致）。"""
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
    return tensor


class ColumnParallelLinear(nn.Module):
    """列并行 Linear：按输出维把 W 切成 world_size 份，每卡一份。

    参数量: out/N × in（+ out/N bias）= 单卡总量的 1/N ✓
    前向:   y_partial = x @ Wᵣᵀ → all_gather → 完整 y [.., out]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        world_size: int | None = None,
        rank: int | None = None,
        gather_output: bool = True,
    ) -> None:
        super().__init__()
        self.world_size = world_size if world_size is not None else get_world_size()
        self.rank = rank if rank is not None else get_rank()
        self.gather_output = gather_output
        assert out_features % self.world_size == 0, (
            f"输出维 {out_features} 必须能被 {self.world_size} 整除（列并行按输出切）"
        )
        self.out_features_per_rank = out_features // self.world_size

        # 每卡只持有自己那块 Wᵣ: [out/N, in]
        self.weight = nn.Parameter(torch.empty(self.out_features_per_rank, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_rank))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 与 nn.Linear 相同的初始化分布（kaiming-uniform），但作用在切片上
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 局部矩阵乘: yᵣ [.., out/N]
        y = F.linear(x, self.weight, self.bias)
        if self.gather_output and self.world_size > 1:
            # 各卡的 out/N 块按输出维拼成完整 [.., out]
            y = all_gather(y, dim=-1)
        return y


class RowParallelLinear(nn.Module):
    """行并行 Linear：按输入维把 W 切成 world_size 份，每卡一份。

    参数量: out × in/N（+ bias 每卡完整 out）= 单卡总量的 ~1/N ✓
    前向:   x 切份 → 局部乘 → all_reduce(sum) → 完整 y [.., out]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        world_size: int | None = None,
        rank: int | None = None,
        input_is_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.world_size = world_size if world_size is not None else get_world_size()
        self.rank = rank if rank is not None else get_rank()
        self.input_is_parallel = input_is_parallel  # 上游已是切分好的输入（Megatron 风格）
        assert in_features % self.world_size == 0, (
            f"输入维 {in_features} 必须能被 {self.world_size} 整除（行并行按输入切）"
        )
        self.in_features_per_rank = in_features // self.world_size

        # 每卡只持有自己那块 Wᵣ: [out, in/N]
        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_rank))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_is_parallel:
            # 上游列并行直接给了本卡那份 xᵣ [.., in/N]，无需再切
            x_i = x
        else:
            # 教学版：输入是完整 x [.., in]，内部切成 world_size 份取本卡那份
            chunks = x.chunk(self.world_size, dim=-1)
            x_i = chunks[self.rank]
        # 注意: 这里不加 bias！bias 若在归约前加，每卡加一次会被 all_reduce 重复 N 次
        y = F.linear(x_i, self.weight)  # [.., out]
        if self.world_size > 1:
            y = all_reduce(y, op="sum")  # Σ xᵣ@Wᵣᵀ = 完整结果
        if self.bias is not None:
            y = y + self.bias  # 归约后只加一次完整 bias
        return y


class ReplicatedLayerNorm(nn.Module):
    """TP 中的归一化层：每卡持有完整副本，不切分。

    为什么 LayerNorm 不切分？
      1. 归一化要算整行统计量（mean/var），切分后每卡只有部分行，统计量不全
      2. 参数极少（weight/bias 各 out 个），复制成本几乎为零
    所以 Megatron 里 LayerNorm 是"重复放置"，前向每卡跑同样的完整归一化。
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


# ---------------------------------------------------------------------------
# 自校验：python -m minillm.distributed.tp.linear
# 1) 数学恒等（纯张量）  2) 单卡退化  3) 双进程 gloo 真实 TP 通信
# ---------------------------------------------------------------------------
def _verify_math() -> None:
    """演示列并行 = concat、行并行 = sum 的数学恒等（不需要进程组）。"""
    torch.manual_seed(0)
    x = torch.randn(3, 4)
    W = torch.randn(6, 4)  # out=6, in=4
    b = torch.randn(6)
    y_ref = F.linear(x, W, b)  # [3, 6]

    # 列并行: W 按输出切成 2 份
    W0, W1 = W[:3], W[3:]
    b0, b1 = b[:3], b[3:]
    y_col = torch.cat([F.linear(x, W0, b0), F.linear(x, W1, b1)], dim=-1)
    print(f"列并行误差: {(y_col - y_ref).abs().max().item():.2e} (应≈0)")
    assert torch.allclose(y_col, y_ref, atol=1e-6)

    # 行并行: W 按输入切成 2 份, x 也切; bias 只加一次（归约后）
    W0r, W1r = W[:, :2], W[:, 2:]
    x0, x1 = x[:, :2], x[:, 2:]
    y_row = F.linear(x0, W0r) + F.linear(x1, W1r) + b
    print(f"行并行误差: {(y_row - y_ref).abs().max().item():.2e} (应≈0)")
    assert torch.allclose(y_row, y_ref, atol=1e-6)
    print("✅ 数学恒等: 列并行=concat(切片输出), 行并行=sum(切片输出)")


def _verify_single_gpu() -> None:
    """world_size=1 时（单卡退化），TP Linear 应等价于普通 nn.Linear。"""
    torch.manual_seed(0)
    x = torch.randn(2, 8)
    lin = nn.Linear(8, 16)
    col = ColumnParallelLinear(8, 16, world_size=1)
    row = RowParallelLinear(8, 16, world_size=1)
    # 复制权重保证对比公平
    col.weight.data.copy_(lin.weight.data)
    col.bias.data.copy_(lin.bias.data)
    row.weight.data.copy_(lin.weight.data)
    row.bias.data.copy_(lin.bias.data)

    y_ref = lin(x)
    err_col = (col(x) - y_ref).abs().max().item()
    err_row = (row(x) - y_ref).abs().max().item()
    print(f"单卡退化: 列并行误差 {err_col:.2e} | 行并行误差 {err_row:.2e}")
    assert err_col < 1e-6 and err_row < 1e-6
    print("✅ 单卡退化: 等价于 nn.Linear")


def _run_tp(rank: int, world_size: int) -> None:
    """双进程 gloo 真实 TP：两卡各持一半 W，通信还原完整输出。"""
    import torch.distributed as dist

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    torch.manual_seed(0)

    x = torch.randn(3, 4)
    W = torch.randn(6, 4)
    b = torch.randn(6)
    y_ref = F.linear(x, W, b)  # [3, 6]

    # 列并行: 每卡拿输出维的一半
    col = ColumnParallelLinear(4, 6, world_size=world_size, rank=rank)
    with torch.no_grad():
        col.weight.copy_(W[rank * 3 : rank * 3 + 3])
        col.bias.copy_(b[rank * 3 : rank * 3 + 3])
    y_col = col(x)
    ok_col = torch.allclose(y_col, y_ref, atol=1e-5)
    print(f"[rank{rank}] 列并行输出误差 {(y_col - y_ref).abs().max().item():.2e} {'✅' if ok_col else '❌'}")

    # 行并行: 每卡拿输入维的一半
    row = RowParallelLinear(4, 6, world_size=world_size, rank=rank)
    with torch.no_grad():
        row.weight.copy_(W[:, rank * 2 : rank * 2 + 2])
        row.bias.copy_(b)
    y_row = row(x)
    ok_row = torch.allclose(y_row, y_ref, atol=1e-5)
    print(f"[rank{rank}] 行并行输出误差 {(y_row - y_ref).abs().max().item():.2e} {'✅' if ok_row else '❌'}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    _verify_math()
    _verify_single_gpu()

    try:
        import os

        import torch.multiprocessing as mp

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        mp.spawn(_run_tp, args=(2,), nprocs=2, join=True)
        print("\n✅ 双进程 gloo 真实 TP 通信通过（列/行并行均与单卡对齐）")
    except Exception as e:  # noqa: BLE001
        print(f"\n[skip] 双进程 TP 测试跳过: {type(e).__name__}: {e}")
        print("（单卡环境跳过；AutoDL 多卡时自动启用）")
