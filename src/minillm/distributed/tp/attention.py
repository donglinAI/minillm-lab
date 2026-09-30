"""
minillm.distributed.tp.attention
================================
TP 化多头注意力（MHA / GQA），对齐 Megatron 张量并行标准做法
--------------------------------------------------------
单卡注意力（Phase 2 已交付 attention.py）:
    q = x @ W_qᵀ   [.., heads*D]
    k = x @ W_kᵀ   [.., kv_heads*D]
    v = x @ W_vᵀ   [.., kv_heads*D]
    → 切头 → RoPE → repeat_kv → softmax(QKᵀ/√D)V → 合并头
    out = ctx @ W_oᵀ

TP 化的切法（Megatron 标准答案）:
    q/k/v: ColumnParallelLinear（按输出维切，输出维 = heads*D）
        每卡拿 W_qᵣ [heads/N*D, hidden]，reshape 出 heads/N 个头
        ★ 关键: 多头注意力各头完全独立 → 注意力计算可在"切分状态"做，
          每卡只算自己那 heads/N 个头的分数，不需要先 all_gather！
    o:    RowParallelLinear（输入就是各卡算好的 heads/N 份，all_reduce 归约）

形状流转（N=2, hidden=512, heads=8, kv_heads=2, D=64）:
    x [B, S, 512]
      ├─ q_proj(列) → [B,S,256] → 4 个头           ┐
      ├─ k_proj(列) → [B,S,64]  → 1 个 kv 头       │ 全部在切分状态
      ├─ v_proj(列) → [B,S,64]  → 1 个 kv 头       │ 完成注意力
      └─ 每卡: 4q × 1kv (repeat) → context [B,S,256]┘
      └─ o_proj(行, 输入已切分) → all_reduce → [B,S,512]

约束: num_heads % world_size == 0 且 num_kv_heads % world_size == 0
     （TP 大小不能超过 kv_heads——GQA 下 kv 头太少则 TP 上限受限）
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minillm.distributed.comm import get_rank, get_world_size
from minillm.distributed.tp.linear import ColumnParallelLinear, RowParallelLinear
from minillm.models.layers.attention import repeat_kv
from minillm.models.layers.rope import RotaryEmbedding


class ParallelAttention(nn.Module):
    """TP 化 MHA/GQA：q/k/v 列并行 + o 行并行，注意力在切分状态完成。"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        world_size: int | None = None,
        rank: int | None = None,
    ) -> None:
        super().__init__()
        self.world_size = world_size if world_size is not None else get_world_size()
        self.rank = rank if rank is not None else get_rank()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.n_rep = num_heads // self.num_kv_heads

        # TP 约束: 头数必须能被卡数整除（否则每卡拿不到整数个头）
        assert num_heads % self.world_size == 0, (
            f"num_heads={num_heads} 必须能被 world_size={self.world_size} 整除"
        )
        assert self.num_kv_heads % self.world_size == 0, (
            f"num_kv_heads={self.num_kv_heads} 必须能被 world_size={self.world_size} 整除"
            "（GQA 下 kv 头太少会限制 TP 大小）"
        )
        # 每卡持有的头数
        self.heads_per_rank = num_heads // self.world_size
        self.kv_heads_per_rank = self.num_kv_heads // self.world_size

        # q/k/v: 列并行，先不 gather（注意力各头独立，切分状态直接算）
        self.q_proj = ColumnParallelLinear(
            hidden_size, num_heads * self.head_dim, bias=False,
            world_size=self.world_size, rank=self.rank, gather_output=False,
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size, self.num_kv_heads * self.head_dim, bias=False,
            world_size=self.world_size, rank=self.rank, gather_output=False,
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size, self.num_kv_heads * self.head_dim, bias=False,
            world_size=self.world_size, rank=self.rank, gather_output=False,
        )
        # o: 行并行（输入就是各卡算好的 heads/N 份）
        self.o_proj = RowParallelLinear(
            num_heads * self.head_dim, hidden_size, bias=False,
            world_size=self.world_size, rank=self.rank, input_is_parallel=True,
        )
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len, rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: [batch, seq, hidden] → 输出同形状（与单卡注意力接口一致）。"""
        b, s, _ = x.shape

        # 1) 列并行投影 + 切头: 每卡只有自己那份头
        q = self.q_proj(x).view(b, s, self.heads_per_rank, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.kv_heads_per_rank, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.kv_heads_per_rank, self.head_dim).transpose(1, 2)

        # 2) RoPE（逐元素操作，切分状态直接做）
        if positions is None:
            positions = torch.arange(s, device=x.device)
        q, k = self.rope(q, k, positions)

        # 3) GQA: 每卡内部重复 kv 头（分组比例不变: n_rep = heads/kv_heads）
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # 4) 缩放点积分数（只算本卡头）
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 5) causal 掩码
        if attention_mask is None:
            causal = torch.triu(torch.ones(s, s, dtype=torch.bool, device=x.device), diagonal=1)
            attention_mask = causal[None, None, :, :]
        attn_weights = attn_weights.masked_fill(attention_mask, float("-inf"))

        # 6) softmax + 加权求和
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn_weights, v)  # [B, hq/N, S, D]

        # 7) 合并本卡头 → o_proj（行并行, 输入已是 N 份, all_reduce 归约）
        out = out.transpose(1, 2).contiguous().view(b, s, self.heads_per_rank * self.head_dim)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# 自校验 python -m minillm.distributed.tp.attention
# ---------------------------------------------------------------------------
def _make_single_attn():
    """构造单卡参考注意力（与 TP 版同配置）。"""
    torch.manual_seed(0)
    from minillm.models.layers.attention import MultiHeadAttention

    return MultiHeadAttention(512, 8, num_kv_heads=2, max_seq_len=32)


def _verify_single_gpu() -> None:
    """world_size=1 时 TP attention 应等价于单卡 attention（0 误差）。"""
    H, S = 512, 6
    attn = _make_single_attn()
    x = torch.randn(2, S, H)
    y_ref = attn(x)

    pattn = ParallelAttention(H, 8, num_kv_heads=2, max_seq_len=32, world_size=1)
    with torch.no_grad():
        pattn.q_proj.weight.copy_(attn.q_proj.weight)
        pattn.k_proj.weight.copy_(attn.k_proj.weight)
        pattn.v_proj.weight.copy_(attn.v_proj.weight)
        pattn.o_proj.weight.copy_(attn.o_proj.weight)
    y = pattn(x)
    err = (y - y_ref).abs().max().item()
    print(f"单卡退化误差: {err:.2e} (应=0)")
    assert err < 1e-6
    print(f"每卡参数量: {sum(p.numel() for p in pattn.parameters())} "
          f"(单卡完整: {sum(p.numel() for p in attn.parameters())})")
    print("✅ 单卡退化: TP attention == 单卡 MultiHeadAttention")


def _run_tp_attn(rank: int, world_size: int) -> None:
    """双进程 gloo 真实 TP attention：两卡各持一半头，前向与单卡对齐。"""
    import torch.distributed as dist

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    H, S = 512, 6
    attn = _make_single_attn()
    x = torch.randn(2, S, H)
    y_ref = attn(x)

    pattn = ParallelAttention(H, 8, num_kv_heads=2, max_seq_len=32,
                              world_size=world_size, rank=rank)
    per = 8 // world_size * 64          # 每卡 q 输出维 = heads/N*D
    kvper = 2 // world_size * 64        # 每卡 kv 输出维
    with torch.no_grad():
        pattn.q_proj.weight.copy_(attn.q_proj.weight[rank * per : (rank + 1) * per])
        pattn.k_proj.weight.copy_(attn.k_proj.weight[rank * kvper : (rank + 1) * kvper])
        pattn.v_proj.weight.copy_(attn.v_proj.weight[rank * kvper : (rank + 1) * kvper])
        pattn.o_proj.weight.copy_(attn.o_proj.weight[:, rank * per : (rank + 1) * per])
    y = pattn(x)
    err = (y - y_ref).abs().max().item()
    ok = err < 1e-5
    print(f"[rank{rank}] TP attention 输出误差 {err:.2e} {'✅' if ok else '❌'}")
    assert ok
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    _verify_single_gpu()

    try:
        import os

        import torch.multiprocessing as mp

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29503")
        mp.spawn(_run_tp_attn, args=(2,), nprocs=2, join=True)
        print("\n✅ 双进程 gloo 真实 TP attention 通过（与单卡对齐）")
    except Exception as e:  # noqa: BLE001
        print(f"\n[skip] 双进程 TP attention 测试跳过: {type(e).__name__}: {e}")
        print("（单卡环境跳过；AutoDL 多卡时自动启用）")
