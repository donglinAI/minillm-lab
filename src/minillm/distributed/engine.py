"""
minillm.distributed.engine
==========================
3D 并行训练引擎: DP × PP × TP 组合（对应 Megatron-LM 的 3D 并行）
------------------------------------------------------------------
为什么要组合三种并行?
    单一并行方式都不够:
      DP (数据并行):   每卡一份完整模型 → 显存完全不减, 只靠多卡凑算力
      TP (张量并行):   每层权重切到多卡 → 显存减到 1/tp, 但每层前向都要
                      通信（all-reduce/all-gather）, 通信密集 → 只适合
                      单机内高速互联（NVLink/PCIe）, 跨机带宽不够
      PP (流水线并行): 按层切 → 显存减到 1/pp, 段间只传激活（低频少量）,
                      但流水线有 bubble, micro-batch 数要够大
    3D 并行 = 三者的乘积:
      TP 组内: 每层权重切分（NVLink 扛住高频通信）
      PP 组间: 段间传递激活（跨机也能忍, 频率低）
      DP 组间: 整模型梯度 all-reduce（每步一次, 量=模型大小）
      例: 64 卡 = DP4 × PP4 × TP4 → 显存减 16x, 算力 64x, 通信分层管理

本文件:
    1. make_process_grid(): rank → (dp, pp, tp) 坐标 + 三个通信组
    2. ParallelEngine: 虚拟世界模拟 3D 并行训练（8 卡 = DP2×PP2×TP2）
       每卡只持有"自己的 TP 权重切片 + 自己的 PP 段", 数据按 DP 切
    3. 核心验证: 3D 并行训练 K 步 == 单卡全模型全数据训练（误差 0）
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# 1. 进程网格
# ---------------------------------------------------------------------------
def make_process_grid(dp: int, pp: int, tp: int):
    """把 world_size = dp×pp×tp 张卡映射成三维网格。

    rank 排序: dp 优先（rank = d*(pp*tp) + p*tp + t）。
    返回 (grid, tp_groups, pp_groups, dp_groups):
      grid[rank] = (dp_rank, pp_rank, tp_rank)
      tp_groups[(d,p)] = [rank(d,p,t) for t]     # 同一层切分的卡
      pp_groups[d]     = [rank(d,p,0) for p]     # 同一模型的流水段卡
      dp_groups[(p,t)] = [rank(d,p,t) for d]     # 同一位置的数据副本卡
    """
    world = dp * pp * tp
    grid = {}
    for d in range(dp):
        for p in range(pp):
            for t in range(tp):
                grid[d * (pp * tp) + p * tp + t] = (d, p, t)

    tp_groups = {(d, p): [d * (pp * tp) + p * tp + t for t in range(tp)] for d in range(dp) for p in range(pp)}
    pp_groups = {d: [d * (pp * tp) + p * tp for p in range(pp)] for d in range(dp)}
    dp_groups = {(p, t): [d * (pp * tp) + p * tp + t for d in range(dp)] for p in range(pp) for t in range(tp)}
    return grid, tp_groups, pp_groups, dp_groups


def print_process_grid(dp: int, pp: int, tp: int) -> None:
    grid, tp_g, pp_g, dp_g = make_process_grid(dp, pp, tp)
    world = dp * pp * tp
    print(f"进程网格 (world={world} 卡 = DP{dp} × PP{pp} × TP{tp}):")
    for rank in range(world):
        d, p, t = grid[rank]
        print(f"  rank {rank:2d}: (dp={d}, pp={p}, tp={t})")
    print(f"\n  TP 组 (层内切分, 高频通信, 须同机 NVLink):")
    for k, v in tp_g.items():
        print(f"    (dp={k[0]},pp={k[1]}): ranks {v}")
    print(f"  PP 组 (段间流水, 低频少量):")
    for k, v in pp_g.items():
        print(f"    dp={k}: ranks {v}")
    print(f"  DP 组 (梯度归约, 每步一次):")
    for k, v in dp_g.items():
        print(f"    (pp={k[0]},tp={k[1]}): ranks {v}")


# ---------------------------------------------------------------------------
# 2. 3D 并行训练引擎（虚拟世界模拟）
# ---------------------------------------------------------------------------
class ParallelEngine:
    """模拟 8 卡 (DP2×PP2×TP2) 训练一个 4 层 MLP, 与单卡全量训练对比。

    模型: L=4 层 Linear(8,8)（无 bias, 简化）, 按
          PP=2: 每段 2 层;  TP=2: 每层权重按输出维切半（ColumnParallelLinear）
    数据: DP=2 份, 每份 [B, 8]
    每卡持有: 自己的 PP 段 × 每层自己的 TP 权重切片 + m/v 切片
    """

    def __init__(self, dp=2, pp=2, tp=2, d=8, layers=4, lr=0.01, seed=0):
        torch.manual_seed(seed)
        self.dp, self.pp, self.tp, self.d, self.layers = dp, pp, tp, d, layers
        self.lpp = layers // pp                      # 每段层数
        self.lr = lr
        self.grid, self.tp_g, self.pp_g, self.dp_g = make_process_grid(dp, pp, tp)

        # 每层全量权重（参考基准, 同时用于初始化分片）
        torch.manual_seed(seed)
        self.W_full = [torch.randn(d, d) * 0.1 for _ in range(layers)]

        # 每卡状态: W/m/v 都是 {本地层: 分片张量}
        self.W = {}
        self.m = {}
        self.v = {}
        for rank in range(dp * pp * tp):
            d_r, p_r, t_r = self.grid[rank]
            self.W[rank] = {}
            self.m[rank] = {}
            self.v[rank] = {}
            for li in range(self.lpp):
                W = self.W_full[p_r * self.lpp + li]
                col = d // tp
                self.W[rank][li] = W[t_r * col:(t_r + 1) * col].clone()   # 输出维切半
                self.m[rank][li] = torch.zeros_like(self.W[rank][li])
                self.v[rank][li] = torch.zeros_like(self.W[rank][li])

    # ---- 虚拟通信原语 ----
    @staticmethod
    def _tp_all_gather(parts):
        """同 TP 组各卡的输出分片拼成完整输出（按输出维 concat）。"""
        return torch.cat(parts, dim=-1)

    @staticmethod
    def _sum_tensors(vals):
        return sum(vals)

    # ---- 训练一步（虚拟并行） ----
    def train_step(self, X_list, Y_list):
        """X_list[dp_r]: [B, d]; Y_list[dp_r]: [B, d]。每个 DP 副本独立跑流水。"""
        dW = {rank: {} for rank in self.W}   # 每卡每层的梯度分片

        for d_r in range(self.dp):
            X, Y = X_list[d_r], Y_list[d_r]
            # ---- 前向: PP 段依次, 段内逐层 TP 协作 ----
            h = X
            acts = []                         # (h_prev, 完整输出 Y) 供后向用
            for p_r in range(self.pp):
                for li in range(self.lpp):
                    h_prev = h
                    parts = []
                    for t_r in range(self.tp):
                        rank = self.grid_id(d_r, p_r, t_r)
                        w = self.W[rank][li]                 # [d/tp, d]
                        parts.append(h @ w.T)                # [B, d/tp]
                    h = self._tp_all_gather(parts)           # [B, d] all-gather
                    acts.append((h_prev, h))

            # ---- loss ----
            loss = ((h - Y) ** 2).mean()

            # ---- 后向: 反向流回, 每 TP 卡算自己的 dW 分片 ----
            dY = 2 * (h - Y) / X.shape[0]
            for step in range(self.layers - 1, -1, -1):
                p_r, li = divmod(step, self.lpp)
                h_prev, _ = acts[step]
                dx_parts = []
                for t_r in range(self.tp):
                    rank = self.grid_id(d_r, p_r, t_r)
                    w = self.W[rank][li]
                    dY_r = dY[:, t_r * (self.d // self.tp):(t_r + 1) * (self.d // self.tp)]
                    dw = dY_r.T @ h_prev                      # [d/tp, d] 本地精确
                    dW[rank][li] = dW[rank].get(li, torch.zeros_like(dw)) + dw
                    dx_parts.append(dY_r @ w)                 # 部分梯度
                dY = self._sum_tensors(dx_parts)              # TP all-reduce: 完整 dX

        # ---- DP 梯度归约（平均, 标准 DDP 语义） + 分片更新 ----
        for rank in self.W:
            d_r, p_r, t_r = self.grid[rank]
            for li in range(self.lpp):
                g = torch.zeros_like(dW[rank][li])
                for other in self.dp_g[(p_r, t_r)]:           # DP 组内求和
                    g = g + dW[other][li]
                g = g / self.dp                               # 平均 = 全 batch 平均梯度
                # Adam 分片更新（m/v 逐元素独立 → 分片可行）
                self.m[rank][li] = 0.9 * self.m[rank][li] + 0.1 * g
                self.v[rank][li] = 0.999 * self.v[rank][li] + 0.001 * g ** 2
                mh = self.m[rank][li] / (1 - 0.9 ** 2)
                vh = self.v[rank][li] / (1 - 0.999 ** 2)
                self.W[rank][li] = self.W[rank][li] - self.lr * mh / (vh.sqrt() + 1e-8)
        return loss.item()

    def grid_id(self, d, p, t):
        return d * (self.pp * self.tp) + p * self.tp + t

    def gather_full(self):
        """把所有卡的权重切片拼回全量模型（验证用）。"""
        full = []
        for l in range(self.layers):
            p_r, li = divmod(l, self.lpp)
            cols = []
            for t_r in range(self.tp):
                rank = self.grid_id(0, p_r, t_r)   # dp=0 的卡（dp 组内参数一致）
                cols.append(self.W[rank][li])
            full.append(torch.cat(cols, dim=0))
        return full


# ---------------------------------------------------------------------------
# 3. 参考: 单卡全模型全数据训练
# ---------------------------------------------------------------------------
def run_single_card(W0, X_all, Y_all, steps=5, lr=0.01):
    torch.manual_seed(0)
    W = [w.clone() for w in W0]
    m = [torch.zeros_like(w) for w in W]
    v = [torch.zeros_like(w) for w in W]
    for _ in range(steps):
        h = X_all
        hs = []
        for w in W:
            hs.append(h)
            h = h @ w.T
        dY = 2 * (h - Y_all) / X_all.shape[0]
        for l in range(len(W) - 1, -1, -1):
            g = dY.T @ hs[l]
            dX = dY @ W[l]                 # 用旧 W 算 dX（前向用的权重）
            m[l] = 0.9 * m[l] + 0.1 * g
            v[l] = 0.999 * v[l] + 0.001 * g ** 2
            mh = m[l] / (1 - 0.9 ** 2)
            vh = v[l] / (1 - 0.999 ** 2)
            W[l] = W[l] - lr * mh / (vh.sqrt() + 1e-8)
            dY = dX
    return W


# ---------------------------------------------------------------------------
# 自校验
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 62)
    print_process_grid(2, 2, 2)

    print("\n" + "=" * 62)
    print("核心验证: 3D 并行训练 ≡ 单卡全模型训练")
    print("=" * 62)
    torch.manual_seed(0)
    engine = ParallelEngine(dp=2, pp=2, tp=2, d=8, layers=4)
    W0 = [w.clone() for w in engine.W_full]

    # 数据: 每 DP 副本一份; 单卡参考用全量数据
    X_all = torch.randn(4, 8)
    Y_all = torch.randn(4, 8)
    X_list = [X_all[:2], X_all[2:]]
    Y_list = [Y_all[:2], Y_all[2:]]

    ref = run_single_card(W0, X_all, Y_all)

    for step in range(5):
        engine.train_step(X_list, Y_list)
    got = engine.gather_full()

    err = max((a - b).abs().max().item() for a, b in zip(got, ref))
    print(f"  8 卡 (DP2×PP2×TP2) 训练 5 步后拼接权重 vs 单卡全量: 最大误差 {err:.2e}")
    print("  ✅ 3D 并行只是'切存储 + 通信拼接', 训练数学与单卡完全一致"
          if err < 1e-6 else "  ❌ 不一致")
    assert err < 1e-6

    print("\n结论:")
    print("  · TP 每层切分 → 高频 all-gather（组内须 NVLink）")
    print("  · PP 段间只传激活 → 低频点对点（可跨机）")
    print("  · DP 组间每步一次梯度归约（量 = 模型大小）")
    print("  · 三者组合: 显存减到 1/(tp×pp), 算力 ×world, 通信分层管理")
