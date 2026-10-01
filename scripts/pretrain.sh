#!/bin/bash
# ============================================================
# 一键预训练（真实数据，单卡 / 多卡 DDP）
# 用法:
#   bash scripts/pretrain.sh                          # 单卡
#   bash scripts/pretrain.sh --nproc 4                # 4 卡 DDP
#   bash scripts/pretrain.sh --mini --steps 5 --split "train[:200]" --seq_len 64   # CPU 冒烟
# ============================================================
set -e
cd "$(dirname "$0")/.."

# 用户要求无条件开启网络加速
source /etc/network_turbo

export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf_cache}
export HF_HUB_DISABLE_XET=1
export PYTHONPATH=src:${PYTHONPATH}

NPROC=1
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nproc) NPROC="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

echo "[pretrain.sh] 启动预训练 (nproc=$NPROC)"
if [ "$NPROC" -gt 1 ]; then
  # 真实 DDP: torchrun 启动 N 个进程, 每进程一张卡(NCCL)
  torchrun --nproc_per_node="$NPROC" \
    -m minillm.pretrain.train \
    --config configs/train/pretrain_ddp.yaml "${EXTRA_ARGS[@]}"
else
  python -m minillm.pretrain.train \
    --config configs/train/pretrain_single_gpu.yaml "${EXTRA_ARGS[@]}"
fi

echo "[pretrain.sh] 完成。权重在 output/pretrain_*/final.pt"
