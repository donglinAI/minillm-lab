#!/bin/bash
# ============================================================
# 一键 SFT（真数据）：下载中文指令数据 → 单卡训练 → 生成验证
# 用法:
#   bash scripts/sft.sh                              # 正式训练(500 步)
#   bash scripts/sft.sh --mini --limit 300 --steps 15 --seq_len 64  # CPU 冒烟
# 说明: 多卡 DDP 待 distributed 真实验证后接入（engine.py 目前是模拟器）
# ============================================================
set -e
cd "$(dirname "$0")/.."

# 用户要求无条件开启网络加速
source /etc/network_turbo

export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf_cache}
export HF_HUB_DISABLE_XET=1
export PYTHONPATH=src:${PYTHONPATH}

# 1) 下载 SFT 数据（已存在则跳过）
if [ ! -f data_cache/sft_zh.jsonl ]; then
  echo "[sft.sh] 下载 SFT 数据 → data_cache/sft_zh.jsonl"
  python scripts/download_data.py --dataset sft --limit 50000
else
  echo "[sft.sh] 数据已存在: data_cache/sft_zh.jsonl"
fi

# 2) 训练（单卡）
echo "[sft.sh] 开始 SFT 训练"
python -m minillm.posttrain.sft.train \
  --config configs/train/sft_single_gpu.yaml "$@"

echo "[sft.sh] 完成。checkpoint 在 output/sft_tiny/"
