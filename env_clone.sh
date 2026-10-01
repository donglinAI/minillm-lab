#!/bin/bash
# 环境配置 + 下载项目脚本
# 功能：
#   1. 加载 turbo 网络加速（并写入 .bashrc）
#   2. 设置 HF 缓存路径到数据盘（并写入 .bashrc）
#   3. clone git@github.com:donglinAI/minillm-lab.git
# 执行：chmod +x env_clone.sh && ./env_clone.sh
# 注意：clone 前需先配好 SSH 密钥并添加到 GitHub（见 init_git_ssh.sh）

echo "===== 1. 加载AutoDL网络加速 ====="
source /etc/network_turbo
# 写入 .bashrc，让每次新开终端自动启用 turbo
echo 'source /etc/network_turbo' >> ~/.bashrc
echo "✅ turbo 已写入 ~/.bashrc（新开终端自动启用）"

echo "===== 2. 启动conda环境 ====="
echo 'source /root/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc 


echo "===== 3. 设置HF缓存路径到数据盘 ====="
export HF_HOME=/root/autodl-tmp/hf_cache
# 目录已存在则不再创建（mkdir -p 本身幂等，这里显式跳过）
if [ ! -d "$HF_HOME" ]; then
    mkdir -p "$HF_HOME"
    echo "✅ 已创建缓存目录 $HF_HOME"
else
    echo "✅ 缓存目录已存在 $HF_HOME，跳过 mkdir"
fi
# 写入 .bashrc，新开终端自动生效（系统盘小，HF 缓存放数据盘）
echo 'export HF_HOME=/root/autodl-tmp/hf_cache' >> ~/.bashrc
echo "✅ HF_HOME=$HF_HOME 已写入 ~/.bashrc"

echo "===== 3. Clone minillm-lab仓库到 autodl-tmp ====="
cd /root/autodl-tmp
git clone git@github.com:donglinAI/minillm-lab.git
echo -e "\n==================== 完成 ===================="
echo "仓库位置：/root/autodl-tmp/minillm-lab"
echo "✅ 请判断执行：bash scripts/install_deps.sh "