#!/usr/bin/env bash
# ============================================================
# AutoDL 环境初始化脚本
# 用法：
#   bash scripts/setup_env.sh             # turbo 优先（GitHub 场景）
#   bash scripts/setup_env.sh --no-turbo  # 不用 turbo，纯国内镜像
#   bash scripts/setup_env.sh --full      # 装全部依赖（含 Phase 8）
#
# 网络策略（重要）：
#   - 默认优先 source /etc/network_turbo（GitHub/HF 直连走代理）
#   - HF 下载若失败，在终端执行 `hf_mirror` 一键切国内镜像
#   - 想切回代理执行 `hf_turbo`，查看状态执行 `hf_status`
# ============================================================

set -e

# ---- 1. 激活 conda ----
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate base
fi

# ---- 2. 网络策略 ----
# 默认开 turbo（除非 --no-turbo）
if [ "$1" == "--no-turbo" ]; then
    echo "[setup] 跳过 turbo，使用国内镜像"
    export HF_ENDPOINT=https://hf-mirror.com
else
    if [ -f "/etc/network_turbo" ]; then
        source /etc/network_turbo || true
        echo "[setup] 已开启 turbo 代理（GitHub/HF 走代理直连）"
        # 开了代理就不设 HF_ENDPOINT，保持默认 huggingface.co
        # 若 HF 下载失败，之后执行 `hf_mirror` 切换
    else
        echo "[setup] 未找到 /etc/network_turbo，回退国内镜像"
        export HF_ENDPOINT=https://hf-mirror.com
    fi
fi

# ---- 3. HuggingFace 缓存 ----
export HF_HOME=/root/autodl-tmp/hf_cache
mkdir -p "$HF_HOME"
# # 禁用 Xet（国内无论走代理还是镜像都可能 401，统一禁掉）
# export HF_HUB_DISABLE_XET=1
echo "[setup] HF_HOME=$HF_HOME"
# echo "[setup] HF_HUB_DISABLE_XET=1"

# ---- 4. 安装依赖（pip 走清华源；若开了代理，临时关掉避免绕远）----
if [ "$1" == "--full" ] || [ "$2" == "--full" ]; then
    REQ=requirements-full.txt
    echo "[setup] 安装全部依赖: $REQ"
else
    REQ=requirements.txt
    echo "[setup] 安装核心依赖: $REQ"
fi

# pip 安装临时关代理（清华源在国内，走代理反而慢）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY || true
pip install -r "$REQ" -i https://pypi.tuna.tsinghua.edu.cn/simple

# 恢复 turbo（如果刚才开着）
if [ -f "/etc/network_turbo" ] && [ "$1" != "--no-turbo" ]; then
    source /etc/network_turbo || true
fi

# ---- 5. 写 .envrc（每次新开终端 source .envrc）----
cat > .envrc <<'EOF'
# minillm-lab 环境变量 + 网络切换函数
# 用法：在项目根目录执行 source .envrc

export PYTHONPATH=./src:$PYTHONPATH
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_DISABLE_XET=1

# 切换函数（随时可用）：
hf_turbo() {
    source /etc/network_turbo 2>/dev/null && unset HF_ENDPOINT
    echo "[env] turbo 代理：HF 走 huggingface.co（经代理）| GitHub 可用"
}

hf_mirror() {
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null
    export HF_ENDPOINT=https://hf-mirror.com
    echo "[env] 国内镜像：HF 走 hf-mirror.com（代理已关）"
}

hf_status() {
    echo "http_proxy    : ${http_proxy:-无}"
    echo "https_proxy   : ${https_proxy:-无}"
    echo "HF_ENDPOINT   : ${HF_ENDPOINT:-无（默认 huggingface.co）}"
    echo "HF_HOME       : $HF_HOME"
    echo "HF_HUB_DISABLE_XET: $HF_HUB_DISABLE_XET"
}
EOF

echo ""
echo "[setup] 完成。每次新开终端："
echo "  source .envrc"
echo ""
echo "常用命令："
echo "  hf_turbo   # 切到 turbo 代理（GitHub 场景）"
echo "  hf_mirror  # 切到国内镜像（HF 下载失败时）"
echo "  hf_status  # 查看当前网络状态"
