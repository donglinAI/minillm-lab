#!/bin/bash
# 安装 Python 依赖脚本
# 用法：
#   bash scripts/install_deps.sh          # 装核心依赖 requirements.txt
#   bash scripts/install_deps.sh --full   # 装全部依赖 requirements-full.txt
#
# 背景：turbo 代理会拖慢 pip（清华源在国内，走代理绕远路、易超时），
#       所以安装前临时解除代理变量（仅当前 shell 生效，不影响 .bashrc），
#       安装完成后在当前终端恢复 turbo。
set -e

echo "===== 1. 临时解除 turbo 代理 ====="
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
echo "✅ 已解除代理变量（仅当前终端）"

echo "===== 2. 选择依赖文件 ====="
if [ "$1" == "--full" ]; then
    REQ=requirements-full.txt
    echo "✅ 安装全部依赖: $REQ"
else
    REQ=requirements.txt
    echo "✅ 安装核心依赖: $REQ"
fi

echo "===== 3. 安装依赖（清华源）====="
pip install -r "$REQ" -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "===== 4. 恢复 turbo 代理 ====="
source /etc/network_turbo
echo "✅ 已恢复 turbo 代理（当前终端可继续 git 操作）"
echo ""
echo "安装完成"
