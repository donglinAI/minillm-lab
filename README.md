
# ---- 1. 激活 conda ----
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate base
fi

# ---- 2. 尝试开启 AutoDL 学术加速（不存在则跳过）----
if [ -f "/etc/network_turbo" ]; then
    source /etc/network_turbo
    echo "[setup] AutoDL 学术加速已开启"
else
    echo "[setup] 未找到 /etc/network_turbo，回退到 HF 镜像"
    export HF_ENDPOINT=https://hf-mirror.com
fi


可以关闭学术代理并强制使用国内镜像
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com
重新执行；


# ---- 3. 装核心依赖 ----
pip install transformers datasets tokenizers pyyaml tqdm \
    -i https://pypi.tuna.tsinghua.edu.cn/simple