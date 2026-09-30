
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate base
fi

autodl代理加速：source /etc/network_turbo； 影响pip

可以关闭学术代理并强制使用hf国内镜像，也可以pip安装
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com 

pip install transformers datasets tokenizers pyyaml tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
