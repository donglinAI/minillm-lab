#!/bin/bash
# Git 配置 + SSH 密钥生成脚本（仅此两项）
# 执行：chmod +x init_git_ssh.sh && ./init_git_ssh.sh
# 注意：不设任何判断，线性执行。SSH 密钥每次都会重新生成（覆盖旧的），
#       重跑后需把新公钥重新添加到 GitHub。

echo "===== 1. 配置Git全局用户信息 ====="
git config --global user.name "donglinAI"
git config --global user.email "donglin_cui@163.com"
echo "✅ Git config done."

echo "===== 2. 生成SSH密钥 ed25519 ====="
SSH_PRIVATE_KEY="$HOME/.ssh/id_ed25519"
# 删除旧密钥对，避免 ssh-keygen 交互式询问是否覆盖
rm -f "$SSH_PRIVATE_KEY" "$SSH_PRIVATE_KEY.pub"
# -N "" 代表密钥不设置密码；-f 指定密钥文件路径，避免交互式询问
ssh-keygen -t ed25519 -C "donglin_cui@163.com" -N "" -f "$SSH_PRIVATE_KEY"

echo "===== 公钥内容（复制下面全部，添加到GitHub SSH Keys） ====="
cat "$SSH_PRIVATE_KEY.pub"
echo -e "\n==================== 完成 ===================="
echo "测试连接：ssh -T git@github.com"