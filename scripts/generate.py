"""
scripts/generate.py
====================
加载预训练 / SFT checkpoint，给定 prompt 生成文本，直观评估模型质量。

用法:
    python scripts/generate.py --ckpt output/pretrain_tiny/final.pt
    python scripts/generate.py --ckpt output/pretrain_tiny/ckpt_step1000.pt \
        --prompt "人工智能是" --topk 50 --device auto

三把尺子判断预训练好坏:
    1. loss 曲线: 初始 ≈ ln(vocab)=11.93, 应单调下降
    2. PPL = exp(loss): <50 算"能读"
    3. 本脚本: 生成文本是否语法正确、不纯重复(最终裁判)
"""

from __future__ import annotations

import argparse
import os

import torch

# 允许 scripts/ 下直接运行
os.environ.setdefault("PYTHONPATH", "src")

from minillm.models.transformer import MiniLLM, MiniLLMConfig
from minillm.tokenizer.tokenizer import Tokenizer


def build_model(model_cfg, tokenizer) -> MiniLLM:
    real_vocab = max(tokenizer.vocab_size, tokenizer.eos_token_id) + 1
    if model_cfg.vocab_size != real_vocab:
        print(f"  [warn] 配置 vocab={model_cfg.vocab_size} 与 tokenizer 实际 "
              f"{real_vocab} 不一致, 自动修正为 {real_vocab}")
    return MiniLLM(MiniLLMConfig(
        vocab_size=real_vocab,
        hidden_size=model_cfg.hidden_size,
        intermediate_size=model_cfg.intermediate_size,
        num_hidden_layers=model_cfg.num_hidden_layers,
        num_attention_heads=model_cfg.num_attention_heads,
        num_key_value_heads=model_cfg.num_key_value_heads,
        max_position_embeddings=model_cfg.max_position_embeddings,
        rope_theta=model_cfg.rope_theta,
        norm_eps=model_cfg.norm_eps,
        initializer_range=model_cfg.initializer_range,
        tie_word_embeddings=True,
    ))


def load_ckpt_weights(model, ckpt_path: str, device: str) -> None:
    """加载 checkpoint 中的模型权重（兼容 DDP 的 module. 前缀）。"""
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = sd["model"] if "model" in sd else sd
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    step = sd.get("step", "?")
    print(f"  [ckpt] 加载 {ckpt_path} (step={step})")


@torch.no_grad()
def generate(model, tokenizer, prompt: str, *, max_new: int = 80,
             topk: int = 0, temperature: float = 0.8, device: str = "cpu"):
    """自回归生成: topk=0 时 greedy; topk>0 时 top-k 采样。

    greedy 适合看"模型最自信的预测"(易重复);
    top-k 采样更像人写出来的文本(用于展示效果更好)。
    """
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    gen = list(ids)
    for _ in range(max_new):
        logits = model(torch.tensor([gen], device=device))[0, -1]  # [V]
        if topk > 0:
            logits = logits / temperature
            # top-k: 只保留概率最高的 k 个再 softmax
            v, idx = logits.topk(topk)
            probs = torch.softmax(v, dim=-1)
            nxt = idx[torch.multinomial(probs, 1).item()].item()
        else:
            nxt = logits.argmax().item()
        if nxt == tokenizer.eos_token_id:
            break
        gen.append(nxt)
    return tokenizer.decode(gen[len(ids):])


def main() -> None:
    ap = argparse.ArgumentParser(description="加载 ckpt 生成文本, 评估模型质量")
    ap.add_argument("--ckpt", required=True, help="checkpoint 路径")
    ap.add_argument("--prompt", default="人工智能是", help="生成起点")
    ap.add_argument("--max_new", type=int, default=80, help="生成的最大新 token 数")
    ap.add_argument("--topk", type=int, default=0,
                    help="0=greedy(看最自信预测); >0 用 top-k 采样(看真实能力)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--hidden_size", type=int, default=None,
                    help="覆盖 hidden_size(加载 mini 冒烟 ckpt 时用, 如 96)")
    ap.add_argument("--num_layers", type=int, default=None,
                    help="覆盖层数(加载 mini 冒烟 ckpt 时用, 如 2)")
    ap.add_argument("--num_heads", type=int, default=None,
                    help="覆盖 attention 头数(加载 mini 冒烟 ckpt 时用, 如 4)")
    ap.add_argument("--kv_heads", type=int, default=None,
                    help="覆盖 KV 头数(加载 mini 冒烟 ckpt 时用, 如 1)")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device

    print(f"[generate] device={device}")
    tok = Tokenizer()
    import yaml
    from minillm.config.train_config import ModelConfig
    mcfg = ModelConfig.from_dict(
        yaml.safe_load(open("configs/model/tiny.yaml", encoding="utf-8")))
    if args.hidden_size:
        mcfg.hidden_size = args.hidden_size
        mcfg.intermediate_size = max(256, args.hidden_size * 2 + 64)
    if args.num_layers:
        mcfg.num_hidden_layers = args.num_layers
    if args.num_heads:
        mcfg.num_attention_heads = args.num_heads
    if args.kv_heads:
        mcfg.num_key_value_heads = args.kv_heads
    model = build_model(mcfg, tok).to(device)
    load_ckpt_weights(model, args.ckpt, device)
    model.eval()

    print("\n" + "-" * 66)
    print("prompt: " + args.prompt)
    g = generate(model, tok, args.prompt, max_new=args.max_new, topk=0, device=device)
    print("greedy: " + g)
    if args.topk > 0:
        s = generate(model, tok, args.prompt, max_new=args.max_new, topk=args.topk,
                     temperature=args.temperature, device=device)
        print(f"top-{args.topk}: " + s)
    print("-" * 66)
    print("解读: 乱码=没学好 | 高频字词重复=学了一点 | "
          "语法通顺=语言能力成型")


if __name__ == "__main__":
    main()
