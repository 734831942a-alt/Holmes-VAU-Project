"""
merge_lora_balanced.py

将 HolmesVAU_lora_balanced（未合并 LoRA checkpoint）与基座模型合并，
输出可直接用 AutoModel.from_pretrained 推理的目录。

用法（服务器上）：
    cd /root/autodl-tmp/HolmesVAU-master
    python merge_lora_balanced.py
    # 或覆盖默认路径：
    python merge_lora_balanced.py \
        --lora-ckpt ckpts/HolmesVAU_lora_balanced \
        --base-model ckpts/HolmesVAU-2B \
        --output    ckpts/HolmesVAU_lora_balanced_merged

LoRA 超参（来自 run_autodl_balanced_train.sh）：
    --use_llm_lora 64  →  lora_r = 64, lora_alpha = 2*64 = 128, scale = 2.0
"""

import argparse
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


# ---------------------------------------------------------------------------
# LoRA 超参（与训练脚本对齐）
# ---------------------------------------------------------------------------
LORA_R = 64
LORA_ALPHA = 128
SCALE = LORA_ALPHA / LORA_R  # = 2.0


def clean_key(key: str) -> str:
    """去掉 PEFT 在键名中插入的 .base_model.model 前缀。

    Examples
    --------
    language_model.base_model.model.model.layers.0.attention.wo.base_layer.weight
      → language_model.model.layers.0.attention.wo.weight  (after strip .base_layer)

    language_model.base_model.model.output.weight
      → language_model.output.weight
    """
    return re.sub(r"\.base_model\.model", "", key)


def merge_lora(lora_ckpt: Path, base_model: Path, output: Path) -> None:
    print(f"[1/5] Loading LoRA checkpoint: {lora_ckpt}")
    ckpt_path = lora_ckpt / "model.safetensors"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {lora_ckpt}")
    ckpt = load_file(str(ckpt_path))

    # ------------------------------------------------------------------
    # Organize LoRA keys
    # base_layer  → base weights (with LoRA layers applied on top)
    # lora_A / lora_B  → adapter matrices
    # other  → non-LoRA weights (vision encoder, mlp, embeddings, …)
    # ------------------------------------------------------------------
    print("[2/5] Analysing checkpoint keys …")
    base_weights: dict[str, torch.Tensor] = {}
    lora_A_map: dict[str, torch.Tensor] = {}
    lora_B_map: dict[str, torch.Tensor] = {}
    other_weights: dict[str, torch.Tensor] = {}

    for key, tensor in ckpt.items():
        if ".base_layer.weight" in key:
            # Strip .base_layer.weight → .weight, remove PEFT prefix
            clean = clean_key(key).replace(".base_layer.weight", ".weight")
            base_weights[clean] = tensor
        elif ".lora_A." in key:
            # e.g.  …wo.lora_A.default.weight  →  …wo.weight  (map key)
            clean = re.sub(r"\.lora_A\.default\.weight$", ".weight", clean_key(key))
            lora_A_map[clean] = tensor
        elif ".lora_B." in key:
            clean = re.sub(r"\.lora_B\.default\.weight$", ".weight", clean_key(key))
            lora_B_map[clean] = tensor
        else:
            other_weights[clean_key(key)] = tensor

    n_lora = len(base_weights)
    n_other = len(other_weights)
    print(f"    LoRA layers : {n_lora}")
    print(f"    Other keys  : {n_other}")
    print(f"    scale       : {SCALE} (r={LORA_R}, alpha={LORA_ALPHA})")

    # ------------------------------------------------------------------
    # Merge: W_merged = W_base + (B @ A) * scale
    # ------------------------------------------------------------------
    print("[3/5] Merging LoRA deltas …")
    merged: dict[str, torch.Tensor] = {}

    for key, base in base_weights.items():
        if key in lora_A_map and key in lora_B_map:
            A = lora_A_map[key]  # shape (r, in_features)
            B = lora_B_map[key]  # shape (out_features, r)
            delta = (B.to(torch.float32) @ A.to(torch.float32)) * SCALE
            merged[key] = (base.to(torch.float32) + delta).to(base.dtype)
        else:
            # LoRA key exists as base_layer but no adapter → keep as-is
            merged[key] = base

    # Non-LoRA weights (vision encoder, mlp, embedding, …)
    merged.update(other_weights)

    # ------------------------------------------------------------------
    # Fallback: load base model for any keys still missing
    # (typically not needed if checkpoint is a full model save)
    # ------------------------------------------------------------------
    base_ckpt_path = base_model / "model.safetensors"
    if base_ckpt_path.exists():
        print("[3b/5] Loading base model for any missing keys …")
        base_ckpt = load_file(str(base_ckpt_path))
        missing = [k for k in base_ckpt if k not in merged]
        if missing:
            print(f"    Filling {len(missing)} keys from base model: {missing[:5]} …")
            for k in missing:
                merged[k] = base_ckpt[k]
        else:
            print("    No missing keys — checkpoint is self-contained.")
    else:
        print(f"    [warn] base model safetensors not found at {base_ckpt_path}, skipping fallback")

    # ------------------------------------------------------------------
    # Save merged weights
    # ------------------------------------------------------------------
    print(f"[4/5] Saving merged model → {output}")
    output.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(output / "model.safetensors"), metadata={"format": "pt"})
    print(f"    Saved {len(merged)} tensors")

    # ------------------------------------------------------------------
    # Copy config / tokenizer files from base model (not from lora_ckpt,
    # because lora_ckpt config.json may still reference LoRA structure)
    # ------------------------------------------------------------------
    print("[5/5] Copying config & tokenizer files from base model …")
    COPY_EXTS = {".json", ".py", ".model", ".txt", ".bin"}
    for src in base_model.iterdir():
        if src.is_file() and src.suffix in COPY_EXTS and src.name != "model.safetensors":
            dst = output / src.name
            shutil.copy2(src, dst)
            print(f"    copied {src.name}")

    # config.json from lora_ckpt may have use_llm_lora field; after merge
    # we don't need it. Overwrite with base model's config.json (already done above).
    print(f"\nDone. Merged model at: {output}")
    print("Test with:")
    print(f"  python batch_test_traffic.py --model-path {output} --max-samples 1 ...")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-ckpt", default="ckpts/HolmesVAU_lora_balanced",
                        help="Path to unmerged LoRA checkpoint directory")
    parser.add_argument("--base-model", default="ckpts/HolmesVAU-2B",
                        help="Path to base model directory")
    parser.add_argument("--output", default="ckpts/HolmesVAU_lora_balanced_merged",
                        help="Output directory for merged model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_lora(
        lora_ckpt=Path(args.lora_ckpt),
        base_model=Path(args.base_model),
        output=Path(args.output),
    )
