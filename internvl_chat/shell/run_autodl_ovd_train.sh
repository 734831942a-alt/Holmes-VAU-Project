#!/usr/bin/env bash
# ============================================================
# run_autodl_ovd_train.sh
# E4 实验：OVD训练-推理一致性实验
# 用 OVD-ZH 增强训练集重新微调模型，测试时也用 OVD-ZH prompt
#
# 用法：
#   cd /root/autodl-tmp/HolmesVAU-master
#   bash internvl_chat/shell/run_autodl_ovd_train.sh
#
# 关键变量可覆盖：
#   TRAIN_OVD_SUMMARY  训练集 OVD 摘要 jsonl（如不存在则跳过注入，退化为 E1）
#   MAX_STEPS          默认 500（与 balanced 训练对齐）
# ============================================================
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/HolmesVAU-master}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/高架桥数据}

# ── 输入数据 ──
TRAIN_JSONL=${TRAIN_JSONL:-${DATA_ROOT}/traffic_train.jsonl}
TRAIN_OVD_SUMMARY=${TRAIN_OVD_SUMMARY:-${DATA_ROOT}/traffic_train_ovd_summary.jsonl}
VIDEO_ROOT=${VIDEO_ROOT:-${DATA_ROOT}/video_fixed}

# ── 中间产物 ──
TRAIN_OVD_ZH=${TRAIN_OVD_ZH:-${DATA_ROOT}/traffic_train_ovd_zh.jsonl}
BY_CLASS_DIR=${BY_CLASS_DIR:-${DATA_ROOT}/by_class_train_ovd}
BALANCED_META=${BALANCED_META:-${PROJECT_ROOT}/internvl_chat/shell/holmesvau_data_ovd_balanced.json}

# ── 输出 ──
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/ckpts/HolmesVAU_lora_ovd}

PYTHON_BIN=${PYTHON_BIN:-python}
GPUS=${GPUS:-1}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
BATCH_SIZE=${BATCH_SIZE:-256}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
MASTER_PORT=${MASTER_PORT:-34230}
MAX_STEPS=${MAX_STEPS:-500}

REPEAT_DUOCHE=${REPEAT_DUOCHE:-8}
REPEAT_POSA=${REPEAT_POSA:-9}

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/internvl_chat:${PYTHONPATH:-}"

echo "========================================================"
echo "E4 OVD Training Experiment"
echo "  TRAIN_JSONL       = ${TRAIN_JSONL}"
echo "  TRAIN_OVD_SUMMARY = ${TRAIN_OVD_SUMMARY}"
echo "  TRAIN_OVD_ZH      = ${TRAIN_OVD_ZH}"
echo "  OUTPUT_DIR        = ${OUTPUT_DIR}"
echo "========================================================"

[[ -f "${TRAIN_JSONL}" ]] || { echo "ERROR: Missing ${TRAIN_JSONL}"; exit 1; }

# ─────────────────────────────────────────────────────────────
# STEP 1: 构建 OVD-ZH 增强训练集
# ─────────────────────────────────────────────────────────────
echo ""
echo "[1/4] Building OVD-ZH augmented training set..."

if [[ -f "${TRAIN_OVD_SUMMARY}" ]]; then
    python build_ovd_prompt_zh.py \
        --input  "${TRAIN_JSONL}" \
        --ovd    "${TRAIN_OVD_SUMMARY}" \
        --output "${TRAIN_OVD_ZH}"
    echo "    OVD-ZH train jsonl saved: ${TRAIN_OVD_ZH}"
    ACTUAL_TRAIN_JSONL="${TRAIN_OVD_ZH}"
else
    echo "    WARNING: ${TRAIN_OVD_SUMMARY} not found."
    echo "    Falling back to original train jsonl (no OVD injection)."
    echo "    To run true E4, generate OVD summaries for training videos first:"
    echo "      python ovd_video_summary.py --video-dir ${VIDEO_ROOT} \\"
    echo "        --output ${TRAIN_OVD_SUMMARY}"
    ACTUAL_TRAIN_JSONL="${TRAIN_JSONL}"
fi

export ACTUAL_TRAIN_JSONL

# ─────────────────────────────────────────────────────────────
# STEP 2: 按类别拆分训练集
# ─────────────────────────────────────────────────────────────
echo ""
echo "[2/4] Splitting train jsonl by class..."
mkdir -p "${BY_CLASS_DIR}"
export BY_CLASS_DIR ACTUAL_TRAIN_JSONL

"${PYTHON_BIN}" - <<'PY'
import json, os
from pathlib import Path

src = Path(os.environ["ACTUAL_TRAIN_JSONL"])
out_dir = Path(os.environ["BY_CLASS_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)

writers = {}
for line in src.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    obj = json.loads(line)
    from pathlib import Path as P
    video = P(obj.get("video", "")).name
    cls = video.split("_", 1)[0] if "_" in video else P(video).stem
    dst = out_dir / f"{cls}.jsonl"
    if cls not in writers:
        writers[cls] = dst.open("w", encoding="utf-8")
    writers[cls].write(json.dumps(obj, ensure_ascii=False) + "\n")

for fp in writers.values():
    fp.close()
print("split done:", out_dir, "classes:", list(writers.keys()))
PY

# ─────────────────────────────────────────────────────────────
# STEP 3: 构建 balanced meta json（重用 balanced 训练的重加权逻辑）
# ─────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Building balanced meta json..."

export BY_CLASS_DIR VIDEO_ROOT BALANCED_META REPEAT_DUOCHE REPEAT_POSA

"${PYTHON_BIN}" - <<'PY'
import json, os
from pathlib import Path

by_class_dir = Path(os.environ["BY_CLASS_DIR"])
video_root = Path(os.environ["VIDEO_ROOT"]).resolve()
out_meta = Path(os.environ["BALANCED_META"])
repeat_duoche = int(os.environ["REPEAT_DUOCHE"])
repeat_posa   = int(os.environ["REPEAT_POSA"])

# 与 run_autodl_balanced_train.sh 保持一致的重加权比例
repeat_map = {
    "拥堵":         1,
    "二轮车辆闯入": 1,
    "占道施工":     1,
    "异常停车":     1,
    "多车事故":     repeat_duoche,
    "抛洒物":       repeat_posa,
}

datasets = {}
for jsonl_file in sorted(by_class_dir.glob("*.jsonl")):
    cls = jsonl_file.stem
    repeat = repeat_map.get(cls, 1)
    datasets[cls] = {
        "root": str(video_root),
        "annotation": str(jsonl_file),
        "data_augment": False,
        "repeat_time": repeat,
        "length": sum(1 for l in jsonl_file.read_text().splitlines() if l.strip()),
    }

out_meta.parent.mkdir(parents=True, exist_ok=True)
out_meta.write_text(json.dumps(datasets, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"meta saved: {out_meta}")
for k, v in datasets.items():
    print(f"  {k}: {v['length']} samples x{v['repeat_time']}")
PY

# ─────────────────────────────────────────────────────────────
# STEP 4: 微调训练
# ─────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Starting fine-tuning (E4 OVD model)..."
mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_ROOT}/internvl_chat"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "${PROJECT_ROOT}/ckpts/HolmesVAU_lora_balanced_merged" \
  --conv_style "internlm2-chat" \
  --output_dir "${OUTPUT_DIR}" \
  --meta_path "${BALANCED_META}" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --use_llm_lora 64 \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACC}" \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 100 \
  --save_total_limit 1 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 4096 \
  --do_train True \
  --max_steps "${MAX_STEPS}" \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version "v2" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

echo ""
echo "Training done. Next: merge LoRA and eval."
echo "  python merge_lora_balanced.py \\"
echo "    --lora-ckpt ${OUTPUT_DIR} \\"
echo "    --base-model ${PROJECT_ROOT}/ckpts/HolmesVAU-2B \\"
echo "    --output    ${OUTPUT_DIR}_merged"
