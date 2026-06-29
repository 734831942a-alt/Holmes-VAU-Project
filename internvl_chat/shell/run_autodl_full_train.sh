#!/usr/bin/env bash
set -euo pipefail

# AutoDL defaults based on user-provided paths.
DATA_ROOT=${DATA_ROOT:-autodl-tmp/高架桥数据}
EXCEL_PATH=${EXCEL_PATH:-${DATA_ROOT}/标注.xlsx}
VIDEO_ROOT=${VIDEO_ROOT:-${DATA_ROOT}/video}
OUTPUT_DIR=${OUTPUT_DIR:-${DATA_ROOT}}

# Optional split config.
TRAIN_RATIO=${TRAIN_RATIO:-0.8}
SPLIT_COLUMN=${SPLIT_COLUMN:-}

# Training config.
GPUS=${GPUS:-1}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "[1/3] Building full train/test jsonl + meta from Excel..."
PREP_ARGS=(
  --excel "${EXCEL_PATH}"
  --video-root "${VIDEO_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --train-ratio "${TRAIN_RATIO}"
)
if [[ -n "${SPLIT_COLUMN}" ]]; then
  PREP_ARGS+=(--split-column "${SPLIT_COLUMN}")
fi
python prepare_traffic_dataset.py "${PREP_ARGS[@]}"

echo "[2/3] Start LoRA training using generated meta..."
cd internvl_chat
export GPUS PER_DEVICE_BATCH_SIZE BATCH_SIZE NUM_TRAIN_EPOCHS
export META_PATH="${OUTPUT_DIR}/traffic_meta.json"

bash shell/internvl2_2b_finetune_lora.sh

echo "[3/3] Done."
echo "Meta used: ${META_PATH}"
