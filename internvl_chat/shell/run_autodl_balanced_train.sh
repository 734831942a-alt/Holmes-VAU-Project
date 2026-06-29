#!/usr/bin/env bash
set -euo pipefail

# Paths based on your latest server layout.
PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/HolmesVAU-master}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/HolmesVAU-master/examples/HIVAU-70k}

TRAIN_JSONL=${TRAIN_JSONL:-${DATA_ROOT}/traffic_train.jsonl}
VIDEO_ROOT=${VIDEO_ROOT:-}
BY_CLASS_DIR=${BY_CLASS_DIR:-${DATA_ROOT}/by_class_train}
BALANCED_META=${BALANCED_META:-${PROJECT_ROOT}/internvl_chat/shell/holmesvau_data_balanced.json}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/ckpts/HolmesVAU_lora_balanced}
BASE_MODEL=${BASE_MODEL:-../ckpts/HolmesVAU-2B}

PYTHON_BIN=${PYTHON_BIN:-python}
GPUS=${GPUS:-1}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
BATCH_SIZE=${BATCH_SIZE:-256}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
MASTER_PORT=${MASTER_PORT:-34229}
MAX_STEPS=${MAX_STEPS:-500}
LEARNING_RATE=${LEARNING_RATE:-4e-5}

# Class reweighting via repeat_time.
REPEAT_DUOCHE=${REPEAT_DUOCHE:-8}  # 多车事故
REPEAT_POSA=${REPEAT_POSA:-9}      # 抛洒物

cd "${PROJECT_ROOT}"

echo "[0/3] Check paths..."
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "TRAIN_JSONL=${TRAIN_JSONL}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "LEARNING_RATE=${LEARNING_RATE}"
[[ -f "${TRAIN_JSONL}" ]] || { echo "Missing: ${TRAIN_JSONL}"; exit 1; }

# Needed by Python fallback probe below.
export TRAIN_JSONL

# Auto-detect VIDEO_ROOT if not provided.
if [[ -z "${VIDEO_ROOT}" ]]; then
  if [[ -d "${DATA_ROOT}/video" ]]; then
    VIDEO_ROOT="${DATA_ROOT}/video"
  elif [[ -d "${DATA_ROOT}/videos" ]]; then
    VIDEO_ROOT="${DATA_ROOT}/videos"
  else
    # Fallback: locate by first video filename in train jsonl.
    FIRST_VIDEO=$("${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
p = Path(os.environ["TRAIN_JSONL"])
for line in p.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    obj = json.loads(line)
    v = obj.get("video", "")
    if v:
        print(Path(v).name)
        break
PY
)
    if [[ -n "${FIRST_VIDEO}" ]]; then
      CANDIDATE=$(find "${DATA_ROOT}" -type f -name "${FIRST_VIDEO}" 2>/dev/null | head -n 1 || true)
      if [[ -n "${CANDIDATE}" ]]; then
        VIDEO_ROOT=$(dirname "${CANDIDATE}")
      fi
    fi
  fi
fi

echo "VIDEO_ROOT=${VIDEO_ROOT}"
[[ -n "${VIDEO_ROOT}" && -d "${VIDEO_ROOT}" ]] || { echo "Missing video dir. Set VIDEO_ROOT explicitly."; exit 1; }

# Export variables for Python subprocesses.
export PROJECT_ROOT DATA_ROOT TRAIN_JSONL VIDEO_ROOT BY_CLASS_DIR BALANCED_META OUTPUT_DIR
export REPEAT_DUOCHE REPEAT_POSA

echo "[1/3] Split train jsonl by class..."
mkdir -p "${BY_CLASS_DIR}"
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

src = Path(os.environ["TRAIN_JSONL"])
out_dir = Path(os.environ["BY_CLASS_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)

writers = {}
for line in src.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    obj = json.loads(line)
    video = Path(obj.get("video", "")).name
    cls = video.split("_", 1)[0] if "_" in video else Path(video).stem
    dst = out_dir / f"{cls}.jsonl"
    if cls not in writers:
        writers[cls] = dst.open("w", encoding="utf-8")
    writers[cls].write(json.dumps(obj, ensure_ascii=False) + "\n")

for fp in writers.values():
    fp.close()

print("split done:", out_dir)
PY

echo "[2/3] Build balanced meta json..."
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

by_class_dir = Path(os.environ["BY_CLASS_DIR"])
video_root = Path(os.environ["VIDEO_ROOT"]).resolve()
out_meta = Path(os.environ["BALANCED_META"])
repeat_duoche = int(os.environ["REPEAT_DUOCHE"])
repeat_posa = int(os.environ["REPEAT_POSA"])

repeat_map = {
    "拥堵": 1,
    "二轮车辆闯入": 1,
    "占道施工": 1,
    "异常停车": 1,
    "多车事故": repeat_duoche,
    "抛洒物": repeat_posa,
}

meta = {}
for p in sorted(by_class_dir.glob("*.jsonl")):
    cls = p.stem
    n = sum(1 for _ in p.open("r", encoding="utf-8"))
    meta[cls] = {
        "root": str(video_root).replace("\\", "/"),
        "annotation": str(p.resolve()).replace("\\", "/"),
        "data_augment": False,
        "repeat_time": repeat_map.get(cls, 1),
        "length": n,
    }

out_meta.parent.mkdir(parents=True, exist_ok=True)
out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print("saved meta:", out_meta)
print(json.dumps(meta, ensure_ascii=False, indent=2))
PY

echo "[3/3] Start balanced LoRA training..."
cd "${PROJECT_ROOT}/internvl_chat"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export MASTER_PORT
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

mkdir -p "${OUTPUT_DIR}"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "${BASE_MODEL}" \
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
  --learning_rate "${LEARNING_RATE}" \
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

echo "Done. Output: ${OUTPUT_DIR}"
