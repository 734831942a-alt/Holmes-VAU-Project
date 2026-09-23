#!/usr/bin/env bash
set -euo pipefail

# ============================================
# TRB Baseline 500步训练实验
# 纯Baseline，无OVD，无balanced采样
# ============================================

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/HolmesVAU-master}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/高架桥数据}
TRAIN_JSONL=${TRAIN_JSONL:-${DATA_ROOT}/traffic_train.jsonl}
VIDEO_ROOT=${VIDEO_ROOT:-${DATA_ROOT}/video}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/ckpts/HolmesVAU_lora_baseline_500}
BASE_MODEL=${BASE_MODEL:-../ckpts/HolmesVAU-2B}

PYTHON_BIN=${PYTHON_BIN:-python}
GPUS=${GPUS:-1}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
BATCH_SIZE=${BATCH_SIZE:-256}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
MASTER_PORT=${MASTER_PORT:-34229}
MAX_STEPS=500
LEARNING_RATE=4e-5

cd "${PROJECT_ROOT}"

echo "============================================"
echo "TRB Baseline 500步训练实验"
echo "============================================"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "TRAIN_JSONL=${TRAIN_JSONL}"
echo "VIDEO_ROOT=${VIDEO_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "============================================"

# 检查训练数据
[[ -f "${TRAIN_JSONL}" ]] || { echo "❌ Missing: ${TRAIN_JSONL}"; exit 1; }
[[ -d "${VIDEO_ROOT}" ]] || { echo "❌ Missing: ${VIDEO_ROOT}"; exit 1; }

# 创建简单的meta配置 (无balanced, repeat_time=1)
META_JSON="${PROJECT_ROOT}/internvl_chat/shell/holmesvau_data_baseline.json"
cat > "${META_JSON}" <<EOF
{
  "TrafficAccident": {
    "root": "${VIDEO_ROOT}",
    "annotation": "${TRAIN_JSONL}",
    "data_augment": false,
    "repeat_time": 1,
    "length": 0
  }
}
EOF

echo "✅ 生成配置文件: ${META_JSON}"
cat "${META_JSON}"
echo "============================================"

# 开始训练
cd "${PROJECT_ROOT}/internvl_chat"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export MASTER_PORT
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

mkdir -p "${OUTPUT_DIR}"

echo "🚀 启动训练 ($(date))..."
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
  --meta_path "${META_JSON}" \
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

echo "============================================"
echo "✅ 训练完成！($(date))"
echo "输出目录: ${OUTPUT_DIR}"
echo "训练记录: ${OUTPUT_DIR}/trainer_state.json"
echo "============================================"

# 自动打包结果文件
RESULT_ARCHIVE="${OUTPUT_DIR}/training_results.tar.gz"
echo "📦 打包结果文件..."
cd "${OUTPUT_DIR}"
tar -czf training_results.tar.gz \
  trainer_state.json \
  training_log.txt \
  checkpoint-500/trainer_state.json \
  2>/dev/null || true

if [ -f "${RESULT_ARCHIVE}" ]; then
  echo "✅ 结果已打包: ${RESULT_ARCHIVE}"
  echo "   文件大小: $(du -h ${RESULT_ARCHIVE} | cut -f1)"
fi

echo "============================================"
echo "下一步:"
echo "1. 下载 trainer_state.json 到本地"
echo "2. 运行训练曲线绘图脚本"
echo "============================================"
