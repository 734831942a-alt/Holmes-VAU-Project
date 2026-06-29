#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-autodl-tmp/高架桥数据}
META_PATH=${META_PATH:-${DATA_ROOT}/traffic_meta.json}
TEST_JSONL=${TEST_JSONL:-${DATA_ROOT}/traffic_test.jsonl}
MODEL_PATH=${MODEL_PATH:-./ckpts/HolmesVAU-2B}
SAMPLER_PATH=${SAMPLER_PATH:-./holmesvau/ATS/anomaly_scorer.pth}
DEVICE=${DEVICE:-cuda:0}
SELECT_FRAMES=${SELECT_FRAMES:-12}
MAX_SAMPLES=${MAX_SAMPLES:-0}
USE_ATS=${USE_ATS:-1}

OUT_JSONL=${OUT_JSONL:-${DATA_ROOT}/traffic_test_pred.jsonl}
SUMMARY_JSON=${SUMMARY_JSON:-${DATA_ROOT}/traffic_test_summary.json}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

CMD=(
  python batch_test_traffic.py
  --meta "${META_PATH}"
  --test-jsonl "${TEST_JSONL}"
  --model-path "${MODEL_PATH}"
  --sampler-path "${SAMPLER_PATH}"
  --device "${DEVICE}"
  --select-frames "${SELECT_FRAMES}"
  --max-samples "${MAX_SAMPLES}"
  --output-jsonl "${OUT_JSONL}"
  --summary-json "${SUMMARY_JSON}"
)

if [[ "${USE_ATS}" == "1" ]]; then
  CMD+=(--use-ats)
fi

"${CMD[@]}"
