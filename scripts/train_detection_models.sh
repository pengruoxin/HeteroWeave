#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 VOC_ROOT BYOL_CHECKPOINT PROTOCOL_JSON OUTPUT_DIR" >&2
  exit 2
fi

VOC_ROOT="$1"
BYOL_CHECKPOINT="$2"
PROTOCOL_JSON="$3"
OUTPUT_DIR="$4"
PYTHON_BIN="${PYTHON:-python}"
mkdir -p "${OUTPUT_DIR}/checkpoints"

while read -r NAME POSITION GAMMA; do
  "${PYTHON_BIN}" tools/train_detection_fixed_long.py \
    --smoke-script tools/run_fasterrcnn_heteroweave.py \
    --protocol "${PROTOCOL_JSON}" --data "${VOC_ROOT}" \
    --byol "${BYOL_CHECKPOINT}" --candidate "${NAME}" \
    --position "${POSITION}" --gate "${GAMMA}" \
    --output "${OUTPUT_DIR}/${NAME}.csv" \
    --checkpoint-dir "${OUTPUT_DIR}/checkpoints"
done <<'MODELS'
resnet50 none 0.0
heteroweave_d1 layer3 0.001
heteroweave_d2 layer3 0.0025
heteroweave_d3 layer4 0.0025
MODELS

