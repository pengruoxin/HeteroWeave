#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 MAE_CHECKPOINT DINO_CHECKPOINT OUTPUT_DIR" >&2
  exit 2
fi

MAE_CHECKPOINT="$1"
DINO_CHECKPOINT="$2"
OUTPUT_DIR="$3"
PYTHON_BIN="${PYTHON:-python}"
MANIFEST="${OUTPUT_DIR}/retrieval_paper_models.json"
mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" tools/export_paper_manifest.py \
  configs/retrieval/paper_models.py "${MANIFEST}"

for SEED in 0 1 2; do
  "${PYTHON_BIN}" tools/run_clip_heteroweave_search_pool.py \
    --manifest "${MANIFEST}" \
    --mae "${MAE_CHECKPOINT}" \
    --dino "${DINO_CHECKPOINT}" \
    --steps 1000 --batch-size 32 --lr 0.003 --seed "${SEED}" \
    --output "${OUTPUT_DIR}/seed${SEED}.csv"
done

