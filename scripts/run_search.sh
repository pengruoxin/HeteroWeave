#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:?usage: bash scripts/run_search.sh /path/to/imagenet/train [output-dir]}"
OUTPUT_DIR="${2:-work_dirs/heteroweave_search}"
PYTHON_BIN="${PYTHON:-python}"

PYTHONPATH="${PWD}" "${PYTHON_BIN}" tools/search_nsga3_heteroweave.py \
  configs/imagenet/dery_baseline_100e.py \
  --assignment assets/component_pool/assignment_hybrid_4.pkl \
  --data-config configs/_base_/datasets/imagenet_bs64_swin_224.py \
  --data-prefix "${DATA_ROOT}" \
  --proxy CLAS \
  --swap-image-count 32 \
  --proxy-data-seed 11 \
  --proxy-model-seed 11 \
  --population-size 96 \
  --branch-first-population-size 96 \
  --branch-first-generations 40 \
  --generations 120 \
  --branch-layers 1 2 3 \
  --max-branch-layers 3 \
  --max-components-per-position 2 \
  --operators sum \
  --structured-sampling \
  --output-dir "${OUTPUT_DIR}"
