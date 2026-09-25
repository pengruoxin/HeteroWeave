#!/usr/bin/env bash
set -euo pipefail

GPUS="${1:-8}"
MODEL="${2:-all}"
PYTHON_BIN="${PYTHON:-python}"

declare -A CONFIGS=(
  [heteroweave_p]="configs/imagenet/heteroweave_p_10m3g_100e.py"
  [heteroweave_e]="configs/imagenet/heteroweave_e_10m3g_100e.py"
  [heteroweave]="configs/imagenet/heteroweave_main_100e.py"
  [dery_10m3g]="configs/imagenet/dery_10m3g_100e.py"
  [dery_30m6g]="configs/imagenet/dery_baseline_100e.py"
  [model_stitching_10m3g]="configs/baselines/imagenet/model_stitching_10m3g_100e.py"
  [model_stitching_30m6g]="configs/baselines/imagenet/model_stitching_30m6g_100e.py"
  [side_tuning_10m3g]="configs/baselines/imagenet/side_tuning_10m3g_100e.py"
  [side_tuning_30m6g]="configs/baselines/imagenet/side_tuning_30m6g_100e.py"
  [snnet_3ti9s]="configs/baselines/imagenet/snnet_3ti9s_100e.py"
  [snnet_9ti3s]="configs/baselines/imagenet/snnet_9ti3s_100e.py"
  [regnet_y_800mf]="configs/references/imagenet/regnet_y_800mf_100e.py"
  [mobilenet_v3_large]="configs/references/imagenet/mobilenet_v3_large_100e.py"
  [regnet_y_3_2gf]="configs/references/imagenet/regnet_y_3_2gf_100e.py"
  [resnet50]="configs/references/imagenet/resnet50_100e.py"
  [swin_tiny]="configs/references/imagenet/swin_tiny_100e.py"
  [capacity_serial]="configs/analysis/capacity_serial_100e.py"
  [capacity_parallel]="configs/analysis/capacity_parallel_100e.py"
)

if [[ "${MODEL}" == "all" ]]; then
  MODELS=(heteroweave_p heteroweave_e heteroweave dery_10m3g dery_30m6g \
    model_stitching_10m3g model_stitching_30m6g \
    side_tuning_10m3g side_tuning_30m6g snnet_3ti9s snnet_9ti3s \
    regnet_y_800mf mobilenet_v3_large regnet_y_3_2gf resnet50 swin_tiny \
    capacity_serial capacity_parallel)
else
  MODELS=("${MODEL}")
fi

for NAME in "${MODELS[@]}"; do
  CONFIG="${CONFIGS[${NAME}]:-}"
  if [[ -z "${CONFIG}" ]]; then
    echo "Unknown model: ${NAME}" >&2
    exit 2
  fi
  for SEED in 11 23 47; do
    PYTHONPATH="${PWD}" "${PYTHON_BIN}" tools/dist_train.py \
      "${CONFIG}" "${GPUS}" --seed "${SEED}" --deterministic \
      --work-dir "work_dirs/${NAME}_seed${SEED}"
  done
done
