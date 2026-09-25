#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
  echo "Usage: $0 PETS_ROOT PROTOCOL_JSON ANCHOR BYOL DINO SWAV OUTPUT_DIR" >&2
  exit 2
fi

"${PYTHON:-python}" tools/train_segmentation_paper_models.py \
  --data "$1" --protocol "$2" --anchor "$3" \
  --byol "$4" --dino "$5" --swav "$6" --output-dir "$7"

