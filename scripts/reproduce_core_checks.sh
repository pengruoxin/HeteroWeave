#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
PYTHONPATH="${PWD}" "${PYTHON_BIN}" -m unittest discover -s tests -v
"${PYTHON_BIN}" -m compileall -q heteroweave blocklize mmcls_addon simlarity tools

TEMP_DIR="$(mktemp -d)"
trap 'find "${TEMP_DIR}" -type f -delete; rmdir "${TEMP_DIR}"' EXIT
cat > "${TEMP_DIR}/archive.csv" <<'CSV'
candidate,total_quality,size,flops
a,10,4,4
b,9,3,3
c,8,5,5
d,10,5,4
CSV
"${PYTHON_BIN}" tools/export_three_objective_pareto.py \
  --input "${TEMP_DIR}/archive.csv" \
  --output "${TEMP_DIR}/front.csv" \
  --front-only
"${PYTHON_BIN}" - "${TEMP_DIR}/front.csv" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    names = [row["candidate"] for row in csv.DictReader(handle)]
if set(names) != {"a", "b"}:
    raise SystemExit(f"unexpected Pareto front: {names}")
print("core checks passed")
PY
