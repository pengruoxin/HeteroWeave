#!/usr/bin/env python3
"""Launch distributed MMClassification training without host-specific logic."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="MMClassification config path")
    parser.add_argument("gpus", type=int, help="number of local GPU processes")
    parser.add_argument(
        "training_args", nargs=argparse.REMAINDER,
        help="arguments forwarded to tools/train.py",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpus < 1:
        raise SystemExit("gpus must be positive")
    repo_root = Path(__file__).resolve().parents[1]
    train_script = repo_root / "tools" / "train.py"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node",
        str(args.gpus),
        "--master_port",
        os.environ.get("MASTER_PORT", "29500"),
        str(train_script),
        args.config,
        "--launcher",
        "pytorch",
        *args.training_args,
    ]
    subprocess.run(command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
