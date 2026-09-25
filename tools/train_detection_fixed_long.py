#!/usr/bin/env python3
"""Fixed-configuration long training for the detection transfer study."""

import argparse
import copy
import csv
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def load_module(path):
    spec = importlib.util.spec_from_file_location("detection_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_write(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-script", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--byol", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--position", choices=["none", "layer3", "layer4"], required=True)
    parser.add_argument("--gate", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    detection = load_module(args.smoke_script)
    protocol = json.loads(Path(args.protocol).read_text())
    full = detection.VOCDetection(args.data, year="2007", image_set="trainval", download=False)
    index_by_id = {Path(path).stem: index for index, path in enumerate(full.images)}
    train_indices = [index_by_id[value] for value in protocol["data"]["final_train_ids"]]
    validation_indices = [index_by_id[value] for value in protocol["data"]["search_validation_ids"]]
    final = protocol["training"]["final"]
    lr = float(protocol["training"]["short"]["learning_rate"])
    donor_state = detection.unwrap(args.byol)
    torch.manual_seed(0)
    template = detection.detector_template()
    base_state = copy.deepcopy(template.state_dict())
    del template

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows, completed = [], set()
    if args.resume and output.exists():
        rows = list(csv.DictReader(output.open(newline="")))
        completed = {int(row["seed"]) for row in rows}

    device = torch.device("cuda")
    position = None if args.position == "none" else args.position
    for seed in map(int, final["seeds"]):
        if seed in completed:
            continue
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        train_loader = DataLoader(
            detection.VOCSubset(args.data, train_indices, True, seed),
            batch_size=int(final["batch_size"]), shuffle=True, num_workers=2,
            collate_fn=detection.collate, pin_memory=True)
        validation_loader = DataLoader(
            detection.VOCSubset(args.data, validation_indices), batch_size=2,
            shuffle=False, num_workers=2, collate_fn=detection.collate,
            pin_memory=True)
        model = detection.build_model(
            base_state, donor_state, position, float(args.gate),
            stat_align=position is not None).to(device)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.time()
        history = detection.train(model, train_loader, device, int(final["steps"]), lr)
        ap50, class_ap50 = detection.evaluate_ap50(model, validation_loader, device)
        checkpoint = checkpoint_dir / f"{args.candidate}__seed{seed}__last.pth"
        torch.save({"model": model.state_dict(), "candidate": args.candidate,
                    "seed": seed, "steps": int(final["steps"])}, checkpoint)
        row = {
            "candidate": args.candidate, "position": args.position,
            "gate": args.gate, "seed": seed, "steps": int(final["steps"]),
            "train_samples": len(train_indices),
            "validation_samples": len(validation_indices),
            "validation_ap50": ap50, "class_ap50": json.dumps(class_ap50),
            "history": json.dumps(history),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
            "elapsed_sec": time.time() - started,
            "optimizer": "SGD", "learning_rate": lr,
            "checkpoint_rule": final["checkpoint_rule"],
            "checkpoint": str(checkpoint)}
        rows.append(row)
        atomic_write(output, rows)
        print("RESULT", json.dumps({k: v for k, v in row.items()
                                    if k not in {"class_ap50", "history"}}), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
