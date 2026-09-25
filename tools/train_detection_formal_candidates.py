#!/usr/bin/env python3
"""Uniform short training for a frozen HeteroWeave VOC candidate pool."""

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
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-script", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--byol", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--swav", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    detection = load_module(args.smoke_script)
    protocol = json.loads(Path(args.protocol).read_text())
    candidates = list(csv.DictReader(open(args.manifest, newline="")))
    candidates = [candidate for index, candidate in enumerate(candidates)
                  if index % args.shard_count == args.shard_index]
    full = detection.VOCDetection(
        args.data, year="2007", image_set="trainval", download=False)
    index_by_id = {Path(path).stem: index for index, path in enumerate(full.images)}
    train_indices = [index_by_id[value]
                     for value in protocol["data"]["short_train_ids"]]
    validation_indices = [index_by_id[value]
                          for value in protocol["data"]["search_validation_ids"]]
    short = protocol["training"]["short"]
    train_loader = DataLoader(
        detection.VOCSubset(args.data, train_indices, True, short["seeds"][0]),
        batch_size=short["batch_size"], shuffle=True, num_workers=2,
        collate_fn=detection.collate, pin_memory=True)
    validation_loader = DataLoader(
        detection.VOCSubset(args.data, validation_indices), batch_size=2,
        shuffle=False, num_workers=2, collate_fn=detection.collate,
        pin_memory=True)
    source_states = {
        "byol": detection.unwrap(args.byol),
        "dino": detection.unwrap(args.dino),
        "swav": detection.unwrap(args.swav)}
    seed = int(short["seeds"][0])
    torch.manual_seed(seed)
    template = detection.detector_template()
    base_state = copy.deepcopy(template.state_dict())
    del template
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, completed = [], set()
    if args.resume and output.exists():
        rows = list(csv.DictReader(output.open(newline="")))
        completed = {row["candidate"] for row in rows}
    device = torch.device("cuda")
    for candidate in candidates:
        name = candidate["candidate"]
        if name in completed:
            continue
        position = None if candidate["position"] == "none" else candidate["position"]
        sources = json.loads(candidate["sources"])
        donor_states = [source_states[source] for source in sources]
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = detection.build_model(
            base_state, donor_states, position, float(candidate["gate"]),
            stat_align=position is not None).to(device)
        # Reset after candidate-dependent construction so every candidate sees
        # identical sampler order and stochastic detector operations.
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.time()
        history = detection.train(
            model, train_loader, device, int(short["steps"]),
            float(short["learning_rate"]))
        ap50, class_ap50 = detection.evaluate_ap50(
            model, validation_loader, device)
        row = {
            **candidate,
            "seed": seed,
            "steps": int(short["steps"]),
            "train_samples": len(train_indices),
            "validation_samples": len(validation_indices),
            "validation_ap50": ap50,
            "class_ap50": json.dumps(class_ap50),
            "history": json.dumps(history),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters()
                                               if p.requires_grad),
            "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
            "elapsed_sec": time.time() - started,
            "checkpoint_rule": short["checkpoint_rule"]}
        rows.append(row)
        atomic_write(output, rows)
        print("RESULT", json.dumps({key: value for key, value in row.items()
                                    if key not in {"class_ap50", "history"}}), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
