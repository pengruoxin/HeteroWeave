#!/usr/bin/env python3
"""Evaluate the frozen detection pool with CLAS and resource metrics."""

import argparse
import csv
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis
from torch.utils.data import DataLoader


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_rows(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-script", required=True)
    parser.add_argument("--proxy-script", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--byol", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--swav", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--panel-size", type=int, default=16)
    parser.add_argument("--panel-seeds", nargs="+", type=int, default=[11, 23, 37])
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--candidate-ids", nargs="+", default=None)
    args = parser.parse_args()

    detection = load_module("detection_smoke", args.smoke_script)
    proxy = load_module("detection_proxy", args.proxy_script)
    candidates = list(csv.DictReader(open(args.manifest, newline="")))
    if args.candidate_ids:
        requested = set(args.candidate_ids)
        candidates = [row for row in candidates if row["candidate"] in requested]
        if len(candidates) != len(requested):
            resolved = {row["candidate"] for row in candidates}
            raise RuntimeError(f"Missing requested candidates: {sorted(requested - resolved)}")
    source_states = {
        "byol": detection.unwrap(args.byol),
        "dino": detection.unwrap(args.dino),
        "swav": detection.unwrap(args.swav)}
    full = detection.VOCDetection(
        args.data, year="2007", image_set="trainval", download=False)
    protocol = json.loads(Path(args.protocol).read_text())
    index_by_id = {Path(path).stem: index for index, path in enumerate(full.images)}
    panels = []
    frozen_panels = protocol["data"]["proxy_panel_ids"]
    if len(frozen_panels) != len(args.panel_seeds):
        raise RuntimeError("Protocol panel count differs from requested panel seeds")
    for ids in frozen_panels:
        if len(ids) != args.panel_size:
            raise RuntimeError("Protocol panel size differs from requested panel size")
        indices = [index_by_id[image_id] for image_id in ids]
        loader = DataLoader(
            detection.VOCSubset(args.data, indices), batch_size=args.panel_size,
            shuffle=False, num_workers=2, collate_fn=detection.collate)
        panels.append(next(iter(loader))[0])

    torch.manual_seed(args.model_seed)
    template = detection.detector_template()
    base_state = {key: value.detach().cpu().clone()
                  for key, value in template.state_dict().items()}
    del template
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    completed = set()
    if args.resume and output.exists():
        rows = list(csv.DictReader(output.open(newline="")))
        completed = {row["candidate"] for row in rows}
    device = torch.device("cuda")
    for candidate in candidates:
        name = candidate["candidate"]
        if name in completed:
            continue
        started = time.time()
        position = None if candidate["position"] == "none" else candidate["position"]
        sources = json.loads(candidate["sources"])
        donor_states = [source_states[source] for source in sources]
        torch.manual_seed(args.model_seed)
        model = detection.build_model(
            base_state, donor_states, position, float(candidate["gate"]),
            stat_align=str(candidate["stat_align"]).lower() == "true").to(device)
        scores = []
        diagnostic_totals = []
        position_details = []
        layer_counts = []
        for images in panels:
            total_score, count, details = proxy.score_panel(model, images, device)
            diagnostic_totals.append(total_score)
            position_details.append(details)
            if position is not None:
                # Every candidate in this frozen pool modifies exactly one
                # position, so use the paper-specified natural special case.
                scores.append(float(details[position]["sqrt"]))
            layer_counts.append(count)
        one_image = [panels[0][0].to(device)]
        transformed, _ = model.transform(one_image, None)
        flop_counter = FlopCountAnalysis(model.backbone, transformed.tensors)
        flop_counter.unsupported_ops_warnings(False)
        flop_counter.uncalled_modules_warnings(False)
        backbone_flops = float(flop_counter.total())
        row = {
            **candidate,
            "panel_scores": json.dumps(scores),
            "CLAS_mean": "" if not scores else float(np.mean(scores)),
            "CLAS_std": "" if not scores else float(np.std(scores)),
            "baseline_or_diagnostic_position_details": json.dumps(position_details),
            "forbidden_all_position_sum_diagnostic": json.dumps(diagnostic_totals),
            "activation_layer_count": json.dumps(layer_counts),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters()
                                               if p.requires_grad),
            "backbone_flops": backbone_flops,
            "elapsed_sec": time.time() - started}
        rows.append(row)
        write_rows(output, rows)
        print("PROXY", json.dumps({key: value for key, value in row.items()
                                   if key not in {"panel_scores"}}), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
