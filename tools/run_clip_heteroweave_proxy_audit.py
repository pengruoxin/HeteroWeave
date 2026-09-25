#!/usr/bin/env python3
"""Audit multimodal HeteroWeave proxies on a fixed 24-candidate pool."""

import argparse
import copy
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.stats import spearmanr
from transformers import CLIPModel, CLIPProcessor

import timm

from run_clip_heteroweave_multimodal import (
    CLIPCollaborativeLayer, batches, symmetric_clip_loss, unwrap_state_dict)


def load_donor(path):
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    state = unwrap_state_dict(torch.load(path, map_location="cpu"))
    incompatible = model.load_state_dict(state, strict=False)
    bad_missing = [x for x in incompatible.missing_keys if not x.startswith(("head", "fc_norm"))]
    if bad_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"donor mismatch for {path}: missing={bad_missing[:8]} "
            f"unexpected={incompatible.unexpected_keys[:8]}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def encode_texts(model, processor, dataset, device, batch_size=128):
    captions = []
    for row in dataset:
        captions.extend(row[f"caption_{j}"] for j in range(5))
    output = []
    for start in range(0, len(captions), batch_size):
        tokens = processor(text=captions[start:start + batch_size],
                           return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        output.append(F.normalize(model.get_text_features(**tokens), dim=-1).cpu())
    return torch.cat(output)


@torch.inference_mode()
def retrieval(model, processor, dataset, text_features, device, batch_size=64):
    model.eval()
    image_features = []
    for start in range(0, len(dataset), batch_size):
        rows = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        pixels = processor(images=[x["image"].convert("RGB") for x in rows],
                           return_tensors="pt")["pixel_values"].to(device)
        image_features.append(F.normalize(
            model.get_image_features(pixel_values=pixels), dim=-1).cpu())
    images = torch.cat(image_features)
    similarity = images @ text_features.t()
    i2t_order = similarity.argsort(dim=1, descending=True)
    t2i_order = similarity.t().argsort(dim=1, descending=True)
    text_targets = torch.arange(len(dataset)).repeat_interleave(5)
    result = {}
    for k in (1, 5, 10):
        i2t = []
        for index in range(len(dataset)):
            valid = set(range(index * 5, index * 5 + 5))
            i2t.append(any(int(x) in valid for x in i2t_order[index, :k]))
        result[f"i2t_r{k}"] = 100.0 * sum(i2t) / len(i2t)
        result[f"t2i_r{k}"] = 100.0 * (
            t2i_order[:, :k] == text_targets[:, None]).any(1).float().mean().item()
    result["mean_recall"] = float(np.mean(list(result.values())))
    return result


@torch.inference_mode()
def proxy_scores(model, teacher, collaboration, processor, dataset, device,
                 swap_panels, pair_indices, batch_size=64):
    """Compute reproducible proxies on fixed panels, independently of test recall."""
    swap_counts = []
    for panel in swap_panels:
        rows = [dataset[i] for i in panel]
        pixels = processor(images=[x["image"].convert("RGB") for x in rows],
                           return_tensors="pt")["pixel_values"].to(device)
        captured = []
        handle = collaboration.register_forward_hook(
            lambda _m, _i, output: captured.append(output[0].detach().cpu()))
        model.get_image_features(pixel_values=pixels)
        handle.remove()
        bits = (captured[0] > 0).flatten(1).to(torch.uint8).numpy()
        swap_counts.append(int(np.unique(np.packbits(bits, axis=0), axis=1).shape[1]))

    image_embeddings, text_embeddings, teacher_embeddings = [], [], []
    for start in range(0, len(pair_indices), batch_size):
        indices = pair_indices[start:start + batch_size]
        rows = [dataset[i] for i in indices]
        inputs = processor(images=[x["image"].convert("RGB") for x in rows],
                           text=[x["caption_0"] for x in rows],
                           return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        teacher_images = teacher.get_image_features(pixel_values=inputs["pixel_values"])
        image_embeddings.append(F.normalize(outputs.image_embeds, dim=-1).cpu())
        text_embeddings.append(F.normalize(outputs.text_embeds, dim=-1).cpu())
        teacher_embeddings.append(F.normalize(teacher_images, dim=-1).cpu())
    images = torch.cat(image_embeddings)
    texts = torch.cat(text_embeddings)
    teacher_images = torch.cat(teacher_embeddings)
    logit_scale = model.logit_scale.exp().detach().cpu()
    logits = logit_scale * images @ texts.t()
    swap_sqrt = np.sqrt(np.asarray(swap_counts, dtype=float))
    return {
        "fused_CLAS": float(swap_sqrt.mean()),
        "fused_CLAS_std": float(swap_sqrt.std(ddof=1)),
        "fused_pattern_count_mean": float(np.mean(swap_counts)),
        "fused_pattern_counts": json.dumps(swap_counts),
        "contrastive_loss": float(symmetric_clip_loss(logits)),
        "alignment_drift": float((1.0 - F.cosine_similarity(
            images, teacher_images, dim=-1)).mean()),
    }


def train_short(model, teacher, collaboration, processor, dataset, device,
                steps, batch_size, lr, seed):
    trainable = [p for p in collaboration.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    model.train(); teacher.eval()
    step = 0
    while step < steps:
        for inputs in batches(dataset, batch_size, processor, device, True, seed + step):
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                teacher_images = teacher.get_image_features(pixel_values=inputs["pixel_values"])
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(**inputs)
                contrastive = symmetric_clip_loss(outputs.logits_per_image)
                alignment = (1.0 - F.cosine_similarity(
                    outputs.image_embeds, teacher_images, dim=-1)).mean()
                loss = contrastive + 0.1 * alignment
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            step += 1
            if step >= steps:
                break


def zscore(values):
    values = np.asarray(values, dtype=float)
    std = values.std()
    return (values - values.mean()) / (std if std > 0 else 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default="openai/clip-vit-base-patch16")
    parser.add_argument("--mae", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--blocks", type=int, nargs="+", default=[8, 9, 10, 11])
    parser.add_argument("--gates", type=float, nargs="+", default=[0.01, 0.025, 0.05])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--train-images", type=int, default=6000)
    parser.add_argument("--validation-images", type=int, default=1000)
    parser.add_argument("--swap-panel-count", type=int, default=3)
    parser.add_argument("--swap-panel-size", type=int, default=32)
    parser.add_argument("--proxy-pairs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda")
    processor = CLIPProcessor.from_pretrained(args.clip)
    model = CLIPModel.from_pretrained(args.clip).to(device)
    teacher = CLIPModel.from_pretrained(args.clip).to(device).eval()
    for network in (model, teacher):
        for parameter in network.parameters():
            parameter.requires_grad_(False)
    donors = {"mae": load_donor(args.mae), "dino": load_donor(args.dino)}
    data = load_dataset("jxie/flickr8k")
    train = data["train"].select(range(min(args.train_images, len(data["train"]))))
    validation = data["validation"].select(
        range(min(args.validation_images, len(data["validation"]))))
    rng = np.random.default_rng(args.seed)
    proxy_order = rng.permutation(len(validation)).tolist()
    swap_total = args.swap_panel_count * args.swap_panel_size
    if swap_total > len(validation) or args.proxy_pairs > len(validation):
        raise ValueError("proxy panels exceed the validation set")
    swap_panels = [proxy_order[i * args.swap_panel_size:(i + 1) * args.swap_panel_size]
                   for i in range(args.swap_panel_count)]
    pair_indices = proxy_order[:args.proxy_pairs]
    text_features = encode_texts(teacher, processor, validation, device)
    base_layers = {i: copy.deepcopy(model.vision_model.encoder.layers[i])
                   for i in args.blocks}

    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["candidate", "block", "donor", "gate", "fused_CLAS",
              "fused_CLAS_std", "fused_pattern_count_mean",
              "fused_pattern_counts", "contrastive_loss", "alignment_drift",
              "i2t_r1", "t2i_r1", "i2t_r5", "t2i_r5", "i2t_r10",
              "t2i_r10", "mean_recall", "elapsed_sec", "status"]
    rows = []
    with open(output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for block in args.blocks:
            for donor_name, donor in donors.items():
                for gate in args.gates:
                    started = time.time()
                    torch.manual_seed(args.seed)
                    collaboration = CLIPCollaborativeLayer(
                        copy.deepcopy(base_layers[block]),
                        copy.deepcopy(donor.blocks[block]), gate=gate).to(device)
                    model.vision_model.encoder.layers[block] = collaboration
                    proxies = proxy_scores(
                        model, teacher, collaboration, processor, validation, device,
                        swap_panels, pair_indices)
                    train_short(model, teacher, collaboration, processor, train,
                                device, args.steps, args.batch_size, args.lr, args.seed)
                    metrics = retrieval(
                        model, processor, validation, text_features, device)
                    row = {
                        "candidate": f"b{block}_{donor_name}_g{gate:g}",
                        "block": block, "donor": donor_name, "gate": gate,
                        **proxies, **metrics, "elapsed_sec": time.time() - started,
                        "status": "ok"}
                    writer.writerow(row); handle.flush(); rows.append(row)
                    print("RESULT", json.dumps(row), flush=True)
                    model.vision_model.encoder.layers[block] = copy.deepcopy(
                        base_layers[block]).to(device)
                    del collaboration; torch.cuda.empty_cache()

    target = np.asarray([x["mean_recall"] for x in rows])
    swap = np.asarray([x["fused_CLAS"] for x in rows])
    loss = np.asarray([x["contrastive_loss"] for x in rows])
    drift = np.asarray([x["alignment_drift"] for x in rows])
    composite = zscore(swap) - zscore(loss) - zscore(drift)
    summary = {
        "candidate_count": len(rows),
        "seed": args.seed,
        "validation_images": len(validation),
        "swap_panel_indices": swap_panels,
        "proxy_pair_indices": pair_indices,
        "spearman_CLAS": float(spearmanr(swap, target).statistic),
        "spearman_negative_contrastive": float(spearmanr(-loss, target).statistic),
        "spearman_negative_alignment": float(spearmanr(-drift, target).statistic),
        "spearman_equal_weight_composite": float(spearmanr(composite, target).statistic),
        "best_candidate": rows[int(target.argmax())]["candidate"],
        "best_mean_recall": float(target.max()),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
