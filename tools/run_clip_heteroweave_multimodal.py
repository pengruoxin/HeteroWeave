#!/usr/bin/env python3
"""Minimal multimodal HeteroWeave transfer on CLIP and Flickr8K."""

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
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor

import timm


def unwrap_state_dict(obj):
    if isinstance(obj, dict) and "model" in obj:
        obj = obj["model"]
    return {(k[7:] if k.startswith("module.") else k): v
            for k, v in obj.items()}


class CLIPCollaborativeLayer(nn.Module):
    """Fuse residual updates from a CLIP layer and a frozen external ViT block."""

    def __init__(self, clip_layer, donor_block, dim=768, gate=0.025):
        super().__init__()
        self.clip_layer = clip_layer
        self.donor_block = donor_block
        for module in (clip_layer, donor_block):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.pre_scale = nn.Parameter(torch.ones(dim))
        self.pre_bias = nn.Parameter(torch.zeros(dim))
        self.out_scale = nn.Parameter(torch.ones(dim))
        self.out_bias = nn.Parameter(torch.zeros(dim))
        value = math.log(gate / (1.0 - gate))
        self.register_buffer("gate_logit", torch.tensor(value))

    @property
    def gate(self):
        return torch.sigmoid(self.gate_logit)

    def forward(self, hidden_states, attention_mask, causal_attention_mask,
                output_attentions=False):
        clip_outputs = self.clip_layer(
            hidden_states, attention_mask, causal_attention_mask,
            output_attentions=output_attentions)
        donor_input = hidden_states * self.pre_scale + self.pre_bias
        donor_update = self.donor_block(donor_input) - donor_input
        fused = clip_outputs[0] + self.gate * (
            donor_update * self.out_scale + self.out_bias)
        return (fused,) + clip_outputs[1:]


def symmetric_clip_loss(logits_per_image):
    target = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
    return 0.5 * (
        F.cross_entropy(logits_per_image, target) +
        F.cross_entropy(logits_per_image.t(), target))


def batches(dataset, batch_size, processor, device, shuffle, seed):
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        rows = [dataset[i] for i in indices[start:start + batch_size]]
        images = [row["image"].convert("RGB") for row in rows]
        texts = [row["caption_0"] for row in rows]
        inputs = processor(images=images, text=texts, return_tensors="pt",
                           padding=True, truncation=True)
        yield {k: v.to(device) for k, v in inputs.items()}


@torch.inference_mode()
def retrieval_metrics(model, processor, dataset, device, image_batch=32,
                      text_batch=128):
    model.eval()
    image_features = []
    all_captions = []
    for start in range(0, len(dataset), image_batch):
        rows = [dataset[i] for i in range(start, min(start + image_batch, len(dataset)))]
        images = [row["image"].convert("RGB") for row in rows]
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        features = model.get_image_features(pixel_values=pixels)
        image_features.append(F.normalize(features, dim=-1).cpu())
        for row in rows:
            all_captions.extend(row[f"caption_{j}"] for j in range(5))
    text_features = []
    for start in range(0, len(all_captions), text_batch):
        tokens = processor(text=all_captions[start:start + text_batch],
                           return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        features = model.get_text_features(**tokens)
        text_features.append(F.normalize(features, dim=-1).cpu())
    images = torch.cat(image_features)
    texts = torch.cat(text_features)
    similarity = images @ texts.t()
    image_order = similarity.argsort(dim=1, descending=True)
    text_order = similarity.t().argsort(dim=1, descending=True)
    image_targets = torch.arange(len(dataset))
    text_targets = torch.arange(len(dataset)).repeat_interleave(5)

    result = {}
    for k in (1, 5, 10):
        correct_i2t = []
        for image_index in range(len(dataset)):
            valid = set(range(5 * image_index, 5 * image_index + 5))
            correct_i2t.append(any(int(x) in valid for x in image_order[image_index, :k]))
        result[f"i2t_r{k}"] = 100.0 * sum(correct_i2t) / len(correct_i2t)
        result[f"t2i_r{k}"] = 100.0 * (
            text_order[:, :k] == text_targets[:, None]).any(dim=1).float().mean().item()
    result["mean_recall"] = float(np.mean(list(result.values())))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default="openai/clip-vit-base-patch16")
    parser.add_argument("--donor", required=True)
    parser.add_argument("--block", type=int, default=10)
    parser.add_argument("--gate", type=float, default=0.025)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--distill-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-images", type=int, default=6000)
    parser.add_argument("--test-images", type=int, default=1000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    processor = CLIPProcessor.from_pretrained(args.clip)
    model = CLIPModel.from_pretrained(args.clip).to(device)
    teacher = CLIPModel.from_pretrained(args.clip).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    donor = timm.create_model("vit_base_patch16_224", pretrained=False)
    donor.load_state_dict(unwrap_state_dict(torch.load(
        args.donor, map_location="cpu")), strict=False)
    donor.eval()
    original = model.vision_model.encoder.layers[args.block]
    collaboration = CLIPCollaborativeLayer(
        original, copy.deepcopy(donor.blocks[args.block]), gate=args.gate)
    model.vision_model.encoder.layers[args.block] = collaboration
    model.to(device)
    trainable = [p for p in collaboration.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    data = load_dataset("jxie/flickr8k")
    train_data = data["train"].select(range(min(args.train_images, len(data["train"]))))
    test_data = data["test"].select(range(min(args.test_images, len(data["test"]))))
    before = retrieval_metrics(model, processor, test_data, device)
    baseline = retrieval_metrics(teacher, processor, test_data, device)
    print("BASELINE", json.dumps(baseline), flush=True)
    print("BEFORE", json.dumps(before), flush=True)

    model.train()
    iterator_seed = args.seed
    step = 0
    history = []
    started = time.time()
    while step < args.steps:
        for inputs in batches(train_data, args.batch_size, processor,
                              device, shuffle=True, seed=iterator_seed):
            iterator_seed += 1
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                teacher_images = teacher.get_image_features(
                    pixel_values=inputs["pixel_values"])
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(**inputs)
                contrastive = symmetric_clip_loss(outputs.logits_per_image)
                alignment = (1.0 - F.cosine_similarity(
                    outputs.image_embeds, teacher_images, dim=-1)).mean()
                loss = contrastive + args.distill_weight * alignment
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step in {1, 50, 100, args.steps}:
                item = {"step": step, "loss": float(loss.detach()),
                        "contrastive": float(contrastive.detach()),
                        "alignment": float(alignment.detach())}
                history.append(item)
                print("TRAIN", json.dumps(item), flush=True)
            if step >= args.steps:
                break

    after = retrieval_metrics(model, processor, test_data, device)
    result = {
        "block": args.block, "gate": args.gate, "steps": args.steps,
        "seed": args.seed, "trainable_parameters": sum(p.numel() for p in trainable),
        "elapsed_sec": time.time() - started,
        **{f"baseline_{k}": v for k, v in baseline.items()},
        **{f"before_{k}": v for k, v in before.items()},
        **{f"after_{k}": v for k, v in after.items()},
        "history": json.dumps(history),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result.keys())
        writer.writeheader(); writer.writerow(result)
    torch.save({
        "block": args.block, "gate": args.gate,
        "adapter": {k: v.cpu() for k, v in collaboration.state_dict().items()
                    if k.startswith(("pre_", "out_"))}}, args.checkpoint)
    print("RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
