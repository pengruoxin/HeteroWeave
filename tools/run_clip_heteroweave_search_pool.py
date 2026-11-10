#!/usr/bin/env python3
"""Evaluate a frozen multimodal HeteroWeave candidate manifest on Flickr8K val."""

import argparse
import copy
import csv
import hashlib
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

from run_clip_heteroweave_multimodal import batches, symmetric_clip_loss
from run_clip_heteroweave_proxy_audit import load_donor, encode_texts, retrieval


class DonorAdapter(nn.Module):
    def __init__(self, donor_block, dim, adapter, width, adapter_only=False):
        super().__init__()
        self.donor_block = donor_block
        for p in self.donor_block.parameters():
            p.requires_grad_(False)
        self.adapter = adapter
        self.adapter_only = adapter_only
        if adapter == "affine":
            self.pre_scale = nn.Parameter(torch.ones(dim))
            self.pre_bias = nn.Parameter(torch.zeros(dim))
            self.out_scale = nn.Parameter(torch.ones(dim))
            self.out_bias = nn.Parameter(torch.zeros(dim))
        elif adapter == "bottleneck":
            self.down = nn.Linear(dim, width)
            self.up = nn.Linear(width, dim)
            nn.init.trunc_normal_(self.down.weight, std=0.02)
            nn.init.zeros_(self.down.bias)
            nn.init.trunc_normal_(self.up.weight, std=0.02)
            nn.init.zeros_(self.up.bias)
        else:
            raise ValueError(adapter)

    def forward(self, hidden_states):
        if self.adapter == "affine":
            donor_input = hidden_states * self.pre_scale + self.pre_bias
            update = self.donor_block(donor_input) - donor_input
            return update * self.out_scale + self.out_bias
        update = (hidden_states if self.adapter_only else
                  self.donor_block(hidden_states) - hidden_states)
        return self.up(F.gelu(self.down(update)))


class MultiDonorCollaborativeLayer(nn.Module):
    def __init__(self, clip_layer, donor_blocks, dim, gate, adapter, width, fusion,
                 adapter_only_flags=None):
        super().__init__()
        self.clip_layer = clip_layer
        for p in self.clip_layer.parameters():
            p.requires_grad_(False)
        flags = adapter_only_flags or [False] * len(donor_blocks)
        self.branches = nn.ModuleList([
            DonorAdapter(block, dim, adapter, width, flag)
            for block, flag in zip(donor_blocks, flags)])
        self.register_buffer("gate_value", torch.tensor(float(gate)))
        self.fusion = fusion

    def forward(self, hidden_states, attention_mask, causal_attention_mask,
                output_attentions=False):
        clip_outputs = self.clip_layer(
            hidden_states, attention_mask, causal_attention_mask,
            output_attentions=output_attentions)
        updates = torch.stack([branch(hidden_states) for branch in self.branches]).sum(0)
        if self.fusion == "normalized":
            updates = updates / math.sqrt(len(self.branches))
        return (clip_outputs[0] + self.gate_value * updates,) + clip_outputs[1:]


def train_short(model, teacher, module, processor, dataset, device,
                steps, batch_size, lr, seed):
    trainable = [p for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    model.eval(); teacher.eval()
    step = 0
    while step < steps:
        for inputs in batches(dataset, batch_size, processor, device, True, seed + step):
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                target = F.normalize(teacher.get_image_features(
                    pixel_values=inputs["pixel_values"]), dim=-1)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(**inputs)
                contrastive = symmetric_clip_loss(outputs.logits_per_image)
                alignment = (1.0 - F.cosine_similarity(
                    F.normalize(outputs.image_embeds, dim=-1), target, dim=-1)).mean()
                loss = contrastive + 0.1 * alignment
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            step += 1
            if step >= steps:
                break


@torch.inference_mode()
def proxies(model, teacher, module, processor, dataset, device,
            swap_panels, pair_indices, batch_size=64):
    counts = []
    for panel in swap_panels:
        rows = [dataset[i] for i in panel]
        pixels = processor(images=[r["image"].convert("RGB") for r in rows],
                           return_tensors="pt")["pixel_values"].to(device)
        captured = []
        hook = module.register_forward_hook(
            lambda _m, _i, o: captured.append(o[0].detach().cpu()))
        model.get_image_features(pixel_values=pixels)
        hook.remove()
        bits = (captured[0] > 0).flatten(1).to(torch.uint8).numpy()
        counts.append(int(np.unique(np.packbits(bits, axis=0), axis=1).shape[1]))
    image_embeds, text_embeds, teacher_embeds = [], [], []
    for start in range(0, len(pair_indices), batch_size):
        ids = pair_indices[start:start + batch_size]
        rows = [dataset[i] for i in ids]
        inputs = processor(images=[r["image"].convert("RGB") for r in rows],
                           text=[r["caption_0"] for r in rows],
                           return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = model(**inputs)
        target = teacher.get_image_features(pixel_values=inputs["pixel_values"])
        image_embeds.append(F.normalize(output.image_embeds, dim=-1).cpu())
        text_embeds.append(F.normalize(output.text_embeds, dim=-1).cpu())
        teacher_embeds.append(F.normalize(target, dim=-1).cpu())
    images, texts, targets = map(torch.cat, (image_embeds, text_embeds, teacher_embeds))
    logits = model.logit_scale.exp().detach().cpu() * images @ texts.t()
    sqrt_counts = np.sqrt(np.asarray(counts, dtype=float))
    return {
        "CLAS": float(sqrt_counts.mean()),
        "CLAS_panel_std": float(sqrt_counts.std(ddof=1)),
        "pattern_counts": json.dumps(counts),
        "contrastive_loss": float(symmetric_clip_loss(logits)),
        "alignment_drift": float((1.0 - F.cosine_similarity(images, targets)).mean()),
    }


def cost(module, token_count=197, dim=768):
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    added = sum(p.numel() for b in module.branches for p in b.parameters())
    flops = 0
    for branch in module.branches:
        # ViT attention + MLP multiply-accumulate estimate for one donor block.
        if not branch.adapter_only:
            flops += 12 * token_count * dim * dim + 2 * token_count * token_count * dim
        if branch.adapter == "affine":
            flops += 4 * token_count * dim
        else:
            flops += 2 * token_count * dim * branch.down.out_features
    return trainable, added, flops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mae", required=True)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--train-images", type=int, default=6000)
    parser.add_argument("--validation-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only-tags", nargs="*", default=None)
    parser.add_argument("--only-ids", nargs="*", default=None)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    manifest = json.loads(Path(args.manifest).read_text())
    candidates = manifest["candidates"]
    if args.only_tags:
        candidates = [c for c in candidates if c["tag"] in args.only_tags]
    if args.only_ids:
        ids = set(args.only_ids); candidates = [c for c in candidates if c["id"] in ids]
    if args.limit is not None:
        candidates = candidates[:args.limit]

    device = torch.device("cuda")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device).eval()
    teacher = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device).eval()
    for network in (model, teacher):
        for p in network.parameters(): p.requires_grad_(False)
    random_donor = timm.create_model("vit_base_patch16_224", pretrained=False)
    random_donor.eval()
    for p in random_donor.parameters(): p.requires_grad_(False)
    donors = {"mae": load_donor(args.mae), "dino": load_donor(args.dino),
              "random": random_donor, "adapter_only": random_donor}
    data = load_dataset("jxie/flickr8k")
    train = data["train"].select(range(min(args.train_images, len(data["train"]))))
    validation = data["validation"].select(range(min(
        args.validation_images, len(data["validation"]))))
    order = np.random.default_rng(0).permutation(len(validation)).tolist()
    swap_panels = [order[i * 32:(i + 1) * 32] for i in range(3)]
    pair_indices = order[:256]
    text_features = encode_texts(teacher, processor, validation, device)
    baseline = retrieval(teacher, processor, validation, text_features, device)

    fields = ["candidate", "tag", "block", "donors", "gate", "adapter", "width",
              "fusion", "trainable_params", "added_params", "added_flops",
              "CLAS", "CLAS_panel_std", "pattern_counts",
              "contrastive_loss", "alignment_drift", "i2t_r1", "t2i_r1",
              "i2t_r5", "t2i_r5", "i2t_r10", "t2i_r10", "mean_recall",
              "elapsed_sec", "seed", "manifest_sha256", "status"]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for c in candidates:
            started = time.time()
            # Make initialization a function of (training seed, candidate id),
            # rather than the candidate's position in a manifest.
            digest = int(hashlib.sha256(c["id"].encode()).hexdigest()[:8], 16)
            candidate_seed = (args.seed * 1_000_003 + digest) % (2**31 - 1)
            random.seed(candidate_seed)
            np.random.seed(candidate_seed)
            torch.manual_seed(candidate_seed)
            if not c["donors"]:
                row = {"candidate": c["id"], "tag": c["tag"], "block": "",
                       "donors": "", "gate": 0, "adapter": "none", "width": 0,
                       "fusion": "none", "trainable_params": 0, "added_params": 0,
                       "added_flops": 0, "CLAS": "",
                       "CLAS_panel_std": "", "pattern_counts": "",
                       "contrastive_loss": "", "alignment_drift": "", **baseline,
                       "elapsed_sec": time.time() - started, "seed": args.seed,
                       "manifest_sha256": manifest["manifest_sha256"], "status": "ok"}
            else:
                block = c["block"]
                original = copy.deepcopy(model.vision_model.encoder.layers[block])
                donor_blocks = [
                    nn.Identity() if d == "adapter_only" else
                    copy.deepcopy(donors[d].blocks[block]) for d in c["donors"]]
                module = MultiDonorCollaborativeLayer(
                    original, donor_blocks,
                    768, c["gate"], c["adapter"], c["width"], c["fusion"],
                    [d == "adapter_only" for d in c["donors"]]).to(device)
                model.vision_model.encoder.layers[block] = module
                proxy = proxies(model, teacher, module, processor, validation, device,
                                swap_panels, pair_indices)
                train_short(model, teacher, module, processor, train, device, args.steps,
                            args.batch_size, args.lr, args.seed)
                metrics = retrieval(model, processor, validation, text_features, device)
                trainable, added, added_flops = cost(module)
                row = {"candidate": c["id"], "tag": c["tag"], "block": block,
                       "donors": "+".join(c["donors"]), "gate": c["gate"],
                       "adapter": c["adapter"], "width": c["width"],
                       "fusion": c["fusion"], "trainable_params": trainable,
                       "added_params": added, "added_flops": added_flops,
                       **proxy, **metrics, "elapsed_sec": time.time() - started,
                       "seed": args.seed, "manifest_sha256": manifest["manifest_sha256"],
                       "status": "ok"}
                model.vision_model.encoder.layers[block] = original.to(device)
                del module; torch.cuda.empty_cache()
            writer.writerow(row); f.flush()
            print("RESULT", json.dumps(row), flush=True)
    print("COMPLETE", json.dumps({"candidate_count": len(candidates),
                                  "baseline": baseline}), flush=True)


if __name__ == "__main__":
    main()
