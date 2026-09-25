#!/usr/bin/env python3
"""Build and score a segmentation HeteroWeave candidate pool without test data."""

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CHANNELS = {"layer2": 512, "layer3": 1024, "layer4": 2048}


def load_state(path):
    value = torch.load(path, map_location="cpu")
    if isinstance(value, dict) and "state_dict" in value:
        value = value["state_dict"]
    result = {}
    for key, tensor in value.items():
        for prefix in ("module.", "encoder_q.", "backbone.", "trunk."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        if key.startswith("fc.") or key.startswith("projection_head."):
            continue
        result[key] = tensor
    if "conv1.weight" not in result:
        raise RuntimeError(f"Checkpoint {path} did not expose a ResNet conv1.weight")
    return result


def complete_names(root):
    root = Path(root)
    listed = [line.split()[0] for line in
              (root / "annotations" / "trainval.txt").read_text().splitlines()]
    return [name for name in listed
            if (root / "images" / f"{name}.jpg").is_file()
            and (root / "annotations" / "trimaps" / f"{name}.png").is_file()]


class ImagePanel(Dataset):
    def __init__(self, root, names, size):
        self.root, self.names, self.size = Path(root), list(names), size

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        image = Image.open(self.root / "images" / f"{self.names[index]}.jpg").convert("RGB")
        image = TF.resize(image, [self.size, self.size], InterpolationMode.BILINEAR)
        return TF.normalize(TF.to_tensor(image), MEAN, STD)


class BottleneckAdapter(nn.Module):
    def __init__(self, channels, bottleneck, seed):
        super().__init__()
        state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        self.down = nn.Conv2d(channels, bottleneck, 1)
        self.up = nn.Conv2d(bottleneck, channels, 1)
        nn.init.normal_(self.up.weight, std=1e-3)
        nn.init.zeros_(self.up.bias)
        torch.random.set_rng_state(state)

    def forward(self, inputs):
        return self.up(torch.nn.functional.gelu(self.down(inputs)))


class MultiBlockStage(nn.Module):
    """A reference block and a variable-size set of same-position blocks."""

    def __init__(self, reference, collaborators, channels, bottleneck, gate,
                 fusion, seed):
        super().__init__()
        self.reference = reference
        self.collaborators = nn.ModuleList(collaborators)
        self.adapters = nn.ModuleList([
            BottleneckAdapter(channels, bottleneck, seed + index)
            for index in range(len(collaborators))])
        self.register_buffer("gate", torch.tensor(float(gate)))
        self.fusion = fusion
        if fusion == "normalized_weighted_sum":
            self.logits = nn.Parameter(torch.zeros(len(collaborators)))
        for block in [reference, *collaborators]:
            for parameter in block.parameters():
                parameter.requires_grad_(False)

    def forward(self, inputs):
        reference = self.reference(inputs)
        if not self.collaborators:
            return reference
        contributions = [adapter(block(inputs))
                         for block, adapter in zip(self.collaborators, self.adapters)]
        if self.fusion == "normalized_weighted_sum":
            weights = self.logits.softmax(0)
            combined = sum(weight * value for weight, value in zip(weights, contributions))
        else:
            combined = sum(contributions)
        return reference + self.gate * combined


def build_model(anchor_state, component_states, candidate, model_seed):
    torch.manual_seed(model_seed)
    model = deeplabv3_resnet50(
        weights=None, weights_backbone=None, num_classes=3, aux_loss=False)
    incompatible = model.backbone.load_state_dict(anchor_state, strict=False)
    if "conv1.weight" in incompatible.missing_keys:
        raise RuntimeError("Anchor checkpoint did not load into the ResNet backbone")
    position = candidate.get("position")
    if position:
        blocks = []
        for component in candidate["components"]:
            donor = resnet50(weights=None, replace_stride_with_dilation=[False, True, True])
            if component_states.get(component) is not None:
                incompatible = donor.load_state_dict(component_states[component], strict=False)
                if "conv1.weight" in incompatible.missing_keys:
                    raise RuntimeError(f"Component {component} did not load into ResNet-50")
            blocks.append(copy.deepcopy(getattr(donor, position)))
        stage = MultiBlockStage(
            copy.deepcopy(getattr(model.backbone, position)), blocks,
            CHANNELS[position], candidate["adapter_bottleneck"], candidate["gate"],
            candidate["fusion"], model_seed + 1000)
        setattr(model.backbone, position, stage)
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    if position:
        for parameter in stage.adapters.parameters():
            parameter.requires_grad_(True)
        if hasattr(stage, "logits"):
            stage.logits.requires_grad_(True)
    return model


ACTIVATION_TYPES = (
    nn.ReLU, nn.LeakyReLU, nn.GELU, nn.PReLU, nn.Hardswish, nn.ELU,
    nn.SELU, nn.Mish, nn.SiLU)


class PositionLayerSwapCollector:
    """CLAS over fused outputs at the registered search positions.

    Each hook observes the output *after* all same-position collaborators have
    been fused.  Pattern counts are computed independently per position, then
    square-rooted before summation.  Internal block activations are deliberately
    excluded: summing their individual SWAP scores would not measure the
    candidate's position-level combination output.
    """

    def __init__(self, model, positions=("layer2", "layer3", "layer4")):
        self.patterns = {name: [] for name in positions}
        self.handles = []
        for name in positions:
            module = getattr(model.backbone, name)
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def capture(_module, _inputs, output):
            if not isinstance(output, torch.Tensor) or output.ndim < 2:
                return
            output = output.detach()
            # Each image is one observation. Each channel-spatial coordinate
            # has one binary response code across the fixed image panel.
            binary = (output > 0).reshape(output.shape[0], -1)
            self.patterns[name].append(binary.cpu().numpy())
        return capture

    def scores(self):
        result = {}
        for name, chunks in self.patterns.items():
            if not chunks:
                continue
            for call_index, observations in enumerate(chunks):
                if observations.shape[0] <= 32:
                    powers = np.left_shift(
                        np.uint32(1), np.arange(observations.shape[0], dtype=np.uint32))
                    codes = observations.T.astype(np.uint32, copy=False) @ powers
                    unique = int(np.unique(codes).size)
                else:
                    packed = np.packbits(observations.T, axis=1)
                    unique = int(np.unique(packed, axis=0).shape[0])
                if len(chunks) != 1:
                    raise RuntimeError(
                        f"Search position {name} executed {len(chunks)} times; expected once")
                result[name] = unique
        if set(result) != set(self.patterns):
            raise RuntimeError(
                f"Missing position outputs: {sorted(set(self.patterns) - set(result))}")
        result["position_sqrt"] = {
            name: math.sqrt(value) for name, value in result.items()}
        result["CLAS"] = float(sum(result["position_sqrt"].values()))
        result["selected_position_count"] = len(self.patterns)
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def dense_swap(model, loader, device):
    collector = PositionLayerSwapCollector(model)
    model.eval()
    for images in loader:
        model(images.to(device, non_blocking=True))
    scores = collector.scores()
    collector.close()
    return scores


@torch.inference_mode()
def count_compute(model, device, size):
    macs = 0
    handles = []

    def conv_hook(module, _inputs, output):
        nonlocal macs
        kernel = module.kernel_size[0] * module.kernel_size[1]
        per_output = kernel * module.in_channels // module.groups
        macs += output.numel() * per_output

    def linear_hook(module, _inputs, output):
        nonlocal macs
        macs += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    model.eval()(torch.zeros(1, 3, size, size, device=device))
    for handle in handles:
        handle.remove()
    return macs, 2 * macs


def make_protocol(data, output, seed, panel_count, panel_size):
    names = complete_names(data)
    if len(names) != 3680:
        raise RuntimeError(f"Expected 3680 complete trainval pairs, found {len(names)}")
    rng = random.Random(seed)
    rng.shuffle(names)
    validation_count = 800
    validation = names[:validation_count]
    training = names[validation_count:]
    needed = panel_count * panel_size
    if needed > len(training):
        raise RuntimeError("Proxy panels exceed the training split")
    panels = [training[i * panel_size:(i + 1) * panel_size]
              for i in range(panel_count)]
    protocol = dict(seed=seed, dataset="Oxford-IIIT Pets", train=training,
                    search_validation=validation, proxy_panels=panels,
                    test_split="official test.txt; prohibited during search")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


def make_manifest(components, output, seed, limit):
    component_names = sorted(components)
    candidates = [dict(candidate_id="baseline", position=None, components=[],
                       total_blocks=1, gate=0.0, adapter_bottleneck=0,
                       fusion="none")]
    for position in CHANNELS:
        for collaborator_count in (1, 2):
            for selected in itertools.combinations(component_names, collaborator_count):
                for gate in (0.005, 0.0125, 0.025):
                    for bottleneck in (8, 16):
                        for fusion in ("residual_sum", "normalized_weighted_sum"):
                            key = f"{position}|{'+'.join(selected)}|{gate}|{bottleneck}|{fusion}"
                            candidate_id = hashlib.sha1(key.encode()).hexdigest()[:12]
                            candidates.append(dict(
                                candidate_id=candidate_id, position=position,
                                components=list(selected), total_blocks=1 + collaborator_count,
                                gate=gate, adapter_bottleneck=bottleneck, fusion=fusion,
                                is_anchor=False))
    candidates[0]["is_anchor"] = True
    anchor_specs = [
        ("layer2", ["byol"], 0.0125, 16, "normalized_weighted_sum"),
        ("layer3", ["byol"], 0.0125, 16, "normalized_weighted_sum"),
        ("layer4", ["byol"], 0.025, 16, "residual_sum"),
    ]
    anchors = []
    for spec in anchor_specs:
        for candidate in candidates[1:]:
            observed = (candidate["position"], candidate["components"], candidate["gate"],
                        candidate["adapter_bottleneck"], candidate["fusion"])
            if observed == spec:
                candidate["is_anchor"] = True
                anchors.append(candidate)
                break
    rng = random.Random(seed)
    anchor_ids = {candidate["candidate_id"] for candidate in anchors}
    body = [candidate for candidate in candidates[1:]
            if candidate["candidate_id"] not in anchor_ids]
    rng.shuffle(body)
    if limit:
        body = body[:max(0, limit - 1 - len(anchors))]
    candidates = [candidates[0], *anchors, *body]
    payload = dict(seed=seed, frozen=True, components=components, candidates=candidates)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def score(args):
    protocol = json.loads(Path(args.protocol).read_text())
    manifest = json.loads(Path(args.manifest).read_text())
    anchor_state = load_state(args.anchor)
    component_states = {name: (load_state(path) if path is not None else None)
                        for name, path in manifest["components"].items()}
    loaders = [DataLoader(ImagePanel(args.data, names, args.size),
                          batch_size=len(names), shuffle=False, num_workers=0)
               for names in protocol["proxy_panels"]]
    candidates = manifest["candidates"]
    if args.candidate_csv:
        with open(args.candidate_csv, newline="") as handle:
            requested = {row["candidate_id"] for row in csv.DictReader(handle)}
        candidates = [candidate for candidate in candidates
                      if candidate["candidate_id"] in requested]
        if len(candidates) != len(requested):
            raise RuntimeError(
                f"Resolved {len(candidates)} of {len(requested)} requested candidates")
    if args.shard_count > 1:
        candidates = [candidate for index, candidate in enumerate(candidates)
                      if index % args.shard_count == args.shard_index]
    rows, device = [], torch.device("cuda")
    for candidate in candidates:
        model = build_model(anchor_state, component_states, candidate, args.model_seed).to(device)
        panel_scores = [dense_swap(model, loader, device) for loader in loaders]
        macs, flops = count_compute(model, device, args.size)
        values = [item["CLAS"] for item in panel_scores]
        row = dict(candidate)
        row.update(
            components=json.dumps(candidate["components"]),
            panel_scores=json.dumps(values),
            CLAS_mean=float(np.mean(values)),
            CLAS_panel_std=float(np.std(values)),
            params=sum(parameter.numel() for parameter in model.parameters()),
            trainable_params=sum(parameter.numel() for parameter in model.parameters()
                                 if parameter.requires_grad),
            macs=macs, flops=flops,
            layer_scores=json.dumps(panel_scores))
        rows.append(row)
        print("SCORE", json.dumps(row), flush=True)
        del model
        torch.cuda.empty_cache()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    protocol = subparsers.add_parser("protocol")
    protocol.add_argument("--data", required=True)
    protocol.add_argument("--output", required=True)
    protocol.add_argument("--seed", type=int, default=20260915)
    protocol.add_argument("--panel-count", type=int, default=3)
    protocol.add_argument("--panel-size", type=int, default=32)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--components-json", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--seed", type=int, default=20260915)
    manifest.add_argument("--limit", type=int, default=0)
    scorer = subparsers.add_parser("score")
    scorer.add_argument("--data", required=True)
    scorer.add_argument("--anchor", required=True)
    scorer.add_argument("--protocol", required=True)
    scorer.add_argument("--manifest", required=True)
    scorer.add_argument("--output", required=True)
    scorer.add_argument("--candidate-csv", default=None)
    scorer.add_argument("--shard-count", type=int, default=1)
    scorer.add_argument("--shard-index", type=int, default=0)
    scorer.add_argument("--size", type=int, default=192)
    scorer.add_argument("--batch-size", type=int, default=8)
    scorer.add_argument("--model-seed", type=int, default=11)
    args = parser.parse_args()
    if args.command == "protocol":
        make_protocol(args.data, args.output, args.seed, args.panel_count, args.panel_size)
    elif args.command == "manifest":
        make_manifest(json.loads(args.components_json), args.output, args.seed, args.limit)
    else:
        score(args)


if __name__ == "__main__":
    main()
