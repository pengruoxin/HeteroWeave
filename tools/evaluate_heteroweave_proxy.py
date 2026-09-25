#!/usr/bin/env python3
"""Evaluate zero-training signals aimed at future trained performance.

No optimizer step is taken.  The script combines three independent signals:
block-normalized ZiCo, cross-batch gradient alignment, and a small empirical
NTK kernel-regression forecast of held-out target margins.
"""

import argparse
import copy
import csv
import glob
import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import Config, DictAction
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier

from mmcls_addon import *  # noqa: F401,F403
from simlarity.zero_nas.heteroweave_composite import block_expressivity
from simlarity.zero_nas.zico import (
    calculate_block_zico, calculate_gradient_alignment, collect_zico_grad,
    logical_gradient_group)


class LogitWrapper(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, images):
        return forward_logits(self.classifier, images)


def forward_logits(model, images):
    features = model.extract_feat(images)
    if isinstance(features, tuple):
        features = features[-1]
    return model.head.fc(features)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configs', nargs='*')
    parser.add_argument('--input-csv', action='append', default=[])
    parser.add_argument('--config-column', default='config_path')
    parser.add_argument('--candidate-dir')
    parser.add_argument('--pattern', default='pareto_*.py')
    parser.add_argument(
        '--base-config',
        default='configs/imagenet/heteroweave_main_100e.py')
    parser.add_argument(
        '--data-config', default='configs/_base_/datasets/imagenet_bs64.py')
    parser.add_argument('--data-prefix', default=None)
    parser.add_argument('--ann-file', default=None)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gradient-batches', type=int, default=4)
    parser.add_argument('--block-dispersion-penalty', type=float, default=0.25)
    parser.add_argument('--ntk-train-samples', type=int, default=4)
    parser.add_argument('--ntk-holdout-samples', type=int, default=4)
    parser.add_argument('--ntk-sketch-per-tensor', type=int, default=16)
    parser.add_argument('--ntk-ridge', type=float, default=0.1)
    parser.add_argument('--ntk-target-margin', type=float, default=2.0)
    parser.add_argument('--expressivity-max-vectors', type=int, default=256)
    parser.add_argument('--skip-expressivity', action='store_true')
    parser.add_argument('--skip-ntk-future', action='store_true')
    parser.add_argument('--skip-gradients', action='store_true')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_float(value, default=float('nan')):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def collect_candidates(args):
    candidates = {}

    def add(path, metadata=None):
        normalized = os.path.normpath(path)
        if not os.path.isfile(normalized):
            raise FileNotFoundError(normalized)
        if normalized not in candidates:
            candidates[normalized] = dict(metadata or {})

    for path in args.configs:
        add(path)
    if args.candidate_dir:
        for path in glob.glob(os.path.join(args.candidate_dir, args.pattern)):
            add(path)
    for csv_path in args.input_csv:
        with open(csv_path, newline='', encoding='utf-8') as file:
            for row in csv.DictReader(file):
                path = row.get(args.config_column)
                if path:
                    add(path, row)
    if not candidates:
        raise RuntimeError('No candidate configs found.')
    return list(candidates.items())


def is_model_only_config(cfg):
    return 'model' in cfg and ('data' not in cfg or 'optimizer' not in cfg)


def load_candidate_config(path, args):
    cfg = Config.fromfile(path)
    if is_model_only_config(cfg):
        base = Config.fromfile(args.base_config)
        full = Config(copy.deepcopy(base._cfg_dict))
        full.model = copy.deepcopy(cfg.model)
        if 'train_cfg' not in full.model and 'train_cfg' in base.model:
            full.model.train_cfg = copy.deepcopy(base.model.train_cfg)
        cfg = full
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    return cfg


def data_loader(args):
    cfg = Config.fromfile(args.data_config)
    if args.data_prefix is not None:
        cfg.data.train.data_prefix = args.data_prefix
    if args.ann_file is not None:
        cfg.data.train.ann_file = args.ann_file
    if getattr(args, 'drop_albu', False):
        cfg.data.train.pipeline = [
            transform for transform in cfg.data.train.pipeline
            if transform.get('type') != 'Albu']
    cfg.data.samples_per_gpu = args.batch_size
    cfg.data.workers_per_gpu = args.workers
    dataset = build_dataset(cfg.data.train)
    return build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=True,
        round_up=True,
        # Proxy evaluation immediately copies only a few cached batches.
        # Pinning them needlessly initializes CUDA in the loader thread and
        # can surface unrelated GPU/ECC failures before model evaluation.
        pin_memory=False,
        persistent_workers=cfg.data.workers_per_gpu > 0)


def cache_batches(loader, count):
    batches = []
    for index, data in enumerate(loader):
        if index >= count:
            break
        images = data['img'].detach().cpu()
        labels = data['gt_label'].detach().view(-1).long().cpu()
        batches.append((images, labels))
    if len(batches) < count:
        raise RuntimeError(f'Needed {count} batches, found {len(batches)}.')
    return batches


def collect_batch_gradients(wrapper, batches):
    """Collect full Conv/Linear gradients without changing parameters."""
    was_training = wrapper.training
    wrapper.eval()
    criterion = nn.CrossEntropyLoss()
    gradients = {}
    losses = []
    try:
        for step, (images, labels) in enumerate(batches):
            wrapper.zero_grad()
            logits = wrapper(images.cuda(non_blocking=True))
            loss = criterion(logits, labels.cuda(non_blocking=True))
            loss.backward()
            losses.append(float(loss.detach().item()))
            gradients = collect_zico_grad(wrapper, gradients, step)
    finally:
        wrapper.zero_grad()
        wrapper.train(was_training)
    return gradients, losses


def target_margin(logits, labels):
    row = torch.arange(logits.shape[0], device=logits.device)
    true_logits = logits[row, labels]
    masked = logits.clone()
    masked[row, labels] = -torch.inf
    return true_logits - torch.logsumexp(masked, dim=1)


def sampled_gradient_sketch(model, per_tensor, index_cache):
    """Cheap deterministic gradient sketch with equal logical-block weight."""
    grouped = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        gradient = module.weight.grad
        if gradient is None:
            continue
        flat = gradient.detach().reshape(-1).float()
        count = min(max(1, int(per_tensor)), flat.numel())
        key = (module_name, flat.numel(), count, flat.device)
        indices = index_cache.get(key)
        if indices is None:
            indices = torch.linspace(
                0, flat.numel() - 1, steps=count,
                device=flat.device).long()
            index_cache[key] = indices
        # Scale keeps the sampled tensor norm on the same order as its full norm.
        sampled = flat.index_select(0, indices) * math.sqrt(flat.numel() / count)
        grouped.setdefault(logical_gradient_group(module_name), []).append(sampled)

    result = {}
    for group_name, parts in grouped.items():
        vector = torch.cat(parts)
        norm = torch.linalg.vector_norm(vector)
        if bool(torch.isfinite(norm).item()) and float(norm.item()) > 0:
            vector = vector / norm
        result[group_name] = vector.detach().cpu()
    if not result:
        raise RuntimeError('No gradients available for NTK sketch.')
    return result


def combine_group_sketches(sample_sketches):
    groups = sorted(set().union(*(sample.keys() for sample in sample_sketches)))
    dimensions = {}
    for group in groups:
        for sample in sample_sketches:
            if group in sample:
                dimensions[group] = sample[group].numel()
                break
    rows = []
    scale = 1.0 / math.sqrt(max(1, len(groups)))
    for sample in sample_sketches:
        parts = []
        for group in groups:
            vector = sample.get(group)
            if vector is None:
                vector = torch.zeros(dimensions[group], dtype=torch.float32)
            elif vector.numel() != dimensions[group]:
                raise RuntimeError(f'Inconsistent sketch size for {group}.')
            parts.append(vector * scale)
        row = torch.cat(parts).double()
        row = row / torch.linalg.vector_norm(row).clamp_min(1e-12)
        rows.append(row)
    return torch.stack(rows, dim=0)


def ntk_future_forecast(model, images, labels, args):
    """Forecast held-out target margins via empirical tangent-kernel regression."""
    train_count = int(args.ntk_train_samples)
    holdout_count = int(args.ntk_holdout_samples)
    total = train_count + holdout_count
    if train_count < 2 or holdout_count < 1:
        raise ValueError('NTK forecast needs >=2 train and >=1 holdout samples.')
    if images.shape[0] < total:
        raise RuntimeError(f'NTK needs {total} samples, got {images.shape[0]}.')

    was_training = model.training
    model.eval()
    sketches = []
    margins = []
    index_cache = {}
    try:
        for index in range(total):
            model.zero_grad()
            image = images[index:index + 1].cuda(non_blocking=True)
            label = labels[index:index + 1].cuda(non_blocking=True)
            logits = forward_logits(model, image)
            margin = target_margin(logits, label)[0]
            margin.backward()
            margins.append(float(margin.detach().item()))
            sketches.append(sampled_gradient_sketch(
                model, args.ntk_sketch_per_tensor, index_cache))
    finally:
        model.zero_grad()
        model.train(was_training)

    features = combine_group_sketches(sketches)
    kernel = features @ features.t()
    k_train = kernel[:train_count, :train_count]
    k_holdout = kernel[train_count:, :train_count]
    train_margin = torch.tensor(margins[:train_count], dtype=torch.float64)
    holdout_margin = torch.tensor(margins[train_count:], dtype=torch.float64)
    target = torch.full_like(train_margin, float(args.ntk_target_margin))
    ridge = float(args.ntk_ridge)
    system = k_train + ridge * torch.eye(train_count, dtype=torch.float64)
    coefficients = torch.linalg.solve(system, target - train_margin)
    predicted = holdout_margin + k_holdout @ coefficients

    eigenvalues = torch.linalg.eigvalsh(k_train)
    minimum = max(float(eigenvalues[0].item()), 1e-8)
    maximum = max(float(eigenvalues[-1].item()), 1e-8)
    condition = maximum / minimum
    # log(sigmoid(margin)) is a smooth, bounded classification-margin surrogate score.
    score = float(F.logsigmoid(predicted.float()).mean().item())
    return dict(
        score=score,
        predicted_margin=float(predicted.mean().item()),
        positive_rate=float((predicted > 0).double().mean().item()),
        condition=float(condition),
        train_margin=float(train_margin.mean().item()),
        holdout_margin_before=float(holdout_margin.mean().item()),
        holdout_margin_after=json.dumps([float(value) for value in predicted]),
        kernel=json.dumps(kernel.tolist()))


def flatten_cached_samples(batches, count):
    images = torch.cat([batch[0] for batch in batches], dim=0)
    labels = torch.cat([batch[1] for batch in batches], dim=0)
    return images[:count], labels[:count]


def evaluate(path, metadata, batches, args):
    set_seed(args.seed)
    cfg = load_candidate_config(path, args)
    model = build_classifier(cfg.model)
    model.init_weights()
    size = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    model.cuda()
    wrapper = LogitWrapper(model).cuda()

    block_zico = dict(
        score=float('nan'), center=float('nan'), dispersion=float('nan'),
        group_scores={})
    alignment = dict(
        score=float('nan'), center=float('nan'), dispersion=float('nan'),
        group_scores={})
    batch_losses = []
    if not args.skip_gradients:
        gradients, batch_losses = collect_batch_gradients(
            wrapper, batches[:args.gradient_batches])
        block_zico = calculate_block_zico(
            gradients, dispersion_penalty=args.block_dispersion_penalty)
        alignment = calculate_gradient_alignment(
            gradients, dispersion_penalty=args.block_dispersion_penalty)
        # Release the full gradient history before per-sample NTK calculations.
        del gradients

    expressivity = dict(
        expressivity=float('nan'), progressivity=float('nan'), block_entropy=[])
    if not args.skip_expressivity:
        expressivity = block_expressivity(
            model, batches[0][0].cuda(non_blocking=True),
            max_vectors=args.expressivity_max_vectors)

    ntk = dict(
        score=float('nan'), predicted_margin=float('nan'),
        positive_rate=float('nan'), condition=float('nan'),
        train_margin=float('nan'), holdout_margin_before=float('nan'),
        holdout_margin_after='[]', kernel='[]')
    if not args.skip_ntk_future:
        total = args.ntk_train_samples + args.ntk_holdout_samples
        images, labels = flatten_cached_samples(batches, total)
        ntk = ntk_future_forecast(model, images, labels, args)

    row = dict(
        candidate=os.path.splitext(os.path.basename(path))[0],
        config_path=path,
        source_size=metadata.get('size', ''),
        source_flops=metadata.get('flops', ''),
        source_proxy_only_performance=metadata.get('proxy_only_performance', ''),
        source_proxy_only_raw=metadata.get('proxy_only_raw', ''),
        source_ntk_condition=metadata.get('ntk_condition', ''),
        source_zico=metadata.get('zico', ''),
        size=size,
        block_zico_score=block_zico['score'],
        block_zico_center=block_zico['center'],
        block_zico_dispersion=block_zico['dispersion'],
        block_zico_groups=json.dumps(block_zico['group_scores'], sort_keys=True),
        gradient_alignment_score=alignment['score'],
        gradient_alignment_center=alignment['center'],
        gradient_alignment_dispersion=alignment['dispersion'],
        gradient_alignment_groups=json.dumps(
            alignment['group_scores'], sort_keys=True),
        ntk_future_score=ntk['score'],
        ntk_future_predicted_margin=ntk['predicted_margin'],
        ntk_future_positive_rate=ntk['positive_rate'],
        ntk_condition=ntk['condition'],
        ntk_train_margin=ntk['train_margin'],
        ntk_holdout_margin_before=ntk['holdout_margin_before'],
        ntk_holdout_margin_after=ntk['holdout_margin_after'],
        ntk_kernel=ntk['kernel'],
        expressivity=expressivity['expressivity'],
        progressivity=expressivity['progressivity'],
        block_entropy=json.dumps(expressivity['block_entropy']),
        gradient_batch_losses=json.dumps(batch_losses),
        seed=args.seed,
        error='')
    del wrapper
    del model
    torch.cuda.empty_cache()
    return row


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required.')
    if not args.skip_gradients and args.gradient_batches < 2:
        raise ValueError('--gradient-batches must be at least 2.')
    total_ntk = args.ntk_train_samples + args.ntk_holdout_samples
    required_batches = max(
        1 if args.skip_gradients else args.gradient_batches,
        0 if args.skip_ntk_future else
        int(math.ceil(total_ntk / float(args.batch_size))))
    set_seed(args.seed)
    batches = cache_batches(data_loader(args), required_batches)
    candidates = collect_candidates(args)
    if args.num_shards < 1:
        raise ValueError('--num-shards must be positive.')
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError('--shard-index must satisfy 0 <= index < num-shards.')
    candidates = candidates[args.shard_index::args.num_shards]
    print(
        f'shard={args.shard_index}/{args.num_shards} candidates={len(candidates)}',
        flush=True)

    rows = []
    completed = set()
    if args.resume and os.path.isfile(args.output_csv):
        with open(args.output_csv, newline='', encoding='utf-8') as file:
            rows = list(csv.DictReader(file))
        completed = {
            row.get('config_path') for row in rows if not row.get('error')}

    for index, (path, metadata) in enumerate(candidates, 1):
        if path in completed:
            print(f'[{index}/{len(candidates)}] skip completed {path}', flush=True)
            continue
        print(f'[{index}/{len(candidates)}] evaluate {path}', flush=True)
        try:
            row = evaluate(path, metadata, batches, args)
        except Exception as exc:
            row = dict(
                candidate=os.path.splitext(os.path.basename(path))[0],
                config_path=path,
                source_size=metadata.get('size', ''),
                source_flops=metadata.get('flops', ''),
                source_proxy_only_performance=metadata.get(
                    'proxy_only_performance', ''),
                source_proxy_only_raw=metadata.get('proxy_only_raw', ''),
                source_ntk_condition=metadata.get('ntk_condition', ''),
                source_zico=metadata.get('zico', ''),
                error=f'{type(exc).__name__}: {exc}')
            torch.cuda.empty_cache()
        rows = [old for old in rows if old.get('config_path') != path]
        rows.append(row)
        write_csv(args.output_csv, rows)
        print(
            '  block_zico={} align={} ntk_future={} ntk_cond={} error={}'.format(
                row.get('block_zico_score'),
                row.get('gradient_alignment_score'),
                row.get('ntk_future_score'), row.get('ntk_condition'),
                row.get('error')), flush=True)
    print(f'wrote {len(rows)} candidates: {args.output_csv}')


if __name__ == '__main__':
    main()
