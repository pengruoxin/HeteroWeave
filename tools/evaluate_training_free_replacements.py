#!/usr/bin/env python3
"""Evaluate candidate replacements for ZiCo without optimizer updates.

Scores:
  * InTrain (CVPR 2026): geometric participation ratio multiplied by
    cumulative gradient health.
  * L-SWAG (CVPR 2025): inverse gradient-dispersion trainability multiplied
    by sample-wise activation-pattern cardinality.
  * SWAP (ICLR 2024): sample-wise activation-pattern cardinality.
  * NASWOT (ICML 2021): activation-code kernel log determinant.

The script never changes model parameters.  HeteroWeave candidates are evaluated at
their actual constructed initialization, including imported pretrained blocks.
"""

import argparse
import csv
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
from mmcv import Config, DictAction
from mmcls.datasets import build_dataset
from mmcls.models import build_classifier

from mmcls_addon import *  # noqa: F401,F403
from simlarity.zero_nas.heteroweave_composite import first_feature_tensor, feature_matrix
from simlarity.zero_nas.zico import (
    calculate_zico, collect_zico_grad, logical_gradient_group)
from tools.evaluate_heteroweave_proxy import (
    LogitWrapper, cache_batches, collect_candidates, data_loader,
    load_candidate_config, set_seed, write_csv)


ACTIVATION_TYPES = (
    nn.ReLU, nn.LeakyReLU, nn.GELU, nn.PReLU, nn.Hardswish, nn.ELU,
    nn.SELU, nn.Mish, nn.SiLU)


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
    parser.add_argument('--microbatch-size', type=int, default=4)
    parser.add_argument('--microbatches', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument(
        '--balanced-classes', type=int, default=0,
        help='Use a fixed class-balanced batch with this many classes.')
    parser.add_argument(
        '--samples-per-class', type=int, default=4,
        help='Images per class for --balanced-classes.')
    parser.add_argument('--intrain-batch-size', type=int, default=8)
    parser.add_argument('--intrain-resolution', type=int, default=224)
    parser.add_argument('--intrain-max-observations', type=int, default=128)
    parser.add_argument(
        '--activation-only', action='store_true',
        help='Compute only SWAP and NASWOT (used for extra-seed checks).')
    parser.add_argument(
        '--zico-only', action='store_true',
        help='Compute activation metrics plus original ZiCo, skipping others.')
    parser.add_argument(
        '--drop-albu', action='store_true',
        help=(
            'Remove Albumentations transforms from the proxy input pipeline. '
            'Useful on evaluation hosts without the optional dependency.'))
    parser.add_argument(
        '--gradient-consistent-swap', action='store_true',
        help=(
            'Compute GC-SWAP from class-matched activation gradients. '
            'Requires --balanced-classes and an even --samples-per-class.'))
    parser.add_argument(
        '--model-seed', type=int, default=None,
        help=(
            'Optional fixed model/head initialization seed. This separates '
            'panel sampling variance from classifier-head variance.'))
    parser.add_argument(
        '--reinitialize-all', action='store_true',
        help='Reset supported modules after loading imported block weights.')
    parser.add_argument(
        '--skip-block-swap', action='store_true',
        help='Skip memory-heavy per-structure-group activation code sets.')
    parser.add_argument(
        '--skip-junction-swap', action='store_true',
        help='Skip effective-rank health at top-level HeteroWeave block junctions.')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def activation_modules(model):
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, ACTIVATION_TYPES)]


class ActivationPatternMonitor:
    """Accumulate SWAP cardinality and the NASWOT kernel in one forward."""

    def __init__(self, model, batch_size, collect_groups=True,
                 retain_activations=False):
        self.batch_size = int(batch_size)
        self.collect_groups = bool(collect_groups)
        self.retain_activations = bool(retain_activations)
        self.kernel = torch.zeros(
            self.batch_size, self.batch_size, dtype=torch.float64)
        self.codes = set()
        self.layer_pattern_counts = []
        self.relative_layer_pattern_counts = []
        self.activation_records = []
        self.group_codes = {}
        self.handles = []
        self.module_count = 0
        for name, module in activation_modules(model):
            self.handles.append(
                module.register_forward_hook(self._capture(name)))

    def _capture(self, name):
        def hook(_module, _inputs, output):
            tensor = first_feature_tensor(output)
            if (tensor is None or tensor.ndim < 2 or
                    tensor.shape[0] != self.batch_size):
                return
            values = tensor.detach().reshape(self.batch_size, -1)
            binary = values > 0
            if binary.numel() == 0:
                return
            positive_count = binary.sum(dim=0).clamp_min(1)
            positive_mean = values.clamp_min(0).sum(dim=0) / positive_count
            strong = binary & (values > positive_mean.unsqueeze(0))
            numeric = binary.to(dtype=torch.float64)
            self.kernel.add_((numeric @ numeric.t()).cpu())
            inverse = 1.0 - numeric
            self.kernel.add_((inverse @ inverse.t()).cpu())

            if self.batch_size <= 62:
                powers = (2 ** torch.arange(
                    self.batch_size, dtype=torch.int64,
                    device=binary.device)).view(self.batch_size, 1)
                packed = (binary.to(torch.int64) * powers).sum(dim=0)
                codes = [int(value) for value in torch.unique(packed).cpu()]
                strong_packed = (
                    strong.to(torch.int64) * powers).sum(dim=0)
                if self.batch_size <= 32:
                    on_uint = packed.cpu().numpy().astype(np.uint64)
                    strong_uint = strong_packed.cpu().numpy().astype(np.uint64)
                    relative_count = int(np.unique(
                        on_uint | (strong_uint << np.uint64(self.batch_size))
                    ).size)
                else:
                    relative_bits = torch.cat([binary, strong], dim=0)
                    relative_packed = np.packbits(
                        relative_bits.t().contiguous().cpu().numpy(), axis=1)
                    relative_count = int(np.unique(
                        relative_packed, axis=0).shape[0])
            else:
                packed = np.packbits(
                    binary.t().contiguous().cpu().numpy(), axis=1)
                packed = np.unique(packed, axis=0)
                codes = [row.tobytes() for row in packed]
                relative_bits = torch.cat([binary, strong], dim=0)
                relative_packed = np.packbits(
                    relative_bits.t().contiguous().cpu().numpy(), axis=1)
                relative_count = int(np.unique(
                    relative_packed, axis=0).shape[0])
            self.layer_pattern_counts.append(len(codes))
            self.relative_layer_pattern_counts.append(relative_count)
            if self.retain_activations:
                self.activation_records.append(
                    (name, tensor, int(len(codes))))
            self.codes.update(codes)
            if self.collect_groups:
                group = logical_gradient_group(name)
                self.group_codes.setdefault(group, set()).update(codes)
            self.module_count += 1
        return hook

    @staticmethod
    def _class_reduced_gradient(gradient, indices, labels, classes):
        """Reduce an activation gradient to a class-by-channel matrix."""
        selected = gradient.index_select(0, indices)
        if selected.ndim == 4:
            selected = selected.mean(dim=(2, 3))
        elif selected.ndim == 3:
            # HeteroWeave token features use [batch, tokens, channels].
            selected = selected.mean(dim=1)
        elif selected.ndim > 2:
            selected = selected.reshape(selected.shape[0], -1)
        elif selected.ndim == 1:
            selected = selected.unsqueeze(1)
        selected_labels = labels.index_select(0, indices)
        rows = []
        for label in classes:
            mask = selected_labels.eq(label)
            if not bool(mask.any().item()):
                return None
            rows.append(selected[mask].mean(dim=0))
        return torch.stack(rows, dim=0).reshape(-1).float()

    def gradient_consistent_scores(self, logits, labels, eps=1e-12):
        """Layer-SWAP gated by class-matched activation-gradient agreement.

        The panel contains an even number of images from every class. Half of
        each class forms split A and the other half split B. No parameters are
        updated; two activation-gradient queries share the same forward graph.
        """
        if not self.retain_activations or not self.activation_records:
            raise RuntimeError('GC-SWAP requires retained activation outputs.')
        labels = labels.to(device=logits.device, dtype=torch.long).view(-1)
        classes = torch.unique(labels, sorted=True)
        left_indices = []
        right_indices = []
        for label in classes:
            indices = torch.nonzero(labels.eq(label), as_tuple=False).view(-1)
            if indices.numel() < 2 or indices.numel() % 2:
                raise ValueError(
                    'GC-SWAP needs an even number >=2 for every class.')
            midpoint = indices.numel() // 2
            left_indices.extend(indices[:midpoint].tolist())
            right_indices.extend(indices[midpoint:].tolist())
        left = torch.tensor(left_indices, device=logits.device, dtype=torch.long)
        right = torch.tensor(
            right_indices, device=logits.device, dtype=torch.long)
        criterion = nn.CrossEntropyLoss()
        left_loss = criterion(
            logits.index_select(0, left), labels.index_select(0, left))
        right_loss = criterion(
            logits.index_select(0, right), labels.index_select(0, right))
        tensors = [record[1] for record in self.activation_records]
        usable = [index for index, tensor in enumerate(tensors)
                  if tensor.requires_grad]
        if not usable:
            raise RuntimeError('No differentiable activation outputs captured.')
        usable_tensors = [tensors[index] for index in usable]
        left_gradients = torch.autograd.grad(
            left_loss, usable_tensors, retain_graph=True, allow_unused=True)
        right_gradients = torch.autograd.grad(
            right_loss, usable_tensors, retain_graph=False, allow_unused=True)
        agreements = []
        weighted_terms = []
        layer_rows = []
        class_values = classes.tolist()
        for record_index, left_gradient, right_gradient in zip(
                usable, left_gradients, right_gradients):
            if left_gradient is None or right_gradient is None:
                continue
            left_vector = self._class_reduced_gradient(
                left_gradient, left, labels, class_values)
            right_vector = self._class_reduced_gradient(
                right_gradient, right, labels, class_values)
            if left_vector is None or right_vector is None:
                continue
            denominator = (
                torch.linalg.vector_norm(left_vector) *
                torch.linalg.vector_norm(right_vector))
            if (not bool(torch.isfinite(denominator).item()) or
                    float(denominator.item()) <= eps):
                continue
            cosine = torch.dot(left_vector, right_vector) / denominator
            cosine = float(cosine.clamp(-1.0, 1.0).item())
            agreement = 0.5 * (1.0 + cosine)
            name, _tensor, pattern_count = self.activation_records[record_index]
            term = math.sqrt(pattern_count) * agreement
            agreements.append(agreement)
            weighted_terms.append(term)
            layer_rows.append(dict(
                name=name, patterns=pattern_count,
                cosine=cosine, agreement=agreement, term=term))
        if not agreements:
            raise RuntimeError('No valid activation-gradient agreements.')
        baseline_captured = sum(
            math.sqrt(self.activation_records[index][2]) for index in usable)
        return {
            'gc_layer_swap_score': float(sum(weighted_terms)),
            'gc_gradient_agreement_mean': float(np.mean(agreements)),
            'gc_gradient_agreement_min': float(np.min(agreements)),
            'gc_gradient_layer_count': len(agreements),
            'gc_gradient_coverage': float(
                len(agreements) / max(1, len(self.activation_records))),
            'gc_differentiable_layer_swap_sqrt_sum': float(baseline_captured),
            'gc_left_loss': float(left_loss.detach().item()),
            'gc_right_loss': float(right_loss.detach().item()),
            'gc_layer_details': json.dumps(layer_rows),
        }

    def scores(self):
        if self.module_count == 0:
            raise RuntimeError('No activation outputs were captured.')
        sign, logdet = np.linalg.slogdet(self.kernel.numpy())
        if sign <= 0 or not np.isfinite(logdet):
            logdet = float('-inf')
        group_counts = {
            name: len(codes) for name, codes in self.group_codes.items()
            if codes
        }
        result = {
            'swap_score': float(len(self.codes)),
            'naswot_score': float(logdet),
            'activation_module_count': self.module_count,
            'layer_swap_sum': float(sum(self.layer_pattern_counts)),
            'layer_swap_sqrt_sum': float(sum(
                math.sqrt(value) for value in self.layer_pattern_counts)),
            'layer_swap_logsum': float(sum(
                math.log1p(value) for value in self.layer_pattern_counts)),
            'relative_layer_swap_sqrt_sum': float(sum(
                math.sqrt(value)
                for value in self.relative_layer_pattern_counts)),
        }
        if not group_counts:
            return result
        group_logs = [math.log1p(value) for value in group_counts.values()]
        group_mean = float(np.mean(group_logs))
        group_std = float(np.std(group_logs))
        result.update({
            'block_swap_mean_log': group_mean,
            'block_swap_median_log': float(np.median(group_logs)),
            'block_swap_min_log': float(np.min(group_logs)),
            'block_swap_mean_stable': group_mean - 0.25 * group_std,
            'block_swap_group_std': group_std,
            'block_swap_group_counts': json.dumps(
                group_counts, sort_keys=True),
        })
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _data_value(value):
    """Unwrap tensors returned directly by an MMCV dataset pipeline."""
    return value.data if hasattr(value, 'data') and not torch.is_tensor(value) else value


def class_balanced_batch(args):
    """Build one deterministic N-class x K-image proxy panel."""
    cfg = Config.fromfile(args.data_config)
    if args.data_prefix is not None:
        cfg.data.train.data_prefix = args.data_prefix
    if args.ann_file is not None:
        cfg.data.train.ann_file = args.ann_file
    dataset = build_dataset(cfg.data.train)
    by_class = {}
    for index, info in enumerate(dataset.data_infos):
        label = int(np.asarray(info['gt_label']).reshape(-1)[0])
        by_class.setdefault(label, []).append(index)
    per_class = max(1, int(args.samples_per_class))
    eligible = sorted(
        label for label, indices in by_class.items()
        if len(indices) >= per_class)
    class_count = int(args.balanced_classes)
    if class_count < 2 or len(eligible) < class_count:
        raise RuntimeError(
            f'Need {class_count} eligible classes, found {len(eligible)}.')
    rng = np.random.RandomState(args.seed)
    chosen_classes = rng.choice(eligible, size=class_count, replace=False)
    chosen_indices = []
    for label in chosen_classes:
        chosen_indices.extend(rng.choice(
            by_class[int(label)], size=per_class, replace=False).tolist())
    rng.shuffle(chosen_indices)
    images = []
    labels = []
    for index in chosen_indices:
        sample = dataset[int(index)]
        image = _data_value(sample['img'])
        label = _data_value(sample['gt_label'])
        if not torch.is_tensor(image):
            image = torch.as_tensor(image)
        if torch.is_tensor(label):
            label = int(label.reshape(-1)[0].item())
        else:
            label = int(np.asarray(label).reshape(-1)[0])
        images.append(image)
        labels.append(label)
    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)


def centered_kernel_alignment(kernel, target, eps=1e-12):
    """Centered kernel alignment for two square sample kernels."""
    kernel = kernel.double()
    target = target.double()
    kernel = (
        kernel - kernel.mean(dim=0, keepdim=True) -
        kernel.mean(dim=1, keepdim=True) + kernel.mean())
    target = (
        target - target.mean(dim=0, keepdim=True) -
        target.mean(dim=1, keepdim=True) + target.mean())
    numerator = torch.sum(kernel * target)
    denominator = torch.linalg.vector_norm(kernel) * torch.linalg.vector_norm(target)
    if (not bool(torch.isfinite(denominator).item()) or
            float(denominator.item()) <= eps):
        return 0.0
    value = float((numerator / denominator).item())
    return max(0.0, min(1.0, value))


class StageLabelAlignmentMonitor:
    """Measure class alignment of the four fused HeteroWeave stage outputs."""

    def __init__(self, wrapper, labels):
        backbone = getattr(getattr(wrapper, 'model', None), 'backbone', None)
        if backbone is None:
            backbone = getattr(
                getattr(wrapper, 'classifier', None), 'backbone', None)
        blocks = getattr(backbone, 'blocks', None)
        if blocks is None:
            raise TypeError('Expected a HeteroWeave classifier with backbone.blocks.')
        labels = labels.detach().view(-1).long().cpu()
        self.target = labels[:, None].eq(labels[None, :]).to(torch.float64)
        self.alignments = []
        self.handles = [
            block.register_forward_hook(self._capture(index))
            for index, block in enumerate(blocks)]

    def _capture(self, _index):
        def hook(_module, _inputs, output):
            tensor = first_feature_tensor(output)
            if tensor is None or tensor.ndim < 2:
                return
            batch = self.target.shape[0]
            if tensor.shape[0] != batch:
                return
            binary = tensor.detach().reshape(batch, -1) > 0
            if binary.numel() == 0:
                return
            numeric = binary.to(dtype=torch.float32)
            kernel = numeric @ numeric.t()
            inverse = 1.0 - numeric
            kernel.add_(inverse @ inverse.t())
            self.alignments.append(centered_kernel_alignment(
                kernel.cpu(), self.target))
        return hook

    def scores(self, layer_swap_sqrt_sum):
        if not self.alignments:
            raise RuntimeError('No HeteroWeave stage outputs were captured.')
        mean_alignment = float(np.mean(self.alignments))
        return {
            'stage_label_alignment_mean': mean_alignment,
            'stage_label_alignments': json.dumps(self.alignments),
            'alignment_gated_layer_swap': float(
                layer_swap_sqrt_sum * mean_alignment),
        }

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


class ParticipationRatioMonitor:
    """Compute layer-wise participation ratios during a forward pass."""

    def __init__(self, model, max_observations=128, eps=1e-10):
        self.max_observations = max(2, int(max_observations))
        self.eps = float(eps)
        self.values = []
        self.names = []
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.handles.append(
                    module.register_forward_hook(self._capture(name)))

    def _capture(self, name):
        def hook(_module, _inputs, output):
            tensor = first_feature_tensor(output)
            if tensor is None or tensor.ndim < 2:
                return
            matrix = feature_matrix(tensor.detach()).float()
            if matrix.shape[0] > self.max_observations:
                indices = torch.linspace(
                    0, matrix.shape[0] - 1,
                    steps=self.max_observations,
                    device=matrix.device).long()
                matrix = matrix.index_select(0, indices)
            if matrix.shape[0] < 2 or matrix.shape[1] < 2:
                return
            matrix = matrix - matrix.mean(dim=0, keepdim=True)
            trace = matrix.square().sum()
            # The non-zero eigenvalues of X^T X and X X^T are identical.
            # Using the smaller observation Gram avoids a CxC covariance.
            gram = matrix @ matrix.t()
            trace_square = gram.square().sum()
            if (not bool(torch.isfinite(trace).item()) or
                    not bool(torch.isfinite(trace_square).item()) or
                    float(trace_square.item()) <= self.eps):
                return
            ratio = trace.square() / trace_square.clamp_min(self.eps)
            value = float(ratio.item())
            if math.isfinite(value) and value > 0:
                self.values.append(value)
                self.names.append(name)
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def sample_kernel_metrics(value, eps=1e-6):
    """Scale-neutral sample Gram participation ratio and normalized logdet."""
    tensor = first_feature_tensor(value)
    if tensor is None or tensor.ndim < 2 or tensor.shape[0] < 2:
        return None
    matrix = tensor.detach().reshape(tensor.shape[0], -1).float()
    if matrix.shape[1] < 2:
        return None
    gram = (matrix @ matrix.t()) / float(matrix.shape[1])
    # Center across samples without materializing a centered activation tensor.
    row_mean = gram.mean(dim=1, keepdim=True)
    gram = gram - row_mean - row_mean.t() + gram.mean()
    gram = 0.5 * (gram + gram.t())
    trace = torch.trace(gram)
    if (not bool(torch.isfinite(trace).item()) or
            float(trace.item()) <= eps):
        return None
    trace_square = gram.square().sum()
    if (not bool(torch.isfinite(trace_square).item()) or
            float(trace_square.item()) <= eps):
        return None
    participation = trace.square() / trace_square.clamp_min(eps)
    normalized = gram / (trace / gram.shape[0]).clamp_min(eps)
    eigenvalues = torch.linalg.eigvalsh(normalized.double())
    logdet = torch.log(eigenvalues.clamp_min(eps)).sum()
    result = (float(participation.item()), float(logdet.item()))
    return result if all(math.isfinite(item) for item in result) else None


class JunctionHealthMonitor:
    """Measure sample-separability changes across top-level HeteroWeave blocks."""

    def __init__(self, model):
        classifier = getattr(model, 'classifier', model)
        backbone = getattr(classifier, 'backbone', None)
        blocks = getattr(backbone, 'blocks', None)
        if blocks is None:
            raise TypeError('Expected a HeteroWeave classifier with backbone.blocks.')
        self.entries = []
        self.handles = [
            block.register_forward_hook(self._capture(index))
            for index, block in enumerate(blocks)
        ]

    def _capture(self, index):
        def hook(_module, inputs, output):
            before = sample_kernel_metrics(inputs)
            after = sample_kernel_metrics(output)
            if before is not None and after is not None:
                self.entries.append({
                    'block': index,
                    'pr_before': before[0],
                    'pr_after': after[0],
                    'logdet_before': before[1],
                    'logdet_after': after[1],
                })
        return hook

    def scores(self, swap_score):
        if not self.entries:
            raise RuntimeError('No valid HeteroWeave junction metrics were captured.')
        pr_delta = [
            math.log(max(row['pr_after'], 1e-12) /
                     max(row['pr_before'], 1e-12))
            for row in self.entries
        ]
        # Normalize logdet changes by sample count indirectly through the
        # number of eigenvalues represented in each logdet.  The batch size is
        # constant within a run, so this remains rank-comparable.
        logdet_delta = [
            row['logdet_after'] - row['logdet_before']
            for row in self.entries
        ]
        pr_collapse = [max(0.0, -value) for value in pr_delta]
        logdet_collapse = [max(0.0, -value) for value in logdet_delta]
        pr_health_mean = math.exp(-float(np.mean(pr_collapse)))
        pr_health_worst = math.exp(-float(np.max(pr_collapse)))
        # Logdet deltas can scale with batch size.  Dividing by 32 keeps the
        # gate smooth for the default SWAP batch while preserving ordering.
        logdet_health = math.exp(-float(np.mean(logdet_collapse)) / 32.0)
        return {
            'junction_pr_delta_mean': float(np.mean(pr_delta)),
            'junction_pr_delta_min': float(np.min(pr_delta)),
            'junction_pr_health_mean': pr_health_mean,
            'junction_pr_health_worst': pr_health_worst,
            'junction_logdet_delta_mean': float(np.mean(logdet_delta)),
            'junction_logdet_health': logdet_health,
            'junction_swap_pr_mean': float(swap_score) * pr_health_mean,
            'junction_swap_pr_worst': float(swap_score) * pr_health_worst,
            'junction_swap_logdet': float(swap_score) * logdet_health,
            'junction_details': json.dumps(self.entries, sort_keys=True),
        }

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def intrain_score(wrapper, batch_size, resolution, max_observations, seed):
    """CVPR 2026 InTrain equations (1)-(8), with no optimizer step."""
    set_seed(seed)
    monitor = ParticipationRatioMonitor(
        wrapper, max_observations=max_observations)
    was_training = wrapper.training
    wrapper.eval()
    try:
        images = torch.randn(
            int(batch_size), 3, int(resolution), int(resolution),
            device='cuda')
        wrapper.zero_grad()
        logits = wrapper(images)
        targets = torch.randint(
            0, logits.shape[-1], (logits.shape[0],), device=logits.device)
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()

        gamma = sum(math.log(max(value, 1e-10)) for value in monitor.values)
        health = []
        for parameter in wrapper.parameters():
            gradient = parameter.grad
            if gradient is None or gradient.numel() < 2:
                continue
            gradient = gradient.detach().float()
            maximum = gradient.abs().max()
            if (not bool(torch.isfinite(maximum).item()) or
                    float(maximum.item()) <= 1e-10):
                health.append(0.0)
                continue
            deviation = gradient.std(unbiased=False)
            value = min(1.0, float((deviation / (maximum + 1e-10)).item()))
            health.append(value if math.isfinite(value) else 0.0)
        resilience = sum(health)
        depth = len(monitor.values)
        if depth == 0 or not health:
            raise RuntimeError('InTrain captured no valid layers or gradients.')
        score = gamma * (1.0 + resilience) / math.log(depth + 1.0)
        return {
            'intrain_score': score,
            'intrain_geometric_capacity': gamma,
            'intrain_gradient_health': resilience,
            'intrain_depth': depth,
            'intrain_parameter_tensors': len(health),
            'intrain_synthetic_loss': float(loss.detach().item()),
        }
    finally:
        monitor.close()
        wrapper.zero_grad()
        wrapper.train(was_training)


def collect_real_gradient_batches(wrapper, images, labels, microbatch_size,
                                  microbatches):
    criterion = nn.CrossEntropyLoss()
    gradients = {}
    losses = []
    was_training = wrapper.training
    wrapper.eval()
    try:
        for step in range(int(microbatches)):
            left = step * int(microbatch_size)
            right = left + int(microbatch_size)
            wrapper.zero_grad()
            logits = wrapper(images[left:right].cuda(non_blocking=True))
            loss = criterion(logits, labels[left:right].cuda(non_blocking=True))
            loss.backward()
            losses.append(float(loss.detach().item()))
            collect_zico_grad(wrapper, gradients, step)
    finally:
        wrapper.zero_grad()
        wrapper.train(was_training)
    return gradients, losses


def lswag_trainability(gradients, eps=1e-12):
    """L-SWAG equation (1), using all Conv/Linear layers.

    The paper selects a depth interval per benchmark.  No such interval is
    available for these reassembly candidates, so the label-free all-layer form is reported rather
    than selecting layers using the short-training labels.
    """
    module_scores = []
    for values in gradients.values():
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] < 2:
            continue
        variance = np.var(np.abs(array), axis=0)
        valid = np.isfinite(variance) & (variance > eps)
        if not np.any(valid):
            continue
        inverse_std_sum = np.sum(1.0 / np.sqrt(variance[valid]))
        if np.isfinite(inverse_std_sum) and inverse_std_sum > 0:
            module_scores.append(float(np.log(inverse_std_sum)))
    if not module_scores:
        raise RuntimeError('No valid gradient variance for L-SWAG.')
    return float(sum(module_scores)), module_scores


def evaluate(path, metadata, images, labels, args):
    model_seed = args.seed if args.model_seed is None else args.model_seed
    set_seed(model_seed)
    cfg = load_candidate_config(path, args)
    model = build_classifier(cfg.model)
    model.init_weights()
    if args.reinitialize_all:
        for module in model.modules():
            reset = getattr(module, 'reset_parameters', None)
            if callable(reset):
                try:
                    reset()
                except TypeError:
                    # A small number of custom containers expose a reset
                    # method with extra arguments; their child layers are
                    # still reset independently by this traversal.
                    pass
    size = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    model.cuda().eval()
    wrapper = LogitWrapper(model).cuda().eval()
    try:
        junction_monitor = (
            None if args.skip_junction_swap else JunctionHealthMonitor(wrapper))
        alignment_monitor = (
            StageLabelAlignmentMonitor(wrapper, labels)
            if args.balanced_classes > 0 else None)
        pattern_monitor = ActivationPatternMonitor(
            wrapper, batch_size=images.shape[0],
            collect_groups=not args.skip_block_swap,
            retain_activations=args.gradient_consistent_swap)
        try:
            if args.gradient_consistent_swap:
                proxy_images = images.cuda(non_blocking=True).detach()
                proxy_images.requires_grad_(True)
                logits = wrapper(proxy_images)
                gc_swap = pattern_monitor.gradient_consistent_scores(
                    logits, labels.cuda(non_blocking=True))
            else:
                with torch.no_grad():
                    _ = wrapper(images.cuda(non_blocking=True))
                gc_swap = {}
            pattern = pattern_monitor.scores()
            junction = (
                junction_monitor.scores(pattern['swap_score'])
                if junction_monitor is not None else {})
            alignment = (
                alignment_monitor.scores(pattern['layer_swap_sqrt_sum'])
                if alignment_monitor is not None else {})
        finally:
            pattern_monitor.close()
            if junction_monitor is not None:
                junction_monitor.close()
            if alignment_monitor is not None:
                alignment_monitor.close()

        common = dict(
            candidate=os.path.splitext(os.path.basename(path))[0],
            config_path=path,
            source_size=metadata.get('size', ''),
            source_flops=metadata.get('flops', ''),
            source_eval_acc1=metadata.get('eval_acc1', ''),
            source_short_accuracy_rank=metadata.get('short_accuracy_rank', ''),
            size=size,
            swap_score=pattern['swap_score'],
            naswot_score=pattern['naswot_score'],
            activation_module_count=pattern['activation_module_count'],
            layer_swap_sum=pattern['layer_swap_sum'],
            layer_swap_sqrt_sum=pattern['layer_swap_sqrt_sum'],
            layer_swap_logsum=pattern['layer_swap_logsum'],
            relative_layer_swap_sqrt_sum=pattern[
                'relative_layer_swap_sqrt_sum'],
            seed=args.seed,
            model_seed=model_seed,
            pretrained_blocks=not args.reinitialize_all,
            error='')
        for key in (
                'block_swap_mean_log', 'block_swap_median_log',
                'block_swap_min_log', 'block_swap_mean_stable',
                'block_swap_group_std', 'block_swap_group_counts'):
            if key in pattern:
                common[key] = pattern[key]
        common.update(junction)
        common.update(alignment)
        common.update(gc_swap)
        if args.activation_only:
            return common

        gradients, gradient_losses = collect_real_gradient_batches(
            wrapper, images, labels, args.microbatch_size, args.microbatches)
        if args.zico_only:
            row = dict(common)
            row.update(dict(
                zico_score=calculate_zico(gradients),
                gradient_batch_losses=json.dumps(gradient_losses),
                microbatch_size=args.microbatch_size,
                microbatches=args.microbatches))
            return row
        lswag_train, lswag_modules = lswag_trainability(gradients)
        swap_score = pattern['swap_score']
        lswag = lswag_train * swap_score

        intrain = intrain_score(
            wrapper,
            batch_size=args.intrain_batch_size,
            resolution=args.intrain_resolution,
            max_observations=args.intrain_max_observations,
            seed=args.seed)

        row = dict(common)
        row.update(dict(
            lswag_score=lswag,
            lswag_trainability=lswag_train,
            lswag_gradient_module_count=len(lswag_modules),
            lswag_module_scores=json.dumps(lswag_modules),
            gradient_batch_losses=json.dumps(gradient_losses),
            microbatch_size=args.microbatch_size,
            microbatches=args.microbatches,
        ))
        row.update(intrain)
        return row
    finally:
        wrapper.zero_grad()
        del wrapper, model
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    if args.activation_only and args.zico_only:
        raise ValueError('--activation-only and --zico-only are exclusive.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required.')
    if args.microbatches < 2 or args.microbatch_size < 1:
        raise ValueError('Need >=2 microbatches and >=1 image per microbatch.')
    required = args.microbatch_size * args.microbatches
    if args.batch_size != required:
        args.batch_size = required
    set_seed(args.seed)
    if args.balanced_classes > 0:
        expected = int(args.balanced_classes) * int(args.samples_per_class)
        if expected != args.batch_size:
            raise ValueError(
                f'Balanced panel has {expected} samples, expected batch-size '
                f'{args.batch_size}.')
        images, labels = class_balanced_batch(args)
    else:
        if args.gradient_consistent_swap:
            raise ValueError(
                'GC-SWAP requires --balanced-classes with class labels.')
        images, labels = cache_batches(data_loader(args), 1)[0]
    if args.gradient_consistent_swap:
        if args.samples_per_class < 2 or args.samples_per_class % 2:
            raise ValueError(
                'GC-SWAP requires an even --samples-per-class >=2.')
    candidates = collect_candidates(args)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Invalid sharding arguments.')
    candidates = candidates[args.shard_index::args.num_shards]
    print(
        f'candidates={len(candidates)} samples={images.shape[0]} '
        f'seed={args.seed} shard={args.shard_index}/{args.num_shards}',
        flush=True)

    rows = []
    completed = set()
    if args.resume and os.path.isfile(args.output_csv):
        rows = read_csv(args.output_csv)
        completed = {row.get('config_path') for row in rows
                     if not row.get('error')}

    for index, (path, metadata) in enumerate(candidates, 1):
        if path in completed:
            print(f'[{index}/{len(candidates)}] skip {path}', flush=True)
            continue
        print(f'[{index}/{len(candidates)}] evaluate {path}', flush=True)
        try:
            row = evaluate(path, metadata, images, labels, args)
        except Exception as exc:
            if args.fail_fast:
                raise
            row = dict(
                candidate=os.path.splitext(os.path.basename(path))[0],
                config_path=path,
                source_size=metadata.get('size', ''),
                source_flops=metadata.get('flops', ''),
                source_eval_acc1=metadata.get('eval_acc1', ''),
                source_short_accuracy_rank=metadata.get(
                    'short_accuracy_rank', ''),
                seed=args.seed,
                error=f'{type(exc).__name__}: {exc}')
            torch.cuda.empty_cache()
        rows = [old for old in rows if old.get('config_path') != path]
        rows.append(row)
        write_csv(args.output_csv, rows)
        print(
            '  intrain={} lswag={} swap={} naswot={} error={}'.format(
                row.get('intrain_score'), row.get('lswag_score'),
                row.get('swap_score'), row.get('naswot_score'),
                row.get('error')), flush=True)

    print(f'output={args.output_csv}', flush=True)


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


if __name__ == '__main__':
    main()
