import numpy as np
import torch
from torch import nn
import re


# Adapted from the official ZiCo implementation:
# https://github.com/SLDGroup/ZiCo/blob/main/ZeroShotProxy/compute_zico.py
# The scoring formula is kept the same; only the data-batch interface is adapted
# to this repo's mmcls-style {'img', 'gt_label'} batches.


def collect_zico_grad(model, grad_dict, step_iter=0):
    """Collect Conv/Linear weight gradients for ZiCo over multiple batches."""
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if module.weight.grad is None:
            continue
        grad = module.weight.grad.detach().reshape(-1).cpu().numpy()
        if step_iter == 0:
            grad_dict[name] = [grad]
        elif name in grad_dict:
            grad_dict[name].append(grad)
        else:
            grad_dict[name] = [grad]
    return grad_dict


def calculate_zico(grad_dict):
    for modname in grad_dict.keys():
        grad_dict[modname] = np.array(grad_dict[modname])

    nsr_mean_sum_abs = 0
    for modname in grad_dict.keys():
        nsr_std = np.std(grad_dict[modname], axis=0)
        nonzero_idx = np.nonzero(nsr_std)[0]
        nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
        tmpsum = np.sum(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
        if tmpsum == 0:
            pass
        else:
            nsr_mean_sum_abs += np.log(tmpsum)
    return float(nsr_mean_sum_abs)


def _finite_gradient_ratio(values, eps=1e-8):
    gradients = np.asarray(values, dtype=np.float32)
    if gradients.ndim != 2 or gradients.shape[0] < 2:
        return None
    mean_abs = np.mean(np.abs(gradients), axis=0)
    std = np.std(gradients, axis=0)
    valid = np.isfinite(mean_abs) & np.isfinite(std) & (mean_abs > eps)
    if not np.any(valid):
        return None
    ratio = (mean_abs[valid] + eps) / (std[valid] + eps)
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if ratio.size == 0:
        return None
    return ratio


def calculate_normalized_zico(
        grad_dict, eps=1e-8, top_fraction=0.2, clip_quantile=0.95,
        dispersion_penalty=0.10):
    """ZiCo with parameter-count-neutral internal aggregation.

    Original ZiCo computes ``log(sum(mean(|g|) / std(g)))`` per layer and
    sums this value over layers.  The ``sum`` term injects an explicit
    parameter-count factor: a wider layer can score higher even when the
    average gradient signal quality is unchanged.

    This variant keeps the same gradient signal ratio but changes only the
    internal aggregation:

    - ``mean``: average ratio inside each Conv/Linear, then log.
    - ``top``: average of the strongest stable-gradient ratios inside each
      module; this keeps useful high-signal channels without counting all
      parameters.
    - ``clip``: winsorized average ratio, limiting extreme gradient outliers.

    Module scores are then averaged inside each logical DeRy group, and groups
    are averaged equally.  No size/FLOPs term is used.
    """
    grouped = {}
    module_scores = {}
    top_fraction = float(top_fraction)
    clip_quantile = float(clip_quantile)
    for module_name, values in grad_dict.items():
        ratio = _finite_gradient_ratio(values, eps=eps)
        if ratio is None:
            continue

        mean_score = float(np.log(np.mean(ratio) + eps))

        top_count = max(1, int(np.ceil(ratio.size * top_fraction)))
        if top_count >= ratio.size:
            top_values = ratio
        else:
            top_values = np.partition(ratio, ratio.size - top_count)[-top_count:]
        top_score = float(np.log(np.mean(top_values) + eps))

        threshold = float(np.quantile(ratio, clip_quantile))
        clipped = np.minimum(ratio, threshold)
        clip_score = float(np.log(np.mean(clipped) + eps))

        if not all(np.isfinite(v) for v in [mean_score, top_score, clip_score]):
            continue
        scores = dict(mean=mean_score, top=top_score, clip=clip_score)
        module_scores[module_name] = scores
        grouped.setdefault(logical_gradient_group(module_name), []).append(
            scores)

    if not grouped:
        raise RuntimeError('No valid Conv/Linear gradients for normalized ZiCo.')

    group_scores = {}
    for group_name, entries in grouped.items():
        group_scores[group_name] = {
            key: float(np.mean([entry[key] for entry in entries]))
            for key in ['mean', 'top', 'clip']
        }

    result = dict(group_scores=group_scores, module_scores=module_scores)
    for key in ['mean', 'top', 'clip']:
        values = np.asarray(
            [scores[key] for scores in group_scores.values()],
            dtype=np.float64)
        center = float(np.mean(values))
        dispersion = float(np.std(values))
        result[key] = center
        result[f'{key}_center'] = center
        result[f'{key}_dispersion'] = dispersion
        result[f'{key}_stable'] = center - float(dispersion_penalty) * dispersion
    result['balanced'] = float(
        0.5 * result['mean'] + 0.5 * result['top'] -
        float(dispersion_penalty) *
        np.std([scores['mean'] for scores in group_scores.values()]))
    return result


def logical_gradient_group(module_name):
    """Map Conv/Linear modules to equal-weight DeRy structural groups."""
    name = module_name
    if name.startswith('classifier.'):
        name = name[len('classifier.'):]
    match = re.search(r'backbone\.blocks\.(\d+)(?:\.|$)', name)
    if match:
        return f'block_{int(match.group(1)):02d}'
    match = re.search(r'backbone\.adapters\.(\d+)(?:\.|$)', name)
    if match:
        return f'adapter_{int(match.group(1)):02d}'
    if 'backbone.base_adapter' in name:
        return 'base_adapter'
    if 'backbone.base' in name:
        return 'stem'
    if '.neck' in name or name.startswith('neck'):
        return 'neck'
    if '.head' in name or name.startswith('head'):
        return 'head'
    return 'other'


def calculate_block_zico(
        grad_dict, eps=1e-8, dispersion_penalty=0.25):
    """Parameter-count/depth-resistant ZiCo diagnostic.

    Original ZiCo uses ``log(sum(parameter ratios))`` for every layer and then
    sums over layers.  That contains explicit width and depth terms.  Here we
    take a median log ratio inside each module, a median inside each logical
    DeRy block, and finally a median across blocks.  Every logical block has
    equal weight regardless of parameter count or number of internal layers.
    """
    grouped = {}
    module_scores = {}
    for module_name, values in grad_dict.items():
        gradients = np.asarray(values, dtype=np.float32)
        if gradients.ndim != 2 or gradients.shape[0] < 2:
            continue
        mean_abs = np.mean(np.abs(gradients), axis=0)
        std = np.std(gradients, axis=0)
        valid = np.isfinite(mean_abs) & np.isfinite(std) & (mean_abs > eps)
        if not np.any(valid):
            continue
        log_ratio = np.log((mean_abs[valid] + eps) / (std[valid] + eps))
        score = float(np.median(log_ratio))
        if not np.isfinite(score):
            continue
        module_scores[module_name] = score
        grouped.setdefault(logical_gradient_group(module_name), []).append(score)

    group_scores = {
        name: float(np.median(values)) for name, values in grouped.items()
        if values
    }
    values = np.asarray(list(group_scores.values()), dtype=np.float64)
    if values.size == 0:
        raise RuntimeError('No valid Conv/Linear gradients for block ZiCo.')
    center = float(np.median(values))
    dispersion = float(np.std(values))
    return dict(
        score=center - float(dispersion_penalty) * dispersion,
        center=center,
        dispersion=dispersion,
        group_scores=group_scores,
        module_scores=module_scores)


def calculate_gradient_alignment(
        grad_dict, eps=1e-12, dispersion_penalty=0.25):
    """Cross-batch gradient cosine, aggregated with equal block weights."""
    grouped = {}
    num_batches = None
    for module_name, values in grad_dict.items():
        gradients = np.asarray(values, dtype=np.float32)
        if gradients.ndim != 2 or gradients.shape[0] < 2:
            continue
        if num_batches is None:
            num_batches = gradients.shape[0]
        if gradients.shape[0] != num_batches:
            continue
        grouped.setdefault(logical_gradient_group(module_name), []).append(
            gradients)

    group_scores = {}
    for group_name, tensors in grouped.items():
        pair_scores = []
        for left in range(num_batches):
            for right in range(left + 1, num_batches):
                dot = 0.0
                left_sq = 0.0
                right_sq = 0.0
                for gradients in tensors:
                    # Keep float32 here. Converting every full-model gradient
                    # to float64 for every pair dominates runtime and memory.
                    a = gradients[left]
                    b = gradients[right]
                    dot += float(np.dot(a, b))
                    left_sq += float(np.dot(a, a))
                    right_sq += float(np.dot(b, b))
                denom = np.sqrt(max(left_sq, eps) * max(right_sq, eps))
                pair_scores.append(dot / denom)
        if pair_scores:
            group_scores[group_name] = float(np.mean(pair_scores))

    values = np.asarray(list(group_scores.values()), dtype=np.float64)
    if values.size == 0:
        raise RuntimeError('No valid Conv/Linear gradients for alignment.')
    center = float(np.median(values))
    dispersion = float(np.std(values))
    return dict(
        score=center - float(dispersion_penalty) * dispersion,
        center=center,
        dispersion=dispersion,
        group_scores=group_scores)


def collect_zico_batch(model, x, target, criterion, grad_dict, step_iter=0):
    model.zero_grad()
    output = model(x)
    target = target.view(-1).long()
    loss = criterion(output, target)
    loss.backward()
    return collect_zico_grad(model, grad_dict, step_iter)


def zico(model, x, target, criterion):
    grad_dict = {}
    collect_zico_batch(model, x, target, criterion, grad_dict, 0)
    return calculate_zico(grad_dict)
