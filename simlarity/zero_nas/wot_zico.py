"""NASWOT-conditioned ZiCo without combining two final scalar scores.

NASWOT's activation agreement kernel is used as a spectral filter over the
micro-batch axis of ZiCo gradients.  ZiCo is then calculated only once from
the filtered gradients.  This keeps the result a single, internally coupled
zero-shot proxy instead of ``a * ZiCo + b * NASWOT``.
"""

import numpy as np
from torch import nn

from simlarity.zero_nas.heteroweave_composite import first_feature_tensor
from simlarity.zero_nas.zico import logical_gradient_group


def activation_rate(feature, max_features=4096):
    """Return one deterministic activation-sign vector for a micro-batch."""
    tensor = feature.detach().float()
    if tensor.ndim < 2:
        tensor = tensor.reshape(1, -1)
    else:
        tensor = tensor.reshape(tensor.shape[0], -1)
    if tensor.shape[1] > int(max_features):
        indices = np.linspace(
            0, tensor.shape[1] - 1, int(max_features), dtype=np.int64)
        tensor = tensor[:, indices]
    # A rate, rather than a single sample bit, makes the filter less dependent
    # on which images happen to share a micro-batch.
    return (tensor > 0).float().mean(dim=0).cpu().numpy().astype(np.float32)


def activation_code(feature, max_features=4096):
    """Return per-image binary activation codes for a top-level block."""
    tensor = feature.detach().float().reshape(feature.shape[0], -1)
    if tensor.shape[1] > int(max_features):
        indices = np.linspace(
            0, tensor.shape[1] - 1, int(max_features), dtype=np.int64)
        tensor = tensor[:, indices]
    return (tensor > 0).float().cpu().numpy().astype(np.float32)


def activation_agreement_filter(rates, eps=1e-6):
    """Build a normalized NASWOT agreement kernel and its shrinkage filter."""
    h = np.asarray(rates, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] < 2:
        raise ValueError('Need at least two activation-rate observations.')
    kernel = h @ h.T + (1.0 - h) @ (1.0 - h).T
    diagonal = np.sqrt(np.maximum(np.diag(kernel), eps))
    kernel = kernel / np.outer(diagonal, diagonal)
    kernel = (kernel + kernel.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    positive = eigenvalues[eigenvalues > eps]
    tau = float(np.median(positive)) if positive.size else float(eps)
    gains = eigenvalues / (eigenvalues + tau)
    projection = (eigenvectors * gains[None, :]) @ eigenvectors.T
    sign, logdet = np.linalg.slogdet(kernel + eps * np.eye(kernel.shape[0]))
    return dict(
        projection=projection.astype(np.float32),
        eigenvalues=eigenvalues.tolist(),
        tau=tau,
        effective_rank=float(np.sum(eigenvalues) ** 2 /
                             max(np.sum(eigenvalues ** 2), eps)),
        logdet=float(logdet if sign > 0 else -np.inf))


def calculate_wot_conditioned_zico(
        grad_dict, activation_rates_by_block, eps=1e-8):
    """Calculate equal-block ZiCo after NASWOT spectral gradient filtering.

    Each logical HeteroWeave block receives one NASWOT filter.  Conv/Linear gradient
    observations from that block are filtered across micro-batches before the
    ZiCo signal-to-noise statistic is evaluated.  Medians prevent parameter
    count and layer count from increasing the score mechanically.
    """
    filters = {}
    diagnostics = {}
    for block_name, rates in activation_rates_by_block.items():
        result = activation_agreement_filter(rates, eps=eps)
        filters[block_name] = result.pop('projection')
        diagnostics[block_name] = result

    module_scores = {}
    grouped = {}
    for module_name, values in grad_dict.items():
        block_name = logical_gradient_group(module_name)
        projection = filters.get(block_name)
        if projection is None:
            continue
        gradients = np.asarray(values, dtype=np.float32)
        if (gradients.ndim != 2 or gradients.shape[0] < 2 or
                gradients.shape[0] != projection.shape[0]):
            continue
        filtered = projection @ gradients
        mean_abs = np.mean(np.abs(filtered), axis=0)
        std = np.std(filtered, axis=0)
        valid = (np.isfinite(mean_abs) & np.isfinite(std) &
                 (mean_abs > eps))
        if not np.any(valid):
            continue
        log_ratio = np.log((mean_abs[valid] + eps) / (std[valid] + eps))
        score = float(np.median(log_ratio))
        if np.isfinite(score):
            module_scores[module_name] = score
            grouped.setdefault(block_name, []).append(score)

    block_scores = {
        name: float(np.median(values))
        for name, values in grouped.items() if values}
    if not block_scores:
        raise RuntimeError('No valid block gradients for WOT-conditioned ZiCo.')
    # Every replacement block has equal influence, independent of its width,
    # depth, parameter count, or number of internal Conv/Linear modules.
    score = float(np.mean(list(block_scores.values())))
    return dict(
        score=score,
        block_scores=block_scores,
        module_scores=module_scores,
        kernel_diagnostics=diagnostics)


class ChannelActivationMonitor:
    """Collect NASWOT-style on/off rates aligned with gradient output axes."""

    def __init__(self, model):
        self.rates = {}
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.handles.append(
                    module.register_forward_hook(self._hook(name, module)))

    def _hook(self, name, module):
        def capture(_module, _inputs, output):
            tensor = first_feature_tensor(output)
            if tensor is None or tensor.ndim < 2:
                return
            tensor = tensor.detach()
            channel_axis = 1 if isinstance(module, nn.Conv2d) else tensor.ndim - 1
            reduce_axes = tuple(
                axis for axis in range(tensor.ndim) if axis != channel_axis)
            rate = (tensor > 0).float().mean(dim=reduce_axes)
            self.rates.setdefault(name, []).append(
                rate.float().cpu().numpy().astype(np.float32))
        return capture

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def calculate_channel_wot_zico(grad_dict, activation_rates, eps=1e-8):
    """ZiCo whose gradient dimensions are gated by aligned activation diversity.

    For each output channel, ZiCo's gradient signal-to-noise ratio is multiplied
    by the normalized binary activation entropy of that exact channel.  The
    module and logical-block medians remove width/depth/parameter-count terms.
    """
    module_scores = {}
    grouped = {}
    channel_diagnostics = {}
    for gradient_name, values in grad_dict.items():
        activation_name = gradient_name
        if activation_name.startswith('classifier.'):
            activation_name = activation_name[len('classifier.'):]
        rates = activation_rates.get(activation_name)
        if rates is None:
            continue
        gradients = np.asarray(values, dtype=np.float32)
        rates = np.asarray(rates, dtype=np.float32)
        if (gradients.ndim != 2 or rates.ndim != 2 or
                gradients.shape[0] != rates.shape[0] or
                gradients.shape[0] < 2):
            continue
        channels = rates.shape[1]
        if channels < 1 or gradients.shape[1] % channels:
            continue
        gradients = gradients.reshape(gradients.shape[0], channels, -1)
        mean_abs = np.mean(np.abs(gradients), axis=0)
        std = np.std(gradients, axis=0)
        snr = (mean_abs + eps) / (std + eps)
        channel_snr = np.median(snr, axis=1)
        probability = np.clip(np.mean(rates, axis=0), eps, 1.0 - eps)
        entropy = -(probability * np.log(probability) +
                    (1.0 - probability) * np.log(1.0 - probability))
        entropy /= np.log(2.0)
        channel_score = np.log((channel_snr + eps) * (entropy + eps))
        valid = np.isfinite(channel_score)
        if not np.any(valid):
            continue
        score = float(np.median(channel_score[valid]))
        group = logical_gradient_group(gradient_name)
        module_scores[gradient_name] = score
        grouped.setdefault(group, []).append(score)
        channel_diagnostics[gradient_name] = dict(
            median_snr=float(np.median(channel_snr[valid])),
            median_activation_entropy=float(np.median(entropy[valid])),
            active_channels=int(np.sum(valid)),
            channels=int(channels))

    block_scores = {
        name: float(np.median(values))
        for name, values in grouped.items() if values}
    if not block_scores:
        raise RuntimeError('No aligned gradients for channel WOT-ZiCo.')
    return dict(
        score=float(np.mean(list(block_scores.values()))),
        block_scores=block_scores,
        module_scores=module_scores,
        channel_diagnostics=channel_diagnostics)


def calculate_sample_wot_zico(grad_dict, activation_codes_by_block, eps=1e-6):
    """Join ZiCo and a true sample-level NASWOT kernel inside each block.

    The joint block score is the log of a product: the block's robust gradient
    SNR times its normalized NASWOT kernel volume.  Consequently no fitted or
    hand-selected linear weight is introduced.
    """
    grouped = {}
    for module_name, values in grad_dict.items():
        block_name = logical_gradient_group(module_name)
        if not block_name.startswith('block_'):
            continue
        gradients = np.asarray(values, dtype=np.float32)
        if gradients.ndim != 2 or gradients.shape[0] < 2:
            continue
        mean_abs = np.mean(np.abs(gradients), axis=0)
        std = np.std(gradients, axis=0)
        valid = np.isfinite(mean_abs) & np.isfinite(std) & (mean_abs > eps)
        if not np.any(valid):
            continue
        module_score = float(np.median(np.log(
            (mean_abs[valid] + eps) / (std[valid] + eps))))
        if np.isfinite(module_score):
            grouped.setdefault(block_name, []).append(module_score)

    zico_blocks = {
        name: float(np.median(values))
        for name, values in grouped.items() if values}
    joint_blocks = {}
    kernel_diagnostics = {}
    for block_name, batches in activation_codes_by_block.items():
        zico_score = zico_blocks.get(block_name)
        if zico_score is None:
            continue
        codes = np.concatenate(batches, axis=0).astype(np.float64)
        kernel = codes @ codes.T + (1.0 - codes) @ (1.0 - codes).T
        diagonal = np.sqrt(np.maximum(np.diag(kernel), eps))
        kernel = kernel / np.outer(diagonal, diagonal)
        kernel = (kernel + kernel.T) * 0.5
        regularized = kernel + eps * np.eye(kernel.shape[0])
        sign, logdet = np.linalg.slogdet(regularized)
        if sign <= 0 or not np.isfinite(logdet):
            continue
        log_volume_per_sample = float(logdet / kernel.shape[0])
        joint_blocks[block_name] = zico_score + log_volume_per_sample
        eigenvalues = np.maximum(np.linalg.eigvalsh(kernel), 0.0)
        kernel_diagnostics[block_name] = dict(
            zico=zico_score,
            logdet=float(logdet),
            log_volume_per_sample=log_volume_per_sample,
            effective_rank=float(np.sum(eigenvalues) ** 2 /
                                 max(np.sum(eigenvalues ** 2), eps)),
            samples=int(kernel.shape[0]))
    if not joint_blocks:
        raise RuntimeError('No valid blocks for sample WOT-ZiCo.')
    return dict(
        score=float(np.mean(list(joint_blocks.values()))),
        block_scores=joint_blocks,
        raw_block_zico=zico_blocks,
        kernel_diagnostics=kernel_diagnostics)


def calculate_signed_block_zico(grad_dict, eps=1e-8):
    """ZiCo with an internal GradSign-style direction-consistency gate.

    Original ZiCo rewards parameters whose gradient magnitude is consistently
    large relative to its cross-batch standard deviation.  That can still favor
    wide/deep architectures because many noisy parameters can accumulate score.
    Here each parameter's ZiCo ratio is multiplied before aggregation by the
    agreement of gradient signs across micro-batches:

        mean(abs(g)) / std(g) * abs(mean(sign(g)))

    A second, softer variant uses ``0.5 + 0.5 * abs(mean(sign(g)))`` to avoid
    making the score too discrete when only a few micro-batches are available.
    Module medians and equal block aggregation remove explicit width/depth
    accumulation.
    """
    strict_grouped = {}
    soft_grouped = {}
    sign_grouped = {}
    module_scores = {}
    for module_name, values in grad_dict.items():
        block_name = logical_gradient_group(module_name)
        if not block_name.startswith('block_'):
            continue
        gradients = np.asarray(values, dtype=np.float32)
        if gradients.ndim != 2 or gradients.shape[0] < 2:
            continue
        mean_abs = np.mean(np.abs(gradients), axis=0)
        std = np.std(gradients, axis=0)
        sign_consistency = np.abs(np.mean(np.sign(gradients), axis=0))
        valid = (np.isfinite(mean_abs) & np.isfinite(std) &
                 np.isfinite(sign_consistency) & (mean_abs > eps))
        if not np.any(valid):
            continue
        base = (mean_abs[valid] + eps) / (std[valid] + eps)
        strict = np.log(base * (sign_consistency[valid] + eps))
        soft_gate = 0.5 + 0.5 * sign_consistency[valid]
        soft = np.log(base * soft_gate)
        strict = strict[np.isfinite(strict)]
        soft = soft[np.isfinite(soft)]
        if strict.size == 0 or soft.size == 0:
            continue
        strict_score = float(np.median(strict))
        soft_score = float(np.median(soft))
        sign_score = float(np.median(sign_consistency[valid]))
        strict_grouped.setdefault(block_name, []).append(strict_score)
        soft_grouped.setdefault(block_name, []).append(soft_score)
        sign_grouped.setdefault(block_name, []).append(sign_score)
        module_scores[module_name] = dict(
            strict=strict_score,
            soft=soft_score,
            sign_consistency=sign_score)

    strict_blocks = {
        name: float(np.median(values))
        for name, values in strict_grouped.items() if values}
    soft_blocks = {
        name: float(np.median(values))
        for name, values in soft_grouped.items() if values}
    sign_blocks = {
        name: float(np.median(values))
        for name, values in sign_grouped.items() if values}
    if not strict_blocks or not soft_blocks:
        raise RuntimeError('No valid block gradients for signed block ZiCo.')

    def summarize(block_scores):
        values = np.asarray(list(block_scores.values()), dtype=np.float64)
        return dict(
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            lower_half_mean=float(np.mean(np.sort(values)[:max(
                1, int(np.ceil(values.size / 2.0)))])),
            minimum=float(np.min(values)),
            dispersion=float(np.std(values)))

    strict_summary = summarize(strict_blocks)
    soft_summary = summarize(soft_blocks)
    sign_summary = summarize(sign_blocks)
    return dict(
        signed_block_zico=strict_summary['mean'],
        signed_block_zico_median=strict_summary['median'],
        signed_block_zico_lowerq=strict_summary['lower_half_mean'],
        signed_block_zico_min=strict_summary['minimum'],
        signed_block_zico_dispersion=strict_summary['dispersion'],
        soft_signed_block_zico=soft_summary['mean'],
        soft_signed_block_zico_median=soft_summary['median'],
        soft_signed_block_zico_lowerq=soft_summary['lower_half_mean'],
        soft_signed_block_zico_min=soft_summary['minimum'],
        soft_signed_block_zico_dispersion=soft_summary['dispersion'],
        grad_sign_block=sign_summary['mean'],
        grad_sign_block_median=sign_summary['median'],
        strict_block_scores=strict_blocks,
        soft_block_scores=soft_blocks,
        sign_block_scores=sign_blocks,
        module_scores=module_scores)
