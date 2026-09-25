"""Zero-training proxy components for heterogeneous HeteroWeave candidates."""

import math

import torch


def first_feature_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = first_feature_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_feature_tensor(item)
            if tensor is not None:
                return tensor
    return None


def feature_matrix(tensor):
    """Convert CNN/token features to [observations, channels]."""
    if tensor.ndim == 4:
        return tensor.permute(0, 2, 3, 1).reshape(-1, tensor.shape[1])
    if tensor.ndim == 3:
        return tensor.reshape(-1, tensor.shape[-1])
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim > 2:
        return tensor.reshape(tensor.shape[0], -1)
    return tensor.reshape(-1, 1)


def normalized_spectral_entropy(tensor, max_vectors=256, eps=1e-12):
    """Normalized covariance-spectrum entropy in [0, 1]."""
    matrix = feature_matrix(tensor.detach()).float()
    if matrix.shape[0] > max_vectors:
        indices = torch.linspace(
            0, matrix.shape[0] - 1, steps=max_vectors,
            device=matrix.device).long()
        matrix = matrix.index_select(0, indices)
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return 0.0
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(matrix)
    eigenvalues = singular_values.square()
    total = eigenvalues.sum()
    if not bool(torch.isfinite(total).item()) or float(total.item()) <= eps:
        return 0.0
    probabilities = eigenvalues / total
    entropy = -(probabilities * torch.log(probabilities.clamp_min(eps))).sum()
    max_entropy = math.log(max(2, min(matrix.shape[0], matrix.shape[1])))
    return max(0.0, min(1.0, float(entropy.item()) / max_entropy))


class BlockFeatureMonitor:
    """Capture primary-block outputs without changing the model forward path."""

    def __init__(self, model):
        backbone = getattr(model, 'backbone', None)
        blocks = getattr(backbone, 'blocks', None)
        if blocks is None:
            raise TypeError('Expected a HeteroWeave classifier with model.backbone.blocks.')
        self.outputs = [None for _ in blocks]
        self.handles = []
        for index, block in enumerate(blocks):
            self.handles.append(block.register_forward_hook(self._hook(index)))

    def _hook(self, index):
        def capture(_module, _inputs, output):
            self.outputs[index] = first_feature_tensor(output)
        return capture

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def block_expressivity(model, images, max_vectors=256):
    """Return block-level entropy, mean expressivity, and progressivity."""
    monitor = BlockFeatureMonitor(model)
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model.extract_feat(images)
        scores = [
            normalized_spectral_entropy(output, max_vectors=max_vectors)
            for output in monitor.outputs if output is not None]
    finally:
        monitor.close()
        model.train(was_training)
    if not scores:
        raise RuntimeError('No HeteroWeave block features were captured.')
    expressivity = sum(scores) / len(scores)
    progressivity = (
        min(scores[index] - scores[index - 1]
            for index in range(1, len(scores)))
        if len(scores) > 1 else 0.0)
    return dict(
        block_entropy=scores,
        expressivity=expressivity,
        progressivity=progressivity)


def empirical_ntk_condition(
        model, images, forward_logits, max_samples=4, eps=1e-6):
    """TE-NAS-style empirical NTK condition using weight gradients."""
    was_training = model.training
    model.eval()
    gradients = []
    try:
        sample_images = images[:max(2, int(max_samples))]
        model.zero_grad()
        logits = forward_logits(model, sample_images)
        for sample_index in range(sample_images.shape[0]):
            model.zero_grad()
            sample_logits = logits[sample_index:sample_index + 1]
            sample_logits.backward(
                torch.ones_like(sample_logits), retain_graph=True)
            parts = [
                parameter.grad.detach().reshape(-1).float()
                for name, parameter in model.named_parameters()
                if 'weight' in name and parameter.grad is not None]
            if parts:
                gradients.append(torch.cat(parts).cpu())
        if len(gradients) < 2:
            raise RuntimeError('Need at least two valid samples for NTK condition.')
        gradient_matrix = torch.stack(gradients, dim=0).double()
        gram = gradient_matrix @ gradient_matrix.t()
        eigenvalues = torch.linalg.eigvalsh(gram)
        min_value = max(float(eigenvalues[0].item()), float(eps))
        max_value = max(float(eigenvalues[-1].item()), float(eps))
        condition = max_value / min_value
        return float(condition) if math.isfinite(condition) else 1e8
    finally:
        model.zero_grad()
        model.train(was_training)

