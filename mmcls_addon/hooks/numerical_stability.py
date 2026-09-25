import torch
import torch.distributed as dist

from mmcv.runner import HOOKS, Hook

from ..models.backbones.heteroweave import HorizontalCompositeBlock, NeuralAdapter


def _iter_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


@HOOKS.register_module()
class NumericalStabilityHook(Hook):
    """Stop all workers before a non-finite loss reaches the optimizer."""

    def __init__(self, monitor_activations=True):
        self.monitor_activations = monitor_activations
        self.handles = []
        self.first_nonfinite = ''

    def _record_output(self, name):
        def hook(_module, _inputs, output):
            if self.first_nonfinite:
                return
            for tensor in _iter_tensors(output):
                if tensor.numel() and not bool(torch.isfinite(tensor).all().item()):
                    self.first_nonfinite = name
                    return
        return hook

    def before_run(self, runner):
        if not self.monitor_activations:
            return

        for name, module in runner.model.named_modules():
            if isinstance(module, NeuralAdapter):
                self.handles.append(module.register_forward_hook(
                    self._record_output(name)))
            if isinstance(module, HorizontalCompositeBlock):
                for index, branch in enumerate(module.blocks):
                    label = f'{name}.branch[{index}].block'
                    self.handles.append(branch.register_forward_hook(
                        self._record_output(label)))
                self.handles.append(module.register_forward_hook(
                    self._record_output(name)))

    def before_train_iter(self, runner):
        self.first_nonfinite = ''

    def after_train_iter(self, runner):
        loss = runner.outputs.get('loss')
        local_bad = bool(self.first_nonfinite)
        if torch.is_tensor(loss):
            local_bad = local_bad or not bool(torch.isfinite(loss).all().item())

        device = loss.device if torch.is_tensor(loss) else torch.device('cuda')
        bad_flag = torch.tensor(int(local_bad), device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(bad_flag, op=dist.ReduceOp.MAX)

        if int(bad_flag.item()) == 0:
            return

        location = self.first_nonfinite or 'loss/another distributed worker'
        runner.logger.error(
            'NumericalStabilityHook stopped training at epoch=%d iter=%d; '
            'first non-finite location: %s',
            runner.epoch + 1, runner.inner_iter + 1, location)
        raise FloatingPointError(
            f'Non-finite value detected before optimizer step: {location}')

    def after_run(self, runner):
        for handle in self.handles:
            handle.remove()
        self.handles = []
