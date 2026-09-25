import math
import copy
import hashlib
import json
import os
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from mmcv.runner import load_checkpoint
from urllib.parse import urlparse

from torchvision import models

import third_package.timm as mytimm
from mmcls.models.builder import BACKBONES, build_backbone

try:
    from ..utils.feature_extraction import (create_sub_network,
                                            create_sub_network_transformer)
except:
    pass
import mmcv

from blocklize import MODEL_STATS
from .base_backbone import BaseBackbone


TOKEN_BLOCK_TYPES = {'vit', 'swin'}
_SUBNET_CACHE = {}


def resolve_checkpoint_path(ckp_path):
    if not isinstance(ckp_path, str):
        return ckp_path
    if not ckp_path.startswith(('http://', 'https://')):
        return ckp_path

    filename = os.path.basename(urlparse(ckp_path).path)
    if not filename:
        return ckp_path

    search_dirs = []
    pretrained_dir = os.environ.get('DERY_PRETRAINED_DIR')
    if pretrained_dir:
        search_dirs.append(pretrained_dir)

    torch_home = os.environ.get('TORCH_HOME')
    if torch_home:
        search_dirs.extend([
            os.path.join(torch_home, 'hub', 'checkpoints'),
            os.path.join(torch_home, 'checkpoints'),
        ])

    home = os.path.expanduser('~')
    search_dirs.extend([
        os.path.join(home, '.cache', 'torch', 'hub', 'checkpoints'),
        os.path.join(home, '.cache', 'torch', 'checkpoints'),
        os.path.join(os.getcwd(), 'pretrained_cache'),
        os.path.join(os.getcwd(), 'checkpoint', 'pretrained'),
    ])

    seen = set()
    for directory in search_dirs:
        if not directory:
            continue
        directory = os.path.abspath(os.path.expanduser(directory))
        if directory in seen:
            continue
        seen.add(directory)
        candidate = os.path.join(directory, filename)
        if os.path.isfile(candidate):
            return candidate
    return ckp_path


def _cache_tuple(value):
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _subnet_cache_key(model_name, block_input, block_output, backend, prefix=None, ckp_path=None):
    return {
        'model_name': model_name,
        'block_input': _cache_tuple(block_input),
        'block_output': _cache_tuple(block_output),
        'backend': backend,
        'prefix': prefix,
        'ckp_path': ckp_path,
    }


def get_subnet_cache_path(cache_dir, model_name, block_input, block_output,
                          backend, prefix=None, ckp_path=None):
    key = _subnet_cache_key(
        model_name, block_input, block_output, backend, prefix, ckp_path)
    payload = json.dumps(key, sort_keys=True, default=str)
    digest = hashlib.sha1(payload.encode('utf-8')).hexdigest()
    safe_name = model_name.replace('/', '_').replace('.', '_')
    return os.path.join(cache_dir, f'{safe_name}_{digest}.pt')


def _load_cached_subnet(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _save_cached_subnet(path, subnet):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f'{path}.tmp.{os.getpid()}'
    torch.save(subnet.cpu(), tmp_path)
    os.replace(tmp_path, path)


def canonical_feature_type(block_type):
    return 'cnn' if block_type == 'cnn' else 'vit'


def build_reassembly_block(block_cfg):
    if isinstance(block_cfg, dict):
        model = block_cfg['model_name']
        block = network_to_module_subnet(**block_cfg)
    elif isinstance(block_cfg, list) and len(block_cfg) == 4:
        model, block_input, block_output, backend = block_cfg
        block = network_to_module_subnet(
            model, block_input, block_output, backend)
    elif isinstance(block_cfg, list) and len(block_cfg) == 5:
        model, block_input, block_output, backend, ckp_path = block_cfg
        block = network_to_module_subnet(
            model, block_input, block_output, backend, ckp_path=ckp_path)
    else:
        raise AssertionError('block_cfg type not supported')
    return block, MODEL_STATS[model]['type']


def forward_reassembly_block(block, block_type, x, out_shape):
    if getattr(block, 'is_composite_block', False):
        return block(x, out_shape)
    if block_type == 'swin':
        x, out_shape = block(x, out_shape)
    elif block_type == 'cnn':
        x = block(x)
        if isinstance(x, dict):
            x = list(x.values())[0]
        out_shape = (x.shape[2], x.shape[3])
    elif block_type == 'vit':
        x = block(x)
    else:
        raise AssertionError(f'Unsupported block type: {block_type}')
    return x, out_shape


def network_to_module_subnet(model_name, block_input, block_output, backend, prefix=None, ckp_path=None):
    cache_key = (
        model_name,
        _cache_tuple(block_input),
        _cache_tuple(block_output),
        backend,
        prefix,
        ckp_path,
    )
    use_cache = os.environ.get('DERY_DISABLE_BLOCK_CACHE', '0') != '1'
    disk_cache_dir = os.environ.get('DERY_BLOCK_CACHE_DIR')
    disk_cache_path = None
    if use_cache and disk_cache_dir:
        disk_cache_path = get_subnet_cache_path(
            disk_cache_dir, model_name, block_input, block_output,
            backend, prefix, ckp_path)
        if os.path.exists(disk_cache_path):
            subnet = _load_cached_subnet(disk_cache_path)
            _SUBNET_CACHE[cache_key] = subnet.cpu()
            return copy.deepcopy(_SUBNET_CACHE[cache_key])

    if use_cache and cache_key in _SUBNET_CACHE:
        return copy.deepcopy(_SUBNET_CACHE[cache_key])

    # Architecture-only utilities (for example FLOPs/parameter counting) do
    # not need pretrained tensors.  Keeping this opt-in avoids network access
    # and checkpoint I/O while preserving the original training behaviour.
    load_pretrained = os.environ.get('DERY_DISABLE_PRETRAINED', '0') != '1'

    # print(model_name, block_input, block_output, backend)
    if backend == 'timm':
        if ckp_path is not None:
            backbone = timm.create_model(
                model_name, pretrained=False, scriptable=True)
            if os.path.isfile(ckp_path):
                state_dict = torch.load(ckp_path, map_location='cpu')
                print(f'Loading checkpoint from {ckp_path}')
                miss_keys = backbone.load_state_dict(state_dict, strict=False)
                print(miss_keys)
            else:
                print(f'{ckp_path} does not exists')
        else:
            backbone = timm.create_model(
                model_name, pretrained=load_pretrained, scriptable=True)
    elif backend == 'mytimm':
        if ckp_path is not None:
            backbone = mytimm.create_model(
                model_name, pretrained=False, scriptable=True)
            if os.path.isfile(ckp_path) and ckp_path.endswith('pth'):
                print(f'Loading checkpoint from {ckp_path}')
                state_dict = torch.load(ckp_path, map_location='cpu')
                if 'state_dict' in state_dict.keys():
                    state_dict = state_dict['state_dict']
                keys = list(state_dict.keys())
                for key in keys:
                    if key in ['head.weight', 'head.bias', 'fc.weight', 'fc.bias']:
                        print(f'removing {key}')
                        del state_dict[key]

                if prefix is not None:
                    new_state_dict = OrderedDict()
                    if prefix.endswith('.'):
                        pass
                    else:
                        prefix += '.'
                    for k, v in state_dict.items():
                        # strip `module.` prefix
                        name = k[len(prefix):] if k.startswith(prefix) else k
                        new_state_dict[name] = v
                    state_dict = new_state_dict

                miss_keys = backbone.load_state_dict(state_dict, strict=False)
                print(miss_keys)
            elif os.path.isfile(ckp_path) and ckp_path.endswith('npz'):
                mytimm.models.vision_transformer._load_weights(
                    backbone, ckp_path)
            else:
                print(f'{ckp_path} does not exists')
        else:
            backbone = mytimm.create_model(
                model_name, pretrained=load_pretrained, scriptable=False)
    elif backend == 'mmcv':
        config = MODEL_STATS[model_name]['cfg']
        cfg = mmcv.Config.fromfile(config)
        backbone = build_backbone(cfg.model.backbone)
        if ckp_path is not None:
            ckp_path = resolve_checkpoint_path(ckp_path)
            if os.path.isfile(ckp_path):
                print(f'Loading checkpoint from {ckp_path}')
                load_checkpoint(backbone, ckp_path, revise_keys=[(r'^module\.backbone\.', ''),
                                                                 (r'^backbone\.', '')])
            else:
                print(f'{ckp_path} does not exists')
        else:
            if load_pretrained:
                ckp_path = resolve_checkpoint_path(MODEL_STATS[model_name]['load_from'])
                load_checkpoint(backbone, ckp_path, revise_keys=[(r'^module\.backbone\.', ''),
                                                                 (r'^backbone\.', '')])

    elif backend == 'pytorch':
        if ckp_path is not None:
            backbone = getattr(models, model_name)(pretrained=False)
            if os.path.isfile(ckp_path):
                state_dict = torch.load(ckp_path, map_location='cpu')
                print(f'Loading checkpoint from {ckp_path}')
                miss_keys = backbone.load_state_dict(state_dict, strict=False)
                print(miss_keys)
            else:
                print(f'{ckp_path} does not exists')
        else:
            backbone = getattr(models, model_name)(pretrained=load_pretrained)

    if isinstance(block_input, str):
        block_input = [block_input]
    elif isinstance(block_input, tuple):
        block_input = list(block_input)
    elif isinstance(block_input, list):
        block_input = block_input
    else:
        TypeError('Block input should be a string or tuple or list')

    if isinstance(block_output, str):
        block_output = [block_output]
    elif isinstance(block_output, tuple):
        block_output = list(block_output)
    elif isinstance(block_output, list):
        block_output = block_output
    else:
        TypeError('Block output should be a string or tuple or list')

    if model_name.startswith('swin_') or model_name.startswith('vit'):
        subnet = create_sub_network_transformer(
            backbone, model_name, block_input, block_output)
    else:
        subnet = create_sub_network(backbone, block_input, block_output)
    if use_cache:
        _SUBNET_CACHE[cache_key] = subnet.cpu()
        if disk_cache_path is not None and not os.path.exists(disk_cache_path):
            _save_cached_subnet(disk_cache_path, _SUBNET_CACHE[cache_key])
        return copy.deepcopy(_SUBNET_CACHE[cache_key])
    return subnet


@BACKBONES.register_module()
class DeRy(BaseBackbone):
    """
    """

    def __init__(
        self,
        block_list,
        adapter_list=None,
        base_adapter=None,
        block_fixed=True,
        train_adapters_only=False,
        train_stem=True,
        all_fixed=False,
        base_pool=True,
        base_channels=64,
        in_channels=3,
        out_indices=(3, ),
        hw_ratio=1,
        capacity_control=None,
        init_cfg=None,
        **kwargs,
    ):
        super(DeRy, self).__init__(init_cfg)
        if train_adapters_only and all_fixed:
            raise ValueError(
                'train_adapters_only and all_fixed cannot both be enabled')
        assert isinstance(block_list, list), 'block_list should be a list'
        assert isinstance(adapter_list, list), 'adapter_list should be a list'
        assert len(
            block_list)-1 == len(adapter_list), 'len(block_list)-1 should be len(adapter_list)'
        base_layers = [
            ('conv0', nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3,
                                bias=False)),
            ('norm0', nn.BatchNorm2d(base_channels)),
            ('relu0', nn.ReLU(inplace=True)),
        ]
        if base_pool:
            base_layers.append(
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)))
        self.base = nn.Sequential(OrderedDict(base_layers))

        blocks = []
        block_types = []
        for block_cfg in block_list:
            if isinstance(block_cfg, dict) and block_cfg.get('type') == 'CompositeBlock':
                block = HorizontalCompositeBlock(**block_cfg)
                block_type = block.out_type
            else:
                block, block_type = build_reassembly_block(block_cfg)
            block_types.append(block_type)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.block_types = block_types

        if adapter_list is not None:
            adapters = []
            for adapter_cfg in adapter_list:
                adapters.append(NeuralAdapter(**adapter_cfg)
                                if adapter_cfg is not None else FeatureIdentity())
            self.adapters = nn.ModuleList(adapters)
        else:
            self.adapters = None

        if base_adapter is not None:
            self.base_adapter = NeuralAdapter(**base_adapter)
        else:
            self.base_adapter = None

        self.out_indices = out_indices
        self.hw_ratio = hw_ratio
        self.capacity_control = (
            ResidualCapacityControl(**capacity_control)
            if capacity_control is not None else None)
        self.train_adapters_only = train_adapters_only
        self.train_stem = train_stem

        if block_fixed:
            for param in self.blocks.parameters():
                param.requires_grad = False

        if train_adapters_only:
            self._enable_adapter_only_training()

        if all_fixed:
            for param in self.parameters():
                param.requires_grad = False

    def _enable_adapter_only_training(self):
        """Freeze feature extractors and leave structural adapters trainable.

        ``block_fixed`` predates composite branches.  It freezes the complete
        ``self.blocks`` tree (including branch adapters and gate parameters).
        Adapter-only training freezes imported feature extractors while
        allowing the randomly initialized input stem, top-level adapters,
        branch adapters, and fusion gates to learn.  ``train_stem=False`` is
        available for experiments that initialize the stem separately.
        """
        for param in self.parameters():
            param.requires_grad = False

        if self.train_stem:
            for param in self.base.parameters():
                param.requires_grad = True
        if self.adapters is not None:
            for param in self.adapters.parameters():
                param.requires_grad = True
        if self.base_adapter is not None:
            for param in self.base_adapter.parameters():
                param.requires_grad = True

        for block in self.blocks:
            if not isinstance(block, HorizontalCompositeBlock):
                continue
            for param in block.input_adapters.parameters():
                param.requires_grad = True
            for param in block.output_adapters.parameters():
                param.requires_grad = True
            if hasattr(block, 'gate_logits'):
                block.gate_logits.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if self.train_adapters_only:
            # Frozen BatchNorm/dropout layers must remain in inference mode;
            # requires_grad=False alone does not stop their running state from
            # changing during training.
            if not self.train_stem:
                self.base.eval()
            for block in self.blocks:
                if isinstance(block, HorizontalCompositeBlock):
                    block.blocks.eval()
                else:
                    block.eval()
        return self

    def forward(self, x):
        x = self.base(x)
        out_shape = (x.shape[2], x.shape[3])
        if self.base_adapter is not None:
            x, out_shape = self.base_adapter(x, out_shape)
        outs = []
        for i, block in enumerate(self.blocks):
            if i > 0 and self.adapters is not None:

                x, out_shape = self.adapters[i-1](x, out_shape)

            # print(x.shape)
            # print(out_shape)
            x, out_shape = forward_reassembly_block(
                block, self.block_types[i], x, out_shape)

            if i in self.out_indices:
                out = x
                if self.block_types[i] == 'cnn':
                    if self.capacity_control is not None:
                        out = self.capacity_control(out)
                    outs.append(out)
                else:
                    token_num = out.shape[1]
                    out_channels = out.shape[2]
                    w, h = out_shape

                    if token_num == w * h + 1:
                        out = out[:, 1:, :]
                        token_num -= 1
                    elif w * h != token_num:
                        side = math.isqrt(token_num)
                        if side * side == token_num:
                            w, h = side, side
                    torch._assert(w*h == token_num,
                                  'When VIT to CNN, w x h == token_num')
                    out = out.view(-1, w, h,
                                   out_channels).permute(0, 3, 1, 2)
                    if self.capacity_control is not None:
                        out = self.capacity_control(out)
                    outs.append(out)
        return tuple(outs)


class ResidualCapacityControl(nn.Module):
    """Single-path capacity control used by the branch-mechanism ablation.

    The block deliberately contains no imported feature extractor and no
    parallel branch.  At a 7x7, 768-channel output, ``hidden_channels=610``
    adds 4.28586M parameters and about 0.210G multiply-accumulates, matching
    the efficient branch model within 1% in both parameters and FLOPs.
    """

    def __init__(self, in_channels, hidden_channels, activation='relu'):
        super().__init__()
        if activation == 'relu':
            act = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            act = nn.GELU()
        else:
            raise ValueError(f'Unsupported activation: {activation}')
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            act,
            nn.Conv2d(hidden_channels, hidden_channels, 3,
                      padding=1, bias=False),
            copy.deepcopy(act),
            nn.Conv2d(hidden_channels, in_channels, 1, bias=False),
        )

    def forward(self, x):
        return x + self.body(x)


class NeuralAdapter(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 num_fc=0,
                 num_conv=1,
                 mode='cnn2cnn',
                 stride=1) -> None:
        super().__init__()
        assert (num_fc > 0 and num_conv == 0) or (
            num_fc == 0 and num_conv > 0), \
            'num_fc and num_conv can not be both positive.'

        assert mode in ['cnn2cnn', 'cnn2vit', 'vit2cnn', 'vit2vit'], 'mode is not recognized'
        layers = []
        self.mode = mode
        if num_fc > 0:
            layers.append(nn.LayerNorm(input_channel))
            for i in range(num_fc):
                if i == 0:
                    layers.append(
                        nn.Linear(input_channel, output_channel, bias=False))
                else:
                    layers.append(
                        nn.Linear(output_channel, output_channel, bias=False))
            layers.append(nn.LeakyReLU(0.1, inplace=True))

        elif num_conv > 0:
            layers.append(nn.BatchNorm2d(input_channel))
            for i in range(num_conv):
                if i == 0:
                    layers.append(nn.Conv2d(input_channel, output_channel,
                                            kernel_size=stride, stride=stride,
                                            padding=0, bias=False))
                else:
                    layers.append(nn.Conv2d(output_channel, output_channel,
                                            kernel_size=1, stride=1,
                                            padding=0, bias=False))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.adapter = nn.Sequential(*layers)

    def forward(self, x, input_shape=None):
        if self.mode == 'cnn2vit':
            # CNN 2 Vsion Transformer(VIT)
            x = self.adapter(x)
            return x.flatten(2).transpose(1, 2), (x.shape[2], x.shape[3])

        elif self.mode == 'cnn2cnn':
            # CNN 2 CNN
            x = self.adapter(x)
            return x, (x.shape[2], x.shape[3])

        elif self.mode == 'vit2cnn':
            # VIT 2 CNN
            out_channels = x.shape[2]
            token_num = x.shape[1]
            w, h = input_shape
            if token_num == w * h + 1:
                x = x[:, 1:, :]
                token_num -= 1
            elif w * h != token_num:
                side = math.isqrt(token_num)
                if side * side == token_num:
                    w, h = side, side
            torch._assert(w*h == token_num,
                          'When VIT to CNN, w x h == token_num')
            
            x = x.view(-1, w, h, out_channels).permute(0, 3, 1, 2)
            x = self.adapter(x)
            return x, (x.shape[2], x.shape[3])

        elif self.mode == 'vit2vit':
            # VIT/Swin 2 VIT/Swin
            return self.adapter(x), input_shape


class FeatureIdentity(nn.Module):
    def forward(self, x, input_shape=None):
        return x, input_shape


class HorizontalCompositeBlock(nn.Module):
    is_composite_block = True

    def __init__(self,
                 branches,
                 operator='sum',
                 out_type='cnn',
                 type=None) -> None:
        super().__init__()
        assert isinstance(branches, list) and len(branches) > 0, \
            'CompositeBlock requires at least one branch'
        assert operator in ['sub', 'sum', 'concat', 'gate'], \
            'CompositeBlock operator is not recognized'
        assert out_type in ['cnn', 'vit'], 'CompositeBlock out_type is not recognized'

        self.operator = operator
        self.out_type = out_type
        self.audit_mode = 'full'
        self.blocks = nn.ModuleList()
        self.input_adapters = nn.ModuleList()
        self.output_adapters = nn.ModuleList()
        self.block_types = []

        for branch_cfg in branches:
            block_cfg = branch_cfg['block']
            block, block_type = build_reassembly_block(block_cfg)
            if branch_cfg.get('randomize', False):
                # Exact-architecture control for testing whether an auxiliary
                # path benefits from inherited representations rather than
                # merely adding parameters and nonlinear computation.
                for module in block.modules():
                    if hasattr(module, 'reset_parameters'):
                        module.reset_parameters()
            self.blocks.append(block)
            self.block_types.append(block_type)

            input_adapter_cfg = branch_cfg.get('input_adapter')
            output_adapter_cfg = branch_cfg.get('output_adapter')
            self.input_adapters.append(
                NeuralAdapter(**input_adapter_cfg)
                if input_adapter_cfg is not None else FeatureIdentity())
            self.output_adapters.append(
                NeuralAdapter(**output_adapter_cfg)
                if output_adapter_cfg is not None else FeatureIdentity())

        if self.operator == 'gate':
            self.gate_logits = nn.Parameter(torch.zeros(len(branches)))

    def _to_target_type(self, x, out_shape):
        if self.out_type == 'cnn':
            if x.dim() == 4:
                return x, (x.shape[2], x.shape[3])

            token_num = x.shape[1]
            out_channels = x.shape[2]
            w, h = out_shape
            if token_num == w * h + 1:
                x = x[:, 1:, :]
                token_num -= 1
            elif w * h != token_num:
                side = math.isqrt(token_num)
                if side * side == token_num:
                    w, h = side, side
            torch._assert(w * h == token_num,
                          'When token features are fused as CNN, w x h must equal token_num')
            x = x.view(-1, w, h, out_channels).permute(0, 3, 1, 2).contiguous()
            return x, (w, h)

        if x.dim() == 3:
            return x, out_shape
        return x.flatten(2).transpose(1, 2), (x.shape[2], x.shape[3])

    def _resize_tokens(self, x, source_shape, target_shape):
        target_tokens = target_shape[0] * target_shape[1]
        if x.shape[1] == target_tokens:
            return x

        if source_shape is not None and source_shape[0] * source_shape[1] == x.shape[1]:
            bsz, _, channels = x.shape
            src_h, src_w = source_shape
            x = x.view(bsz, src_h, src_w, channels).permute(0, 3, 1, 2)
            x = F.interpolate(x, size=target_shape, mode='bilinear', align_corners=False)
            return x.flatten(2).transpose(1, 2)

        return F.interpolate(
            x.transpose(1, 2), size=target_tokens, mode='linear',
            align_corners=False).transpose(1, 2)

    def _align_outputs(self, branch_outputs):
        target_shape = branch_outputs[0][1]
        aligned = []
        for x, out_shape in branch_outputs:
            if self.out_type == 'cnn':
                if out_shape != target_shape:
                    x = F.interpolate(
                        x, size=target_shape, mode='bilinear',
                        align_corners=False)
                aligned.append(x)
            else:
                aligned.append(self._resize_tokens(x, out_shape, target_shape))
        return aligned, target_shape

    def _fuse(self, tensors):
        if self.audit_mode == 'main_only':
            return 0.5 * tensors[0]
        if self.audit_mode == 'branch_only':
            return 0.5 * tensors[1]
        if self.audit_mode == 'shuffle_branch':
            shuffled = list(tensors)
            shuffled[1] = torch.roll(shuffled[1], shifts=1, dims=0)
            tensors = shuffled

        if self.operator == 'sub':
            return tensors[0]

        if self.operator == 'concat':
            cat_dim = 1 if self.out_type == 'cnn' else 2
            return torch.cat(tensors, dim=cat_dim)

        channels = [x.shape[1] if self.out_type == 'cnn' else x.shape[2]
                    for x in tensors]
        torch._assert(
            len(set(channels)) == 1,
            'sum/gate fusion requires all branch channels to be aligned')

        stacked = torch.stack(tensors, dim=0)
        # Historical compatibility: operator='sum' denotes arithmetic mean
        # fusion in this implementation. Search uses this parameter-free mode;
        # selected architectures may use operator='gate' during full training.
        if self.operator == 'gate':
            weights = torch.softmax(self.gate_logits, dim=0)
            view_shape = (-1,) + (1,) * (stacked.dim() - 1)
            return (stacked * weights.view(view_shape)).sum(dim=0)
        return stacked.mean(dim=0)

    def forward(self, x, out_shape=None):
        branch_outputs = []
        for block, block_type, input_adapter, output_adapter in zip(
                self.blocks, self.block_types, self.input_adapters,
                self.output_adapters):
            branch_x, branch_shape = input_adapter(x, out_shape)
            branch_x, branch_shape = forward_reassembly_block(
                block, block_type, branch_x, branch_shape)
            branch_x, branch_shape = output_adapter(branch_x, branch_shape)
            branch_x, branch_shape = self._to_target_type(branch_x, branch_shape)
            branch_outputs.append((branch_x, branch_shape))

        aligned, target_shape = self._align_outputs(branch_outputs)
        return self._fuse(aligned), target_shape
