"""Resource-matched structural-reuse baselines for ImageNet experiments.

These models intentionally implement small, auditable instances of two prior
ideas rather than reproducing their original multi-task experimental suites:

* model stitching: a pretrained prefix is connected to a pretrained suffix
  through a learned 1x1 transformation;
* side tuning: frozen/trainable pretrained feature paths are projected to a
  common space and combined by a learned scalar gate.
"""

import torch
import torch.nn as nn
from torchvision import models
import third_package.timm as mytimm

from mmcls.models.builder import BACKBONES
from .base_backbone import BaseBackbone


def _torchvision_model(name, pretrained):
    constructor = getattr(models, name)
    return constructor(pretrained=pretrained)


def _set_trainable(module, trainable):
    for parameter in module.parameters():
        parameter.requires_grad = trainable


class _ResNetPrefix(nn.Module):

    def __init__(self, model, stop_stage):
        super().__init__()
        layers = [
            model.conv1, model.bn1, model.relu, model.maxpool, model.layer1
        ]
        if stop_stage >= 2:
            layers.append(model.layer2)
        if stop_stage >= 3:
            layers.append(model.layer3)
        if stop_stage >= 4:
            layers.append(model.layer4)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class _ResNetSuffix(nn.Module):

    def __init__(self, model, start_stage):
        super().__init__()
        self.layers = nn.Sequential(*[
            getattr(model, f'layer{stage}')
            for stage in range(start_stage, 5)
        ])

    def forward(self, x):
        return self.layers(x)


@BACKBONES.register_module()
class ModelStitchingBaseline(BaseBackbone):
    """Two fixed resource-matched model-stitching structures."""

    def __init__(self,
                 variant,
                 pretrained=True,
                 freeze_sources=False,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.variant = variant

        if variant == 'low_r18_to_mbv3l':
            source = _torchvision_model('resnet18', pretrained)
            target = _torchvision_model('mobilenet_v3_large', pretrained)
            # R18 layer3: 256 x 14 x 14. MBV3-L block 7 expects a
            # 40-channel 14x14 tensor and supplies the remaining suffix.
            self.prefix = _ResNetPrefix(source, stop_stage=3)
            self.stitch = nn.Sequential(
                nn.Conv2d(256, 40, kernel_size=1, bias=False),
                nn.BatchNorm2d(40),
                nn.ReLU(inplace=True),
            )
            self.suffix = nn.Sequential(*list(target.features.children())[7:])
            self.out_channels = 960
        elif variant == 'high_r18_to_r50':
            source = _torchvision_model('resnet18', pretrained)
            target = _torchvision_model('resnet50', pretrained)
            # Both tensors are 28x28 at this boundary.
            self.prefix = _ResNetPrefix(source, stop_stage=2)
            self.stitch = nn.Sequential(
                nn.Conv2d(128, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
            self.suffix = _ResNetSuffix(target, start_stage=3)
            self.out_channels = 2048
        else:
            raise ValueError(f'Unknown stitching variant: {variant}')

        if freeze_sources:
            _set_trainable(self.prefix, False)
            _set_trainable(self.suffix, False)

    def forward(self, x):
        x = self.prefix(x)
        x = self.stitch(x)
        x = self.suffix(x)
        return (x, )


class _FeaturePath(nn.Module):

    def __init__(self, model_name, pretrained):
        super().__init__()
        model = _torchvision_model(model_name, pretrained)
        if model_name.startswith('resnet'):
            self.features = nn.Sequential(
                model.conv1, model.bn1, model.relu, model.maxpool,
                model.layer1, model.layer2, model.layer3, model.layer4)
            self.out_channels = model.fc.in_features
        elif model_name.startswith('mobilenet_v3'):
            self.features = model.features
            self.out_channels = model.classifier[0].in_features
        else:
            raise ValueError(f'Unsupported side-tuning path: {model_name}')
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        return self.pool(x).flatten(1)


@BACKBONES.register_module()
class SideTuningBaseline(BaseBackbone):
    """Additive two-path feature side tuning with a learned scalar gate."""

    def __init__(self,
                 variant,
                 fusion_channels=512,
                 pretrained=True,
                 freeze_base=True,
                 init_cfg=None):
        super().__init__(init_cfg)
        if variant == 'low_mbv3s_mbv3l':
            base_name, side_name = 'mobilenet_v3_small', 'mobilenet_v3_large'
        elif variant == 'high_r34_mbv3s':
            base_name, side_name = 'resnet34', 'mobilenet_v3_small'
        else:
            raise ValueError(f'Unknown side-tuning variant: {variant}')

        self.base_path = _FeaturePath(base_name, pretrained)
        self.side_path = _FeaturePath(side_name, pretrained)
        self.base_projection = nn.Linear(
            self.base_path.out_channels, fusion_channels)
        self.side_projection = nn.Linear(
            self.side_path.out_channels, fusion_channels)
        self.gate_logit = nn.Parameter(torch.zeros(()))
        self.out_channels = fusion_channels

        if freeze_base:
            _set_trainable(self.base_path, False)

    def train(self, mode=True):
        super().train(mode)
        if not any(parameter.requires_grad
                   for parameter in self.base_path.parameters()):
            self.base_path.eval()
        return self

    def forward(self, x):
        base = self.base_projection(self.base_path(x))
        side = self.side_projection(self.side_path(x))
        alpha = torch.sigmoid(self.gate_logit)
        fused = alpha * base + (1.0 - alpha) * side
        return (fused[:, :, None, None], )


@BACKBONES.register_module()
class SNNetDeiTBaseline(BaseBackbone):
    """Fixed DeiT-Ti/DeiT-S stitch used by the two SN-Net controls."""

    def __init__(self,
                 tiny_blocks,
                 small_blocks,
                 pretrained=True,
                 init_cfg=None):
        super().__init__(init_cfg)
        if tiny_blocks + small_blocks != 12:
            raise ValueError('SN-Net DeiT controls must contain 12 blocks')

        tiny = mytimm.create_model(
            'vit_tiny_patch16_224', pretrained=pretrained)
        small = mytimm.create_model(
            'vit_small_patch16_224', pretrained=pretrained)

        self.patch_embed = tiny.patch_embed
        self.cls_token = tiny.cls_token
        self.pos_embed = tiny.pos_embed
        self.pos_drop = tiny.pos_drop
        self.tiny_prefix = nn.Sequential(
            *list(tiny.blocks.children())[:tiny_blocks])
        self.stitch = nn.Linear(tiny.embed_dim, small.embed_dim)
        self.small_suffix = nn.Sequential(
            *list(small.blocks.children())[12 - small_blocks:])
        self.norm = small.norm
        self.out_channels = small.embed_dim

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.tiny_prefix(x)
        x = self.stitch(x)
        x = self.small_suffix(x)
        x = self.norm(x)
        return (x[:, 0, :, None, None], )
