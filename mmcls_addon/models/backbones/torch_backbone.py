from collections import OrderedDict

import torch

from mmcls.models.builder import BACKBONES
from ..builder import BACKBONES as ADDON_BACKBONES
from .base_backbone import BaseBackbone
from torchvision import models

@ADDON_BACKBONES.register_module(force=True)
@BACKBONES.register_module(force=True)
class TORCHBackbone(BaseBackbone):
    """Wrapper to use torchvision backbones inside MMClassification configs.

    Args:
        model_name (str): Name of torchvision model to instantiate.
        pretrained (bool): Load pretrained weights if True.
        checkpoint_path (str): Path of checkpoint to load after
            model is initialized.
        in_channels (int): Number of input image channels. Default: 3.
        freeze_model (bool): Freeze the wrapped backbone and keep it in eval.
        init_cfg (dict, optional): Initialization config dict
        **kwargs: Other torchvision model specific arguments.
    """

    def __init__(
        self,
        model_name,
        pretrained=False,
        checkpoint_path='',
        in_channels=3,
        freeze_model=False,
        init_cfg=None,
        **kwargs,
    ):
        super(TORCHBackbone, self).__init__(init_cfg)
        self.model_name = model_name
        self.freeze_model = freeze_model
        self.torch_model = getattr(models, model_name)(
            pretrained=pretrained, **kwargs)
        self._reset_classifier()
        if checkpoint_path:
            state_dict = self._load_state_dict(checkpoint_path)
            keys = self.torch_model.load_state_dict(state_dict, strict=False)
            print('Miss Keys', keys)

        if freeze_model:
            self.torch_model.eval()
            for param in self.torch_model.parameters():
                param.requires_grad = False

        if pretrained or checkpoint_path:
            self._is_init = True

    @staticmethod
    def _load_state_dict(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            elif 'model' in checkpoint:
                checkpoint = checkpoint['model']
        state_dict = OrderedDict()
        for key, value in checkpoint.items():
            if key.startswith('module.'):
                key = key[7:]
            state_dict[key] = value
        return state_dict

    def _reset_classifier(self):
        for attr in ('fc', 'classifier', 'head'):
            if hasattr(self.torch_model, attr):
                setattr(self.torch_model, attr, torch.nn.Identity())

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_model:
            self.torch_model.eval()
        return self

    def forward(self, x):
        if self.model_name.startswith('eff'):
            features = self.torch_model.features(x)
            return (features, )
        elif self.model_name.startswith('regnet'):
            x = self.torch_model.stem(x)
            x = self.torch_model.trunk_output(x)
            return (x, )
        elif self.model_name.startswith(('resnet', 'resnext', 'wide_resnet')):
            x = self.torch_model.conv1(x)
            x = self.torch_model.bn1(x)
            x = self.torch_model.relu(x)
            x = self.torch_model.maxpool(x)
            x = self.torch_model.layer1(x)
            x = self.torch_model.layer2(x)
            x = self.torch_model.layer3(x)
            x = self.torch_model.layer4(x)
            return (x, )
        raise NotImplementedError(f'Unsupported torchvision model: {self.model_name}')
