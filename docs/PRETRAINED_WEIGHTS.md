# Pretrained weights

No checkpoint is included in this repository. The reusable blocks are loaded
from public model providers through torchvision, timm, the vendored legacy
timm definitions, or OpenMMLab configurations.

Before a full run:

1. download the upstream weights under their original terms;
2. verify the model name and checkpoint checksum;
3. place files in a local directory excluded by `.gitignore`;
4. set `DERY_PRETRAINED_DIR` to that directory, or provide the checkpoint path
   in the candidate configuration;
5. keep the same weight source for every candidate in a comparison.

Useful environment variables:

```bash
export DERY_PRETRAINED_DIR=/path/to/pretrained
export DERY_BLOCK_CACHE_DIR=/path/to/block_cache
```

Set `DERY_DISABLE_PRETRAINED=1` only for architecture construction or resource
inspection. Scores and training results produced without pretrained blocks are
not comparable to the paper experiments.

## Released final ImageNet model

`configs/imagenet/heteroweave_main_100e.py` uses three public sources:

- `swsl_resnext50_32x4d`: the timm/Facebook semi-weakly-supervised ResNeXt-50
  checkpoint ending in `semi_weakly_supervised_resnext50_32x4-72679e44.pth`;
- `resnet50`: torchvision's ImageNet-1K ResNet-50 weights;
- `vit_small_patch16_224`: the timm AugReg ViT-S/16 source identified by
  `S_16-i21k-300ep`, `imagenet2012-steps_20k`, and `res_224`.

The DeRy reference additionally uses the OpenMMLab Swin-S checkpoint ending in
`cc7a01c9.pth`.
