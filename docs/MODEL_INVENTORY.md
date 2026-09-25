# Model inventory

This file maps every model displayed in the paper to its released definition.

## ImageNet shared protocol

- `configs/imagenet/heteroweave_p_10m3g_100e.py`
- `configs/imagenet/heteroweave_e_10m3g_100e.py`
- `configs/imagenet/heteroweave_main_100e.py`
- `configs/imagenet/dery_10m3g_100e.py`
- `configs/imagenet/dery_baseline_100e.py`
- `configs/baselines/imagenet/model_stitching_10m3g_100e.py`
- `configs/baselines/imagenet/model_stitching_30m6g_100e.py`
- `configs/baselines/imagenet/side_tuning_10m3g_100e.py`
- `configs/baselines/imagenet/side_tuning_30m6g_100e.py`
- `configs/baselines/imagenet/snnet_3ti9s_100e.py`
- `configs/baselines/imagenet/snnet_9ti3s_100e.py`

## Complete-network references

- `configs/references/imagenet/regnet_y_800mf_100e.py`
- `configs/references/imagenet/mobilenet_v3_large_100e.py`
- `configs/references/imagenet/regnet_y_3_2gf_100e.py`
- `configs/references/imagenet/resnet50_100e.py`
- `configs/references/imagenet/swin_tiny_100e.py`

## Auxiliary tasks

- Retrieval: `configs/retrieval/paper_models.py`
- Semantic segmentation: `configs/segmentation/paper_models.py`
- Object detection: `configs/detection/paper_models.py`

The auxiliary definitions include the task baseline, every HeteroWeave row in
the main table, and the adapter, source, and capacity controls discussed in the
paper.

## Analysis controls

- DeRy serial path: `configs/imagenet/dery_baseline_100e.py`
- Capacity-matched serial extension: `configs/analysis/capacity_serial_100e.py`
- Capacity-matched parallel composition: `configs/analysis/capacity_parallel_100e.py`
