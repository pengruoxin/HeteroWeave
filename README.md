# HeteroWeave

Implementation of **HeteroWeave: Variable-Cardinality Composition in Pretrained
Model Reassembly**, with code for architecture search, training, and evaluation
on ImageNet classification, retrieval, segmentation, and detection.

## Overview

HeteroWeave builds a model by selecting one or more pretrained blocks at each
aligned position. Adapters reconcile heterogeneous interfaces, and the selected
block outputs are fused. A training-free activation-pattern score, **CLAS**,
estimates candidate performance during **NSGA-III** search, which jointly
maximizes CLAS and minimizes parameter count and FLOPs. Selected architectures
are then fully trained.

See the [method-to-code guide](docs/METHOD_TO_CODE.md) for implementation details,
including the composition interfaces, CLAS definition, and search-time versus
full-training fusion protocols.

## Documentation

| To… | Start here |
|---|---|
| Reproduce search and training | [Reproducibility guide](docs/REPRODUCIBILITY.md) |
| Locate paper models and baselines | [Complete model inventory](docs/MODEL_INVENTORY.md) |
| Obtain pretrained source weights | [Checkpoint sources and loading instructions](docs/PRETRAINED_WEIGHTS.md) |
| Prepare datasets | [Dataset guide](docs/DATASETS.md) |
| Inspect the method implementation | [Method-to-code guide](docs/METHOD_TO_CODE.md) |
| Verify full-training proxy results | [20-architecture evidence package](evidence/clas_100e_20/README.md) |

## Getting started

Run commands from the repository root.

### Environment

The main ImageNet experiments use Python 3.8, PyTorch 1.10.2, torchvision 0.11.3,
CUDA 11.3, MMCV 1.4.8, and MMClassification 0.25.0.

```bash
conda create -n heteroweave python=3.8 -y
conda activate heteroweave
pip install -r requirements-core.txt
pip install mmcv-full==1.4.8 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
```

Retrieval, segmentation, and detection additionally require
[requirements-auxiliary.txt](requirements-auxiliary.txt). Datasets and pretrained
weights are obtained separately through the guides above.

### Check the implementation

Run the core checks without downloading data or weights:

```bash
bash scripts/reproduce_core_checks.sh
```

Run a small NSGA-III search using the built-in toy objective:

```bash
python tools/search_nsga3_multiobj.py --toy \
  --population-size 12 --generations 2 --output-dir work_dirs/toy_search
```

### Train an ImageNet model

After preparing ImageNet paths and the required pretrained source weights:

```bash
bash scripts/train_imagenet_models.sh 8 heteroweave
```

This launcher trains HeteroWeave with eight GPUs and seeds 11, 23, and 47.
See the [ImageNet instructions](docs/REPRODUCIBILITY.md#imagenet) for dataset
path overrides and the [auxiliary-task instructions](docs/REPRODUCIBILITY.md#auxiliary-tasks)
for retrieval, segmentation, and detection.

## Paper model inventory

The links below provide the complete architecture configurations for the paper
models and baselines. ImageNet configurations specify source-block selections,
interfaces, and training settings. Auxiliary-task configurations define each
candidate by its ID, insertion position, source components, and fusion settings.

Pretrained source-model identifiers, download links, loading instructions, and
available SHA-256 checksums are listed in the
[checkpoint guide](docs/PRETRAINED_WEIGHTS.md). Dataset preparation and execution
commands are in the [reproducibility guide](docs/REPRODUCIBILITY.md).

### ImageNet classification

| Model | Search resource budget | Complete configuration |
|---|---|---|
| HeteroWeave-P | 10M / 3G | [heteroweave_p_10m3g_100e.py](configs/imagenet/heteroweave_p_10m3g_100e.py) |
| HeteroWeave-E | 10M / 3G | [heteroweave_e_10m3g_100e.py](configs/imagenet/heteroweave_e_10m3g_100e.py) |
| HeteroWeave | 30M / 6G | [heteroweave_main_100e.py](configs/imagenet/heteroweave_main_100e.py) |
| DeRy | 10M / 3G | [dery_10m3g_100e.py](configs/imagenet/dery_10m3g_100e.py) |
| DeRy | 30M / 6G | [dery_baseline_100e.py](configs/imagenet/dery_baseline_100e.py) |
| Model Stitching | 10M / 3G | [model_stitching_10m3g_100e.py](configs/baselines/imagenet/model_stitching_10m3g_100e.py) |
| Model Stitching | 30M / 6G | [model_stitching_30m6g_100e.py](configs/baselines/imagenet/model_stitching_30m6g_100e.py) |
| Side-Tuning | 10M / 3G | [side_tuning_10m3g_100e.py](configs/baselines/imagenet/side_tuning_10m3g_100e.py) |
| Side-Tuning | 30M / 6G | [side_tuning_30m6g_100e.py](configs/baselines/imagenet/side_tuning_30m6g_100e.py) |
| SN-Net (3Ti–9S) | — | [snnet_3ti9s_100e.py](configs/baselines/imagenet/snnet_3ti9s_100e.py) |
| SN-Net (9Ti–3S) | — | [snnet_9ti3s_100e.py](configs/baselines/imagenet/snnet_9ti3s_100e.py) |
| RegNetY-800MF | — | [regnet_y_800mf_100e.py](configs/references/imagenet/regnet_y_800mf_100e.py) |
| MobileNetV3-Large | — | [mobilenet_v3_large_100e.py](configs/references/imagenet/mobilenet_v3_large_100e.py) |
| RegNetY-3.2GF | — | [regnet_y_3_2gf_100e.py](configs/references/imagenet/regnet_y_3_2gf_100e.py) |
| ResNet-50 | — | [resnet50_100e.py](configs/references/imagenet/resnet50_100e.py) |
| Swin-T | — | [swin_tiny_100e.py](configs/references/imagenet/swin_tiny_100e.py) |

A dash indicates that no search-budget label is assigned to that reference row;
it does not denote zero resource use.

### Retrieval, segmentation, and detection

Each linked task configuration contains all candidate IDs listed for that task.

| Task | Model / control | Candidate ID | Configuration |
|---|---|---|---|
| Retrieval | CLIP baseline | `clip` | [retrieval/paper_models.py](configs/retrieval/paper_models.py) |
| Retrieval | HeteroWeave R1 | `heteroweave_r1` | [retrieval/paper_models.py](configs/retrieval/paper_models.py) |
| Retrieval | HeteroWeave R2 | `heteroweave_r2` | [retrieval/paper_models.py](configs/retrieval/paper_models.py) |
| Retrieval | HeteroWeave R3 | `heteroweave_r3` | [retrieval/paper_models.py](configs/retrieval/paper_models.py) |
| Retrieval | Wider adapter control | `wider_adapter` | [retrieval/paper_models.py](configs/retrieval/paper_models.py) |
| Retrieval | Parallel adapters control | `parallel_adapters` | [retrieval/paper_models.py](configs/retrieval/paper_models.py) |
| Segmentation | ResNet-50 baseline | `resnet50` | [segmentation/paper_models.py](configs/segmentation/paper_models.py) |
| Segmentation | HeteroWeave S1 (capacity control) | `heteroweave_s1` | [segmentation/paper_models.py](configs/segmentation/paper_models.py) |
| Segmentation | HeteroWeave S2 (source control) | `heteroweave_s2` | [segmentation/paper_models.py](configs/segmentation/paper_models.py) |
| Segmentation | HeteroWeave S3 | `heteroweave_s3` | [segmentation/paper_models.py](configs/segmentation/paper_models.py) |
| Segmentation | HeteroWeave S4 | `heteroweave_s4` | [segmentation/paper_models.py](configs/segmentation/paper_models.py) |
| Detection | ResNet-50 baseline | `resnet50` | [detection/paper_models.py](configs/detection/paper_models.py) |
| Detection | HeteroWeave D1 | `heteroweave_d1` | [detection/paper_models.py](configs/detection/paper_models.py) |
| Detection | HeteroWeave D2 | `heteroweave_d2` | [detection/paper_models.py](configs/detection/paper_models.py) |
| Detection | HeteroWeave D3 | `heteroweave_d3` | [detection/paper_models.py](configs/detection/paper_models.py) |

### Capacity-matched analysis

| Model / control | Complete configuration |
|---|---|
| DeRy serial path | [dery_baseline_100e.py](configs/imagenet/dery_baseline_100e.py) |
| Capacity-matched serial extension | [capacity_serial_100e.py](configs/analysis/capacity_serial_100e.py) |
| Capacity-matched parallel composition | [capacity_parallel_100e.py](configs/analysis/capacity_parallel_100e.py) |

## Full-training proxy evidence

The [evidence package](evidence/clas_100e_20/README.md) links **20 reassembled
architectures (A01–A20)** to their completed 100-epoch ImageNet-1K training logs
and per-seed proxy scores, including the CLAS square-root ablation.

- [Architecture configurations and final accuracies](evidence/clas_100e_20/architectures.csv)
- [Training runs and log paths](evidence/clas_100e_20/training_runs.csv)
- [Per-seed proxy scores](evidence/clas_100e_20/proxy/scores.csv)
- [Complete correlation table](evidence/clas_100e_20/table.md)

Verify the recorded measurements and regenerate the table with Python 3.8 or
newer; this command needs no third-party packages, GPU, dataset, or checkpoints:

```bash
python3 evidence/clas_100e_20/recompute.py
```

The evidence README documents the cohort, scoring protocol, target aggregation,
and anonymization. The general training launcher above and the archived evidence
runs have separately documented seed lists.

## Attribution

This implementation builds on DeRy, MMClassification, torchvision, timm, and
pymoo. The legacy `third_package/timm` definitions support heterogeneous block
loading. See [third-party notices](docs/THIRD_PARTY.md) for attribution and
redistribution notes.
