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

| Model | Resource budget | Configuration |
|---|---|---|
| HeteroWeave-P | 10M / 3G | [heteroweave_p_10m3g_100e.py](configs/imagenet/heteroweave_p_10m3g_100e.py) |
| HeteroWeave-E | 10M / 3G | [heteroweave_e_10m3g_100e.py](configs/imagenet/heteroweave_e_10m3g_100e.py) |
| HeteroWeave | 30M / 6G | [heteroweave_main_100e.py](configs/imagenet/heteroweave_main_100e.py) |

The [complete inventory](docs/MODEL_INVENTORY.md) also lists DeRy, Model
Stitching, Side-Tuning, SN-Net, complete-network references, auxiliary-task
models, and capacity-matched controls. The
[checkpoint guide](docs/PRETRAINED_WEIGHTS.md) records source-model identifiers,
download links, loading instructions, and available SHA-256 checksums.

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
