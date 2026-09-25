# HeteroWeave

Lightweight implementation package for **HeteroWeave: Variable-Cardinality
Composition in Pretrained Model Reassembly**. This repository is organized for
method inspection and reproduction. It intentionally excludes the paper,
figures, tables, datasets, checkpoints, logs, and archived experiment outputs.

## What is included

- variable-cardinality block composition and heterogeneous interfaces;
- fixed mean fusion during architecture search, plus mean/gated fusion for full training;
- the capacity-control implementation;
- CLAS, the training-free activation-pattern proxy for candidate performance;
- explicit separation between the paper-level performance objective and its search-time CLAS proxy;
- three-objective Pareto utilities and NSGA-III search code;
- all ImageNet models and baselines appearing in the paper;
- explicit R1--R3, S1--S4, and D1--D3 auxiliary-task definitions;
- training entry points for every reported final model;
- retrieval, segmentation, and detection implementations used in the
  auxiliary evaluations;
- small CPU tests for the core mathematical behavior.

## Repository map

| Paper concept | Main implementation |
|---|---|
| Heterogeneous block extraction | `blocklize/block_meta.py`, `mmcls_addon/models/utils/feature_extraction.py` |
| Interfaces and composition | `mmcls_addon/models/backbones/dery.py` |
| Structural baselines | `mmcls_addon/models/backbones/structural_baselines.py` |
| CLAS reference definition | `heteroweave/clas.py` |
| Full proxy hooks | `tools/search_nsga3_multiobj.py`, task-specific scripts in `tools/` |
| Pareto dominance | `heteroweave/pareto.py`, `simlarity/multi_objective.py` |
| NSGA-III search | `tools/search_nsga3_multiobj.py`, `tools/search_nsga3_dery_refine.py` |
| ImageNet architectures | `configs/imagenet/` |
| ImageNet structural baselines | `configs/baselines/imagenet/` |
| Complete-network references | `configs/references/imagenet/` |
| Capacity controls | `configs/analysis/` |
| Retrieval | `tools/run_clip_heteroweave_*.py` |
| Segmentation | `tools/run_deeplab_heteroweave_search.py` |
| Detection | `tools/evaluate_detection_formal_proxy.py`, `tools/train_detection_*.py` |

The inherited internal directory name `simlarity` is intentionally preserved
because existing imports depend on it. The inherited internal name `DeRy` remains in low-level backbone code.
Paper-facing terminology follows HeteroWeave and CLAS throughout the release;
implementation mappings are documented in `docs/METHOD_TO_CODE.md`.

## Fusion protocol

The fusion operator is **not** an architecture variable during search. All
multi-branch search candidates use parameter-free arithmetic mean fusion so
that the search focuses on component identity and cardinality rather than on
additional trainable fusion parameters.

After an architecture has been selected, full training supports two fusion
strategies for a composite block:

- `operator='sum'`: parameter-free arithmetic mean fusion;
- `operator='gate'`: learnable softmax-gated fusion.

The name `sum` is retained for historical compatibility; in the released
implementation it computes `stacked.mean(dim=0)`, not an arithmetic sum. The
operator used for each reported final architecture is specified by its
ImageNet configuration.

| Stage / released model | Fusion used |
|---|---|
| Architecture search | Fixed arithmetic mean (`sum`) |
| HeteroWeave (30M/6G) full training | Arithmetic mean (`sum`) |
| HeteroWeave-P (10M/3G) full training | Learnable softmax gate (`gate`) |
| HeteroWeave-E (10M/3G) full training | Not applicable (single component per position) |

## Environment

The main ImageNet code was developed with Python 3.8, PyTorch 1.10.2,
torchvision 0.11.3, CUDA 11.3, MMCV 1.4.8, and MMCLASsification 0.25.0.
That stack is the reproducibility target for the MMCLASsification pipeline.

```bash
conda create -n heteroweave python=3.8 -y
conda activate heteroweave
pip install -r requirements-core.txt
pip install mmcv-full==1.4.8 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
```

Install `requirements-auxiliary.txt` only for the retrieval, segmentation, or
detection entry points. See `docs/REPRODUCIBILITY.md` for environment notes and
command templates.

## Quick checks

The dependency-light checks do not download data or weights:

```bash
bash scripts/reproduce_core_checks.sh
```

The NSGA-III implementation also provides a toy mode:

```bash
python tools/search_nsga3_multiobj.py --toy \
  --population-size 12 --generations 2 --output-dir work_dirs/toy_search
```

## Paper model inventory

| Task | Paper row | Released definition |
|---|---|---|
| ImageNet 10M/3G | HeteroWeave-P | `configs/imagenet/heteroweave_p_10m3g_100e.py` |
| ImageNet 10M/3G | HeteroWeave-E | `configs/imagenet/heteroweave_e_10m3g_100e.py` |
| ImageNet 30M/6G | HeteroWeave | `configs/imagenet/heteroweave_main_100e.py` |
| ImageNet 10M/3G | DeRy | `configs/imagenet/dery_10m3g_100e.py` |
| ImageNet 30M/6G | DeRy | `configs/imagenet/dery_baseline_100e.py` |
| ImageNet | Model Stitching, Side-Tuning, SN-Net | `configs/baselines/imagenet/` |
| ImageNet | RegNetY, MobileNetV3, ResNet-50, Swin-T | `configs/references/imagenet/` |
| Retrieval | CLIP, R1, R2, R3, adapter controls | `configs/retrieval/paper_models.py` |
| Segmentation | ResNet-50, S1, S2, S3, S4 | `configs/segmentation/paper_models.py` |
| Detection | ResNet-50, D1, D2, D3 | `configs/detection/paper_models.py` |
| Analysis | DeRy, serial extension, parallel composition | `configs/analysis/` |

## Training

After preparing ImageNet and the required public pretrained weights, train the
released final model with seeds 11, 23, and 47:

```bash
bash scripts/train_imagenet_models.sh 8 heteroweave
```

Use `all` as the second argument to run every ImageNet configuration, or pass
one key from `scripts/train_imagenet_models.sh`. Auxiliary tasks have separate
entry points:

```bash
bash scripts/train_retrieval_models.sh MAE.pth DINO.pth work_dirs/retrieval

python tools/run_deeplab_heteroweave_search.py protocol \
  --data /path/to/oxford-iiit-pet --output work_dirs/segmentation_protocol.json
bash scripts/train_segmentation_models.sh \
  /path/to/oxford-iiit-pet work_dirs/segmentation_protocol.json \
  RESNET50.pth BYOL.pth DINO.pth SWAV.pth work_dirs/segmentation

python tools/freeze_detection_protocol.py \
  --voc-root /path/to/VOC --output work_dirs/detection_protocol.json
bash scripts/train_detection_models.sh \
  /path/to/VOC BYOL.pth work_dirs/detection_protocol.json work_dirs/detection
```

Dataset paths can be changed through the MMCLASsification configuration
override mechanism. No private dataset or checkpoint is distributed here.

## Attribution

The implementation builds on DeRy, MMCLASsification, torchvision, timm, and
pymoo. The vendored legacy `third_package/timm` tree is retained because the
heterogeneous block loader relies on its model definitions. See
`docs/THIRD_PARTY.md` before redistribution.
