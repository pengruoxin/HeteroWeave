# Reproducibility guide

## Scope

This lightweight release supports method inspection, search, and training of
the ImageNet, retrieval, segmentation, and detection models reported in the
paper. Datasets and pretrained weights are obtained from their public sources.

## Determinism

Record Python, NumPy, and PyTorch seeds for every run. Clas comparisons must
use the same image panel, preprocessing, initialization boundary, and evaluated
positions for all candidates. Full-training comparisons should report all
seeds used rather than selecting the best seed.

## Search protocol

1. Build the candidate component inventory and record every checkpoint source.
2. Encode a nonempty selection at every aligned position, within its maximum
   cardinality.
3. Evaluate Clas on one fixed image panel.
4. count parameters and FLOPs using the same input resolution and accounting
   boundary;
5. optimize `[maximize Clas, minimize parameters, minimize FLOPs]`;
6. compute the strict nondominated set before choosing operating points;
7. train only representatives selected by the declared rule.

The canonical protocol is recorded in
`configs/search/imagenet_canonical.yaml`. Search scripts preserve the
historical option name `layer_swap_sqrt` for Clas. Position 0 admits one
component; positions 1--3 admit one or two components. The fusion operator is
not searched: all multi-branch candidates use fixed, parameter-free arithmetic
mean fusion during architecture search (`operator='sum'` in the historical
code interface).

After search, the selected component identities and cardinalities are fixed and
the final architecture is fully trained. At this stage the released
implementation supports either arithmetic mean fusion (`sum`) or learnable
softmax-gated fusion (`gate`). Thus, search-time mean fusion does not imply that
every final-training configuration must also use mean fusion.

| Released configuration | Full-training fusion |
|---|---|
| `heteroweave_main_100e.py` | Arithmetic mean (`sum`) |
| `heteroweave_p_10m3g_100e.py` | Learnable softmax gate (`gate`) |
| `heteroweave_e_10m3g_100e.py` | Not applicable; no multi-branch position |

## ImageNet

```bash
PYTHONPATH="$PWD" python tools/dist_train.py \
  configs/imagenet/heteroweave_main_100e.py 8 \
  --cfg-options data.train.data_prefix=/path/to/imagenet/train \
  data.val.data_prefix=/path/to/imagenet/val
```

The archived recipe uses eight GPUs with 128 images per GPU. If hardware forces
a different global batch size, adjust the learning-rate schedule explicitly
and report the change.

The ImageNet launcher fixes seeds 11, 23, and 47 and accepts every model key
listed in `scripts/train_imagenet_models.sh`:

```bash
bash scripts/train_imagenet_models.sh 8 heteroweave
```

## Auxiliary tasks

The paper-model launchers are:

```bash
bash scripts/train_retrieval_models.sh
bash scripts/train_segmentation_models.sh
bash scripts/train_detection_models.sh
```

Detection entry points expect a protocol manifest that defines the candidate
architecture and frozen split. Do not substitute an output table for that
manifest. Retrieval may download Flickr8K and public model weights through
Hugging Face unless an offline cache is configured.

## Resource accounting

Report whether resources are total or added relative to the task backbone.
Within one Pareto calculation, all candidates must share the same accounting
boundary. Search-time mean fusion adds no parameters; its elementwise functional
operations are not included by the inherited MMCV module counter. A final model
trained with gated fusion includes its learnable gate parameters in the model
parameter count. Interface parameters
belong to the candidate and are counted. Total parameters include stem,
selected blocks, interfaces, neck, and head. Feature-extraction FLOPs use a
3 x 224 x 224 input and exclude the classifier head.

`tools/count_params_flops.py` prints these boundaries for a released
configuration. If a block is instantiated twice with separate weights, count
both instances; if weights are shared, count parameter storage once and report
the repeated computation in FLOPs. The released final ImageNet architecture
does not duplicate a component instance.
