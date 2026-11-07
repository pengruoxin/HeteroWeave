# Pretrained weights

Pretrained checkpoints are obtained from their original public providers and are
not redistributed in this repository. The links below identify the source weight
files, not the final trained HeteroWeave checkpoints. Download them under the
providers' original terms.

## Paper architectures and source mapping

Complete layer boundaries, participating branches, interface adapters, and fusion
settings are specified by the configurations linked in the README's
[Paper model inventory](../README.md#paper-model-inventory).

| Reported model | Pretrained block sources |
|---|---|
| HeteroWeave, 30M/6G | `swsl_resnext50_32x4d`, `resnet50`, `vit_small_patch16_224` |
| HeteroWeave-P, 10M/3G | `resnet50`, `regnet_y_1_6gf`, `regnet_y_16gf`, `regnet_y_800mf` |
| HeteroWeave-E, 10M/3G | `resnet50`, `regnet_y_1_6gf`, `regnet_y_800mf` |

The DeRy medium-scale reference additionally uses the OpenMMLab Swin-S source.
The inventory below also covers other ImageNet source weights present in the
three-seed proxy audit. It is a checkpoint inventory, not a claim that every
listed source appears in every final architecture or that it enumerates the
entire possible search library.

## Exact ImageNet checkpoint files and original download links

For the torchvision models, these are the legacy weights selected by
`pretrained=True` in the target torchvision 0.11.3 environment. In particular,
ResNet-50 uses **`resnet50-0676ba61.pth`**, not a newer V2 checkpoint.
The RegNet and MobileNet torchvision entries are ImageNet-1K weights.
The ResNeXt source is the Facebook semi-weakly-supervised checkpoint; the
AugReg ViT files identify ImageNet-21K pretraining followed by ImageNet-1K
fine-tuning. Do not replace these with differently pretrained versions sharing
the same architecture name.

| Source model identifier | Loader | Original-provider checkpoint download |
|---|---|---|
| `vit_base_patch16_224` | vendored timm (`mytimm`) | [`B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz`](https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz) |
| `vit_small_patch16_224` | vendored timm (`mytimm`) | [`S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz`](https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz) |
| `mobilenet_v3_large` | torchvision (`pytorch`) | [`mobilenet_v3_large-8738ca79.pth`](https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth) |
| `mobilenet_v3_small` | torchvision (`pytorch`) | [`mobilenet_v3_small-047dcff4.pth`](https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth) |
| `mobilenetv3_large_100` | timm; use the matching legacy definition | [`mobilenetv3_large_100_ra-f55367f5.pth`](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth) |
| `regnet_y_16gf` | torchvision (`pytorch`) | [`regnet_y_16gf-9e6ed7dd.pth`](https://download.pytorch.org/models/regnet_y_16gf-9e6ed7dd.pth) |
| `regnet_y_1_6gf` | torchvision (`pytorch`) | [`regnet_y_1_6gf-b11a554e.pth`](https://download.pytorch.org/models/regnet_y_1_6gf-b11a554e.pth) |
| `regnet_y_3_2gf` | torchvision (`pytorch`) | [`regnet_y_3_2gf-b5a9779c.pth`](https://download.pytorch.org/models/regnet_y_3_2gf-b5a9779c.pth) |
| `regnet_y_800mf` | torchvision (`pytorch`) | [`regnet_y_800mf-1b27b58c.pth`](https://download.pytorch.org/models/regnet_y_800mf-1b27b58c.pth) |
| `resnet101` | torchvision (`pytorch`) | [`resnet101-63fe2227.pth`](https://download.pytorch.org/models/resnet101-63fe2227.pth) |
| `resnet18` | torchvision (`pytorch`) | [`resnet18-f37072fd.pth`](https://download.pytorch.org/models/resnet18-f37072fd.pth) |
| `resnet34` | torchvision (`pytorch`) | [`resnet34-b627a593.pth`](https://download.pytorch.org/models/resnet34-b627a593.pth) |
| `resnet50` | torchvision (`pytorch`) | [`resnet50-0676ba61.pth`](https://download.pytorch.org/models/resnet50-0676ba61.pth) |
| `swsl_resnext50_32x4d` | vendored timm (`mytimm`) | [`semi_weakly_supervised_resnext50_32x4-72679e44.pth`](https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext50_32x4-72679e44.pth) |
| `swin_small_patch4_window7_224` | OpenMMLab (`mmcv`) | [`swin_small_patch4_window7_224-cc7a01c9.pth`](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_small_patch4_window7_224-cc7a01c9.pth) |

The vendored URLs can be inspected in
[`third_package/timm/models/resnet.py`](../third_package/timm/models/resnet.py)
and [`third_package/timm/models/vision_transformer.py`](../third_package/timm/models/vision_transformer.py).
OpenMMLab model configuration and URL mappings are in
[`blocklize/block_meta.py`](../blocklize/block_meta.py).

## Download and verify

Full SHA-256 values for all files above are provided in
[`PRETRAINED_SHA256SUMS.txt`](PRETRAINED_SHA256SUMS.txt). They were recorded from
the source-weight files used in the three-seed proxy evaluation audit. They
identify those audited files; they do not independently certify every historical
training run. The short hexadecimal suffix in a checkpoint filename is not a
substitute for the full checksum.

Example for the ResNet-50 source (run from the repository root):

```bash
mkdir -p pretrained
curl --fail --location --retry 3 \
  https://download.pytorch.org/models/resnet50-0676ba61.pth \
  --output pretrained/resnet50-0676ba61.pth
```

Download the other required files from the table, preserving their filenames.
After downloading every file in the checksum list, verify from that directory:

```bash
cd pretrained
# macOS:
shasum -a 256 -c ../docs/PRETRAINED_SHA256SUMS.txt
# Linux alternative:
sha256sum -c ../docs/PRETRAINED_SHA256SUMS.txt
```

If only a subset was downloaded, verify the corresponding lines in the checksum
file; missing unrelated files are not checksum mismatches. No full checkpoint
download is needed for architecture-only inspection.

## Loading and caches

The actual loader is
[`mmcls_addon/models/backbones/heteroweave.py`](../mmcls_addon/models/backbones/heteroweave.py).
By default it initializes the source model with pretrained weights and then
extracts the consecutive layers specified by the architecture configuration.

- **torchvision (`pytorch`):** calls the torchvision model with
  `pretrained=True`. With torchvision 0.11.3 this selects the legacy file above.
- **Vendored timm (`mytimm`):** calls `third_package.timm.create_model` with
  `pretrained=True`. Its bundled model definitions select the exact URL;
  ViT AugReg `.npz` files use the bundled ViT weight converter.
- **External timm (`timm`):** uses the installed timm model registry. Since
  the external timm version is not pinned in `requirements-core.txt`, do not
  assume its current default weights equal the legacy weights above. Record
  the installed version and use an explicit matching checkpoint when needed.
- **OpenMMLab (`mmcv`):** uses `MODEL_STATS[model_name]['load_from']`, with
  `module.backbone.` and `backbone.` prefixes removed when loading.

Torch Hub-backed downloads use the usual `$TORCH_HOME/hub/checkpoints` cache
(or `~/.cache/torch/hub/checkpoints` when `TORCH_HOME` is unset). For offline
runs, prepopulate the cache with the exact filenames required by each loader.
The OpenMMLab URL resolver additionally looks in
`HETEROWEAVE_PRETRAINED_DIR`. This environment variable is **not a universal
override** for torchvision or timm downloads.

```bash
export TORCH_HOME=/path/to/torch_cache
export HETEROWEAVE_PRETRAINED_DIR=/path/to/pretrained
export HETEROWEAVE_BLOCK_CACHE_DIR=/path/to/block_cache
```

For explicit local source files, the block loader also accepts `ckp_path`.
Its `pytorch` branch expects a directly loadable state dictionary; the `mytimm`
branch supports `.pth` and `.npz`, with optional prefix handling for `.pth`.
Use the existing loader signature and configuration decoding when supplying an
override; simply setting `HETEROWEAVE_PRETRAINED_DIR` does not inject `ckp_path`
into an architecture configuration.

Do not reuse extracted-block caches across changes to source weights at the same
path. Use a fresh block-cache directory (and a fresh process) after changing
weights, or set `HETEROWEAVE_DISABLE_BLOCK_CACHE=1`.

Set `HETEROWEAVE_DISABLE_PRETRAINED=1` only for architecture construction or
resource inspection. Scores or training produced without pretrained blocks are
not comparable to the paper experiments.

## Environment versions

The ImageNet target environment is Python 3.8, PyTorch 1.10.2, torchvision
0.11.3, CUDA 11.3, MMCV 1.4.8, and MMClassification 0.25.0; see
[`requirements-core.txt`](../requirements-core.txt). The legacy timm source (version 0.5.5) is
vendored under `third_package/timm`. External timm, NumPy, and the auxiliary
packages are currently not fully pinned; record their installed versions with
your reproduction results rather than treating this list as a complete lockfile.

## Auxiliary-task weights and loading boundary

The auxiliary tasks use different backbones and checkpoint formats from the
ImageNet reassembly sources. In particular, DINO ViT-B/16 for retrieval and
DINO ResNet-50 for segmentation are not interchangeable.

| Task | Source and released loading interface |
|---|---|
| Retrieval | CLIP model and processor: [`openai/clip-vit-base-patch16`](https://huggingface.co/openai/clip-vit-base-patch16), loaded with `from_pretrained`; MAE and DINO ViT-B/16 files are supplied as the first two arguments to `scripts/train_retrieval_models.sh` |
| Segmentation | ImageNet ResNet-50 anchor plus ResNet-50 SSL donor files supplied to `scripts/train_segmentation_models.sh`; the reported compositions use DINO and SwAV, while the launcher also accepts BYOL |
| Detection | The Faster R-CNN baseline and a ResNet-50 BYOL donor; the donor file is supplied to `scripts/train_detection_models.sh` |

Retrieval donor loading constructs `vit_base_patch16_224` with `pretrained=False`
and loads the supplied state dictionary, rejecting missing backbone keys or
unexpected keys. Segmentation unwraps `state_dict`, strips known prefixes, and
checks for the ResNet `conv1.weight`; it initializes the segmentation head
separately. These supplied donor files are not replaced by a timm default.

**Remaining provenance limitation:** the current release does not yet provide
verified original download URLs and full checksums for every auxiliary donor,
nor an immutable Hugging Face revision for CLIP. Local donor filenames alone
do not identify exact pretrained tensors. The ImageNet checksum list above
must not be presented as covering these auxiliary files. Exact auxiliary
checkpoint provenance and package versions must be retained with a reproduction.

See the README [training commands](../README.md#training) and
[reproducibility guide](REPRODUCIBILITY.md) for the task entry points.
