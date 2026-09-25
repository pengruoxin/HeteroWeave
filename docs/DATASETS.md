# Datasets

Datasets are not redistributed. Obtain them from their official or public
sources and retain their original licenses.

## ImageNet-1K

- Source: <https://www.image-net.org/challenges/LSVRC/2012/>
- Split: standard ILSVRC 2012 train and validation sets.
- Input: 224 x 224.
- Training: random resized crop, horizontal flip, RandAugment, random erasing,
  and ImageNet normalization.
- Validation: resize the short side to 256, center crop to 224, and normalize.
- Expected layout:

```text
data/imagenet/
  train/<class>/*.JPEG
  val/<class>/*.JPEG
  meta/val.txt
```

## Flickr8K retrieval

- Public source: <https://huggingface.co/datasets/jxie/flickr8k>
- Training uses the first 6,000 public training images.
- Validation uses 1,000 images and all five captions per image.
- Images and text use the CLIP ViT-B/16 processor; maximum text length is 77.

## Oxford-IIIT Pet segmentation

- Source: <https://www.robots.ox.ac.uk/~vgg/data/pets/>
- Task: three-class trimap segmentation (pet, background, boundary).
- The entry point accepts explicit image, annotation, and split paths. Preserve
  the frozen validation identifiers used for architecture selection.

## PASCAL VOC 2007 detection

- Source: <http://host.robots.ox.ac.uk/pascal/VOC/voc2007/>
- Task: Faster R-CNN detection.
- Training and validation identifiers must be supplied through the protocol
  manifest; the public test set is not used for architecture selection.

For every run, record the dataset revision or download date, exact split file,
preprocessing configuration, random seed, and checkpoint-selection rule.
