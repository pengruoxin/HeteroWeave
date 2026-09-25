#!/usr/bin/env python3
"""Faster R-CNN model components used by the paper detection runs."""

import copy
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF


VOC_CLASSES = (
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
    'tvmonitor')
CLASS_TO_ID = {name: index for index, name in enumerate(VOC_CLASSES)}
CHANNELS = {'layer2': 512, 'layer3': 1024, 'layer4': 2048}


def unwrap(path):
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict) and 'state_dict' in obj:
        obj = obj['state_dict']
    return {(key[7:] if key.startswith('module.') else key): value
            for key, value in obj.items()}


class VOCSubset(Dataset):
    def __init__(self, root, indices, train=False, seed=0):
        self.dataset = VOCDetection(
            root, year='2007', image_set='trainval', download=False)
        self.indices = list(indices)
        self.train = train
        self.seed = seed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        image, annotation = self.dataset[self.indices[item]]
        objects = annotation['annotation'].get('object', [])
        if isinstance(objects, dict):
            objects = [objects]
        boxes, labels, difficult = [], [], []
        for obj in objects:
            box = obj['bndbox']
            boxes.append([
                float(box['xmin']) - 1, float(box['ymin']) - 1,
                float(box['xmax']) - 1, float(box['ymax']) - 1])
            labels.append(CLASS_TO_ID[obj['name']])
            difficult.append(int(obj.get('difficult', 0)))
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'difficult': torch.tensor(difficult, dtype=torch.bool),
            'image_id': torch.tensor([self.indices[item]]),
        }
        if self.train and random.Random(
                self.seed + self.indices[item]).random() < 0.5:
            image = TF.hflip(image)
            width = image.width
            old = target['boxes'].clone()
            target['boxes'][:, 0] = width - old[:, 2]
            target['boxes'][:, 2] = width - old[:, 0]
        return TF.to_tensor(image), target


def collate(batch):
    return tuple(zip(*batch))


class CollaborativeStage(nn.Module):
    def __init__(self, blocks, channels, gate, stat_align=False):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        for block in self.blocks:
            for parameter in block.parameters():
                parameter.requires_grad_(False)
        self.scale = nn.ParameterList([
            nn.Parameter(torch.ones(channels)) for _ in self.blocks[1:]])
        self.bias = nn.ParameterList([
            nn.Parameter(torch.zeros(channels)) for _ in self.blocks[1:]])
        self.register_buffer('gate', torch.tensor(float(gate)))
        self.stat_align = stat_align

    def forward(self, inputs):
        output = self.blocks[0](inputs)
        for block, scale, bias in zip(
                self.blocks[1:], self.scale, self.bias):
            source = block(inputs)
            if self.stat_align:
                source_mean = source.mean((2, 3), keepdim=True)
                source_std = source.var(
                    (2, 3), keepdim=True, unbiased=False).add(1e-6).sqrt()
                output_mean = output.detach().mean((2, 3), keepdim=True)
                output_std = output.detach().var(
                    (2, 3), keepdim=True, unbiased=False).add(1e-6).sqrt()
                source = ((source - source_mean) / source_std * output_std +
                          output_mean)
            output = output + self.gate * (
                source * scale[None, :, None, None] +
                bias[None, :, None, None])
        return output


def detector_template():
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        min_size=384, max_size=640)
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        features, len(VOC_CLASSES))
    return model


def build_model(base_state, source_state, position=None, gate=0.0,
                stat_align=False):
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None, weights_backbone=None, min_size=384, max_size=640,
        num_classes=len(VOC_CLASSES))
    model.load_state_dict(base_state)
    collaboration = None
    if position:
        states = (source_state if isinstance(source_state, (list, tuple))
                  else [source_state])
        source_blocks = []
        for state in states:
            source = resnet50(weights=None)
            source.load_state_dict(state, strict=False)
            source_blocks.append(copy.deepcopy(getattr(source, position)))
        anchor_block = getattr(model.backbone.body, position)
        collaboration = CollaborativeStage(
            [copy.deepcopy(anchor_block)] + source_blocks,
            CHANNELS[position], gate, stat_align)
        setattr(model.backbone.body, position, collaboration)
    for parameter in model.backbone.body.parameters():
        parameter.requires_grad_(False)
    if collaboration:
        for parameter in collaboration.scale:
            parameter.requires_grad_(True)
        for parameter in collaboration.bias:
            parameter.requires_grad_(True)
    return model


def train(model, loader, device, steps, learning_rate):
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        parameters, lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    iterator = iter(loader)
    model.train()
    history = []
    for step in range(1, steps + 1):
        try:
            images, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, targets = next(iterator)
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()
                    if key != 'difficult'} for target in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            loss = sum(model(images, targets).values())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step in {1, 50, 100, steps}:
            history.append([step, float(loss.detach())])
    return history


def box_iou(box, boxes):
    left_top = torch.maximum(box[:2], boxes[:, :2])
    right_bottom = torch.minimum(box[2:], boxes[:, 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(1)
    area1 = (box[2:] - box[:2]).clamp_min(0).prod()
    area2 = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(1)
    return intersection / (area1 + area2 - intersection).clamp_min(1e-8)


def average_precision(recall, precision):
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(precision) - 1, 0, -1):
        precision[index - 1] = max(precision[index - 1], precision[index])
    changes = np.where(recall[1:] != recall[:-1])[0]
    return np.sum(
        (recall[changes + 1] - recall[changes]) * precision[changes + 1])


@torch.inference_mode()
def evaluate_ap50(model, loader, device):
    model.eval()
    ground_truth = defaultdict(dict)
    detections = defaultdict(list)
    for images, targets in loader:
        outputs = model([image.to(device) for image in images])
        for target, output in zip(targets, outputs):
            image_id = int(target['image_id'].item())
            for class_id in range(1, len(VOC_CLASSES)):
                selected = target['labels'] == class_id
                ground_truth[class_id][image_id] = {
                    'boxes': target['boxes'][selected],
                    'difficult': target['difficult'][selected],
                    'matched': torch.zeros(int(selected.sum()), dtype=torch.bool),
                }
            keep = output['scores'].cpu() >= 0.05
            for box, label, score in zip(
                    output['boxes'].cpu()[keep], output['labels'].cpu()[keep],
                    output['scores'].cpu()[keep]):
                detections[int(label)].append((float(score), image_id, box))
    aps = []
    for class_id in range(1, len(VOC_CLASSES)):
        records = ground_truth[class_id]
        positives = sum((~record['difficult']).sum().item()
                        for record in records.values())
        ranked = sorted(
            detections[class_id], reverse=True, key=lambda item: item[0])
        true_positive, false_positive = [], []
        for _, image_id, box in ranked:
            record = records.get(image_id)
            if record is None or len(record['boxes']) == 0:
                true_positive.append(0)
                false_positive.append(1)
                continue
            overlaps = box_iou(box, record['boxes'])
            best_iou, best = overlaps.max(0)
            best = int(best)
            if best_iou >= 0.5 and record['difficult'][best]:
                continue
            if best_iou >= 0.5 and not record['matched'][best]:
                record['matched'][best] = True
                true_positive.append(1)
                false_positive.append(0)
            else:
                true_positive.append(0)
                false_positive.append(1)
        tp = np.cumsum(true_positive)
        fp = np.cumsum(false_positive)
        recall = tp / max(positives, 1)
        precision = tp / np.maximum(tp + fp, 1e-12)
        aps.append(float(average_precision(recall, precision)))
    return 100 * float(np.mean(aps)), [100 * value for value in aps]
