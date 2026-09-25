#!/usr/bin/env python3
"""Train the semantic-segmentation models reported in the paper."""

import argparse
import csv
import json
import random
import runpy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from run_deeplab_heteroweave_search import build_model, load_state, MEAN, STD


class PetsSegmentation(Dataset):
    def __init__(self, root, names, size, train, seed):
        self.root = Path(root)
        self.names = list(names)
        self.size = size
        self.train = train
        self.seed = seed

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        image = Image.open(
            self.root / 'images' / f'{name}.jpg').convert('RGB')
        mask = Image.open(
            self.root / 'annotations' / 'trimaps' / f'{name}.png')
        image = TF.resize(
            image, [self.size, self.size], InterpolationMode.BILINEAR)
        mask = TF.resize(
            mask, [self.size, self.size], InterpolationMode.NEAREST)
        if self.train and random.Random(self.seed + index).random() < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        image = TF.normalize(TF.to_tensor(image), MEAN, STD)
        target = torch.from_numpy(np.array(mask, dtype=np.int64) - 1)
        return image, target


def train(model, loader, device, steps, learning_rate):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    iterator = iter(loader)
    model.train()
    for _ in range(steps):
        try:
            images, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, targets = next(iterator)
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            loss = F.cross_entropy(model(images)['out'], targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    matrix = torch.zeros(3, 3, dtype=torch.int64)
    for images, targets in loader:
        predictions = model(images.to(device))['out'].argmax(1).cpu()
        valid = (targets >= 0) & (targets < 3)
        indices = 3 * targets[valid] + predictions[valid]
        matrix += torch.bincount(indices, minlength=9).reshape(3, 3)
    intersection = matrix.diag().float()
    union = matrix.sum(0) + matrix.sum(1) - intersection
    return 100 * float((intersection / union.clamp_min(1)).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/segmentation/paper_models.py')
    parser.add_argument('--data', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--anchor', required=True)
    parser.add_argument('--byol', required=True)
    parser.add_argument('--dino', required=True)
    parser.add_argument('--swav', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    config = runpy.run_path(args.config)
    settings = config['training']
    protocol = json.loads(Path(args.protocol).read_text())
    anchor_state = load_state(args.anchor)
    states = {
        'byol': load_state(args.byol),
        'dino': load_state(args.dino),
        'swav': load_state(args.swav),
        'random_a': None,
        'random_b': None,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    device = torch.device('cuda')
    rows = []
    for candidate in config['candidates']:
        model_spec = dict(
            candidate_id=candidate['id'],
            position=candidate['position'],
            components=candidate['components'],
            total_blocks=candidate['total_blocks'],
            gate=candidate['gate'],
            adapter_bottleneck=candidate['adapter_bottleneck'],
            fusion=candidate['fusion'],
        )
        for seed in config['seeds']:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            train_loader = DataLoader(
                PetsSegmentation(args.data, protocol['train'],
                                 settings['resolution'], True, seed),
                batch_size=settings['batch_size'], shuffle=True,
                num_workers=2, pin_memory=True)
            validation_loader = DataLoader(
                PetsSegmentation(args.data, protocol['search_validation'],
                                 settings['resolution'], False, seed),
                batch_size=settings['batch_size'], shuffle=False,
                num_workers=2, pin_memory=True)
            model = build_model(anchor_state, states, model_spec, 11).to(device)
            train(model, train_loader, device, settings['steps'],
                  settings['learning_rate'])
            miou = evaluate(model, validation_loader, device)
            checkpoint = checkpoint_dir / f"{candidate['id']}_seed{seed}.pth"
            torch.save({'model': model.state_dict(), 'candidate': candidate,
                        'seed': seed, 'steps': settings['steps']}, checkpoint)
            rows.append({
                'candidate': candidate['id'],
                'seed': seed,
                'validation_miou': miou,
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(
                    p.numel() for p in model.parameters() if p.requires_grad),
                'checkpoint': str(checkpoint),
            })
            del model
            torch.cuda.empty_cache()
    with (output / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()

