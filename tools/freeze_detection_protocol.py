#!/usr/bin/env python3
"""Create the deterministic VOC train/validation protocol used in the paper."""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    voc = Path(args.voc_root) / 'VOCdevkit' / 'VOC2007'
    identifiers = (voc / 'ImageSets' / 'Main' /
                   'trainval.txt').read_text().split()
    random.Random(20260915).shuffle(identifiers)
    validation = identifiers[:1000]
    training = identifiers[1000:]
    protocol = {
        'protocol_id': 'heteroweave_detection_v1_20260915',
        'split_seed': 20260915,
        'data': {
            'search_train_ids': training,
            'short_train_ids': training[48:848],
            'search_validation_ids': validation,
            'proxy_panel_ids': [training[i:i + 16] for i in (0, 16, 32)],
            'final_train_ids': identifiers,
        },
        'training': {
            'short': {
                'steps': 300, 'seeds': [0], 'batch_size': 2,
                'optimizer': 'SGD', 'learning_rate': 0.005,
                'checkpoint_rule': 'last',
            },
            'final': {
                'steps': 5000, 'seeds': [0, 1, 2], 'batch_size': 2,
                'checkpoint_rule': 'last',
            },
            'input': {'min_size': 384, 'max_size': 640},
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(protocol, indent=2) + '\n')


if __name__ == '__main__':
    main()

