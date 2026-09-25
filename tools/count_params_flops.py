#!/usr/bin/env python3
"""Recompute the resource boundary used by ImageNet search."""

import argparse
import os

from mmcv import Config
from mmcv.cnn.utils import get_model_complexity_info
from mmcls.models import build_classifier

from mmcls_addon import *  # noqa: F401,F403


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--input-shape", type=int, nargs=3, default=(3, 224, 224))
    parser.add_argument(
        "--load-pretrained", action="store_true",
        help="Load pretrained tensors; unnecessary for resource accounting.",
    )
    return parser.parse_args()


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def main():
    args = parse_args()
    if not args.load_pretrained:
        os.environ["DERY_DISABLE_PRETRAINED"] = "1"
    cfg = Config.fromfile(args.config)
    model = build_classifier(cfg.model)
    model.init_weights()

    total = parameter_count(model)
    backbone = parameter_count(model.backbone)
    neck = parameter_count(model.neck) if getattr(model, "neck", None) else 0
    head = parameter_count(model.head) if getattr(model, "head", None) else 0

    original_forward = model.forward
    model.eval()
    model.forward = model.extract_feat
    try:
        flops, _ = get_model_complexity_info(
            model, tuple(args.input_shape), print_per_layer_stat=False,
            as_strings=False,
        )
    finally:
        model.forward = original_forward

    print(f"total_parameters={total}")
    print(f"backbone_parameters={backbone}")
    print(f"neck_parameters={neck}")
    print(f"head_parameters={head}")
    print(f"feature_extraction_flops={int(flops)}")
    print("parameter_boundary=stem+selected_blocks+interfaces+neck+head")
    print("flops_boundary=feature_extraction_at_fixed_input_shape;head_excluded")


if __name__ == "__main__":
    main()
