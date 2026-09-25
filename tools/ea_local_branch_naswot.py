import argparse
import copy
import hashlib
import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import Config
from mmcv.cnn.utils import get_model_complexity_info
from mmcls.datasets.builder import build_dataloader, build_dataset
from mmcls.models import build_classifier


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_DIR = os.path.join(REPO_ROOT, 'simlarity')
for path in [REPO_ROOT, SIM_DIR, os.path.join(REPO_ROOT, 'third_package')]:
    if path not in sys.path:
        sys.path.insert(0, path)

from mmcls_addon import *  # noqa: F401,F403
from simlarity.zero_nas import ZeroNas
from simlarity.multi_objective import objective_summary, rank_items
from blocklize import MODEL_BLOCKS, MODEL_STATS, MODEL_ZOO
from blocklize.block_meta import MODEL_INOUT_SHAPE


INPUT_SHAPE = (3, 224, 224)


@dataclass
class LocalGene:
    branches: list
    operator: str = 'sum'


class FixedBackboneBlock:
    def __init__(self, block_cfg, block_index):
        self.block_cfg = list(block_cfg)
        self.block_index = block_index
        self.model_name = resolve_model_name(block_cfg[0], block_cfg[3])
        self.node_list = node_indices_from_cfg(self.model_name, block_cfg)
        self.group_id = None
        self.size = 0.0
        self.get_inout_size()

    def print_split(self):
        return list(self.block_cfg)

    def get_inout_size(self):
        start_name = self.block_cfg[1]
        end_name = self.block_cfg[2]
        self.in_size = MODEL_INOUT_SHAPE[self.model_name]['in_size'][start_name]
        self.out_size = MODEL_INOUT_SHAPE[self.model_name]['out_size'][end_name]

    def __str__(self):
        return (
            f'{self.model_name}:{self.node_list[0]}-'
            f'{self.node_list[-1]} Stage-{self.block_index}')


class ClassifierForwardWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model.extract_feat(x)
        if isinstance(x, tuple):
            x = x[-1]
        return self.model.head.fc(x)


def classifier_size_flops(model, args):
    """Match tools/analysis_tools/get_flops.py for DeRy FLOPs counting."""
    size = sum(param.numel() for param in model.parameters()) / 1e6
    if args.skip_flops_eval:
        return size, 0.0

    if not hasattr(model, 'extract_feat'):
        raise NotImplementedError(
            'FLOPs counter is currently not supported with '
            f'{model.__class__.__name__}')

    original_forward = model.forward
    model.eval()
    model.forward = model.extract_feat
    try:
        flops, _ = get_model_complexity_info(
            model,
            INPUT_SHAPE,
            print_per_layer_stat=False,
            as_strings=False)
    finally:
        model.forward = original_forward
    return size, round(flops / 10.0 ** 9, 3)


def parse_args():
    parser = argparse.ArgumentParser(
        description='EA local-branch search on a fixed DeRy backbone config with zero-cost proxies.')
    parser.add_argument(
        'backbone_config',
        nargs='?',
        default='configs/dery/imagenet/10m_imagenet_128x8_100e_dery_adamw.py',
        help='Existing DeRy config used as the fixed backbone.')
    parser.add_argument(
        '--assignment',
        default='simlarity/out/assignment/assignment_hybrid_4.pkl',
        help='Block assignment pkl used as the local branch candidate pool.')
    parser.add_argument(
        '--data-config',
        default='configs/_base_/datasets/imagenet_bs64.py',
        help='Dataset config used for zero-shot NASWOT evaluation.')
    parser.add_argument('--data-prefix', default='data/imagenet/train')
    parser.add_argument('--ann-file', default=None)
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='NASWOT batch size. Lower this with --num-batch to reduce search cost.')
    parser.add_argument(
        '--num-batch',
        type=int,
        default=1,
        help='Number of NASWOT batches. Total NASWOT images = batch-size * num-batch.')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--pop-size', type=int, default=12)
    parser.add_argument('--generations', type=int, default=8)
    parser.add_argument(
        '--init-max-rounds',
        type=int,
        default=30,
        help=(
            'Maximum rounds for building the initial valid population. '
            'This prevents tight budgets from looking stuck after the baseline score.'))
    parser.add_argument(
        '--init-log-interval',
        type=int,
        default=1,
        help='Print initial-population progress every N rounds.')
    parser.add_argument(
        '--max-initial-memo-factor',
        type=int,
        default=30,
        help='Stop initial sampling when memo entries exceed pop-size times this factor.')
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Save the top-k candidate configs by the selected objective for short training.')
    parser.add_argument(
        '--selection-mode',
        choices=['score', 'pareto'],
        default='score',
        help='score: old scalar-objective EA; pareto: rank population/top-k by Pareto rank and crowding distance.')
    parser.add_argument(
        '--pareto-objectives',
        nargs='+',
        default=['score', 'size', 'flops'],
        choices=[
            'score', 'objective', 'naswot', 'zico', 'real_score',
            'real_acc1', 'real_loss', 'neg_real_loss',
            'adapter_burden', 'type_switches',
            'params', 'params_m', 'size', 'flops', 'flops_g',
        ],
        help='Objectives for --selection-mode pareto. Scores/proxies are maximized; size/flops/loss/burden are minimized.')
    parser.add_argument('--start-ratio', type=float, default=0.35)
    parser.add_argument('--max-branch-layers', type=int, default=2)
    parser.add_argument('--max-branches-per-layer', type=int, default=2)
    parser.add_argument(
        '--operators',
        nargs='+',
        choices=['sum', 'gate'],
        default=['sum', 'gate'],
        help='Fusion operators. concat is intentionally omitted to keep the fixed backbone adapters unchanged.')
    parser.add_argument('--mutation-rate', type=float, default=0.3)
    parser.add_argument('--crossover-rate', type=float, default=0.8)
    parser.add_argument('--elite-ratio', type=float, default=0.25)
    parser.add_argument('--immigrant-ratio', type=float, default=0.15)
    parser.add_argument('--stagnation-patience', type=int, default=2)
    parser.add_argument('--restart-mutation-rate', type=float, default=0.7)
    parser.add_argument('--restart-immigrant-ratio', type=float, default=0.5)
    parser.add_argument(
        '--score-mode',
        choices=['naswot', 'zico', 'multi', 'hybrid', 'real'],
        default='hybrid',
        help=(
            'naswot: old behavior; zico: ZiCo only; '
            'multi: NASWOT + ZiCo; hybrid: zero-cost proxy + small real-data proxy; '
            'real: real-data proxy only.'))
    parser.add_argument('--naswot-weight', type=float, default=0.5)
    parser.add_argument(
        '--zico-weight',
        type=float,
        default=None,
        help='ZiCo weight. Defaults to 1.0 for --score-mode zico, 0.5 for multi, and 0 otherwise.')
    parser.add_argument(
        '--adapter-burden-weight',
        type=float,
        default=0.0,
        help='Penalty weight for structural adapter burden; larger values prefer easier block connections.')
    parser.add_argument(
        '--max-adapter-burden',
        type=float,
        default=None,
        help='Hard reject candidates whose structural adapter burden exceeds this value.')
    parser.add_argument(
        '--max-type-switches',
        type=int,
        default=None,
        help='Hard reject candidates with more than this many cnn<->vit adapter switches.')
    parser.add_argument(
        '--branch-type-policy',
        choices=['any', 'same-io', 'no-vit'],
        default='any',
        help=(
            'any: old behavior; same-io: branch input/output feature types must match '
            'the primary block; no-vit: reject VIT/Swin branch blocks.'))
    parser.add_argument(
        '--min-real-delta',
        type=float,
        default=None,
        help='Hard reject candidates whose real proxy score is worse than baseline by more than this delta.')
    parser.add_argument('--real-weight', type=float, default=0.5)
    parser.add_argument(
        '--absolute-objective',
        dest='relative_objective',
        action='store_false',
        help='Use the raw weighted objective instead of improvement over the baseline config.')
    parser.set_defaults(relative_objective=True)
    parser.add_argument(
        '--real-metric',
        choices=['neg_loss', 'acc1', 'acc1_minus_loss'],
        default='neg_loss')
    parser.add_argument('--real-batch-size', type=int, default=16)
    parser.add_argument('--real-train-batches', type=int, default=2)
    parser.add_argument('--real-eval-batches', type=int, default=1)
    parser.add_argument('--real-train-steps', type=int, default=4)
    parser.add_argument('--real-lr', type=float, default=1e-3)
    parser.add_argument('--real-weight-decay', type=float, default=0.05)
    parser.add_argument('--real-max-grad-norm', type=float, default=1.0)
    parser.add_argument('--real-data-seed', type=int, default=None)
    parser.add_argument('--real-no-shuffle', action='store_true')
    parser.add_argument(
        '--min-real-classes',
        type=int,
        default=4,
        help='Fail fast if cached real-proxy batches contain too few classes.')
    parser.add_argument('--C', '--maxC', dest='max_params', type=float, default=30.0)
    parser.add_argument('--minC', dest='min_params', type=float, default=None)
    parser.add_argument('--flop-C', '--flop_C', dest='max_flops', type=float, default=10.0)
    parser.add_argument('--minflop-C', '--minflop_C', dest='min_flops', type=float, default=None)
    parser.add_argument('--skip-flops-eval', action='store_true')
    parser.add_argument('--output-dir', default='simlarity/out/local_branch_ea')
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['cuda'], default='cuda')
    return parser.parse_args()


def make_run_dir(output_dir, run_name=None):
    os.makedirs(output_dir, exist_ok=True)
    if run_name is not None:
        run_dir = os.path.join(output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    stamp = time.strftime('%Y%m%d_%H%M%S')
    for index in range(1000):
        suffix = '' if index == 0 else f'_{index:03d}'
        run_dir = os.path.join(output_dir, f'run_{stamp}{suffix}')
        try:
            os.makedirs(run_dir)
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f'Cannot allocate run dir under {output_dir}')


def tupleize(value):
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def resolve_model_name(arch, backend):
    for model_name, stats in MODEL_STATS.items():
        if stats.get('arch') == arch and stats.get('backend') == backend:
            return model_name
    if arch in MODEL_STATS:
        return arch
    raise KeyError(f'Cannot resolve model name from arch={arch}, backend={backend}')


def node_indices_from_cfg(model_name, block_cfg):
    blocks = MODEL_BLOCKS[model_name]
    start = blocks.index(block_cfg[1])
    end = blocks.index(block_cfg[2])
    return list(range(start, end + 1))


def split_signature(block_cfg):
    if hasattr(block_cfg, 'to_dict'):
        block_cfg = block_cfg.to_dict()
    if isinstance(block_cfg, dict) and block_cfg.get('type') == 'CompositeBlock':
        return split_signature(block_cfg['branches'][0]['block'])
    if isinstance(block_cfg, dict):
        return (
            block_cfg.get('model_name'),
            tupleize(block_cfg.get('block_input')),
            tupleize(block_cfg.get('block_output')),
            block_cfg.get('backend'),
        )
    if isinstance(block_cfg, (list, tuple)) and len(block_cfg) >= 4:
        return (
            block_cfg[0],
            tupleize(block_cfg[1]),
            tupleize(block_cfg[2]),
            block_cfg[3],
        )
    raise TypeError(f'Unsupported block cfg: {block_cfg}')


def block_key(block):
    return (
        block.model_name,
        block.block_index,
        tuple(block.node_list),
        block.group_id,
    )


def candidate_signature(candidate):
    signature = []
    for gene in candidate:
        if gene is None or len(gene.branches) == 0:
            signature.append(None)
        else:
            signature.append((
                gene.operator,
                tuple(block_key(block) for block in gene.branches),
            ))
    return tuple(signature)


def stable_int(value):
    digest = hashlib.md5(repr(value).encode('utf-8')).hexdigest()
    return int(digest[:8], 16)


def has_any_branch(candidate):
    return any(gene is not None and len(gene.branches) > 0 for gene in candidate)


def set_eval_seed(base_seed, candidate):
    seed = base_seed if not has_any_branch(candidate) else base_seed + stable_int(candidate_signature(candidate))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def load_assignment(path):
    with open(path, 'rb') as file:
        assignment = pickle.load(file)
    if not hasattr(assignment, 'center2block'):
        raise TypeError(f'{path} is not a Block_Assign-like object')

    blocks_by_index = {}
    for group in assignment.center2block:
        for block in group:
            if (
                    not hasattr(block, 'model_name') or
                    not hasattr(block, 'block_index') or
                    not hasattr(block, 'node_list')):
                continue
            if block.model_name not in MODEL_ZOO:
                continue
            if not hasattr(block, 'in_size') or not hasattr(block, 'out_size'):
                block.get_inout_size()
            blocks_by_index.setdefault(block.block_index, []).append(block)
    return blocks_by_index


def find_matching_block(block_cfg, blocks_by_index, index):
    target = split_signature(block_cfg)
    for block in blocks_by_index.get(index, []):
        if split_signature(block.print_split()) == target:
            return block
    return None


def load_backbone_blocks(config_path, blocks_by_index):
    cfg = Config.fromfile(config_path)
    backbone = cfg.model.backbone
    if backbone.type != 'DeRy':
        raise ValueError(f'{config_path} must use model.backbone.type="DeRy"')

    primary_cfgs = list(backbone.block_list)
    primary_blocks = []
    for index, block_cfg in enumerate(primary_cfgs):
        block = find_matching_block(block_cfg, blocks_by_index, index)
        if block is None:
            block = FixedBackboneBlock(block_cfg, index)
        primary_blocks.append(block)
    return cfg, primary_cfgs, primary_blocks


def feature_type(size):
    return 'vit' if len(size) == 2 else 'cnn'


def channels(size):
    return size[1] if len(size) == 2 else size[0]


def make_adapter_cfg(src_size, dst_size):
    src_type = feature_type(src_size)
    dst_type = feature_type(dst_size)
    input_channel = channels(src_size)
    output_channel = channels(dst_size)

    if src_type == dst_type and input_channel == output_channel:
        return None

    mode = f'{src_type}2{dst_type}'
    if mode == 'vit2vit':
        return dict(
            input_channel=input_channel,
            output_channel=output_channel,
            num_fc=1,
            num_conv=0,
            mode=mode)

    stride = 1
    if src_type == 'cnn' and dst_type == 'cnn' and len(src_size) == 3 and len(dst_size) == 3:
        stride = 1 if src_size[1] / dst_size[1] < 2 else 2
    return dict(
        input_channel=input_channel,
        output_channel=output_channel,
        num_fc=0,
        num_conv=1,
        stride=stride,
        mode=mode)


def adapter_cfg_burden(adapter_cfg):
    if adapter_cfg is None:
        return 0.0

    mode = adapter_cfg.get('mode', 'cnn2cnn')
    input_channel = max(1.0, float(adapter_cfg.get('input_channel', 1)))
    output_channel = max(1.0, float(adapter_cfg.get('output_channel', 1)))
    type_switch_penalty = 1.0 if mode in ['cnn2vit', 'vit2cnn'] else 0.0
    channel_ratio_penalty = abs(math.log(output_channel / input_channel))
    stride_penalty = max(0.0, float(adapter_cfg.get('stride', 1)) - 1.0)
    layer_penalty = 0.1 * (
        float(adapter_cfg.get('num_fc', 0)) + float(adapter_cfg.get('num_conv', 0)))
    return (
        type_switch_penalty +
        0.25 * channel_ratio_penalty +
        0.5 * stride_penalty +
        layer_penalty)


def adapter_cfg_type_switches(adapter_cfg):
    if adapter_cfg is None:
        return 0
    return 1 if adapter_cfg.get('mode') in ['cnn2vit', 'vit2cnn'] else 0


def candidate_adapter_burden(candidate, primary_blocks):
    burden = 0.0
    for layer, gene in enumerate(candidate):
        if gene is None or len(gene.branches) == 0:
            continue
        primary = primary_blocks[layer]
        for branch in gene.branches:
            burden += adapter_cfg_burden(
                make_adapter_cfg(primary.in_size, branch.in_size))
            burden += adapter_cfg_burden(
                make_adapter_cfg(branch.out_size, primary.out_size))
        if gene.operator == 'gate':
            burden += 0.05 * len(gene.branches)
    return burden


def candidate_type_switches(candidate, primary_blocks):
    switches = 0
    for layer, gene in enumerate(candidate):
        if gene is None or len(gene.branches) == 0:
            continue
        primary = primary_blocks[layer]
        for branch in gene.branches:
            switches += adapter_cfg_type_switches(
                make_adapter_cfg(primary.in_size, branch.in_size))
            switches += adapter_cfg_type_switches(
                make_adapter_cfg(branch.out_size, primary.out_size))
    return switches


def branch_allowed_by_type_policy(branch, primary, args):
    if args.branch_type_policy == 'any':
        return True
    if args.branch_type_policy == 'no-vit':
        return (
            feature_type(branch.in_size) != 'vit' and
            feature_type(branch.out_size) != 'vit')
    if args.branch_type_policy == 'same-io':
        return (
            feature_type(branch.in_size) == feature_type(primary.in_size) and
            feature_type(branch.out_size) == feature_type(primary.out_size))
    raise ValueError(f'Unknown branch type policy: {args.branch_type_policy}')


def empty_candidate(num_layers):
    return [None for _ in range(num_layers)]


def allowed_layers(num_layers, args):
    if num_layers <= 0 or args.max_branch_layers <= 0:
        return []
    explicit = getattr(args, 'branch_layers', None)
    if explicit is not None:
        return [
            int(layer) for layer in explicit
            if 0 <= int(layer) < num_layers
        ]
    start = int(num_layers * min(max(args.start_ratio, 0.0), 1.0))
    start = min(max(start, 0), num_layers - 1)
    return list(range(start, num_layers))


def branch_pool(layer, candidate, primary_blocks, blocks_by_index, args):
    primary = primary_blocks[layer]
    used_models = {primary.model_name}
    gene = candidate[layer]
    if gene is not None:
        used_models.update(block.model_name for block in gene.branches)
    return [
        block for block in blocks_by_index.get(layer, [])
        if (
            block.model_name not in used_models and
            branch_allowed_by_type_policy(block, primary, args))
    ]


def enforce_candidate(candidate, primary_blocks, blocks_by_index, args):
    candidate = copy.deepcopy(candidate)
    active_layers = []
    for layer, gene in enumerate(candidate):
        if gene is None:
            continue
        deduped = []
        primary = primary_blocks[layer]
        used_models = {primary.model_name}
        for block in gene.branches:
            if block.model_name in used_models:
                continue
            if block not in blocks_by_index.get(layer, []):
                continue
            if not branch_allowed_by_type_policy(block, primary, args):
                continue
            deduped.append(block)
            used_models.add(block.model_name)
            if len(deduped) >= max(0, args.max_branches_per_layer - 1):
                break
        if len(deduped) == 0:
            candidate[layer] = None
        else:
            operator = gene.operator if gene.operator in args.operators else random.choice(args.operators)
            candidate[layer] = LocalGene(branches=deduped, operator=operator)
            active_layers.append(layer)

    max_layers = max(0, args.max_branch_layers)
    if len(active_layers) > max_layers:
        for layer in random.sample(active_layers, len(active_layers) - max_layers):
            candidate[layer] = None
    return candidate


def sample_candidate(primary_blocks, blocks_by_index, args):
    candidate = empty_candidate(len(primary_blocks))
    layers = allowed_layers(len(primary_blocks), args)
    if len(layers) == 0:
        return candidate
    num_layers = random.randint(1, min(args.max_branch_layers, len(layers)))
    for layer in random.sample(layers, num_layers):
        pool = branch_pool(layer, candidate, primary_blocks, blocks_by_index, args)
        if len(pool) == 0:
            continue
        random.shuffle(pool)
        extra_count = random.randint(1, min(args.max_branches_per_layer - 1, len(pool)))
        candidate[layer] = LocalGene(
            branches=pool[:extra_count],
            operator=random.choice(args.operators))
    return enforce_candidate(candidate, primary_blocks, blocks_by_index, args)


def mutate_candidate(candidate, primary_blocks, blocks_by_index, args, rate=None):
    child = copy.deepcopy(candidate)
    rate = args.mutation_rate if rate is None else rate
    layers = allowed_layers(len(primary_blocks), args)
    changed = False

    for layer in layers:
        if random.random() >= rate:
            continue
        changed = True
        gene = child[layer]
        actions = ['add', 'operator']
        if gene is not None and len(gene.branches) > 0:
            actions.extend(['remove', 'replace'])
        action = random.choice(actions)

        if action == 'add':
            if gene is None:
                gene = LocalGene(branches=[], operator=random.choice(args.operators))
                child[layer] = gene
            if len(gene.branches) < max(0, args.max_branches_per_layer - 1):
                pool = branch_pool(layer, child, primary_blocks, blocks_by_index, args)
                if len(pool) > 0:
                    gene.branches.append(random.choice(pool))
                    gene.operator = random.choice(args.operators)
        elif action == 'remove' and gene is not None and len(gene.branches) > 0:
            del gene.branches[random.randrange(len(gene.branches))]
        elif action == 'replace' and gene is not None and len(gene.branches) > 0:
            old = gene.branches.pop(random.randrange(len(gene.branches)))
            pool = branch_pool(layer, child, primary_blocks, blocks_by_index, args)
            if len(pool) > 0:
                gene.branches.append(random.choice(pool))
            else:
                gene.branches.append(old)
        elif action == 'operator' and gene is not None and len(gene.branches) > 0:
            gene.operator = random.choice(args.operators)

    if not changed and len(layers) > 0:
        layer = random.choice(layers)
        gene = child[layer]
        if gene is None:
            gene = LocalGene(branches=[], operator=random.choice(args.operators))
            child[layer] = gene
        pool = branch_pool(layer, child, primary_blocks, blocks_by_index, args)
        if len(pool) > 0:
            gene.branches.append(random.choice(pool))

    return enforce_candidate(child, primary_blocks, blocks_by_index, args)


def crossover_candidate(parent_a, parent_b, primary_blocks, blocks_by_index, args):
    child = []
    for gene_a, gene_b in zip(parent_a, parent_b):
        child.append(copy.deepcopy(gene_a if random.random() < 0.5 else gene_b))
    return enforce_candidate(child, primary_blocks, blocks_by_index, args)


def build_candidate_config(base_cfg, primary_cfgs, primary_blocks, candidate):
    cfg = copy.deepcopy(base_cfg)
    block_list = copy.deepcopy(primary_cfgs)

    for layer, gene in enumerate(candidate):
        if gene is None or len(gene.branches) == 0:
            continue
        primary_cfg = copy.deepcopy(primary_cfgs[layer])
        primary = primary_blocks[layer]
        branches = [dict(
            block=primary_cfg,
            input_adapter=None,
            output_adapter=None)]
        for branch in gene.branches:
            branches.append(dict(
                block=branch.print_split(),
                input_adapter=make_adapter_cfg(primary.in_size, branch.in_size),
                output_adapter=make_adapter_cfg(branch.out_size, primary.out_size)))
        block_list[layer] = dict(
            type='CompositeBlock',
            branches=branches,
            operator=gene.operator,
            out_type=feature_type(primary.out_size))

    cfg.model.backbone.block_list = block_list
    return cfg


def in_budget(size, flops, args):
    if args.min_params is not None and size <= args.min_params:
        return False
    if args.max_params is not None and size > args.max_params:
        return False
    if not args.skip_flops_eval:
        if args.min_flops is not None and flops <= args.min_flops:
            return False
        if args.max_flops is not None and flops > args.max_flops:
            return False
    return True


def format_candidate(candidate):
    rows = []
    for layer, gene in enumerate(candidate):
        if gene is None or len(gene.branches) == 0:
            continue
        branches = ', '.join(str(block) for block in gene.branches)
        rows.append(f'L{layer}:{gene.operator}[{branches}]')
    return '; '.join(rows) if rows else 'no extra branch'


def make_data_loader(args, batch_size=None, shuffle=False):
    data_cfg = Config.fromfile(args.data_config)
    if args.data_prefix is not None:
        data_cfg.data.train.data_prefix = args.data_prefix
    if args.ann_file is not None:
        data_cfg.data.train.ann_file = args.ann_file
    dataset = build_dataset(data_cfg.data.train)
    data_cfg.data.samples_per_gpu = args.batch_size if batch_size is None else batch_size
    if args.workers is not None:
        data_cfg.data.workers_per_gpu = args.workers
    return build_dataloader(
        dataset,
        samples_per_gpu=data_cfg.data.samples_per_gpu,
        workers_per_gpu=data_cfg.data.workers_per_gpu,
        dist=False,
        shuffle=shuffle,
        round_up=True)


def clone_tensor_batch(data):
    cloned = {}
    for key, value in data.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().cpu().clone()
        else:
            cloned[key] = value
    return cloned


def cache_real_batches(data_loader, total_batches):
    cached = []
    if total_batches <= 0:
        return cached
    for index, data in enumerate(data_loader):
        cached.append(clone_tensor_batch(data))
        if index + 1 >= total_batches:
            break
    return cached


def cached_label_stats(real_batches):
    labels = []
    for data in real_batches:
        if 'gt_label' not in data:
            continue
        label = data['gt_label']
        if torch.is_tensor(label):
            labels.extend(label.view(-1).cpu().tolist())
    return len(labels), len(set(labels))


def batch_img_label(data):
    img = data['img'].cuda(non_blocking=True)
    label = data['gt_label'].cuda(non_blocking=True).view(-1).long()
    return img, label


def real_metric_value(avg_loss, acc1, args):
    if args.real_metric == 'acc1':
        return acc1
    if args.real_metric == 'acc1_minus_loss':
        return acc1 - avg_loss
    return -avg_loss


def active_zero_indicators(args):
    if args.score_mode == 'real':
        return []

    indicators = []
    if args.naswot_weight > 0:
        indicators.append('naswot')
    if args.zico_weight > 0:
        indicators.append('zico')
    return indicators


def objective_value(raw_naswot, raw_zico, adapter_burden, real_score, args):
    score = 0.0
    if raw_naswot is not None:
        naswot_score = raw_naswot
        naswot_ref = getattr(args, 'naswot_reference', None)
        if args.relative_objective and naswot_ref is not None:
            naswot_score = naswot_score - naswot_ref
        score += args.naswot_weight * naswot_score

    if raw_zico is not None:
        zico_score = raw_zico
        zico_ref = getattr(args, 'zico_reference', None)
        if args.relative_objective and zico_ref is not None:
            zico_score = zico_score - zico_ref
        score += args.zico_weight * zico_score

    if adapter_burden is not None and args.adapter_burden_weight > 0:
        burden = adapter_burden
        burden_ref = getattr(args, 'adapter_burden_reference', None)
        if args.relative_objective and burden_ref is not None:
            burden = burden - burden_ref
        score -= args.adapter_burden_weight * burden

    if real_score is not None:
        proxy_score = real_score
        real_ref = getattr(args, 'real_reference_score', None)
        if args.relative_objective and real_ref is not None:
            proxy_score = proxy_score - real_ref
        score += args.real_weight * proxy_score
    return score


def compute_real_proxy(cfg, real_batches, args):
    train_count = max(0, args.real_train_batches)
    eval_count = max(1, args.real_eval_batches)
    if len(real_batches) < train_count + eval_count:
        raise RuntimeError(
            f'Need {train_count + eval_count} cached real batches, got {len(real_batches)}')

    train_batches = real_batches[:train_count]
    eval_batches = real_batches[train_count:train_count + eval_count]
    model = None
    try:
        model = build_classifier(cfg.model)
        model.init_weights()
        wrapped = ClassifierForwardWrapper(model).cuda()

        params = [param for param in wrapped.parameters() if param.requires_grad]
        if len(params) > 0 and train_count > 0:
            optimizer = torch.optim.AdamW(
                params, lr=args.real_lr, weight_decay=args.real_weight_decay)
            train_steps = (
                args.real_train_steps
                if args.real_train_steps is not None else train_count)
            wrapped.train()
            for step in range(max(0, train_steps)):
                img, label = batch_img_label(train_batches[step % train_count])
                optimizer.zero_grad()
                logits = wrapped(img)
                loss = F.cross_entropy(logits, label)
                loss.backward()
                if args.real_max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.real_max_grad_norm)
                optimizer.step()

        wrapped.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for data in eval_batches:
                img, label = batch_img_label(data)
                logits = wrapped(img)
                total_loss += F.cross_entropy(
                    logits, label, reduction='sum').item()
                total_correct += (logits.argmax(dim=1) == label).sum().item()
                total_samples += label.numel()

        avg_loss = total_loss / max(1, total_samples)
        acc1 = 100.0 * total_correct / max(1, total_samples)
        score = real_metric_value(avg_loss, acc1, args)
        return dict(real_score=score, real_loss=avg_loss, real_acc1=acc1)
    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()


def evaluate_candidate(
        candidate, base_cfg, primary_cfgs, primary_blocks, indicator, args,
        memo, real_batches=None):
    sig = candidate_signature(candidate)
    if sig in memo:
        return memo[sig]

    fixed_eval_seed = getattr(args, 'fixed_eval_seed', None)
    if fixed_eval_seed is None:
        eval_seed = set_eval_seed(args.seed, candidate)
    else:
        eval_seed = int(fixed_eval_seed)
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)
    model = None
    try:
        cfg = build_candidate_config(base_cfg, primary_cfgs, primary_blocks, candidate)
        classifier = build_classifier(cfg.model)
        size, flops = classifier_size_flops(classifier, args)
        model = ClassifierForwardWrapper(classifier)

        if not in_budget(size, flops, args):
            result = dict(
                candidate=copy.deepcopy(candidate),
                score=-float('inf'),
                raw_score=-float('inf'),
                naswot=None,
                zico=None,
                adapter_burden=None,
                real_score=None,
                real_loss=None,
                real_acc1=None,
                size=size,
                flops=flops,
                seed=eval_seed,
                error='budget')
        else:
            adapter_burden = candidate_adapter_burden(candidate, primary_blocks)
            type_switches = candidate_type_switches(candidate, primary_blocks)
            structure_error = None
            if (
                    args.max_adapter_burden is not None and
                    adapter_burden > args.max_adapter_burden):
                structure_error = 'adapter_burden'
            elif (
                    args.max_type_switches is not None and
                    type_switches > args.max_type_switches):
                structure_error = 'type_switches'

            if structure_error is not None:
                result = dict(
                    candidate=copy.deepcopy(candidate),
                    score=-float('inf'),
                    raw_score=-float('inf'),
                    naswot=None,
                    zico=None,
                    adapter_burden=adapter_burden,
                    type_switches=type_switches,
                    real_score=None,
                    real_loss=None,
                    real_acc1=None,
                    size=size,
                    flops=flops,
                    seed=eval_seed,
                    error=structure_error)
                memo[sig] = result
                return result

            raw_naswot = None
            raw_zico = None
            zero_score_extras = {}
            zero_indicators = active_zero_indicators(args)
            if len(zero_indicators) > 0:
                zero_scores = indicator.get_score(model)
                raw_naswot = zero_scores.get('naswot')
                raw_zico = zero_scores.get('zico')
                zero_score_extras = {
                    key: value for key, value in zero_scores.items()
                    if key not in ('naswot', 'zico')}
            if model is not None:
                del model
                model = None
                torch.cuda.empty_cache()

            real_values = dict(real_score=None, real_loss=None, real_acc1=None)
            if args.score_mode in ['real', 'hybrid']:
                torch.manual_seed(eval_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(eval_seed)
                real_values = compute_real_proxy(cfg, real_batches or [], args)

            real_ref = getattr(args, 'real_reference_score', None)
            if (
                    args.min_real_delta is not None and
                    real_ref is not None and
                    real_values['real_score'] is not None and
                    real_values['real_score'] - real_ref < args.min_real_delta):
                result = dict(
                    candidate=copy.deepcopy(candidate),
                    score=-float('inf'),
                    raw_score=-float('inf'),
                    naswot=raw_naswot,
                    zico=raw_zico,
                    adapter_burden=adapter_burden,
                    type_switches=type_switches,
                    real_score=real_values['real_score'],
                    real_loss=real_values['real_loss'],
                    real_acc1=real_values['real_acc1'],
                    size=size,
                    flops=flops,
                    seed=eval_seed,
                    error='real_delta')
                memo[sig] = result
                return result

            score = objective_value(
                raw_naswot, raw_zico, adapter_burden,
                real_values['real_score'], args)

            result = dict(
                candidate=copy.deepcopy(candidate),
                score=score,
                raw_score=score,
                naswot=raw_naswot,
                zico=raw_zico,
                adapter_burden=adapter_burden,
                type_switches=type_switches,
                real_score=real_values['real_score'],
                real_loss=real_values['real_loss'],
                real_acc1=real_values['real_acc1'],
                size=size,
                flops=flops,
                seed=eval_seed,
                error=None)
            result.update(zero_score_extras)
    except Exception as exc:
        result = dict(
            candidate=copy.deepcopy(candidate),
            score=-float('inf'),
            raw_score=-float('inf'),
            naswot=None,
            zico=None,
            adapter_burden=None,
            type_switches=None,
            real_score=None,
            real_loss=None,
            real_acc1=None,
            size=0.0,
            flops=0.0,
            seed=eval_seed,
            error=f'{type(exc).__name__}: {exc}')
    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()

    memo[sig] = result
    return result


def evaluate_many(
        candidates, base_cfg, primary_cfgs, primary_blocks, indicator, args,
        memo, real_batches=None):
    return [
        evaluate_candidate(
            candidate, base_cfg, primary_cfgs, primary_blocks, indicator, args,
            memo, real_batches=real_batches)
        for candidate in candidates
    ]


def unique_candidates(candidates):
    seen = set()
    unique = []
    for candidate in candidates:
        sig = candidate_signature(candidate)
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(candidate)
    return unique


def rank_population(items, args):
    if args.selection_mode == 'pareto':
        return rank_items(items, args.pareto_objectives)
    return sorted(items, key=lambda item: item['score'], reverse=True)


def better_item(left, right, args):
    if args.selection_mode == 'pareto':
        return rank_items([left, right], args.pareto_objectives)[0]
    return left if left['score'] >= right['score'] else right


def should_replace_best(candidate, current_best, args):
    if args.selection_mode == 'pareto':
        return better_item(candidate, current_best, args) is candidate
    return candidate['score'] > current_best['score']


def tournament(population, args, k=3):
    candidates = random.sample(population, min(k, len(population)))
    return rank_population(candidates, args)[0]


def save_best(run_dir, best, base_cfg, primary_cfgs, primary_blocks):
    cfg = build_candidate_config(
        base_cfg, primary_cfgs, primary_blocks, best['candidate'])
    config_path = os.path.join(run_dir, 'best_local_branch_model.py')
    with open(config_path, 'w', encoding='utf-8') as file:
        file.write(cfg.pretty_text)

    pkl_path = os.path.join(run_dir, 'best_local_branch.pkl')
    with open(pkl_path, 'wb') as file:
        pickle.dump(best, file, protocol=pickle.HIGHEST_PROTOCOL)
    return config_path, pkl_path


def ranked_valid_items(memo, args):
    items = [
        copy.deepcopy(item)
        for item in memo.values()
        if item['score'] > -float('inf') and item.get('error') is None
    ]
    return rank_population(items, args)


def fmt_optional(value, precision=6):
    if value is None:
        return 'None'
    return f'{value:.{precision}f}'


def memo_error_summary(memo):
    counts = {}
    for item in memo.values():
        error = item.get('error')
        if error is None:
            error = 'valid' if item.get('score', -float('inf')) > -float('inf') else 'invalid'
        elif isinstance(error, str) and ': ' in error:
            error = error.split(': ', 1)[0]
        counts[error] = counts.get(error, 0) + 1
    return ', '.join(f'{key}:{counts[key]}' for key in sorted(counts))


def save_top_k(
        run_dir, items, top_k, base_cfg, primary_cfgs, primary_blocks,
        selection_mode='score', pareto_objectives=None):
    pareto_objectives = pareto_objectives or []
    top_dir = os.path.join(run_dir, 'top_k')
    os.makedirs(top_dir, exist_ok=True)
    summary_path = os.path.join(top_dir, 'top_k_summary.txt')
    summary_json_path = os.path.join(top_dir, 'top_k_summary.json')
    saved = []
    summary_items = []

    with open(summary_path, 'w', encoding='utf-8') as summary:
        for rank, item in enumerate(items[:top_k], start=1):
            cfg = build_candidate_config(
                base_cfg, primary_cfgs, primary_blocks, item['candidate'])
            stem = f'top_{rank:02d}_score_{item["score"]:.6f}'
            config_path = os.path.join(top_dir, f'{stem}.py')
            pkl_path = os.path.join(top_dir, f'{stem}.pkl')
            with open(config_path, 'w', encoding='utf-8') as file:
                file.write(cfg.pretty_text)
            with open(pkl_path, 'wb') as file:
                pickle.dump(item, file, protocol=pickle.HIGHEST_PROTOCOL)
            summary_items.append(dict(
                rank=rank,
                config_path=config_path,
                pkl_path=pkl_path,
                objective=item['score'],
                naswot=item.get('naswot'),
                zico=item.get('zico'),
                adapter_burden=item.get('adapter_burden'),
                real_score=item.get('real_score'),
                real_loss=item.get('real_loss'),
                real_acc1=item.get('real_acc1'),
                size=item['size'],
                flops=item['flops'],
                seed=item['seed'],
                pareto_rank=item.get('_pareto_rank'),
                crowding_distance=item.get('_crowding_distance'),
                selection_mode=selection_mode,
                pareto_objectives=pareto_objectives,
                objective_summary=objective_summary(item, pareto_objectives)
                if pareto_objectives else None,
                branches=format_candidate(item['candidate'])))
            summary.write(
                f'top_{rank:02d}\t'
                f'objective={item["score"]:.6f}\t'
                f'naswot={fmt_optional(item.get("naswot"))}\t'
                f'zico={fmt_optional(item.get("zico"))}\t'
                f'adapter_burden={fmt_optional(item.get("adapter_burden"))}\t'
                f'real_score={fmt_optional(item.get("real_score"))}\t'
                f'real_loss={fmt_optional(item.get("real_loss"))}\t'
                f'real_acc1={fmt_optional(item.get("real_acc1"))}\t'
                f'size={item["size"]:.3f}M\t'
                f'flops={item["flops"]:.3f}G\t'
                f'pareto_rank={item.get("_pareto_rank")}\t'
                f'crowding={item.get("_crowding_distance")}\t'
                f'objectives={objective_summary(item, pareto_objectives) if pareto_objectives else None}\t'
                f'seed={item["seed"]}\t'
                f'config={config_path}\t'
                f'branches={format_candidate(item["candidate"])}\n')
            saved.append((config_path, pkl_path, item))
    with open(summary_json_path, 'w', encoding='utf-8') as file:
        import json
        json.dump(dict(
            run_type='branch',
            selection_mode=selection_mode,
            pareto_objectives=pareto_objectives,
            top_k=summary_items,
        ), file, indent=2)
    return top_dir, summary_path, saved


def save_baseline_config(run_dir, base_cfg, baseline_item):
    top_dir = os.path.join(run_dir, 'top_k')
    os.makedirs(top_dir, exist_ok=True)
    config_path = os.path.join(top_dir, 'baseline_original.py')
    pkl_path = os.path.join(top_dir, 'baseline_original.pkl')
    with open(config_path, 'w', encoding='utf-8') as file:
        file.write(base_cfg.pretty_text)
    with open(pkl_path, 'wb') as file:
        pickle.dump(baseline_item, file, protocol=pickle.HIGHEST_PROTOCOL)
    return config_path, pkl_path


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required because ZeroNas uses cuda tensors.')

    args.pop_size = max(2, args.pop_size)
    args.generations = max(0, args.generations)
    args.init_max_rounds = max(1, args.init_max_rounds)
    args.init_log_interval = max(1, args.init_log_interval)
    args.max_initial_memo_factor = max(1, args.max_initial_memo_factor)
    args.top_k = max(1, args.top_k)
    if args.selection_mode == 'pareto':
        args.pareto_objectives = list(dict.fromkeys(args.pareto_objectives))
    args.max_branch_layers = max(0, args.max_branch_layers)
    args.max_branches_per_layer = max(2, args.max_branches_per_layer)
    args.mutation_rate = min(max(args.mutation_rate, 0.0), 1.0)
    args.crossover_rate = min(max(args.crossover_rate, 0.0), 1.0)
    args.elite_ratio = min(max(args.elite_ratio, 0.0), 1.0)
    args.immigrant_ratio = min(max(args.immigrant_ratio, 0.0), 1.0)
    args.stagnation_patience = max(1, args.stagnation_patience)
    args.restart_mutation_rate = min(max(args.restart_mutation_rate, 0.0), 1.0)
    args.restart_immigrant_ratio = min(max(args.restart_immigrant_ratio, 0.0), 1.0)
    args.naswot_weight = max(0.0, args.naswot_weight)
    if args.zico_weight is None:
        if args.score_mode == 'zico':
            args.zico_weight = 1.0
        elif args.score_mode == 'multi':
            args.zico_weight = 0.5
        else:
            args.zico_weight = 0.0
    args.zico_weight = max(0.0, args.zico_weight)
    args.adapter_burden_weight = max(0.0, args.adapter_burden_weight)
    args.real_weight = max(0.0, args.real_weight)
    args.real_train_batches = max(0, args.real_train_batches)
    args.real_eval_batches = max(1, args.real_eval_batches)
    if args.real_train_steps is not None:
        args.real_train_steps = max(0, args.real_train_steps)
    if args.score_mode == 'real':
        args.naswot_weight = 0.0
        args.zico_weight = 0.0
        args.adapter_burden_weight = 0.0
    elif args.score_mode == 'zico':
        args.naswot_weight = 0.0
        args.real_weight = 0.0
    elif args.score_mode == 'multi':
        args.real_weight = 0.0
    args.zico_num_batch_adjusted = False
    if args.zico_weight > 0 and args.num_batch < 2:
        args.num_batch = 2
        args.zico_num_batch_adjusted = True

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.naswot_reference = None
    args.zico_reference = None
    args.adapter_burden_reference = None
    args.real_reference_score = None

    run_dir = make_run_dir(args.output_dir, args.run_name)
    log_path = os.path.join(run_dir, 'ea_local_branch_log.txt')
    log_file = open(log_path, 'w', encoding='utf-8', buffering=1)

    def emit(message):
        print(message, flush=True)
        log_file.write(message + '\n')

    emit('Starting fixed-backbone local-branch EA')
    emit(f'run_dir: {run_dir}')
    emit(f'backbone_config: {args.backbone_config}')
    emit(f'assignment: {args.assignment}')
    emit(
        f'zero-cost settings: batch_size={args.batch_size}, '
        f'num_batch={args.num_batch}, images={args.batch_size * args.num_batch}, '
        f'seed={args.seed}')
    if args.zico_num_batch_adjusted:
        emit('ZiCo requires gradient variance across batches; num_batch was raised to 2.')
    emit(
        f'score_mode={args.score_mode}, zero_indicators={active_zero_indicators(args)}, '
        f'naswot_weight={args.naswot_weight}, zico_weight={args.zico_weight}, '
        f'adapter_burden_weight={args.adapter_burden_weight}, '
        f'real_weight={args.real_weight}, real_metric={args.real_metric}, '
        f'relative_objective={args.relative_objective}, '
        f'selection_mode={args.selection_mode}, '
        f'pareto_objectives={args.pareto_objectives}')

    blocks_by_index = load_assignment(args.assignment)
    base_cfg, primary_cfgs, primary_blocks = load_backbone_blocks(
        args.backbone_config, blocks_by_index)
    emit(f'fixed backbone layers: {len(primary_blocks)}')
    emit(f'allowed local branch layers: {allowed_layers(len(primary_blocks), args)}')

    zero_indicators = active_zero_indicators(args)
    indicator = None
    if len(zero_indicators) > 0:
        # Zero-cost proxies should see shuffled train batches; otherwise ImageNet
        # folder ordering can dominate the score for small num_batch settings.
        data_loader = make_data_loader(args, shuffle=True)
        indicator = ZeroNas(
            dataloader=data_loader,
            indicator=zero_indicators[0] if len(zero_indicators) == 1 else zero_indicators,
            num_batch=args.num_batch)
    real_batches = None
    if args.score_mode in ['real', 'hybrid']:
        real_batch_size = args.real_batch_size or args.batch_size
        real_total_batches = args.real_train_batches + args.real_eval_batches
        emit(
            f'caching real-data proxy batches: batch_size={real_batch_size}, '
            f'train_batches={args.real_train_batches}, '
            f'eval_batches={args.real_eval_batches}, '
            f'train_steps={args.real_train_steps if args.real_train_steps is not None else args.real_train_batches}, '
            f'lr={args.real_lr}')
        real_data_seed = args.seed if args.real_data_seed is None else args.real_data_seed
        random.seed(real_data_seed)
        torch.manual_seed(real_data_seed)
        torch.cuda.manual_seed_all(real_data_seed)
        real_loader = make_data_loader(
            args,
            batch_size=real_batch_size,
            shuffle=not args.real_no_shuffle)
        real_batches = cache_real_batches(real_loader, real_total_batches)
        if len(real_batches) < real_total_batches:
            raise RuntimeError(
                f'Could only cache {len(real_batches)} real batches, need {real_total_batches}')
        real_samples, real_classes = cached_label_stats(real_batches)
        emit(
            f'cached real-data proxy labels: samples={real_samples}, '
            f'unique_classes={real_classes}, shuffle={not args.real_no_shuffle}, '
            f'seed={real_data_seed}')
        if real_classes < args.min_real_classes:
            raise RuntimeError(
                f'Real-data proxy batches contain only {real_classes} classes. '
                f'Increase --real-batch-size/--real-train-batches/--real-eval-batches '
                f'or keep shuffle enabled.')
    memo = {}

    baseline = empty_candidate(len(primary_blocks))
    baseline_item = evaluate_candidate(
        baseline, base_cfg, primary_cfgs, primary_blocks, indicator, args, memo,
        real_batches=real_batches)
    if baseline_item['score'] == -float('inf'):
        raise RuntimeError(f'Baseline backbone is invalid: {baseline_item}')
    if args.relative_objective:
        args.naswot_reference = baseline_item.get('naswot')
        args.zico_reference = baseline_item.get('zico')
        args.adapter_burden_reference = baseline_item.get('adapter_burden')
        args.real_reference_score = baseline_item.get('real_score')
        baseline_item['score'] = objective_value(
            baseline_item.get('naswot'),
            baseline_item.get('zico'),
            baseline_item.get('adapter_burden'),
            baseline_item.get('real_score'),
            args)
        memo[candidate_signature(baseline)] = baseline_item
        emit(
            f'objective baseline reference: naswot={fmt_optional(args.naswot_reference)}, '
            f'zico={fmt_optional(args.zico_reference)}, '
            f'adapter_burden={fmt_optional(args.adapter_burden_reference)}, '
            f'real_score={fmt_optional(args.real_reference_score)}')
    emit(
        f"[baseline] naswot={fmt_optional(baseline_item.get('naswot'))}, "
        f"zico={fmt_optional(baseline_item.get('zico'))}, "
        f"adapter_burden={fmt_optional(baseline_item.get('adapter_burden'))}, "
        f"objective={baseline_item['score']:.6f}, "
        f"real_score={fmt_optional(baseline_item.get('real_score'))}, "
        f"real_loss={fmt_optional(baseline_item.get('real_loss'))}, "
        f"real_acc1={fmt_optional(baseline_item.get('real_acc1'))}, "
        f"size={baseline_item['size']:.3f}M, flops={baseline_item['flops']:.3f}G, "
        f"seed={baseline_item['seed']}")

    population = [baseline_item]
    init_round = 0
    init_memo_limit = args.pop_size * args.max_initial_memo_factor
    while len(population) < args.pop_size and init_round < args.init_max_rounds:
        init_round += 1
        valid_before = len(population)
        memo_before = len(memo)
        samples = [
            sample_candidate(primary_blocks, blocks_by_index, args)
            for _ in range(args.pop_size)
        ]
        samples = unique_candidates(samples)
        items = evaluate_many(
            samples, base_cfg, primary_cfgs, primary_blocks, indicator, args,
            memo, real_batches=real_batches)
        population.extend(item for item in items if item['score'] > -float('inf'))
        best_by_sig = {}
        for item in population:
            sig = candidate_signature(item['candidate'])
            if sig not in best_by_sig or item['score'] > best_by_sig[sig]['score']:
                best_by_sig[sig] = item
        population = rank_population(list(best_by_sig.values()), args)
        if init_round % args.init_log_interval == 0:
            emit(
                f'[init {init_round}/{args.init_max_rounds}] '
                f'population={len(population)}/{args.pop_size}, '
                f'new_valid={max(0, len(population) - valid_before)}, '
                f'new_memo={len(memo) - memo_before}, memo={len(memo)}, '
                f'errors={memo_error_summary(memo)}')
        if len(memo) > init_memo_limit:
            emit(
                f'[init stop] memo={len(memo)} exceeded limit={init_memo_limit}; '
                f'continue with population={len(population)}/{args.pop_size}.')
            break
    if len(population) < args.pop_size:
        emit(
            f'[init stop] population={len(population)}/{args.pop_size} after '
            f'{init_round} rounds; continue with available valid candidates. '
            f'errors={memo_error_summary(memo)}')
    if len(population) == 0:
        raise RuntimeError('No valid candidate found.')

    population = rank_population(population, args)[:args.pop_size]
    best = copy.deepcopy(population[0])
    stagnation = 0
    best_config_path, best_pkl_path = save_best(
        run_dir, best, base_cfg, primary_cfgs, primary_blocks)
    emit(f'initial best saved: {best_config_path}')

    for generation in range(args.generations):
        population = rank_population(population, args)
        if should_replace_best(population[0], best, args):
            best = copy.deepcopy(population[0])
            stagnation = 0
            best_config_path, best_pkl_path = save_best(
                run_dir, best, base_cfg, primary_cfgs, primary_blocks)
        else:
            stagnation += 1

        stalled = stagnation >= args.stagnation_patience
        current_mutation_rate = (
            args.restart_mutation_rate if stalled else args.mutation_rate)
        current_immigrant_ratio = (
            args.restart_immigrant_ratio if stalled else args.immigrant_ratio)

        emit(
            f"[generation {generation + 1}/{args.generations}] "
            f"best={best['score']:.6f}, naswot={fmt_optional(best.get('naswot'))}, "
            f"zico={fmt_optional(best.get('zico'))}, "
            f"adapter_burden={fmt_optional(best.get('adapter_burden'))}, "
            f"real={fmt_optional(best.get('real_score'))}, "
            f"acc1={fmt_optional(best.get('real_acc1'))}, "
            f"size={best['size']:.3f}M, "
            f"flops={best['flops']:.3f}G, branches={format_candidate(best['candidate'])}, "
            f"stagnation={stagnation}, mut={current_mutation_rate:.2f}, "
            f"immigrant_ratio={current_immigrant_ratio:.2f}, memo={len(memo)}")

        elite_count = max(1, int(round(args.pop_size * args.elite_ratio)))
        next_candidates = [copy.deepcopy(item['candidate']) for item in population[:elite_count]]

        immigrant_count = int(round(args.pop_size * current_immigrant_ratio))
        target_children = args.pop_size - elite_count - immigrant_count
        child_candidates = []
        attempts = 0
        while len(child_candidates) < max(0, target_children) and attempts < args.pop_size * 50:
            attempts += 1
            parent_a = tournament(population, args)['candidate']
            parent_b = tournament(population, args)['candidate']
            if random.random() < args.crossover_rate:
                child = crossover_candidate(parent_a, parent_b, primary_blocks, blocks_by_index, args)
            else:
                child = copy.deepcopy(parent_a)
            child = mutate_candidate(
                child, primary_blocks, blocks_by_index, args,
                rate=current_mutation_rate)
            child_candidates.append(child)

        immigrants = [
            sample_candidate(primary_blocks, blocks_by_index, args)
            for _ in range(max(0, immigrant_count))
        ]
        next_candidates.extend(child_candidates)
        next_candidates.extend(immigrants)
        next_candidates = unique_candidates(next_candidates)
        next_items = evaluate_many(
            next_candidates, base_cfg, primary_cfgs, primary_blocks, indicator,
            args, memo, real_batches=real_batches)
        next_items = [item for item in next_items if item['score'] > -float('inf')]
        next_items.extend(population[:elite_count])

        best_by_sig = {}
        for item in next_items:
            sig = candidate_signature(item['candidate'])
            if sig not in best_by_sig or item['score'] > best_by_sig[sig]['score']:
                best_by_sig[sig] = item
        population = rank_population(list(best_by_sig.values()), args)[:args.pop_size]

    population = rank_population(population, args)
    if len(population) > 0 and should_replace_best(population[0], best, args):
        best = copy.deepcopy(population[0])
    best_config_path, best_pkl_path = save_best(
        run_dir, best, base_cfg, primary_cfgs, primary_blocks)
    emit('Search complete')
    emit(
        f"[final best] objective={best['score']:.6f}, "
        f"naswot={fmt_optional(best.get('naswot'))}, "
        f"zico={fmt_optional(best.get('zico'))}, "
        f"adapter_burden={fmt_optional(best.get('adapter_burden'))}, "
        f"real_score={fmt_optional(best.get('real_score'))}, "
        f"real_loss={fmt_optional(best.get('real_loss'))}, "
        f"real_acc1={fmt_optional(best.get('real_acc1'))}, "
        f"size={best['size']:.3f}M, flops={best['flops']:.3f}G, "
        f"seed={best['seed']}, branches={format_candidate(best['candidate'])}")
    emit(f'best config: {best_config_path}')
    emit(f'best pkl: {best_pkl_path}')
    top_items = ranked_valid_items(memo, args)
    baseline_config_path, baseline_pkl_path = save_baseline_config(
        run_dir, base_cfg, baseline_item)
    top_dir, summary_path, saved_top = save_top_k(
        run_dir, top_items, args.top_k, base_cfg, primary_cfgs, primary_blocks,
        selection_mode=args.selection_mode,
        pareto_objectives=args.pareto_objectives)
    emit(f'baseline original config: {baseline_config_path}')
    emit(f'baseline original pkl: {baseline_pkl_path}')
    emit(f'top-{args.top_k} configs saved to: {top_dir}')
    emit(f'top-{args.top_k} summary: {summary_path}')
    for rank, (config_path, _, item) in enumerate(saved_top, start=1):
        emit(
            f"[top {rank}] objective={item['score']:.6f}, "
            f"naswot={fmt_optional(item.get('naswot'))}, "
            f"zico={fmt_optional(item.get('zico'))}, "
            f"adapter_burden={fmt_optional(item.get('adapter_burden'))}, "
            f"real_score={fmt_optional(item.get('real_score'))}, "
            f"real_loss={fmt_optional(item.get('real_loss'))}, "
            f"real_acc1={fmt_optional(item.get('real_acc1'))}, "
            f"size={item['size']:.3f}M, flops={item['flops']:.3f}G, "
            f"pareto_rank={item.get('_pareto_rank')}, "
            f"config={config_path}")
    log_file.close()


if __name__ == '__main__':
    main()
