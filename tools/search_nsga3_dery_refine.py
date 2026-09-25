"""DeRy-initialized joint backbone/branch NSGA-III refinement.

This script keeps the original DeRy architecture as the initial backbone and
searches local refinements under the same resource budget. A candidate can:

1. keep or replace a small number of original backbone blocks;
2. add local branch blocks with sum/gate fusion;
3. optimize three objectives: maximize proxy-estimated performance, minimize
   parameter count, and minimize FLOPs.

The original GA/EA scripts are not modified. This is a conservative joint
search route for debugging and later short-training validation.
"""

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
from collections import Counter

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_DIR = os.path.join(REPO_ROOT, 'simlarity')
for path in [REPO_ROOT, SIM_DIR, os.path.join(REPO_ROOT, 'third_package')]:
    if path not in sys.path:
        sys.path.insert(0, path)

from tools.search_nsga3_multiobj import (  # noqa: E402
    ensure_output_dirs,
    export_configs,
    final_front,
    import_pymoo,
    make_callback,
    make_ref_dirs,
    normalized,
)


METHOD_NOTE = (
    'We initialize the evolutionary search from the original DeRy architecture '
    'and perform constrained multi-objective refinement. Each candidate is a '
    'local perturbation of the original backbone, optionally augmented with '
    'local branches. The objectives are to maximize proxy-estimated '
    'performance, minimize parameter count, and minimize FLOPs. CLAS is the '
    'canonical HeteroWeave proxy; alternative proxies are retained only for '
    'ablation and diagnostic runs.'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Original-DeRy initialized joint backbone/branch NSGA-III refinement.')
    parser.add_argument(
        'backbone_config',
        nargs='?',
        default='configs/imagenet/dery_baseline_100e.py',
        help='Original DeRy config used as the initialization anchor.')
    parser.add_argument('--assignment', default='assets/component_pool/assignment_hybrid_4.pkl')
    parser.add_argument('--data-config', default='configs/_base_/datasets/imagenet_bs64_swin_224.py')
    parser.add_argument('--data-prefix', default='data/imagenet/train')
    parser.add_argument('--ann-file', default=None)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-batch', type=int, default=5)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--population-size', '--pop-size', dest='population_size', type=int, default=96)
    parser.add_argument('--generations', type=int, default=120)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', default='simlarity/out/nsga3_dery_refine_30m')
    parser.add_argument(
        '--proxy',
        choices=[
            'naswot', 'zico', 'raw_swap', 'layer_swap_sum',
            'CLAS'],
        default='CLAS',
        help=(
            'Training-free proxy used as the first objective. raw_swap counts '
            'global activation patterns; layer_swap_sum sums per-position '
            'pattern counts; CLAS applies square-root aggregation to each '
            'position before summation.'))
    parser.add_argument(
        '--proxy-data-seed', type=int, default=11,
        help='Seed used once to cache the same ImageNet proxy batches.')
    parser.add_argument(
        '--proxy-model-seed', type=int, default=11,
        help='Fixed initialization seed shared by every candidate.')
    parser.add_argument(
        '--swap-image-count', type=int, default=32,
        help='Number of cached images used by SWAP-family proxies.')
    parser.add_argument(
        '--performance-proxy-mode',
        choices=['raw', 'zico_ntk'],
        default='raw',
        help=(
            'raw keeps the original proxy objective. zico_ntk uses a TE-NAS-style '
            'combined proxy: log(ZiCo) - ntk_weight * log(NTK condition).'))
    parser.add_argument(
        '--ntk-weight',
        type=float,
        default=0.2,
        help='Weight for log(NTK condition) in --performance-proxy-mode zico_ntk.')
    parser.add_argument(
        '--ntk-max-samples',
        type=int,
        default=4,
        help=(
            'Maximum samples used for empirical NTK condition. Formula follows '
            'TE-NAS; this cap avoids OOM on ImageNet-sized DeRy models.'))
    parser.add_argument(
        '--ntk-num-batch',
        type=int,
        default=1,
        help='Number of dataloader batches cached for NTK evaluation.')
    parser.add_argument(
        '--ntk-eps',
        type=float,
        default=1e-6,
        help='Eigenvalue floor used when computing NTK condition.')
    parser.add_argument(
        '--max-ntk-condition',
        type=float,
        default=None,
        help=(
            'Optional hard trainability/stability gate applied before NSGA-III '
            'selection. Candidates with empirical NTK condition above this '
            'threshold are marked infeasible. Requires '
            '--performance-proxy-mode zico_ntk.'))
    parser.add_argument(
        '--ntk-train-mode',
        action='store_true',
        help='Use model.train() for NTK. Default keeps TE-NAS code default train_mode=False.')
    parser.add_argument('--objectives', default='performance,parameters,flops')
    parser.add_argument('--ref-partitions', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=None)
    parser.add_argument(
        '--structure-min-quota', type=int, default=0,
        help=(
            'Minimum retained final candidates for each available branch '
            'fusion/replacement family. This affects selection, not scoring.'))
    parser.add_argument('--minC', dest='min_params', type=float, default=None)
    parser.add_argument('--C', '--maxC', dest='max_params', type=float, default=30.0)
    parser.add_argument('--minflop-C', '--minflop_C', dest='min_flops', type=float, default=None)
    parser.add_argument('--flop-C', '--flop_C', dest='max_flops', type=float, default=6.0)
    parser.add_argument('--skip-flops-eval', action='store_true')
    parser.add_argument('--start-ratio', type=float, default=0.35)
    parser.add_argument('--max-branch-layers', type=int, default=3)
    parser.add_argument(
        '--branch-layers', type=int, nargs='+', default=[1, 2, 3],
        help=(
            'Aligned positions that may add a second component. The canonical '
            'four-position space uses K_0=1 and K_1=K_2=K_3=2.'))
    parser.add_argument(
        '--max-components-per-position', dest='max_branches_per_layer',
        type=int, default=2,
        help='Maximum total components at each enabled position.')
    parser.add_argument('--max-backbone-replacements', type=int, default=4)
    parser.add_argument(
        '--operators',
        nargs='+',
        choices=['sum', 'gate'],
        default=['sum'],
        help='The canonical HeteroWeave protocol uses fixed mean fusion (sum).')
    parser.add_argument(
        '--branch-type-policy',
        choices=['any', 'same-io', 'no-vit'],
        default='any')
    parser.add_argument(
        '--backbone-replacement-policy',
        choices=['any', 'same-io', 'no-vit'],
        default='same-io',
        help='Filter policy for backbone replacement blocks. Defaults to same-io for stability.')
    parser.add_argument('--max-adapter-burden', type=float, default=None)
    parser.add_argument('--max-type-switches', type=int, default=None)
    parser.add_argument(
        '--anchor-proxy-penalty',
        '--replacement-penalty',
        type=float,
        default=0.0,
        help=(
            'Training-free anchor-aware penalty applied to the first objective. '
            'The optimized proxy becomes proxy - penalty * num_backbone_replacements. '
            'Default 0 keeps the old raw-proxy behavior.'))
    parser.add_argument(
        '--parameter-proxy-penalty',
        type=float,
        default=0.0,
        help=(
            'Penalty per 1M parameters applied to the first objective. '
            'Useful for correcting ZiCo parameter-count bias. Default 0.'))
    parser.add_argument(
        '--flops-proxy-penalty',
        type=float,
        default=0.0,
        help=(
            'Penalty per 1G FLOPs applied to the first objective. '
            'Useful for correcting ZiCo compute bias. Default 0.'))
    parser.add_argument(
        '--staged-branch-first',
        action='store_true',
        default=True,
        help=(
            'Run a branch-only stage before the joint replacement+branch stage. '
            'This lets the unified Space-C search seriously explore additive '
            'branch refinements without seeding known Space-A winners.'))
    parser.add_argument(
        '--no-staged-branch-first', dest='staged_branch_first',
        action='store_false',
        help='Disable the canonical 40-generation initialization stage.')
    parser.add_argument(
        '--branch-first-generations',
        type=int,
        default=40,
        help=(
            'Generations for the branch-only first stage. Defaults to half of '
            '--generations, at least 1, when --staged-branch-first is enabled.'))
    parser.add_argument(
        '--branch-first-population-size',
        type=int,
        default=96,
        help='Population size for the branch-only first stage. Defaults to --population-size.')
    parser.add_argument(
        '--branch-first-two-branch-ratio',
        type=float,
        default=0.75,
        help='Probability of sampling two branch layers in the branch-only first stage.')
    parser.add_argument(
        '--structured-sampling',
        action='store_true',
        help='Use structure-aware integer initial populations instead of pure float random sampling.')
    parser.add_argument(
        '--joint-branch-only-ratio',
        type=float,
        default=0.25,
        help='Initial population ratio for branch-only candidates in the joint stage.')
    parser.add_argument(
        '--joint-replacement-only-ratio',
        type=float,
        default=0.10,
        help='Initial population ratio for replacement-only candidates in the joint stage.')
    parser.add_argument(
        '--joint-identity-ratio',
        type=float,
        default=0.05,
        help=(
            'Initial population ratio for unchanged original-backbone candidates '
            'without extra branches in the joint stage.'))
    parser.add_argument(
        '--joint-replacement-enable-ratio',
        type=float,
        default=0.85,
        help=(
            'When sampling nominal replacement+branch candidates, probability '
            'of enabling at least one backbone replacement.'))
    parser.add_argument(
        '--joint-branch-enable-ratio',
        type=float,
        default=0.85,
        help=(
            'When sampling nominal replacement+branch candidates, probability '
            'of enabling at least one local branch.'))
    parser.add_argument(
        '--joint-two-branch-ratio',
        type=float,
        default=0.50,
        help='Probability of sampling two branch layers for branch-bearing joint-stage candidates.')
    parser.add_argument('--device', choices=['cuda'], default='cuda')
    return parser.parse_args()


def import_real_search_deps():
    import torch
    from simlarity.zero_nas import ZeroNas
    import tools.ea_local_branch_naswot as ea
    return torch, ZeroNas, ea


SWAP_PROXIES = {'raw_swap', 'layer_swap_sum', 'CLAS'}


def proxy_score_key(proxy_name):
    """Map the paper-facing proxy name to the internal score key."""
    return 'layer_swap_sqrt' if proxy_name == 'CLAS' else proxy_name


def clone_proxy_batch(data):
    cloned = {}
    for key, value in data.items():
        if hasattr(value, 'detach'):
            cloned[key] = value.detach().cpu().clone()
        else:
            cloned[key] = value
    return cloned


def cache_proxy_batches(data_loader, num_batch):
    batches = []
    for index, data in enumerate(data_loader):
        batches.append(clone_proxy_batch(data))
        if index + 1 >= num_batch:
            break
    if len(batches) < num_batch:
        raise RuntimeError(
            f'Only cached {len(batches)} proxy batches; need {num_batch}.')
    return batches


def compute_swap_scores(cfg, context, args):
    """Compute the three SWAP ablations from one fixed-batch forward."""
    from tools.evaluate_dery_future_proxy import LogitWrapper
    from tools.evaluate_training_free_replacements import ActivationPatternMonitor

    torch = context['torch']
    ea = context['ea']
    model = None
    monitor = None
    try:
        torch.manual_seed(int(args.proxy_model_seed))
        torch.cuda.manual_seed_all(int(args.proxy_model_seed))
        model = ea.build_classifier(cfg.model)
        model.init_weights()
        model.cuda().eval()
        wrapper = LogitWrapper(model).cuda().eval()
        images = context['proxy_images'].cuda(non_blocking=True)
        monitor = ActivationPatternMonitor(
            wrapper, batch_size=images.shape[0], collect_groups=False)
        with torch.no_grad():
            _ = wrapper(images)
        pattern = monitor.scores()
        return dict(
            raw_swap=float(pattern['swap_score']),
            layer_swap_sum=float(pattern['layer_swap_sum']),
            layer_swap_sqrt=float(pattern['layer_swap_sqrt_sum']))
    finally:
        if monitor is not None:
            monitor.close()
        if model is not None:
            del model
        torch.cuda.empty_cache()


def is_same_block(left, right):
    return left.print_split() == right.print_split()


def replacement_pool(layer, primary_blocks, blocks_by_index, args):
    primary = primary_blocks[layer]
    pool = []
    for block in blocks_by_index.get(layer, []):
        if is_same_block(block, primary):
            continue
        if not block_allowed_by_policy(block, primary, args.backbone_replacement_policy):
            continue
        pool.append(block)
    return pool


def block_allowed_by_policy(block, primary, policy):
    import tools.ea_local_branch_naswot as ea

    if policy == 'any':
        return True
    if policy == 'no-vit':
        return (
            ea.feature_type(block.in_size) != 'vit' and
            ea.feature_type(block.out_size) != 'vit')
    if policy == 'same-io':
        return (
            ea.feature_type(block.in_size) == ea.feature_type(primary.in_size) and
            ea.feature_type(block.out_size) == ea.feature_type(primary.out_size))
    raise ValueError(f'Unknown replacement policy: {policy}')


def ea_branch_policy(block, primary, args):
    # Import lazily through the already-loaded EA module in runtime paths.
    import tools.ea_local_branch_naswot as ea
    return ea.branch_allowed_by_type_policy(block, primary, args)


def make_backbone_adapter_cfg(src_size, dst_size):
    import tools.ea_local_branch_naswot as ea

    src_type = ea.feature_type(src_size)
    dst_type = ea.feature_type(dst_size)
    input_channel = ea.channels(src_size)
    output_channel = ea.channels(dst_size)
    mode = f'{src_type}2{dst_type}'

    if mode == 'vit2vit':
        return dict(
            input_channel=input_channel,
            output_channel=output_channel,
            num_fc=1,
            num_conv=0,
            mode=mode)

    stride = 1
    if src_type == 'cnn' and len(src_size) == 3 and len(dst_size) == 3:
        stride = 1 if src_size[1] / dst_size[1] < 2 else 2
    return dict(
        input_channel=input_channel,
        output_channel=output_channel,
        stride=stride,
        num_fc=0,
        num_conv=1,
        mode=mode)


def update_backbone_connections(cfg, primary_blocks):
    import tools.ea_local_branch_naswot as ea

    cfg.model.backbone.base_channels = ea.channels(primary_blocks[0].in_size)
    cfg.model.backbone.adapter_list = [
        make_backbone_adapter_cfg(
            primary_blocks[index].out_size,
            primary_blocks[index + 1].in_size)
        for index in range(len(primary_blocks) - 1)
    ]
    cfg.model.head.in_channels = ea.channels(primary_blocks[-1].out_size)
    return cfg


def build_refined_config(base_cfg, primary_blocks, branch_candidate):
    import tools.ea_local_branch_naswot as ea

    primary_cfgs = [block.print_split() for block in primary_blocks]
    cfg = ea.build_candidate_config(
        base_cfg, primary_cfgs, primary_blocks, branch_candidate)
    return update_backbone_connections(cfg, primary_blocks)


def collect_ntk_batches(data_loader, args):
    batches = []
    for index, data in enumerate(data_loader):
        if index >= args.ntk_num_batch:
            break
        if 'img' not in data:
            continue
        img = data['img']
        label = data.get('gt_label')
        if hasattr(img, 'detach'):
            img = img.detach().cpu()
        if label is not None and hasattr(label, 'detach'):
            label = label.detach().view(-1).long().cpu()
        batches.append((img, label))
    return batches


def safe_eigvalsh(matrix, torch):
    if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'eigvalsh'):
        return torch.linalg.eigvalsh(matrix)
    eigenvalues, _ = torch.symeig(matrix, eigenvectors=False)
    return eigenvalues


def compute_tenas_ntk_condition(cfg, context, args):
    """TE-NAS-style empirical NTK condition number.

    The implementation follows the official TE-NAS code path: for each input
    sample, backpropagate a vector of ones from that sample's logits, collect
    gradients of weight parameters, form the NTK Gram matrix by gradient inner
    products, and return lambda_max / lambda_min.
    """
    torch = context['torch']
    ea = context['ea']
    if not context.get('ntk_batches'):
        raise RuntimeError('No cached batches available for NTK evaluation.')

    model = None
    try:
        model = ea.ClassifierForwardWrapper(ea.build_classifier(cfg.model)).cuda()
        if args.ntk_train_mode:
            model.train()
        else:
            model.eval()

        grads = []
        samples_left = max(2, int(args.ntk_max_samples))
        for img, _ in context['ntk_batches']:
            if samples_left <= 0:
                break
            x = img[:samples_left].cuda(non_blocking=True)
            if x.numel() == 0:
                continue
            model.zero_grad()
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[-1]
            for sample_index in range(x.shape[0]):
                model.zero_grad()
                logits[sample_index:sample_index + 1].backward(
                    torch.ones_like(logits[sample_index:sample_index + 1]),
                    retain_graph=True)
                sample_grad = []
                for name, weight in model.named_parameters():
                    if 'weight' in name and weight.grad is not None:
                        sample_grad.append(weight.grad.detach().view(-1).float())
                if sample_grad:
                    grads.append(torch.cat(sample_grad, dim=0).cpu())
                samples_left -= 1
                if samples_left <= 0:
                    break
            del logits
            torch.cuda.empty_cache()

        if len(grads) < 2:
            raise RuntimeError('Need at least two valid samples for NTK condition.')
        grad_matrix = torch.stack(grads, dim=0).double()
        ntk = torch.einsum('nc,mc->nm', grad_matrix, grad_matrix)
        eigenvalues = safe_eigvalsh(ntk, torch)
        min_eval = max(float(eigenvalues[0].item()), float(args.ntk_eps))
        max_eval = max(float(eigenvalues[-1].item()), float(args.ntk_eps))
        condition = max_eval / min_eval
        if not math.isfinite(condition):
            condition = 1e8
        return float(condition)
    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()


def make_context(args):
    torch, ZeroNas, ea = import_real_search_deps()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for real zero-cost evaluation.')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.score_mode = 'swap' if args.proxy in SWAP_PROXIES else args.proxy
    args.naswot_weight = 1.0 if args.proxy == 'naswot' else 0.0
    args.zico_weight = 1.0 if args.proxy == 'zico' else 0.0
    args.adapter_burden_weight = 0.0
    args.real_weight = 0.0
    args.real_metric = 'neg_loss'
    args.real_reference_score = None
    args.min_real_delta = None
    args.relative_objective = False
    args.naswot_reference = None
    args.zico_reference = None
    args.adapter_burden_reference = None
    args.max_branches_per_layer = max(
        1, int(args.max_branches_per_layer))
    args.max_branch_layers = max(0, args.max_branch_layers)
    args.max_backbone_replacements = max(0, args.max_backbone_replacements)
    args.skip_flops_eval = bool(args.skip_flops_eval)
    args.fixed_eval_seed = int(args.proxy_model_seed)
    if args.proxy == 'zico' and args.num_batch < 2:
        args.num_batch = 2

    blocks_by_index = ea.load_assignment(args.assignment)
    base_cfg, base_primary_cfgs, base_primary_blocks = ea.load_backbone_blocks(
        args.backbone_config, blocks_by_index)
    allowed_branch_layers = ea.allowed_layers(len(base_primary_blocks), args)
    backbone_layers = list(range(len(base_primary_blocks)))
    backbone_pools = {
        layer: replacement_pool(layer, base_primary_blocks, blocks_by_index, args)
        for layer in backbone_layers
    }

    # Cache the shuffled batches exactly once. Every architecture and every
    # search method then observes identical tensors; the evolutionary seed no
    # longer changes the proxy data panel.
    random.seed(int(args.proxy_data_seed))
    np.random.seed(int(args.proxy_data_seed))
    torch.manual_seed(int(args.proxy_data_seed))
    torch.cuda.manual_seed_all(int(args.proxy_data_seed))
    data_loader = ea.make_data_loader(args, shuffle=True)
    cached_batches = cache_proxy_batches(data_loader, args.num_batch)
    proxy_images = torch.cat(
        [batch['img'] for batch in cached_batches], dim=0
    )[:int(args.swap_image_count)].contiguous()
    if proxy_images.shape[0] < int(args.swap_image_count):
        raise RuntimeError(
            f'Cached only {proxy_images.shape[0]} SWAP images; requested '
            f'{args.swap_image_count}. Increase --batch-size or --num-batch.')
    ntk_batches = (
        [(batch['img'], batch.get('gt_label')) for batch in cached_batches[
            :args.ntk_num_batch]]
        if args.performance_proxy_mode == 'zico_ntk'
        else [])
    indicator = ZeroNas(
        dataloader=cached_batches,
        indicator=args.proxy if args.proxy in ('naswot', 'zico') else 'zico',
        num_batch=args.num_batch)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    return dict(
        torch=torch,
        ea=ea,
        base_cfg=base_cfg,
        base_primary_cfgs=base_primary_cfgs,
        base_primary_blocks=base_primary_blocks,
        blocks_by_index=blocks_by_index,
        backbone_layers=backbone_layers,
        allowed_branch_layers=allowed_branch_layers,
        backbone_pools=backbone_pools,
        indicator=indicator,
        ntk_batches=ntk_batches,
        proxy_images=proxy_images)


def bounds(context, args):
    xu = []
    for layer in context['backbone_layers']:
        pool = context['backbone_pools'].get(layer, [])
        xu.extend([1, max(0, len(pool) - 1)])
    for layer in context['allowed_branch_layers']:
        pool = context['blocks_by_index'].get(layer, [])
        xu.extend([1, max(0, len(pool) - 1), max(0, len(args.operators) - 1)])
    if not xu:
        xu = [0, 0, 0]
    return np.zeros(len(xu), dtype=float), np.asarray(xu, dtype=float)


def clipped_ratio(value):
    return max(0.0, min(1.0, float(value)))


def make_empty_encoded_individual(context):
    n_backbone = len(context['backbone_layers'])
    n_branch = len(context['allowed_branch_layers'])
    return np.zeros(2 * n_backbone + 3 * n_branch, dtype=float)


def branch_offset(context, layer):
    start = 2 * len(context['backbone_layers'])
    branch_index = context['allowed_branch_layers'].index(layer)
    return start + 3 * branch_index


def sample_replacements_into_x(x, primary_blocks, context, args, rng):
    if args.max_backbone_replacements <= 0:
        return primary_blocks
    candidate_layers = [
        layer for layer in context['backbone_layers']
        if context['backbone_pools'].get(layer)
    ]
    if not candidate_layers:
        return primary_blocks
    max_count = min(args.max_backbone_replacements, len(candidate_layers))
    count = int(rng.integers(1, max_count + 1))
    for layer in rng.choice(candidate_layers, size=count, replace=False):
        layer = int(layer)
        pool = context['backbone_pools'][layer]
        block_index = int(rng.integers(0, len(pool)))
        offset = 2 * layer
        x[offset] = 1.0
        x[offset + 1] = float(block_index)
        primary_blocks[layer] = pool[block_index]
    return primary_blocks


def sample_branches_into_x(x, primary_blocks, context, args, rng, two_branch_ratio):
    ea = context['ea']
    branch_layers = list(context['allowed_branch_layers'])
    max_layers = min(args.max_branch_layers, len(branch_layers))
    if max_layers <= 0:
        return
    if max_layers >= 2 and rng.random() < clipped_ratio(two_branch_ratio):
        count = 2
    else:
        count = 1
    count = min(count, max_layers)
    selected_layers = sorted(rng.choice(branch_layers, size=count, replace=False))
    branch_candidate = ea.empty_candidate(len(primary_blocks))
    for layer in selected_layers:
        layer = int(layer)
        pool = ea.branch_pool(
            layer, branch_candidate, primary_blocks,
            context['blocks_by_index'], args)
        if not pool:
            continue
        branch_index = int(rng.integers(0, len(pool)))
        operator_index = int(rng.integers(0, len(args.operators)))
        offset = branch_offset(context, layer)
        x[offset] = 1.0
        x[offset + 1] = float(branch_index)
        x[offset + 2] = float(operator_index)
        branch_candidate[layer] = ea.LocalGene(
            branches=[pool[branch_index]],
            operator=args.operators[operator_index])


def sample_encoded_individual(context, args, rng, structure):
    x = make_empty_encoded_individual(context)
    primary_blocks = list(context['base_primary_blocks'])
    if structure == 'identity':
        return x

    enable_replacement = structure in ('replacement_only', 'replacement_branch')
    enable_branch = structure in ('branch_only', 'replacement_branch')
    if structure == 'replacement_branch':
        enable_replacement = rng.random() < clipped_ratio(
            args.joint_replacement_enable_ratio)
        enable_branch = rng.random() < clipped_ratio(
            args.joint_branch_enable_ratio)

    if enable_replacement:
        primary_blocks = sample_replacements_into_x(
            x, primary_blocks, context, args, rng)
    if enable_branch:
        two_branch_ratio = (
            args.branch_first_two_branch_ratio
            if structure == 'branch_only'
            else args.joint_two_branch_ratio)
        sample_branches_into_x(
            x, primary_blocks, context, args, rng, two_branch_ratio)
    return x


def structured_initial_population(context, args, stage):
    pop_size = max(1, int(args.population_size))
    rng = np.random.default_rng(args.seed + (17 if stage == 'branch_first' else 29))
    rows = []
    if stage == 'branch_first':
        rows.append(make_empty_encoded_individual(context))
        while len(rows) < pop_size:
            rows.append(sample_encoded_individual(
                context, args, rng, 'branch_only'))
        return np.asarray(rows, dtype=float)

    identity_ratio = clipped_ratio(args.joint_identity_ratio)
    branch_only_ratio = clipped_ratio(args.joint_branch_only_ratio)
    replacement_only_ratio = clipped_ratio(args.joint_replacement_only_ratio)
    explicit_ratio = identity_ratio + branch_only_ratio + replacement_only_ratio
    if explicit_ratio > 0.95:
        scale = 0.95 / explicit_ratio
        identity_ratio *= scale
        branch_only_ratio *= scale
        replacement_only_ratio *= scale

    rows.append(make_empty_encoded_individual(context))
    while len(rows) < pop_size:
        draw = rng.random()
        if draw < identity_ratio:
            structure = 'identity'
        elif draw < identity_ratio + branch_only_ratio:
            structure = 'branch_only'
        elif draw < identity_ratio + branch_only_ratio + replacement_only_ratio:
            structure = 'replacement_only'
        else:
            structure = 'replacement_branch'
        rows.append(sample_encoded_individual(context, args, rng, structure))
    return np.asarray(rows, dtype=float)


def decode_individual(x, context, args):
    ea = context['ea']
    values = [int(round(float(value))) for value in x]
    primary_blocks = list(context['base_primary_blocks'])

    cursor = 0
    enabled_replacements = []
    for layer in context['backbone_layers']:
        enabled_raw = float(x[cursor])
        enabled = values[cursor] > 0
        replacement_index = values[cursor + 1]
        cursor += 2
        pool = context['backbone_pools'].get(layer, [])
        if enabled and pool:
            block = pool[min(max(0, replacement_index), len(pool) - 1)]
            enabled_replacements.append((enabled_raw, layer, block))

    enabled_replacements.sort(key=lambda item: item[0], reverse=True)
    for _, layer, block in enabled_replacements[:args.max_backbone_replacements]:
        primary_blocks[layer] = block

    branch_candidate = ea.empty_candidate(len(primary_blocks))
    for layer in context['allowed_branch_layers']:
        enabled = values[cursor] > 0
        branch_index = values[cursor + 1]
        operator_index = values[cursor + 2]
        cursor += 3
        if not enabled:
            continue
        pool = ea.branch_pool(
            layer, branch_candidate, primary_blocks,
            context['blocks_by_index'], args)
        if not pool:
            continue
        branch = pool[min(max(0, branch_index), len(pool) - 1)]
        operator = args.operators[min(max(0, operator_index), len(args.operators) - 1)]
        branch_candidate[layer] = ea.LocalGene(branches=[branch], operator=operator)

    branch_candidate = ea.enforce_candidate(
        branch_candidate, primary_blocks, context['blocks_by_index'], args)
    return primary_blocks, branch_candidate


def signature(primary_blocks, branch_candidate, ea):
    backbone_sig = tuple(ea.block_key(block) for block in primary_blocks)
    branch_sig = ea.candidate_signature(branch_candidate)
    return backbone_sig, branch_sig


def backbone_delta_summary(primary_blocks, context):
    rows = []
    for index, (base, current) in enumerate(zip(context['base_primary_blocks'], primary_blocks)):
        if is_same_block(base, current):
            continue
        rows.append(f'L{index}:{base}->{current}')
    return '; '.join(rows) if rows else 'keep original backbone'


def count_branch_layers(branch_candidate):
    return sum(
        1 for gene in branch_candidate
        if gene is not None and len(gene.branches) > 0)


def structure_type(num_backbone_replacements, num_branch_layers):
    if num_backbone_replacements == 0 and num_branch_layers == 0:
        return 'baseline'
    if num_backbone_replacements == 0:
        return 'branch_only'
    if num_branch_layers == 0:
        return 'replacement_only'
    return 'replacement_branch'


def detailed_structure_group(num_backbone_replacements, branch_candidate):
    operators = {
        gene.operator for gene in branch_candidate
        if gene is not None and len(gene.branches) > 0}
    if num_backbone_replacements > 0:
        return 'replacement_branch' if operators else 'replacement_only'
    if operators == {'gate'}:
        return 'branch_gate'
    if operators == {'sum'}:
        return 'branch_sum'
    if operators == {'gate', 'sum'}:
        return 'branch_mixed'
    return 'baseline'


def objective_proxy_label(args):
    base = base_objective_proxy_label(args)
    if has_proxy_penalty(args):
        return f'adjusted_{base}'
    return base


def base_objective_proxy_label(args):
    if args.performance_proxy_mode == 'zico_ntk':
        return 'performance_zico_ntk'
    return args.proxy


def has_proxy_penalty(args):
    return (
        args.anchor_proxy_penalty > 0 or
        args.parameter_proxy_penalty > 0 or
        args.flops_proxy_penalty > 0)


def uses_nonraw_objective(args):
    return args.performance_proxy_mode != 'raw' or has_proxy_penalty(args)


def individual_key(x):
    return tuple(int(round(float(value))) for value in x)


def evaluate_refined_candidate(
        primary_blocks, branch_candidate, raw_key, context, args, records,
        source='search'):
    ea = context['ea']
    key = signature(primary_blocks, branch_candidate, ea)
    if key in records:
        return records[key]

    primary_cfgs = [block.print_split() for block in primary_blocks]
    eval_cfg = build_refined_config(
        context['base_cfg'], primary_blocks, branch_candidate)
    eval_base_cfg = copy.deepcopy(context['base_cfg'])
    eval_base_cfg.model = eval_cfg.model
    item = ea.evaluate_candidate(
        branch_candidate,
        eval_base_cfg,
        primary_cfgs,
        primary_blocks,
        context['indicator'],
        args,
        memo={})
    proxy_score = item.get(proxy_score_key(args.proxy))
    swap_scores = dict(
        raw_swap=None, layer_swap_sum=None, layer_swap_sqrt=None)
    proxy_error = None
    if item.get('error') is None and args.proxy in SWAP_PROXIES:
        try:
            swap_scores = compute_swap_scores(eval_cfg, context, args)
            proxy_score = swap_scores[proxy_score_key(args.proxy)]
        except Exception as exc:
            proxy_error = f'swap_{type(exc).__name__}: {exc}'
    num_backbone_replacements = sum(
        not is_same_block(base, current)
        for base, current in zip(context['base_primary_blocks'], primary_blocks))
    num_branch_layers = count_branch_layers(branch_candidate)

    if (item.get('error') is None and proxy_error is None and
            proxy_score is not None):
        proxy_score = float(proxy_score)
        size = float(item['size'])
        flops = float(item['flops'])
        naswot = item.get('naswot')
        zico = item.get('zico')
        naswot = float(naswot) if naswot is not None else None
        zico = float(zico) if zico is not None else None
        ntk_condition = None
        ntk_trainability_score = None
        performance_proxy_score = proxy_score
        performance_error = None
        if args.performance_proxy_mode == 'zico_ntk':
            try:
                ntk_condition = compute_tenas_ntk_condition(
                    eval_cfg, context, args)
                ntk_trainability_score = -math.log(
                    max(float(ntk_condition), float(args.ntk_eps)))
                performance_proxy_score = (
                    math.log(max(float(zico), float(args.ntk_eps))) +
                    args.ntk_weight * ntk_trainability_score)
                if (args.max_ntk_condition is not None and
                        ntk_condition > args.max_ntk_condition):
                    performance_error = (
                        'ntk_stability_gate: condition='
                        f'{ntk_condition:.9g} > max='
                        f'{args.max_ntk_condition:.9g}')
            except Exception as exc:
                performance_error = f'ntk_{type(exc).__name__}: {exc}'

        if performance_error is not None:
            objectives = [1e9, 1e9, 1e9]
            error = performance_error
            adjusted_proxy_score = -1e9
            objective_proxy_score = -1e9
            replacement_proxy_penalty = 0.0
            parameter_proxy_penalty = 0.0
            flops_proxy_penalty = 0.0
            complexity_proxy_penalty = 0.0
        else:
            replacement_proxy_penalty = (
                args.anchor_proxy_penalty * float(num_backbone_replacements))
            parameter_proxy_penalty = args.parameter_proxy_penalty * size
            flops_proxy_penalty = args.flops_proxy_penalty * flops
            complexity_proxy_penalty = parameter_proxy_penalty + flops_proxy_penalty
            adjusted_proxy_score = (
                performance_proxy_score -
                replacement_proxy_penalty -
                complexity_proxy_penalty)
            objective_proxy_score = adjusted_proxy_score
            objectives = [-objective_proxy_score, size, flops]
            error = None
    else:
        objectives = [1e9, 1e9, 1e9]
        error = proxy_error or item.get('error') or 'invalid'
        proxy_score = -1e9
        adjusted_proxy_score = -1e9
        objective_proxy_score = -1e9
        performance_proxy_score = -1e9
        ntk_condition = None
        ntk_trainability_score = None
        replacement_proxy_penalty = 0.0
        parameter_proxy_penalty = 0.0
        flops_proxy_penalty = 0.0
        complexity_proxy_penalty = 0.0
        naswot = item.get('naswot')
        zico = item.get('zico')
        naswot = float(naswot) if naswot is not None else None
        zico = float(zico) if zico is not None else None
        size = float(item.get('size', 1e9))
        flops = float(item.get('flops', 1e9))

    records[key] = dict(
        id=len(records),
        individual=list(raw_key),
        candidate_signature=repr(key),
        primary_blocks=primary_blocks,
        branch_candidate=branch_candidate,
        proxy=args.proxy,
        proxy_score=proxy_score,
        raw_proxy_score=proxy_score,
        performance_proxy_mode=args.performance_proxy_mode,
        performance_proxy_score=performance_proxy_score,
        ntk_condition=ntk_condition,
        ntk_trainability_score=ntk_trainability_score,
        ntk_weight=float(args.ntk_weight),
        adjusted_proxy_score=adjusted_proxy_score,
        objective_proxy_score=objective_proxy_score,
        anchor_proxy_penalty=float(args.anchor_proxy_penalty),
        replacement_proxy_penalty=replacement_proxy_penalty,
        parameter_proxy_penalty=parameter_proxy_penalty,
        flops_proxy_penalty=flops_proxy_penalty,
        complexity_proxy_penalty=complexity_proxy_penalty,
        naswot=naswot,
        zico=zico,
        raw_swap=swap_scores.get('raw_swap'),
        layer_swap_sum=swap_scores.get('layer_swap_sum'),
        layer_swap_sqrt=swap_scores.get('layer_swap_sqrt'),
        size=size,
        flops=flops,
        objectives=objectives,
        block_list_summary='original DeRy initialized backbone/branch refinement',
        backbone_delta_summary=backbone_delta_summary(primary_blocks, context),
        branch_summary=ea.format_candidate(branch_candidate),
        operator_summary=', '.join(
            gene.operator for gene in branch_candidate
            if gene is not None and len(gene.branches) > 0),
        num_backbone_replacements=num_backbone_replacements,
        num_branch_layers=num_branch_layers,
        structure_type=structure_type(num_backbone_replacements, num_branch_layers),
        structure_group=detailed_structure_group(
            num_backbone_replacements, branch_candidate),
        source=source,
        error=error)
    return records[key]


def add_baseline_record(context, args, records):
    ea = context['ea']
    baseline_branch = ea.empty_candidate(len(context['base_primary_blocks']))
    return evaluate_refined_candidate(
        list(context['base_primary_blocks']),
        baseline_branch,
        raw_key=(),
        context=context,
        args=args,
        records=records,
        source='baseline')


def make_problem(args, context, pymoo_api, records, source='search'):
    ea = context['ea']
    xl, xu = bounds(context, args)

    class DeRyRefineProblem(pymoo_api['ElementwiseProblem']):
        def __init__(self):
            super().__init__(n_var=len(xl), n_obj=3, xl=xl, xu=xu)

        def _evaluate(self, x, out, *unused_args, **unused_kwargs):
            raw_key = individual_key(x)
            primary_blocks, branch_candidate = decode_individual(x, context, args)
            record = evaluate_refined_candidate(
                primary_blocks,
                branch_candidate,
                raw_key,
                context,
                args,
                records,
                source=source)
            out['F'] = np.asarray(record['objectives'], dtype=float)

    return DeRyRefineProblem()


def run_refine_minimize(pymoo_api, args, context, problem, callback, stage):
    ref_dirs = make_ref_dirs(pymoo_api, args)
    if args.structured_sampling or args.staged_branch_first:
        sampling = structured_initial_population(context, args, stage)
    else:
        sampling = pymoo_api['FloatRandomSampling']()
    algorithm = pymoo_api['NSGA3'](
        pop_size=args.population_size,
        ref_dirs=ref_dirs,
        sampling=sampling,
        crossover=pymoo_api['SBX'](prob=0.9, eta=15),
        mutation=pymoo_api['PM'](eta=20),
        eliminate_duplicates=True)
    return pymoo_api['minimize'](
        problem,
        algorithm,
        ('n_gen', args.generations),
        seed=args.seed,
        callback=callback,
        verbose=False)


def tagged_callback(pymoo_api, logs, records_fn, stage):
    callback = make_callback(pymoo_api, logs, records_fn)
    previous_notify = callback.notify

    def notify_with_stage(algorithm):
        before = len(logs)
        previous_notify(algorithm)
        for row in logs[before:]:
            row['stage'] = stage

    callback.notify = notify_with_stage
    return callback


def stage_args(args, **updates):
    staged = copy.copy(args)
    for key, value in updates.items():
        setattr(staged, key, value)
    return staged


def select_refine_representatives(front, args):
    if not front:
        return {}
    objective_proxy = np.asarray(
        [item.get('objective_proxy_score', item['proxy_score']) for item in front],
        dtype=float)
    raw_proxy = np.asarray([item['proxy_score'] for item in front], dtype=float)
    size = np.asarray([item['size'] for item in front], dtype=float)
    flops = np.asarray([item['flops'] for item in front], dtype=float)
    balanced = (
        normalized(objective_proxy, higher_better=True) +
        normalized(size, higher_better=False) +
        normalized(flops, higher_better=False))
    reps = {
        f'top_{objective_proxy_label(args)}': int(np.argmax(objective_proxy)),
        'min_parameters': int(np.argmin(size)),
        'min_flops': int(np.argmin(flops)),
        'knee': int(np.argmax(balanced)),
    }
    if uses_nonraw_objective(args):
        reps[f'top_raw_{args.proxy}'] = int(np.argmax(raw_proxy))
    return reps


def save_csv(front, reps, output_dir, args):
    path = os.path.join(output_dir, 'pareto_front.csv')
    proxy_name = objective_proxy_label(args)
    top_field = f'is_top_{proxy_name}'
    raw_top_field = f'is_top_raw_{args.proxy}' if uses_nonraw_objective(args) else None
    fields = [
        'rank', 'id', 'proxy', 'proxy_score', 'raw_proxy_score',
        'performance_proxy_mode', 'performance_proxy_score', 'ntk_condition',
        'ntk_trainability_score', 'ntk_weight',
        'adjusted_proxy_score', 'objective_proxy_score', 'anchor_proxy_penalty',
        'replacement_proxy_penalty', 'parameter_proxy_penalty',
        'flops_proxy_penalty', 'complexity_proxy_penalty',
        'naswot', 'zico', 'raw_swap', 'layer_swap_sum',
        'CLAS', 'parameters', 'flops',
        f'objective_1_neg_{proxy_name}', 'objective_2_parameters', 'objective_3_flops',
        'config_path', 'backbone_delta_summary', 'branch_summary',
        'operator_summary', 'num_backbone_replacements', 'num_branch_layers',
        'structure_type', 'structure_group', 'source',
        'is_pareto_front', 'selection_reason', top_field,
    ]
    if raw_top_field is not None:
        fields.append(raw_top_field)
    fields.extend(['is_min_parameters', 'is_min_flops', 'is_knee', 'error'])
    rep_labels = [f'top_{proxy_name}', 'min_parameters', 'min_flops', 'knee']
    if raw_top_field is not None:
        rep_labels.append(f'top_raw_{args.proxy}')
    rep_indices = {
        label: {idx for name, idx in reps.items() if name == label}
        for label in rep_labels
    }
    with open(path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for rank, item in enumerate(front):
            row = dict(
                rank=rank,
                id=item['id'],
                proxy=item.get('proxy', args.proxy),
                proxy_score=item['proxy_score'],
                raw_proxy_score=item.get('raw_proxy_score', item['proxy_score']),
                performance_proxy_mode=item.get('performance_proxy_mode', 'raw'),
                performance_proxy_score=item.get('performance_proxy_score'),
                ntk_condition=item.get('ntk_condition'),
                ntk_trainability_score=item.get('ntk_trainability_score'),
                ntk_weight=item.get('ntk_weight'),
                adjusted_proxy_score=item.get('adjusted_proxy_score', item['proxy_score']),
                objective_proxy_score=item.get('objective_proxy_score', item['proxy_score']),
                anchor_proxy_penalty=item.get('anchor_proxy_penalty', 0.0),
                replacement_proxy_penalty=item.get('replacement_proxy_penalty', 0.0),
                parameter_proxy_penalty=item.get('parameter_proxy_penalty', 0.0),
                flops_proxy_penalty=item.get('flops_proxy_penalty', 0.0),
                complexity_proxy_penalty=item.get('complexity_proxy_penalty', 0.0),
                naswot=item.get('naswot'),
                zico=item['zico'],
                raw_swap=item.get('raw_swap'),
                layer_swap_sum=item.get('layer_swap_sum'),
                CLAS=item.get('layer_swap_sqrt'),
                parameters=item['size'],
                flops=item['flops'],
                **{f'objective_1_neg_{proxy_name}': item['objectives'][0]},
                objective_2_parameters=item['objectives'][1],
                objective_3_flops=item['objectives'][2],
                config_path=item.get('config_path'),
                backbone_delta_summary=item.get('backbone_delta_summary', ''),
                branch_summary=item.get('branch_summary', ''),
                operator_summary=item.get('operator_summary', ''),
                num_backbone_replacements=item.get('num_backbone_replacements', 0),
                num_branch_layers=item.get('num_branch_layers', 0),
                structure_type=item.get('structure_type', ''),
                structure_group=item.get('structure_group', ''),
                source=item.get('source', ''),
                is_pareto_front=item.get('is_pareto_front', True),
                selection_reason=item.get('selection_reason', 'pareto'),
                **{top_field: rank in rep_indices[f'top_{proxy_name}']},
                is_min_parameters=rank in rep_indices['min_parameters'],
                is_min_flops=rank in rep_indices['min_flops'],
                is_knee=rank in rep_indices['knee'],
                error=item.get('error'))
            if raw_top_field is not None:
                row[raw_top_field] = rank in rep_indices[f'top_raw_{args.proxy}']
            writer.writerow(row)
    return path


def save_records_json(output_dir, records):
    path = os.path.join(output_dir, 'all_evaluations.json')
    serializable = []
    for rec in records:
        item = dict(rec)
        item.pop('primary_blocks', None)
        item.pop('branch_candidate', None)
        serializable.append(item)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(serializable, file, indent=2)
    return path


def select_refine_front(records, top_k, structure_min_quota):
    """Retain small topology quotas without changing proxy values."""
    front = final_front(records)
    rank = lambda item: (
        -item.get('objective_proxy_score', item['proxy_score']),
        item['size'], item['flops'], item.get('id', 10 ** 9))
    front.sort(key=rank)
    pareto_ids = {item['id'] for item in front}
    for item in front:
        item['is_pareto_front'] = True
        item['selection_reason'] = 'pareto'
    if top_k is None:
        return front
    limit = max(1, int(top_k))
    quota = max(0, int(structure_min_quota))
    if quota <= 0:
        return front[:limit]

    valid = [
        item for item in records
        if item.get('error') is None and
        all(math.isfinite(value) for value in item['objectives'])]
    valid.sort(key=rank)
    groups = (
        'branch_gate', 'branch_sum', 'branch_mixed',
        'replacement_only', 'replacement_branch')
    available = [
        group for group in groups
        if any(item.get('structure_group') == group for item in valid)]
    if quota * len(available) > limit:
        raise ValueError(
            '--structure-min-quota exceeds --top-k capacity: '
            f'{quota} * {len(available)} > {limit}')

    selected = []
    selected_ids = set()
    for group in available:
        candidates = [
            item for item in valid
            if item.get('structure_group') == group]
        for item in candidates[:quota]:
            item['is_pareto_front'] = item['id'] in pareto_ids
            item['selection_reason'] = f'structure_quota:{group}'
            selected.append(item)
            selected_ids.add(item['id'])
    for source, reason in ((front, 'pareto'), (valid, 'top_performance_fill')):
        for item in source:
            if item['id'] in selected_ids:
                continue
            item['is_pareto_front'] = item['id'] in pareto_ids
            item['selection_reason'] = reason
            selected.append(item)
            selected_ids.add(item['id'])
            if len(selected) >= limit:
                return selected
    return selected


def write_search_log(output_dir, args, ref_dirs, generation_logs, records):
    path = os.path.join(output_dir, 'search_log.txt')
    failures = Counter()
    proxy_name = objective_proxy_label(args)
    for rec in records:
        if rec.get('error'):
            failures[str(rec['error']).split(': ', 1)[0]] += 1
    with open(path, 'w', encoding='utf-8') as file:
        file.write('NSGA-III original-DeRy initialized joint refinement\n')
        file.write(METHOD_NOTE + '\n\n')
        file.write(f'backbone_config: {args.backbone_config}\n')
        file.write(f'population_size: {args.population_size}\n')
        file.write(f'generations: {args.generations}\n')
        file.write(f'seed: {args.seed}\n')
        file.write(f'proxy: {args.proxy}\n')
        file.write(f'proxy_data_seed: {args.proxy_data_seed}\n')
        file.write(f'proxy_model_seed: {args.proxy_model_seed}\n')
        file.write(f'swap_image_count: {args.swap_image_count}\n')
        file.write(f'performance_proxy_mode: {args.performance_proxy_mode}\n')
        file.write(f'objective_proxy: {proxy_name}\n')
        file.write(f'objectives: minimize [-{proxy_name}, parameters, FLOPs]\n')
        if args.performance_proxy_mode == 'zico_ntk':
            file.write(
                'performance_proxy_formula: log(zico) - '
                f'{args.ntk_weight} * log(ntk_condition)\n')
            file.write('ntk_condition_formula: lambda_max(NTK) / lambda_min(NTK)\n')
            file.write(f'ntk_max_samples: {args.ntk_max_samples}\n')
            file.write(f'ntk_num_batch: {args.ntk_num_batch}\n')
            file.write(f'ntk_eps: {args.ntk_eps}\n')
            file.write(f'ntk_train_mode: {args.ntk_train_mode}\n')
            file.write(f'max_ntk_condition: {args.max_ntk_condition}\n')
        file.write(f'anchor_proxy_penalty: {args.anchor_proxy_penalty}\n')
        if has_proxy_penalty(args):
            file.write(
                f'{proxy_name} = {base_objective_proxy_label(args)} - '
                f'{args.anchor_proxy_penalty} * num_backbone_replacements - '
                f'{args.parameter_proxy_penalty} * parameters_M - '
                f'{args.flops_proxy_penalty} * flops_G\n')
        file.write(f'min_params: {args.min_params}\n')
        file.write(f'max_params: {args.max_params}\n')
        file.write(f'min_flops: {args.min_flops}\n')
        file.write(f'max_flops: {args.max_flops}\n')
        file.write(f'max_backbone_replacements: {args.max_backbone_replacements}\n')
        file.write(f'max_branch_layers: {args.max_branch_layers}\n')
        file.write(f'staged_branch_first: {args.staged_branch_first}\n')
        file.write(f'structured_sampling: {args.structured_sampling}\n')
        file.write(f'structure_min_quota: {args.structure_min_quota}\n')
        if args.staged_branch_first:
            file.write(f'branch_first_generations: {args.branch_first_generations}\n')
            file.write(f'branch_first_population_size: {args.branch_first_population_size}\n')
            file.write(
                f'branch_first_two_branch_ratio: '
                f'{args.branch_first_two_branch_ratio}\n')
            file.write(f'joint_branch_only_ratio: {args.joint_branch_only_ratio}\n')
            file.write(
                f'joint_replacement_only_ratio: '
                f'{args.joint_replacement_only_ratio}\n')
            file.write(f'joint_identity_ratio: {args.joint_identity_ratio}\n')
            file.write(
                f'joint_replacement_enable_ratio: '
                f'{args.joint_replacement_enable_ratio}\n')
            file.write(
                f'joint_branch_enable_ratio: '
                f'{args.joint_branch_enable_ratio}\n')
            file.write(f'joint_two_branch_ratio: {args.joint_two_branch_ratio}\n')
        file.write(f'backbone_replacement_policy: {args.backbone_replacement_policy}\n')
        file.write(f'branch_type_policy: {args.branch_type_policy}\n')
        file.write(f'reference_directions: {len(ref_dirs)}\n')
        file.write(f'ref_partitions: {args.ref_partitions}\n')
        if args.population_size != len(ref_dirs):
            file.write(
                'reference_direction_note: population size and reference '
                'direction count differ; NSGA-III keeps the requested '
                'population size and uses the reference directions for '
                'diversity guidance.\n')
        file.write(f'output_dir: {args.output_dir}\n\n')
        for row in generation_logs:
            file.write(
                f"stage={row.get('stage', 'joint')} "
                f"generation={row['generation']} valid={row['valid']} "
                f"pareto={row['pareto']} best_{proxy_name}={row['best_performance']:.6f} "
                f"min_parameters={row['min_parameters']:.6f} min_flops={row['min_flops']:.6f} "
                f"failed={row['failed']}\n")
        file.write('\nfailure_summary:\n')
        if failures:
            for key, value in failures.items():
                file.write(f'{key}: {value}\n')
        else:
            file.write('none\n')
    return path


def main():
    args = parse_args()
    objectives = [name.strip() for name in args.objectives.split(',') if name.strip()]
    if objectives != ['performance', 'parameters', 'flops']:
        raise SystemExit('Use --objectives performance,parameters,flops.')
    if args.performance_proxy_mode == 'zico_ntk' and args.proxy != 'zico':
        raise SystemExit('--performance-proxy-mode zico_ntk requires --proxy zico.')
    if (args.max_ntk_condition is not None and
            args.performance_proxy_mode != 'zico_ntk'):
        raise SystemExit(
            '--max-ntk-condition requires --performance-proxy-mode zico_ntk.')
    if args.max_ntk_condition is not None and args.max_ntk_condition <= 0:
        raise SystemExit('--max-ntk-condition must be positive.')
    args.anchor_proxy_penalty = max(0.0, args.anchor_proxy_penalty)
    args.parameter_proxy_penalty = max(0.0, args.parameter_proxy_penalty)
    args.flops_proxy_penalty = max(0.0, args.flops_proxy_penalty)
    args.ntk_weight = max(0.0, args.ntk_weight)
    args.ntk_max_samples = max(2, int(args.ntk_max_samples))
    args.ntk_num_batch = max(1, int(args.ntk_num_batch))
    args.ntk_eps = max(1e-12, float(args.ntk_eps))
    args.swap_image_count = max(2, int(args.swap_image_count))
    args.branch_first_two_branch_ratio = clipped_ratio(args.branch_first_two_branch_ratio)
    args.joint_branch_only_ratio = clipped_ratio(args.joint_branch_only_ratio)
    args.joint_replacement_only_ratio = clipped_ratio(args.joint_replacement_only_ratio)
    args.joint_identity_ratio = clipped_ratio(args.joint_identity_ratio)
    args.joint_replacement_enable_ratio = clipped_ratio(
        args.joint_replacement_enable_ratio)
    args.joint_branch_enable_ratio = clipped_ratio(args.joint_branch_enable_ratio)
    args.joint_two_branch_ratio = clipped_ratio(args.joint_two_branch_ratio)
    if args.staged_branch_first:
        if args.branch_first_generations is None:
            args.branch_first_generations = max(1, args.generations // 2)
        else:
            args.branch_first_generations = max(1, args.branch_first_generations)
        if args.branch_first_population_size is None:
            args.branch_first_population_size = args.population_size
        else:
            args.branch_first_population_size = max(1, args.branch_first_population_size)

    os.makedirs(args.output_dir, exist_ok=True)
    config_dir = ensure_output_dirs(args.output_dir)
    pymoo_api = import_pymoo()
    context = make_context(args)
    records_by_key = {}
    baseline = add_baseline_record(context, args, records_by_key)
    if baseline.get('error') is not None:
        print(
            'warning: Original DeRy baseline is outside the active search '
            f"budget or failed evaluation: error={baseline.get('error')}, "
            f"size={baseline.get('size')}, flops={baseline.get('flops')}. "
            'Continuing search; invalid baseline will be excluded from the '
            'final Pareto front.')
    generation_logs = []
    results = []
    if args.staged_branch_first:
        branch_args = stage_args(
            args,
            population_size=args.branch_first_population_size,
            generations=args.branch_first_generations,
            max_backbone_replacements=0,
            structured_sampling=True)
        branch_problem = make_problem(
            branch_args,
            context,
            pymoo_api,
            records_by_key,
            source='branch_first')
        branch_callback = tagged_callback(
            pymoo_api,
            generation_logs,
            lambda: list(records_by_key.values()),
            'branch_first')
        results.append(run_refine_minimize(
            pymoo_api,
            branch_args,
            context,
            branch_problem,
            branch_callback,
            stage='branch_first'))

    problem = make_problem(
        args,
        context,
        pymoo_api,
        records_by_key,
        source='joint')
    callback = tagged_callback(
        pymoo_api,
        generation_logs,
        lambda: list(records_by_key.values()),
        'joint')
    results.append(run_refine_minimize(
        pymoo_api,
        args,
        context,
        problem,
        callback,
        stage='joint'))
    records = list(records_by_key.values())

    def build_config_fn(record):
        return build_refined_config(
            context['base_cfg'],
            record['primary_blocks'],
            record['branch_candidate'])

    front = select_refine_front(
        records, args.top_k, args.structure_min_quota)
    reps = select_refine_representatives(front, args)
    export_configs(front, reps, config_dir, build_config_fn=build_config_fn)
    csv_path = save_csv(front, reps, args.output_dir, args)
    log_path = write_search_log(
        args.output_dir,
        args,
        make_ref_dirs(pymoo_api, args),
        generation_logs,
        records)
    json_path = save_records_json(args.output_dir, records)

    print('NSGA-III DeRy refine search complete')
    print(f'output_dir: {args.output_dir}')
    print(f'pareto_front: {csv_path}')
    print(f'search_log: {log_path}')
    print(f'all_evaluations: {json_path}')
    print(f'configs: {config_dir}')
    if any(result is None for result in results):
        print('warning: pymoo returned no result object for at least one stage')


if __name__ == '__main__':
    main()
