"""NSGA-III three-objective local-branch search for DeRy.

We extend the original single-objective genetic search into a three-objective
optimization problem. The objectives are to maximize the ZICO proxy score,
minimize the model size, and minimize FLOPs. NSGA-III is adopted to obtain a
diverse Pareto front under these conflicting objectives. Representative
architectures, including the highest-ZICO model, the minimum-size model, the
minimum-FLOPs model, and the balanced knee solution, are selected for training
and empirical validation.

The NSGA-III optimizer minimizes objectives, so the real objective vector is:
[-zico, size, flops].
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_DIR = os.path.join(REPO_ROOT, 'simlarity')
for path in [REPO_ROOT, SIM_DIR, os.path.join(REPO_ROOT, 'third_package')]:
    if path not in sys.path:
        sys.path.insert(0, path)


METHOD_NOTE = (
    'We extend the original single-objective genetic search into a '
    'three-objective optimization problem. The objectives are to maximize the '
    'ZICO proxy score, minimize the model size, and minimize FLOPs. NSGA-III '
    'is adopted to obtain a diverse Pareto front under these conflicting '
    'objectives. Representative architectures, including the highest-ZICO '
    'model, the minimum-size model, the minimum-FLOPs model, and the balanced '
    'knee solution, are selected for training and empirical validation.'
)


def import_pymoo():
    try:
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.core.callback import Callback
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.optimize import minimize
        from pymoo.util.ref_dirs import get_reference_directions
    except ImportError as exc:
        raise SystemExit(
            'pymoo is required for NSGA-III search but is not installed.\n'
            'Install it in the active DeRy environment with:\n\n'
            '  pip install pymoo\n\n'
            f'Original import error: {exc}')

    return dict(
        NSGA3=NSGA3,
        Callback=Callback,
        ElementwiseProblem=ElementwiseProblem,
        FloatRandomSampling=FloatRandomSampling,
        SBX=SBX,
        PM=PM,
        minimize=minimize,
        get_reference_directions=get_reference_directions,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='NSGA-III multi-objective search: maximize ZiCo, minimize size and FLOPs.')
    parser.add_argument(
        'backbone_config',
        nargs='?',
        default='configs/imagenet/dery_baseline_100e.py',
        help='Fixed DeRy backbone config for local-branch search.')
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
    parser.add_argument('--output-dir', default='simlarity/out/nsga3_zico_size_flops')
    parser.add_argument('--objectives', default='zico,size,flops')
    parser.add_argument(
        '--quality-mode',
        choices=[
            'zico', 'zico_expr', 'old_zico_ntk', 'zico_expr_ntk',
            'raw_swap', 'layer_swap_sum', 'layer_swap_sqrt',
            'zico_norm_mean', 'zico_norm_top', 'zico_norm_clip',
            'zico_norm_balanced',
            'zico_norm_mean_expr', 'zico_norm_top_expr',
            'zico_norm_clip_expr', 'zico_norm_balanced_expr',
            'robust_zico', 'robust_zico_expr'],
        default='layer_swap_sqrt',
        help=(
            'First objective quality. zico keeps legacy behavior. zico_expr '
            'uses log(ZiCo)+log(expressivity). old_zico_ntk uses '
            'log(ZiCo)-0.5*log(NTK condition). zico_expr_ntk also subtracts '
            '--ntk-penalty*log(NTK condition). zico_norm_* uses '
            'parameter-count-neutral ZiCo internal aggregation. robust_zico '
            'keeps log(ZiCo) and penalizes group-gradient dispersion.'))
    parser.add_argument('--proxy-data-seed', type=int, default=11)
    parser.add_argument('--proxy-model-seed', type=int, default=11)
    parser.add_argument('--swap-image-count', type=int, default=32)
    parser.add_argument('--expr-batch-size', type=int, default=16)
    parser.add_argument('--expressivity-max-vectors', type=int, default=256)
    parser.add_argument('--ntk-max-samples', type=int, default=2)
    parser.add_argument('--ntk-penalty', type=float, default=0.25)
    parser.add_argument('--robust-zico-mean-dispersion-weight', type=float, default=0.15)
    parser.add_argument('--robust-zico-top-dispersion-weight', type=float, default=0.025)
    parser.add_argument('--robust-zico-mean-dispersion-center', type=float, default=0.6215739409396356)
    parser.add_argument('--robust-zico-mean-dispersion-scale', type=float, default=0.20072518915174462)
    parser.add_argument('--robust-zico-top-dispersion-center', type=float, default=0.7762648255625314)
    parser.add_argument('--robust-zico-top-dispersion-scale', type=float, default=0.2143574466017383)
    parser.add_argument('--ref-partitions', type=int, default=12)
    parser.add_argument('--top-k', type=int, default=None)
    parser.add_argument(
        '--structure-min-quota', type=int, default=0,
        help=(
            'Minimum retained final candidates for each available gate-only, '
            'sum-only, and mixed branch family. This changes selection only, '
            'never the quality score.'))
    parser.add_argument('--toy', action='store_true', help='Run with random toy objectives, no DeRy imports.')
    parser.add_argument('--toy-branches', type=int, default=4)
    parser.add_argument('--minC', dest='min_params', type=float, default=None)
    parser.add_argument('--C', '--maxC', dest='max_params', type=float, default=30.0)
    parser.add_argument('--minflop-C', '--minflop_C', dest='min_flops', type=float, default=None)
    parser.add_argument('--flop-C', '--flop_C', dest='max_flops', type=float, default=6.0)
    parser.add_argument('--skip-flops-eval', action='store_true')
    parser.add_argument('--start-ratio', type=float, default=0.35)
    parser.add_argument('--max-branch-layers', type=int, default=3)
    parser.add_argument(
        '--branch-layers', type=int, nargs='+', default=[1, 2, 3],
        help='Positions enabled for a second component in the four-position space.')
    parser.add_argument(
        '--max-components-per-position', dest='max_branches_per_layer',
        type=int, default=2)
    parser.add_argument(
        '--operators',
        nargs='+',
        choices=['sum', 'gate'],
        default=['sum'])
    parser.add_argument(
        '--branch-type-policy',
        choices=['any', 'same-io', 'no-vit'],
        default='any')
    parser.add_argument('--max-adapter-burden', type=float, default=None)
    parser.add_argument('--max-type-switches', type=int, default=None)
    parser.add_argument('--device', choices=['cuda'], default='cuda')
    return parser.parse_args()


def ensure_output_dirs(output_dir):
    config_dir = os.path.join(output_dir, 'configs')
    os.makedirs(config_dir, exist_ok=True)
    return config_dir


def normalized(values, higher_better):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    result = np.zeros_like(values, dtype=float)
    if finite.sum() == 0:
        return result
    lo = values[finite].min()
    hi = values[finite].max()
    span = max(hi - lo, 1e-12)
    if higher_better:
        result[finite] = (values[finite] - lo) / span
    else:
        result[finite] = (hi - values[finite]) / span
    return result


def non_dominated_indices(records):
    keep = []
    for i, left in enumerate(records):
        dominated = False
        for j, right in enumerate(records):
            if i == j:
                continue
            left_obj = left['objectives']
            right_obj = right['objectives']
            if all(r <= l for r, l in zip(right_obj, left_obj)) and any(
                    r < l for r, l in zip(right_obj, left_obj)):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def non_dominated_objective_indices(objectives):
    keep = []
    for i, left in enumerate(objectives):
        dominated = False
        for j, right in enumerate(objectives):
            if i == j:
                continue
            if np.all(right <= left) and np.any(right < left):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def finite_records(records):
    return [
        rec for rec in records
        if rec.get('error') is None and all(math.isfinite(v) for v in rec['objectives'])
    ]


def candidate_structure_group(candidate):
    operators = {
        gene.operator for gene in candidate
        if gene is not None and len(gene.branches) > 0}
    if operators == {'gate'}:
        return 'branch_gate'
    if operators == {'sum'}:
        return 'branch_sum'
    if operators == {'gate', 'sum'}:
        return 'branch_mixed'
    return 'baseline'


def select_representatives(front):
    if not front:
        return {}
    quality = np.asarray(
        [item.get('quality', item.get('zico', -1e9)) for item in front],
        dtype=float)
    zico = np.asarray([item['zico'] for item in front], dtype=float)
    size = np.asarray([item['size'] for item in front], dtype=float)
    flops = np.asarray([item['flops'] for item in front], dtype=float)
    balanced = (
        normalized(quality, higher_better=True) +
        normalized(size, higher_better=False) +
        normalized(flops, higher_better=False))
    return dict(
        top_quality=int(np.argmax(quality)),
        top_zico=int(np.argmax(zico)),
        min_size=int(np.argmin(size)),
        min_flops=int(np.argmin(flops)),
        knee=int(np.argmax(balanced)),
    )


def write_text_config(path, text):
    with open(path, 'w', encoding='utf-8') as file:
        file.write(text)
        if not text.endswith('\n'):
            file.write('\n')


def save_placeholder_config(path, record):
    text = (
        '# Toy NSGA-III placeholder config.\n'
        f'# zico={record["zico"]:.6f}, size={record["size"]:.6f}, '
        f'flops={record["flops"]:.6f}\n'
        f'individual = {record["individual"]!r}\n')
    write_text_config(path, text)


def export_configs(front, reps, config_dir, build_config_fn=None):
    for idx, record in enumerate(front):
        path = os.path.join(config_dir, f'pareto_{idx:03d}.py')
        if build_config_fn is None:
            save_placeholder_config(path, record)
        else:
            cfg = build_config_fn(record)
            write_text_config(path, cfg.pretty_text)
        record['config_path'] = path

    for label, idx in reps.items():
        src = front[idx]
        path = os.path.join(config_dir, f'{label}.py')
        if build_config_fn is None:
            save_placeholder_config(path, src)
        else:
            cfg = build_config_fn(src)
            write_text_config(path, cfg.pretty_text)
        src[f'is_{label}'] = True
        src[f'{label}_config_path'] = path


def save_csv(front, reps, output_dir):
    path = os.path.join(output_dir, 'pareto_front.csv')
    fields = [
        'rank', 'id', 'quality_mode', 'quality', 'zico',
        'raw_swap', 'layer_swap_sum', 'layer_swap_sqrt',
        'normalized_zico_mean', 'normalized_zico_top',
        'normalized_zico_clip', 'normalized_zico_balanced',
        'normalized_zico_mean_dispersion', 'normalized_zico_top_dispersion',
        'normalized_zico_clip_dispersion',
        'expressivity', 'progressivity', 'ntk_condition', 'size', 'flops',
        'objective_1_neg_quality', 'objective_1_neg_zico',
        'objective_2_size', 'objective_3_flops',
        'config_path', 'block_list_summary', 'branch_summary', 'operator_summary',
        'structure_group',
        'is_pareto_front', 'selection_reason',
        'is_top_quality', 'is_top_zico', 'is_min_size', 'is_min_flops',
        'is_knee', 'error',
    ]
    rep_indices = {
        label: {idx for name, idx in reps.items() if name == label}
        for label in ('top_quality', 'top_zico', 'min_size', 'min_flops', 'knee')
    }
    with open(path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for rank, item in enumerate(front):
            writer.writerow(dict(
                rank=rank,
                id=item['id'],
                quality_mode=item.get('quality_mode', 'zico'),
                quality=item.get('quality', item.get('zico')),
                zico=item['zico'],
                raw_swap=item.get('raw_swap', ''),
                layer_swap_sum=item.get('layer_swap_sum', ''),
                layer_swap_sqrt=item.get('layer_swap_sqrt', ''),
                normalized_zico_mean=item.get('normalized_zico_mean', ''),
                normalized_zico_top=item.get('normalized_zico_top', ''),
                normalized_zico_clip=item.get('normalized_zico_clip', ''),
                normalized_zico_balanced=item.get(
                    'normalized_zico_balanced', ''),
                normalized_zico_mean_dispersion=item.get(
                    'normalized_zico_mean_dispersion', ''),
                normalized_zico_top_dispersion=item.get(
                    'normalized_zico_top_dispersion', ''),
                normalized_zico_clip_dispersion=item.get(
                    'normalized_zico_clip_dispersion', ''),
                expressivity=item.get('expressivity', ''),
                progressivity=item.get('progressivity', ''),
                ntk_condition=item.get('ntk_condition', ''),
                size=item['size'],
                flops=item['flops'],
                objective_1_neg_quality=item['objectives'][0],
                objective_1_neg_zico=item['objectives'][0],
                objective_2_size=item['objectives'][1],
                objective_3_flops=item['objectives'][2],
                config_path=item.get('config_path'),
                block_list_summary=item.get('block_list_summary', ''),
                branch_summary=item.get('branch_summary', ''),
                operator_summary=item.get('operator_summary', ''),
                structure_group=item.get('structure_group', ''),
                is_pareto_front=item.get('is_pareto_front', True),
                selection_reason=item.get('selection_reason', 'pareto'),
                is_top_quality=rank in rep_indices['top_quality'],
                is_top_zico=rank in rep_indices['top_zico'],
                is_min_size=rank in rep_indices['min_size'],
                is_min_flops=rank in rep_indices['min_flops'],
                is_knee=rank in rep_indices['knee'],
                error=item.get('error')))
    return path


def write_search_log(output_dir, args, ref_dirs, generation_logs, records):
    path = os.path.join(output_dir, 'search_log.txt')
    failures = Counter()
    for rec in records:
        if rec.get('error'):
            failures[str(rec['error']).split(': ', 1)[0]] += 1

    with open(path, 'w', encoding='utf-8') as file:
        file.write('NSGA-III DeRy local-branch multi-objective search\n')
        file.write(METHOD_NOTE + '\n\n')
        file.write(f'population_size: {args.population_size}\n')
        file.write(f'generations: {args.generations}\n')
        file.write(f'seed: {args.seed}\n')
        file.write(
            f'quality_mode: {getattr(args, "quality_mode", "zico")}\n')
        file.write(f'proxy_data_seed: {args.proxy_data_seed}\n')
        file.write(f'proxy_model_seed: {args.proxy_model_seed}\n')
        file.write(f'swap_image_count: {args.swap_image_count}\n')
        file.write(f'backbone_config: {args.backbone_config}\n')
        file.write(f'min_params: {args.min_params}\n')
        file.write(f'max_params: {args.max_params}\n')
        file.write(f'min_flops: {args.min_flops}\n')
        file.write(f'max_flops: {args.max_flops}\n')
        file.write(f'start_ratio: {args.start_ratio}\n')
        file.write(f'max_branch_layers: {args.max_branch_layers}\n')
        file.write(f'branch_type_policy: {args.branch_type_policy}\n')
        file.write(f'top_k: {args.top_k}\n')
        file.write(f'structure_min_quota: {args.structure_min_quota}\n')
        file.write(f'unique_evaluations: {len(records)}\n')
        file.write(f'valid_evaluations: {len(finite_records(records))}\n')
        if str(getattr(args, 'quality_mode', '')) in (
                'robust_zico', 'robust_zico_expr'):
            file.write(
                'robust_zico_formula: log(zico) - '
                f'{args.robust_zico_mean_dispersion_weight} * '
                'zscore(normalized_zico_mean_dispersion) - '
                f'{args.robust_zico_top_dispersion_weight} * '
                'zscore(normalized_zico_top_dispersion)\n')
            file.write(
                'robust_zico_calibration: '
                f'mean_center={args.robust_zico_mean_dispersion_center}, '
                f'mean_scale={args.robust_zico_mean_dispersion_scale}, '
                f'top_center={args.robust_zico_top_dispersion_center}, '
                f'top_scale={args.robust_zico_top_dispersion_scale}\n')
        file.write('objectives: minimize [-quality, size, flops]\n')
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
                f"generation={row['generation']} valid={row['valid']} "
                f"pareto={row['pareto']} best_zico={row['best_zico']:.6f} "
                f"min_size={row['min_size']:.6f} min_flops={row['min_flops']:.6f} "
                f"failed={row['failed']}\n")
        file.write('\nfailure_summary:\n')
        if failures:
            for key, value in failures.items():
                file.write(f'{key}: {value}\n')
        else:
            file.write('none\n')
    return path


def save_records_json(output_dir, records):
    path = os.path.join(output_dir, 'all_evaluations.json')
    serializable = []
    for rec in records:
        item = dict(rec)
        item.pop('candidate', None)
        item.pop('config', None)
        serializable.append(item)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(serializable, file, indent=2)
    return path


def make_ref_dirs(pymoo_api, args):
    ref_dirs = pymoo_api['get_reference_directions'](
        'das-dennis', 3, n_partitions=args.ref_partitions)
    return ref_dirs


def run_toy(args, pymoo_api):
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    n_var = max(3, args.toy_branches * 3)
    records = {}

    class ToyProblem(pymoo_api['ElementwiseProblem']):
        def __init__(self):
            super().__init__(n_var=n_var, n_obj=3, xl=0.0, xu=1.0)

        def _evaluate(self, x, out, *unused_args, **unused_kwargs):
            key = tuple(round(float(value), 4) for value in x)
            if key not in records:
                branch_strength = sum(x[0::3])
                diversity = sum(abs(x[i] - x[i - 1]) for i in range(1, len(x)))
                zico = 100.0 + 40.0 * branch_strength + 5.0 * rng.random()
                size = 5.0 + 20.0 * np.mean(x[1::3]) + 2.0 * rng.random()
                flops = 1.0 + 5.0 * np.mean(x[2::3]) + 0.2 * diversity
                records[key] = dict(
                    id=len(records),
                    individual=list(map(float, x)),
                    zico=float(zico),
                    size=float(size),
                    flops=float(flops),
                    objectives=[-float(zico), float(size), float(flops)],
                    branch_summary='toy',
                    operator_summary='toy',
                    block_list_summary='toy',
                    error=None)
            out['F'] = np.asarray(records[key]['objectives'], dtype=float)

    generation_logs = []
    callback = make_callback(pymoo_api, generation_logs, lambda: list(records.values()))
    result = run_minimize(pymoo_api, args, ToyProblem(), callback)
    return result, list(records.values()), generation_logs, None


def import_real_search_deps():
    import torch

    from simlarity.zero_nas import ZeroNas
    from simlarity.zero_nas.dery_composite import (
        block_expressivity, empirical_ntk_condition)
    import tools.ea_local_branch_naswot as ea

    return torch, ZeroNas, ea, block_expressivity, empirical_ntk_condition


SWAP_QUALITY_MODES = {'raw_swap', 'layer_swap_sum', 'layer_swap_sqrt'}


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


def compute_swap_scores(candidate, context, args):
    from tools.evaluate_dery_future_proxy import LogitWrapper
    from tools.evaluate_training_free_replacements import ActivationPatternMonitor

    torch = context['torch']
    ea = context['ea']
    cfg = ea.build_candidate_config(
        context['base_cfg'], context['primary_cfgs'],
        context['primary_blocks'], candidate)
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
        scores = monitor.scores()
        return dict(
            raw_swap=float(scores['swap_score']),
            layer_swap_sum=float(scores['layer_swap_sum']),
            layer_swap_sqrt=float(scores['layer_swap_sqrt_sum']))
    finally:
        if monitor is not None:
            monitor.close()
        if model is not None:
            del model
        torch.cuda.empty_cache()


def make_real_context(args):
    torch, ZeroNas, ea, block_expressivity, empirical_ntk_condition = (
        import_real_search_deps())
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for real ZiCo evaluation.')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.score_mode = (
        'swap' if args.quality_mode in SWAP_QUALITY_MODES else 'zico')
    args.naswot_weight = 0.0
    args.zico_weight = 0.0 if args.quality_mode in SWAP_QUALITY_MODES else 1.0
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
    args.branch_type_policy = args.branch_type_policy
    args.skip_flops_eval = bool(args.skip_flops_eval)
    args.fixed_eval_seed = int(args.proxy_model_seed)
    args.zico_num_batch_adjusted = False
    if args.num_batch < 2:
        args.num_batch = 2
        args.zico_num_batch_adjusted = True

    blocks_by_index = ea.load_assignment(args.assignment)
    base_cfg, primary_cfgs, primary_blocks = ea.load_backbone_blocks(
        args.backbone_config, blocks_by_index)
    layers = ea.allowed_layers(len(primary_blocks), args)
    layer_pools = {}
    for layer in layers:
        layer_pools[layer] = ea.branch_pool(
            layer, ea.empty_candidate(len(primary_blocks)),
            primary_blocks, blocks_by_index, args)

    random.seed(int(args.proxy_data_seed))
    np.random.seed(int(args.proxy_data_seed))
    torch.manual_seed(int(args.proxy_data_seed))
    torch.cuda.manual_seed_all(int(args.proxy_data_seed))
    data_loader = ea.make_data_loader(args, shuffle=True)
    cached_batches = cache_proxy_batches(data_loader, args.num_batch)
    indicator = ZeroNas(
        dataloader=cached_batches,
        indicator='zico',
        num_batch=args.num_batch)
    cached_images = torch.cat(
        [batch['img'] for batch in cached_batches], dim=0).contiguous()
    extra_count = max(
        int(args.expr_batch_size), int(args.ntk_max_samples),
        int(args.swap_image_count), 2)
    if cached_images.shape[0] < extra_count:
        raise RuntimeError(
            f'Cached {cached_images.shape[0]} images but proxy needs '
            f'{extra_count}. Increase --batch-size or --num-batch.')
    extra_images = cached_images[:extra_count]
    proxy_images = cached_images[:int(args.swap_image_count)]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    return dict(
        torch=torch,
        ea=ea,
        block_expressivity=block_expressivity,
        empirical_ntk_condition=empirical_ntk_condition,
        base_cfg=base_cfg,
        primary_cfgs=primary_cfgs,
        primary_blocks=primary_blocks,
        blocks_by_index=blocks_by_index,
        layers=layers,
        layer_pools=layer_pools,
        indicator=indicator,
        extra_images=extra_images,
        proxy_images=proxy_images,
        memo={})


def forward_logits_for_classifier(model, images):
    features = model.extract_feat(images)
    if isinstance(features, tuple):
        features = features[-1]
    return model.head.fc(features)


def compute_quality_extras(candidate, context, args):
    if args.quality_mode in (
            'zico', 'raw_swap', 'layer_swap_sum', 'layer_swap_sqrt',
            'zico_norm_mean', 'zico_norm_top',
            'zico_norm_clip', 'zico_norm_balanced', 'robust_zico'):
        return dict(
            expressivity=float('nan'),
            progressivity=float('nan'),
            block_entropy=[],
            ntk_condition=float('nan'))
    torch = context['torch']
    ea = context['ea']
    cfg = ea.build_candidate_config(
        context['base_cfg'], context['primary_cfgs'],
        context['primary_blocks'], candidate)
    model = None
    try:
        model = ea.build_classifier(cfg.model)
        model.init_weights()
        model.cuda().eval()
        images = context['extra_images'].cuda(non_blocking=True)
        expr = context['block_expressivity'](
            model, images[:int(args.expr_batch_size)],
            max_vectors=args.expressivity_max_vectors)
        ntk_condition = float('nan')
        if args.quality_mode in ('old_zico_ntk', 'zico_expr_ntk'):
            ntk_condition = context['empirical_ntk_condition'](
                model, images, forward_logits_for_classifier,
                max_samples=args.ntk_max_samples)
        return dict(
            expressivity=float(expr['expressivity']),
            progressivity=float(expr['progressivity']),
            block_entropy=expr['block_entropy'],
            ntk_condition=ntk_condition)
    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()


def quality_score_key(args):
    mapping = {
        'zico_norm_mean': 'normalized_zico_mean',
        'zico_norm_top': 'normalized_zico_top',
        'zico_norm_clip': 'normalized_zico_clip',
        'zico_norm_balanced': 'normalized_zico_balanced',
        'zico_norm_mean_expr': 'normalized_zico_mean',
        'zico_norm_top_expr': 'normalized_zico_top',
        'zico_norm_clip_expr': 'normalized_zico_clip',
        'zico_norm_balanced_expr': 'normalized_zico_balanced',
        'robust_zico': 'zico',
        'robust_zico_expr': 'zico',
    }
    return mapping.get(args.quality_mode, 'zico')


def robust_zico_quality(raw_zico, item, args):
    if raw_zico is None or not math.isfinite(float(raw_zico)) or float(raw_zico) <= 0:
        return -1e9
    mean_dispersion = float(item.get(
        'normalized_zico_mean_dispersion', float('nan')))
    top_dispersion = float(item.get(
        'normalized_zico_top_dispersion', float('nan')))
    if not math.isfinite(mean_dispersion) or not math.isfinite(top_dispersion):
        return -1e9
    mean_scale = max(float(args.robust_zico_mean_dispersion_scale), 1e-12)
    top_scale = max(float(args.robust_zico_top_dispersion_scale), 1e-12)
    mean_z = (
        mean_dispersion -
        float(args.robust_zico_mean_dispersion_center)) / mean_scale
    top_z = (
        top_dispersion -
        float(args.robust_zico_top_dispersion_center)) / top_scale
    return (
        math.log(max(float(raw_zico), 1e-12)) -
        float(args.robust_zico_mean_dispersion_weight) * mean_z -
        float(args.robust_zico_top_dispersion_weight) * top_z)


def proxy_quality(score_value, extras, args, item=None):
    if score_value is None or not math.isfinite(float(score_value)):
        return -1e9
    score_value = float(score_value)
    if args.quality_mode in SWAP_QUALITY_MODES:
        return score_value
    if args.quality_mode == 'zico':
        if score_value <= 0:
            return -1e9
        return float(score_value)
    expr = float(extras.get('expressivity', float('nan')))
    ntk_condition = float(extras.get('ntk_condition', float('nan')))
    if args.quality_mode == 'zico_expr':
        if score_value <= 0:
            return -1e9
        log_zico = math.log(max(score_value, 1e-12))
        if not math.isfinite(expr) or expr <= 0:
            return -1e9
        return log_zico + math.log(max(expr, 1e-12))
    if args.quality_mode == 'old_zico_ntk':
        if score_value <= 0:
            return -1e9
        log_zico = math.log(max(score_value, 1e-12))
        if not math.isfinite(ntk_condition) or ntk_condition <= 0:
            return -1e9
        return log_zico - 0.5 * math.log(ntk_condition)
    if args.quality_mode == 'zico_expr_ntk':
        if score_value <= 0:
            return -1e9
        log_zico = math.log(max(score_value, 1e-12))
        if (not math.isfinite(expr) or expr <= 0 or
                not math.isfinite(ntk_condition) or ntk_condition <= 0):
            return -1e9
        return (
            log_zico + math.log(max(expr, 1e-12)) -
            float(args.ntk_penalty) * math.log(ntk_condition))
    if args.quality_mode in (
            'zico_norm_mean', 'zico_norm_top', 'zico_norm_clip',
            'zico_norm_balanced'):
        return score_value
    if args.quality_mode in (
            'zico_norm_mean_expr', 'zico_norm_top_expr',
            'zico_norm_clip_expr', 'zico_norm_balanced_expr'):
        if not math.isfinite(expr) or expr <= 0:
            return -1e9
        return score_value + math.log(max(expr, 1e-12))
    if args.quality_mode == 'robust_zico':
        return robust_zico_quality(score_value, item or {}, args)
    if args.quality_mode == 'robust_zico_expr':
        if not math.isfinite(expr) or expr <= 0:
            return -1e9
        robust = robust_zico_quality(score_value, item or {}, args)
        if not math.isfinite(robust):
            return -1e9
        return robust + math.log(max(expr, 1e-12))
    raise ValueError(f'Unsupported quality mode: {args.quality_mode}')


def real_bounds(context, args):
    xu = []
    for layer in context['layers']:
        pool = context['layer_pools'].get(layer, [])
        xu.extend([1, max(0, len(pool) - 1), max(0, len(args.operators) - 1)])
    if len(xu) == 0:
        xu = [0, 0, 0]
    return np.zeros(len(xu), dtype=float), np.asarray(xu, dtype=float)


def decode_real_individual(x, context, args):
    ea = context['ea']
    candidate = ea.empty_candidate(len(context['primary_blocks']))
    values = [int(round(float(value))) for value in x]
    cursor = 0
    for layer in context['layers']:
        enabled = values[cursor] > 0
        branch_index = values[cursor + 1]
        operator_index = values[cursor + 2]
        cursor += 3
        pool = context['layer_pools'].get(layer, [])
        if not enabled or len(pool) == 0:
            continue
        branch = pool[min(max(0, branch_index), len(pool) - 1)]
        operator = args.operators[min(max(0, operator_index), len(args.operators) - 1)]
        candidate[layer] = ea.LocalGene(branches=[branch], operator=operator)
    return ea.enforce_candidate(
        candidate,
        context['primary_blocks'],
        context['blocks_by_index'],
        args)


def real_individual_key(x):
    return tuple(int(round(float(value))) for value in x)


def make_real_problem(args, context, pymoo_api, records):
    ea = context['ea']
    xl, xu = real_bounds(context, args)

    class DeRyLocalBranchProblem(pymoo_api['ElementwiseProblem']):
        def __init__(self):
            super().__init__(n_var=len(xl), n_obj=3, xl=xl, xu=xu)

        def _evaluate(self, x, out, *unused_args, **unused_kwargs):
            raw_key = real_individual_key(x)
            candidate = decode_real_individual(x, context, args)
            key = context['ea'].candidate_signature(candidate)
            if key not in records:
                item = ea.evaluate_candidate(
                    candidate,
                    context['base_cfg'],
                    context['primary_cfgs'],
                    context['primary_blocks'],
                    context['indicator'],
                    args,
                    context['memo'])
                swap_scores = dict(
                    raw_swap=float('nan'), layer_swap_sum=float('nan'),
                    layer_swap_sqrt=float('nan'))
                score_key = quality_score_key(args)
                score_value = item.get(score_key)
                swap_error = None
                if (item.get('error') is None and
                        args.quality_mode in SWAP_QUALITY_MODES):
                    try:
                        swap_scores = compute_swap_scores(
                            candidate, context, args)
                        score_value = swap_scores[args.quality_mode]
                    except Exception as exc:
                        swap_error = f'swap_{type(exc).__name__}: {exc}'
                if (item.get('error') is None and swap_error is None and
                        score_value is not None):
                    try:
                        extras = compute_quality_extras(candidate, context, args)
                        quality = proxy_quality(
                            score_value, extras, args, item)
                        objectives = [
                            -float(quality),
                            float(item['size']),
                            float(item['flops'])]
                        error = None
                        zico_value = item.get('zico')
                        zico = (
                            float(zico_value) if zico_value is not None
                            else float('nan'))
                        normalized_zico_mean = float(item.get(
                            'normalized_zico_mean', float('nan')))
                        normalized_zico_top = float(item.get(
                            'normalized_zico_top', float('nan')))
                        normalized_zico_clip = float(item.get(
                            'normalized_zico_clip', float('nan')))
                        normalized_zico_balanced = float(item.get(
                            'normalized_zico_balanced', float('nan')))
                        normalized_zico_mean_dispersion = float(item.get(
                            'normalized_zico_mean_dispersion', float('nan')))
                        normalized_zico_top_dispersion = float(item.get(
                            'normalized_zico_top_dispersion', float('nan')))
                        normalized_zico_clip_dispersion = float(item.get(
                            'normalized_zico_clip_dispersion', float('nan')))
                        expressivity = extras.get('expressivity', float('nan'))
                        progressivity = extras.get('progressivity', float('nan'))
                        ntk_condition = extras.get(
                            'ntk_condition', float('nan'))
                        size = float(item['size'])
                        flops = float(item['flops'])
                    except Exception as exc:
                        objectives = [1e9, 1e9, 1e9]
                        error = f'quality_{type(exc).__name__}: {exc}'
                        quality = -1e9
                        zico = float(item.get('zico') or -1e9)
                        normalized_zico_mean = float(item.get(
                            'normalized_zico_mean', float('nan')))
                        normalized_zico_top = float(item.get(
                            'normalized_zico_top', float('nan')))
                        normalized_zico_clip = float(item.get(
                            'normalized_zico_clip', float('nan')))
                        normalized_zico_balanced = float(item.get(
                            'normalized_zico_balanced', float('nan')))
                        normalized_zico_mean_dispersion = float(item.get(
                            'normalized_zico_mean_dispersion', float('nan')))
                        normalized_zico_top_dispersion = float(item.get(
                            'normalized_zico_top_dispersion', float('nan')))
                        normalized_zico_clip_dispersion = float(item.get(
                            'normalized_zico_clip_dispersion', float('nan')))
                        expressivity = float('nan')
                        progressivity = float('nan')
                        ntk_condition = float('nan')
                        size = float(item.get('size') or 1e9)
                        flops = float(item.get('flops') or 1e9)
                else:
                    objectives = [1e9, 1e9, 1e9]
                    error = swap_error or item.get('error') or 'invalid'
                    quality = -1e9
                    zico = -1e9
                    normalized_zico_mean = float('nan')
                    normalized_zico_top = float('nan')
                    normalized_zico_clip = float('nan')
                    normalized_zico_balanced = float('nan')
                    normalized_zico_mean_dispersion = float('nan')
                    normalized_zico_top_dispersion = float('nan')
                    normalized_zico_clip_dispersion = float('nan')
                    expressivity = float('nan')
                    progressivity = float('nan')
                    ntk_condition = float('nan')
                    size = float(item.get('size') or 1e9)
                    flops = float(item.get('flops') or 1e9)
                records[key] = dict(
                    id=len(records),
                    individual=list(raw_key),
                    candidate_signature=repr(key),
                    candidate=candidate,
                    quality_mode=args.quality_mode,
                    quality=float(quality),
                    zico=zico,
                    normalized_zico_mean=normalized_zico_mean,
                    normalized_zico_top=normalized_zico_top,
                    normalized_zico_clip=normalized_zico_clip,
                    normalized_zico_balanced=normalized_zico_balanced,
                    normalized_zico_mean_dispersion=normalized_zico_mean_dispersion,
                    normalized_zico_top_dispersion=normalized_zico_top_dispersion,
                    normalized_zico_clip_dispersion=normalized_zico_clip_dispersion,
                    expressivity=expressivity,
                    progressivity=progressivity,
                    ntk_condition=ntk_condition,
                    raw_swap=swap_scores.get('raw_swap'),
                    layer_swap_sum=swap_scores.get('layer_swap_sum'),
                    layer_swap_sqrt=swap_scores.get('layer_swap_sqrt'),
                    size=size,
                    flops=flops,
                    objectives=objectives,
                    block_list_summary='fixed DeRy backbone with optional CompositeBlock local branches',
                    branch_summary=ea.format_candidate(candidate),
                    operator_summary=', '.join(
                        gene.operator for gene in candidate
                        if gene is not None and len(gene.branches) > 0),
                    structure_group=candidate_structure_group(candidate),
                    error=error)
            out['F'] = np.asarray(records[key]['objectives'], dtype=float)

    return DeRyLocalBranchProblem()


def make_callback(pymoo_api, generation_logs, records_fn):
    class SearchCallback(pymoo_api['Callback']):
        def notify(self, algorithm):
            pop_f = np.asarray(algorithm.pop.get('F'), dtype=float)
            finite = np.isfinite(pop_f).all(axis=1)
            valid_f = pop_f[finite & (pop_f[:, 0] < 1e9)]
            if len(valid_f) > 0:
                pareto_count = len(non_dominated_objective_indices(valid_f))
                best_zico = -float(np.min(valid_f[:, 0]))
                min_size = float(np.min(valid_f[:, 1]))
                min_flops = float(np.min(valid_f[:, 2]))
                valid_count = int(len(valid_f))
                failed_count = int(len(pop_f) - len(valid_f))
            else:
                pareto_count = 0
                best_zico = -float('inf')
                min_size = float('inf')
                min_flops = float('inf')
                valid_count = 0
                failed_count = int(len(pop_f))
            generation_logs.append(dict(
                generation=int(algorithm.n_gen),
                valid=valid_count,
                pareto=pareto_count,
                best_zico=float(best_zico),
                min_size=float(min_size),
                min_flops=float(min_flops),
                failed=failed_count))

    return SearchCallback()


def run_minimize(pymoo_api, args, problem, callback):
    ref_dirs = make_ref_dirs(pymoo_api, args)
    algorithm = pymoo_api['NSGA3'](
        pop_size=args.population_size,
        ref_dirs=ref_dirs,
        sampling=pymoo_api['FloatRandomSampling'](),
        crossover=pymoo_api['SBX'](prob=0.9, eta=15),
        mutation=pymoo_api['PM'](eta=20),
        eliminate_duplicates=True)
    result = pymoo_api['minimize'](
        problem,
        algorithm,
        ('n_gen', args.generations),
        seed=args.seed,
        callback=callback,
        verbose=False)
    return result


def final_front(records):
    valid = finite_records(records)
    if not valid:
        return []
    return [valid[index] for index in non_dominated_indices(valid)]


def rank_key(record):
    return (
        -float(record.get('quality', record.get('zico', -1e9))),
        float(record.get('size', 1e9)),
        float(record.get('flops', 1e9)),
        int(record.get('id', 10 ** 9)))


def selected_front(records, top_k=None, structure_min_quota=0):
    front = final_front(records)
    pareto_ids = {item['id'] for item in front}
    for item in front:
        item['is_pareto_front'] = True
        item['selection_reason'] = 'pareto'

    front.sort(key=rank_key)
    if top_k is None:
        return front

    limit = max(1, int(top_k))
    quota = max(0, int(structure_min_quota))
    if quota <= 0:
        if len(front) >= limit:
            return front[:limit]

        selected_ids = {item['id'] for item in front}
        valid = finite_records(records)
        valid.sort(key=rank_key)
        for item in valid:
            if item['id'] in selected_ids:
                continue
            item['is_pareto_front'] = item['id'] in pareto_ids
            item['selection_reason'] = 'top_quality_fill'
            front.append(item)
            selected_ids.add(item['id'])
            if len(front) >= limit:
                break
        return front

    valid = finite_records(records)
    valid.sort(key=rank_key)
    quota_groups = ('branch_gate', 'branch_sum', 'branch_mixed')
    available_groups = [
        group for group in quota_groups
        if any(item.get('structure_group') == group for item in valid)]
    if quota * len(available_groups) > limit:
        raise ValueError(
            '--structure-min-quota exceeds --top-k capacity: '
            f'{quota} * {len(available_groups)} > {limit}')

    selected = []
    selected_ids = set()
    for group in available_groups:
        group_rows = [
            item for item in valid
            if item.get('structure_group') == group]
        for item in group_rows[:quota]:
            item['is_pareto_front'] = item['id'] in pareto_ids
            item['selection_reason'] = f'structure_quota:{group}'
            selected.append(item)
            selected_ids.add(item['id'])

    for source, reason in ((front, 'pareto'), (valid, 'top_quality_fill')):
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


def main():
    args = parse_args()
    args.swap_image_count = max(2, int(args.swap_image_count))
    objectives = [name.strip() for name in args.objectives.split(',') if name.strip()]
    if objectives != ['zico', 'size', 'flops']:
        raise SystemExit('This first NSGA-III version supports --objectives zico,size,flops only.')

    os.makedirs(args.output_dir, exist_ok=True)
    config_dir = ensure_output_dirs(args.output_dir)
    pymoo_api = import_pymoo()

    if args.toy:
        result, records, generation_logs, context = run_toy(args, pymoo_api)
        build_config_fn = None
    else:
        context = make_real_context(args)
        records_by_key = {}
        problem = make_real_problem(args, context, pymoo_api, records_by_key)
        generation_logs = []
        callback = make_callback(
            pymoo_api,
            generation_logs,
            lambda: list(records_by_key.values()))
        result = run_minimize(pymoo_api, args, problem, callback)
        records = list(records_by_key.values())

        def build_config_fn(record):
            return context['ea'].build_candidate_config(
                context['base_cfg'],
                context['primary_cfgs'],
                context['primary_blocks'],
                record['candidate'])

    front = selected_front(
        records, args.top_k, args.structure_min_quota)
    reps = select_representatives(front)
    export_configs(front, reps, config_dir, build_config_fn=build_config_fn)
    csv_path = save_csv(front, reps, args.output_dir)
    log_path = write_search_log(
        args.output_dir,
        args,
        make_ref_dirs(pymoo_api, args),
        generation_logs,
        records)
    json_path = save_records_json(args.output_dir, records)

    print('NSGA-III search complete')
    print(f'output_dir: {args.output_dir}')
    print(f'pareto_front: {csv_path}')
    print(f'search_log: {log_path}')
    print(f'all_evaluations: {json_path}')
    print(f'configs: {config_dir}')
    if result is None:
        print('warning: pymoo returned no result object')


if __name__ == '__main__':
    main()
