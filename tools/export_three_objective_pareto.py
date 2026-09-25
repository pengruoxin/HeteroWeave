#!/usr/bin/env python3
import argparse
import csv
import math
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export a readable Pareto table for total_quality, size, and flops.')
    parser.add_argument('--input', required=True, help='Input summary/final_pareto CSV.')
    parser.add_argument('--output', required=True, help='Output readable Pareto CSV.')
    parser.add_argument(
        '--front-only',
        action='store_true',
        help='Only write the first non-dominated front.')
    return parser.parse_args()


def as_float(value, default=float('nan')):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def truth(value):
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y'}


def row_name(row):
    return row.get('candidate') or row.get('config_path') or ''


def is_eligible(row):
    if 'final_eligible' in row and not truth(row.get('final_eligible')):
        return False
    if row.get('budget_status') == 'out_of_budget':
        return False
    return (
        math.isfinite(as_float(row.get('total_quality'))) and
        math.isfinite(as_float(row.get('size'))) and
        math.isfinite(as_float(row.get('flops'))))


def objective_values(row):
    return (
        as_float(row.get('total_quality')),
        as_float(row.get('size')),
        as_float(row.get('flops')),
    )


def dominates(left, right):
    left_q, left_size, left_flops = objective_values(left)
    right_q, right_size, right_flops = objective_values(right)
    no_worse = (
        left_q >= right_q and
        left_size <= right_size and
        left_flops <= right_flops)
    strictly_better = (
        left_q > right_q or
        left_size < right_size or
        left_flops < right_flops)
    return no_worse and strictly_better


def assign_pareto_ranks(rows):
    remaining = list(range(len(rows)))
    rank = 0
    while remaining:
        front = []
        for index in remaining:
            if not any(
                    dominates(rows[other], rows[index])
                    for other in remaining
                    if other != index):
                front.append(index)
        for index in front:
            rows[index]['three_objective_pareto_rank'] = rank
            rows[index]['is_three_objective_pareto'] = (rank == 0)
        remaining = [index for index in remaining if index not in set(front)]
        rank += 1


def normalized_minimize_values(front):
    values = []
    for row in front:
        quality, size, flops = objective_values(row)
        values.append((-quality, size, flops))
    mins = [min(vector[i] for vector in values) for i in range(3)]
    maxs = [max(vector[i] for vector in values) for i in range(3)]
    normalized = []
    for vector in values:
        normalized.append([
            0.0 if math.isclose(maxs[i], mins[i]) else
            (vector[i] - mins[i]) / (maxs[i] - mins[i])
            for i in range(3)
        ])
    return normalized


def mark_special_points(rows):
    for row in rows:
        row['is_top_total_quality'] = False
        row['is_min_size'] = False
        row['is_min_flops'] = False
        row['is_three_objective_knee'] = False
        row['three_objective_knee_asf'] = ''
        row['selection_role'] = ''

    if not rows:
        return

    max_quality = max(as_float(row.get('total_quality')) for row in rows)
    min_size = min(as_float(row.get('size')) for row in rows)
    min_flops = min(as_float(row.get('flops')) for row in rows)
    eps = 1e-12

    first_front = [
        row for row in rows
        if int(row.get('three_objective_pareto_rank', 999999)) == 0]
    normalized = normalized_minimize_values(first_front)
    asf_values = [max(vector) for vector in normalized]
    knee_index = min(range(len(first_front)), key=lambda index: asf_values[index])

    for index, row in enumerate(first_front):
        row['three_objective_knee_asf'] = asf_values[index]
        if index == knee_index:
            row['is_three_objective_knee'] = True

    for row in rows:
        roles = []
        if abs(as_float(row.get('total_quality')) - max_quality) <= eps:
            row['is_top_total_quality'] = True
            roles.append('top_total_quality')
        if abs(as_float(row.get('size')) - min_size) <= eps:
            row['is_min_size'] = True
            roles.append('min_size')
        if abs(as_float(row.get('flops')) - min_flops) <= eps:
            row['is_min_flops'] = True
            roles.append('min_flops')
        if row.get('is_three_objective_knee'):
            roles.insert(0, 'knee')
        row['selection_role'] = ';'.join(roles)


def sort_rows(rows):
    def key(row):
        return (
            int(row.get('three_objective_pareto_rank', 999999)),
            not truth(row.get('is_three_objective_knee')),
            not truth(row.get('is_top_total_quality')),
            not truth(row.get('is_min_size')),
            not truth(row.get('is_min_flops')),
            -as_float(row.get('total_quality'), -math.inf),
            as_float(row.get('size'), math.inf),
            as_float(row.get('flops'), math.inf),
            row_name(row),
        )
    rows.sort(key=key)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    preferred = [
        'readable_rank',
        'selection_role',
        'candidate',
        'config_path',
        'total_quality',
        'size',
        'flops',
        'three_objective_pareto_rank',
        'is_three_objective_pareto',
        'is_three_objective_knee',
        'three_objective_knee_asf',
        'is_top_total_quality',
        'is_min_size',
        'is_min_flops',
    ]
    fields = []
    for field in preferred:
        if any(field in row for row in rows):
            fields.append(field)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    with open(args.input, newline='', encoding='utf-8') as file:
        rows = [row for row in csv.DictReader(file) if is_eligible(row)]
    assign_pareto_ranks(rows)
    mark_special_points(rows)
    if args.front_only:
        rows = [
            row for row in rows
            if int(row.get('three_objective_pareto_rank', 999999)) == 0]
    sort_rows(rows)
    for index, row in enumerate(rows):
        row['readable_rank'] = index
    write_csv(args.output, rows)
    print(f'wrote three-objective Pareto table: {args.output}')
    for row in rows[:10]:
        role = row.get('selection_role') or 'front'
        print(
            f'{row["readable_rank"]}: {row_name(row)} role={role} '
            f'Q={as_float(row.get("total_quality")):.4f} '
            f'size={as_float(row.get("size")):.6g} '
            f'flops={as_float(row.get("flops")):.6g}')


if __name__ == '__main__':
    main()
