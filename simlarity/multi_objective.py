"""Pareto-ranking utilities for HeteroWeave search candidates."""

import math


MAXIMIZE = {
    'score', 'objective', 'proxy', 'zero_proxy_value',
    'naswot', 'zico', 'real_score', 'real_acc1', 'acc1',
}

MINIMIZE = {
    'size', 'params', 'params_m', 'flops', 'flops_g',
    'real_loss', 'adapter_burden', 'type_switches',
}

ALIASES = {
    'objective': ('objective', 'score', 'raw_score'),
    'score': ('score', 'objective', 'raw_score'),
    'proxy': ('proxy', 'zero_proxy_value', 'value', 'score', 'objective'),
    'zero_proxy_value': ('zero_proxy_value', 'proxy', 'value', 'score'),
    'params': ('params', 'params_m', 'size'),
    'params_m': ('params_m', 'size', 'params'),
    'size': ('size', 'params_m', 'params'),
    'flops': ('flops', 'flops_g'),
    'flops_g': ('flops_g', 'flops'),
    'acc1': ('acc1', 'real_acc1'),
}


def _lookup(container, name):
    if not isinstance(container, dict):
        return None
    for key in ALIASES.get(name, (name,)):
        if key in container:
            return container[key]
    metrics = container.get('metrics')
    if isinstance(metrics, dict):
        for key in ALIASES.get(name, (name,)):
            if key in metrics:
                return metrics[key]
    return None


def metric_value(item, name):
    if name == 'neg_real_loss':
        value = _lookup(item, 'real_loss')
        if value is None:
            return None
        value = -float(value)
    else:
        value = _lookup(item, name)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def objective_direction(name):
    if name == 'neg_real_loss':
        return 'max'
    if name in MAXIMIZE:
        return 'max'
    if name in MINIMIZE:
        return 'min'
    raise ValueError(f'Unknown Pareto objective: {name}')


def has_objectives(item, objectives):
    return all(metric_value(item, name) is not None for name in objectives)


def dominates(left, right, objectives, eps=1e-12):
    better = False
    for name in objectives:
        lv = metric_value(left, name)
        rv = metric_value(right, name)
        if lv is None or rv is None:
            return False
        direction = objective_direction(name)
        if direction == 'max':
            if lv < rv - eps:
                return False
            if lv > rv + eps:
                better = True
        else:
            if lv > rv + eps:
                return False
            if lv < rv - eps:
                better = True
    return better


def non_dominated_fronts(items, objectives):
    valid_items = [item for item in items if has_objectives(item, objectives)]
    dominates_map = {index: [] for index in range(len(valid_items))}
    dominated_count = {index: 0 for index in range(len(valid_items))}
    first_front = []

    for i, left in enumerate(valid_items):
        for j, right in enumerate(valid_items):
            if i == j:
                continue
            if dominates(left, right, objectives):
                dominates_map[i].append(j)
            elif dominates(right, left, objectives):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            first_front.append(i)

    fronts = []
    current = first_front
    while current:
        fronts.append([valid_items[index] for index in current])
        next_front = []
        for index in current:
            for dominated in dominates_map[index]:
                dominated_count[dominated] -= 1
                if dominated_count[dominated] == 0:
                    next_front.append(dominated)
        current = next_front
    return fronts


def crowding_distance(front, objectives):
    if not front:
        return {}
    distances = {id(item): 0.0 for item in front}
    if len(front) <= 2:
        for item in front:
            distances[id(item)] = float('inf')
        return distances

    for name in objectives:
        values = [(metric_value(item, name), item) for item in front]
        values = [(value, item) for value, item in values if value is not None]
        if len(values) <= 2:
            for _, item in values:
                distances[id(item)] = float('inf')
            continue
        values.sort(key=lambda pair: pair[0])
        min_value = values[0][0]
        max_value = values[-1][0]
        distances[id(values[0][1])] = float('inf')
        distances[id(values[-1][1])] = float('inf')
        span = max(max_value - min_value, 1e-12)
        for index in range(1, len(values) - 1):
            if math.isinf(distances[id(values[index][1])]):
                continue
            prev_value = values[index - 1][0]
            next_value = values[index + 1][0]
            distances[id(values[index][1])] += (next_value - prev_value) / span
    return distances


def fallback_score(item):
    for name in ('score', 'objective', 'proxy', 'zero_proxy_value', 'zico', 'naswot'):
        value = metric_value(item, name)
        if value is not None:
            return value
    return -float('inf')


def rank_items(items, objectives):
    ranked = []
    used = set()
    fronts = non_dominated_fronts(items, objectives)
    for rank, front in enumerate(fronts):
        distances = crowding_distance(front, objectives)
        for item in front:
            item['_pareto_rank'] = rank
            item['_crowding_distance'] = distances.get(id(item), 0.0)
            ranked.append(item)
            used.add(id(item))

    missing = [item for item in items if id(item) not in used]
    for item in missing:
        item['_pareto_rank'] = len(fronts) + 1
        item['_crowding_distance'] = -float('inf')
    ranked.extend(missing)
    return sorted(
        ranked,
        key=lambda item: (
            item.get('_pareto_rank', len(fronts) + 1),
            -item.get('_crowding_distance', -float('inf')),
            -fallback_score(item),
        ))


def objective_summary(item, objectives):
    parts = []
    for name in objectives:
        value = metric_value(item, name)
        if value is None:
            parts.append(f'{name}=None')
        else:
            parts.append(f'{name}={value:.6f}')
    return ', '.join(parts)
