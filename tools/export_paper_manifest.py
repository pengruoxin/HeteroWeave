#!/usr/bin/env python3
"""Export a Python paper-model configuration as JSON or CSV."""

import argparse
import csv
import hashlib
import json
import runpy
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('output')
    parser.add_argument('--format', choices=('json', 'csv'), default='json')
    args = parser.parse_args()

    namespace = runpy.run_path(args.config)
    candidates = namespace['candidates']
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == 'json':
        canonical = json.dumps(candidates, sort_keys=True, separators=(',', ':'))
        payload = {
            'manifest_sha256': hashlib.sha256(canonical.encode()).hexdigest(),
            'candidates': candidates,
        }
        output.write_text(json.dumps(payload, indent=2) + '\n')
        return

    rows = []
    for candidate in candidates:
        row = dict(candidate)
        if 'components' in row:
            row['sources'] = json.dumps(row.pop('components'))
        if 'gamma' in row:
            row['gate'] = row.pop('gamma')
        row['candidate'] = row.pop('id')
        rows.append(row)
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()

