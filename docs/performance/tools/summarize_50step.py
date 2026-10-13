"""Usage: python summarize_50step.py OUTPUT_DIR; synchronized intervals, steps11-50."""
import json
import math
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
metadata = [json.loads(p.read_text()) for p in root.glob('rank-*-environment.json')]
assert metadata, 'No rank metadata'
world = int(metadata[0]['world_size'])
assert sorted(m['rank'] for m in metadata) == list(range(world))
assert all(m['world_size'] == world for m in metadata)
rows = []
for rank in range(world):
    data = [json.loads(line) for line in (root / f'rank-{rank}-performance.jsonl').read_text().splitlines()]
    assert [r['step'] for r in data] == list(range(1, 51)), f'Incomplete rank {rank}'
    assert all(r['batch_size'] == 8 and math.isfinite(r['interval_s']) and r['interval_s'] > 0 for r in data)
    rows.append(data)
steady = [r['interval_s'] for r in rows[0][10:]]
print(json.dumps({
    'world_size': world, 'steady_window': [11, 50],
    'rank0_steady_mean_s': statistics.mean(steady),
    'rank0_steady_median_s': statistics.median(steady),
    'rank0_all50_mean_s': statistics.mean(r['interval_s'] for r in rows[0]),
    'rank_steady_means_s': [statistics.mean(r['interval_s'] for r in data[10:]) for data in rows],
    'max_peak_allocated_gib': max(r['peak_allocated_bytes'] for data in rows for r in data) / 2**30,
    'note': 'GBS assumes accumulation=1; additionally check launcher config, exit codes, finite metrics and sampled updates.'
}, indent=2))
