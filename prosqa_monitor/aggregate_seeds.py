"""Merge per-seed results: python aggregate_seeds.py <dir with s*_monitor_results.csv and s*_run_meta.json> [...]
Refuses to merge runs with different code hashes or modes. Reports mean, min, max AUROC over seeds per monitor/outcome."""
import sys, glob, json, os
import pandas as pd
dirs = sys.argv[1:] or ['.']
metas, frames = {}, []
for d in dirs:
    for mp in sorted(glob.glob(os.path.join(d, 's*_run_meta.json'))):
        m = json.load(open(mp)); csv = mp.replace('run_meta.json', 'monitor_results.csv')
        if not os.path.exists(csv): continue
        metas[m['seed']] = m; frames.append(pd.read_csv(csv).assign(seed=m['seed']))
assert frames, 'no per-seed results found'
hashes, modes = {m['code_hash'] for m in metas.values()}, {m['mode'] for m in metas.values()}
assert len(hashes) == 1 and 'unknown' not in hashes, f'runs come from different code: {hashes}'
assert len(modes) == 1, f'runs come from different modes: {modes}'
bad = {s: [n for n, ok in m['gates'] if not ok] for s, m in metas.items()}
print(f"seeds {sorted(metas)} | code {hashes.pop()} | mode {modes.pop()} | failed gates per seed: {bad}")
df = pd.concat(frames)
g = df.groupby('monitor').agg(seeds=('seed', 'nunique'), auroc_mean=('auroc', 'mean'), auroc_min=('auroc', 'min'), auroc_max=('auroc', 'max'), tpr5_mean=('tpr5', 'mean'), n=('n', 'mean'), pos=('pos', 'mean'), neg=('neg', 'mean')).round(3)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 100); pd.set_option('display.max_rows', 500)
print(g.sort_index().to_string())
g.to_csv('seed_summary.csv')
