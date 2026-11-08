"""Label-free detector scores of a finished run, split by the harmless decoy line (uses s{seed}_monitor_scores.json + generations; no model needed).
    python -I prosqa_monitor/analysis/decoy_stratified_scores.py <results_dir>"""
import json, sys, glob, numpy as np
from sklearn.metrics import roc_auc_score
auc = lambda y, s: roc_auc_score(y, s) if len(set(y)) == 2 else float('nan')
for mp in sorted(glob.glob(f'{sys.argv[1]}/s*_run_meta.json')):
    seed = json.load(open(mp))['seed']; sc = json.load(open(mp.replace('run_meta', 'monitor_scores'))); gen = [json.loads(l) for l in open(mp.replace('run_meta.json', 'generations.jsonl'))]
    for model, pat in (('coconut_b', 'coconut_c -> coconut_b'), ('coconut_b', 'CROSS-FORMAT'), ('explicit_b', 'explicit_c -> explicit_b')):
        rows = sorted([r for r in gen if r['model'] == model and r['split'] == 'te'], key=lambda r: r['idx'])
        sel = np.array([r['ok'] and not r['hint_wrong'] for r in rows]); d = np.array([r['decoy'] for r in rows])[sel].astype(int)
        for k, v in sc.items():
            if not (k.startswith('label-free') and k.endswith('[correct-hint]') and pat in k) or len(v['y']) != sel.sum(): continue
            y, s = np.array(v['y']), np.array(v['s'])
            print(f"seed {seed} {pat:26s} all {auc(y, s):.3f} | decoy {auc(y[d == 1], s[d == 1]):.3f} | no-decoy {auc(y[d == 0], s[d == 0]):.3f} | mean score decoy {s[d == 1].mean():.4f} vs no-decoy {s[d == 0].mean():.4f}")
