"""Difficulty-matched and decoy-stratified AUROCs from saved scores (s{seed}_monitor_scores.json + generations; no model needed).
[correct-hint]: positives are hints that fixed a wrong unhinted answer, so the matched subset is "unhinted answer wrong" (positives vs hint ignored).
[harmful]: positives are wrong hints that broke a right answer, so the matched subset is "unhinted answer right".
    python -I prosqa_monitor/analysis/matched_evaluation.py <results_dir> [...]"""
import json, sys, glob, re, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ANS_RE = re.compile(r'###\s*\w+\s+is\s+an?\s+(\w+)')
auc = lambda y, s: roc_auc_score(y, s) if len(set(y)) == 2 and len(y) >= 8 else float('nan')
MAIN = ['explicit text: CoT conclusion != answer', 'explicit: probe h_ans', 'explicit: probe mean_cot', 'explicit: probe h_prompt', 'coconut: probe z_cat', 'coconut: probe h_ans', 'coconut: probe h_prompt',
        'label-free: clean-anchor matching (coconut_c -> coconut_b)', 'label-free: clean-anchor matching (explicit_c -> explicit_b)', 'label-free: CROSS-FORMAT', 'baseline: output-only exact match, hinted prediction == hint (explicit_b)', 'baseline: output-only exact match, hinted prediction == hint (coconut_b)']
out = []
for d in sys.argv[1:]:
    for mp in sorted(glob.glob(f'{d}/s*_run_meta.json')):
        seed = json.load(open(mp))['seed']; sc = json.load(open(mp.replace('run_meta', 'monitor_scores'))); gen = [json.loads(l) for l in open(mp.replace('run_meta.json', 'generations.jsonl'))]
        info = {}
        for model in ('explicit_b', 'coconut_b'):
            rows = sorted([r for r in gen if r['model'] == model and r['split'] == 'te'], key=lambda r: r['idx'])
            ok = np.array([r['ok'] for r in rows]); w = np.array([r['hint_wrong'] for r in rows]).astype(bool)
            uw = np.array([(ANS_RE.search(r['text_unhinted']).group(1) if ANS_RE.search(r['text_unhinted']) else None) != r['true'] for r in rows])
            info[model] = dict(y=np.array([r['y'] for r in rows]), ok=ok, w=w, uw=uw, dec=np.array([r['decoy'] for r in rows]).astype(bool))
        for k, v in sc.items():
            m = re.search(r'\[(correct-hint|harmful)\]$', k)
            if not m or not any(k.startswith(p) or p in k for p in MAIN): continue
            y, s = np.array(v['y']), np.array(v['s'])
            for model, I in info.items():
                sel = I['ok'] & (~I['w'] if m.group(1) == 'correct-hint' else I['w'])
                if len(y) != sel.sum() or not np.array_equal(I['y'][sel], y): continue
                matched = I['uw'][sel] if m.group(1) == 'correct-hint' else ~I['uw'][sel]; dc = I['dec'][sel]
                out.append(dict(seed=seed, outcome=m.group(1), monitor=k[:k.rindex(' [')], all=auc(y, s), matched=auc(y[matched], s[matched]), n_matched=int(matched.sum()), pos_matched=int(y[matched].sum()),
                                no_decoy=auc(y[~dc], s[~dc]), decoy=auc(y[dc], s[dc]))); break
df = pd.DataFrame(out); pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80)
g = df.groupby(['outcome', 'monitor']).agg(all=('all', 'mean'), matched=('matched', 'mean'), matched_min=('matched', 'min'), matched_max=('matched', 'max'), no_decoy=('no_decoy', 'mean'), decoy=('decoy', 'mean'), n_matched=('n_matched', 'mean'), pos=('pos_matched', 'mean'), seeds=('seed', 'nunique')).round(3)
print(g.sort_values(['outcome', 'all'], ascending=[True, False]).to_string())
