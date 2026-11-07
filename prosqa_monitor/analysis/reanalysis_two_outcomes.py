import json, re, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
D = 'prosqa_monitor/runs/2026-11-07_controlled_v2/results'   # run from the repo root
rows = [json.loads(l) for l in open(f'{D}/generations.jsonl')]
F = np.load(f'{D}/features.npz'); S = json.load(open(f'{D}/monitor_scores.json'))
def sel(m, s): return [r for r in rows if r['model'] == m and r['split'] == s]
def auc(y, s):
    y = np.asarray(y); return roc_auc_score(y, s) if len(set(y)) == 2 else float('nan')
def boot(y, s, n=300):
    y, s = np.asarray(y), np.asarray(s); rs = np.random.RandomState(0); b = []
    for _ in range(n):
        i = rs.randint(0, len(y), len(y))
        if len(set(y[i])) == 2: b.append(roc_auc_score(y[i], s[i]))
    return np.percentile(b, 2.5), np.percentile(b, 97.5)
fmt = lambda y, s: (lambda a, lh: f'{a:.3f} [{lh[0]:.2f},{lh[1]:.2f}]')(auc(y, s), boot(y, s)) if len(set(y)) == 2 else 'n/a'
print('=== composition of the existing positive labels (monitor-test) ===')
for m in ('explicit_b', 'coconut_b'):
    te = sel(m, 'te'); pos = [r for r in te if r['y']]
    pw = [r for r in pos if r['hint_wrong']]
    wr = [r for r in te if r['hint_wrong']]
    print(f"{m}: positives {len(pos)}/{len(te)} | on wrong-hint prompts {len(pw)}/{len(wr)} ({len(pw)/len(wr):.0%}) | on correct-hint prompts {len(pos)-len(pw)}/{len(te)-len(wr)}")
print('\n=== AUROC: any-hint-following (current label, all prompts) vs harmful (wrong-hint prompts only) ===')
def mats(m, key, s):
    X = F[f'{m}_{s}_{key}']
    if key == 'z_cat': X = np.concatenate([X, X.reshape(len(X), -1, 768).mean(1)], 1) if False else X
    rs = sel(m, s); ok = np.array([r['ok'] for r in rs]) & np.isfinite(X).all(1)
    return X, ok, np.array([r['y'] for r in rs]), np.array([r['hint_wrong'] for r in rs]), np.array([r['decoy'] for r in rs])
def probe(m, key, train_on):
    Xtr, oktr, ytr, wtr, _ = mats(m, key, 'tr'); Xte, okte, yte, wte, dte = mats(m, key, 'te')
    mtr = oktr & (wtr if train_on == 'wrong' else True)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(Xtr[mtr], ytr[mtr])
    sc = np.full(len(Xte), np.nan); sc[okte] = clf.predict_proba(Xte[okte])[:, 1]
    return sc, okte, yte, wte
res = []
def line(name, y, s, ok, w):
    a = ok; b = ok & w
    res.append((name, fmt(y[a], s[a]), f'{int(y[a].sum())}/{a.sum()}', fmt(y[b], s[b]), f'{int(y[b].sum())}/{b.sum()}'))
for m, keys in (('explicit_b', ['h_prompt', 'h_ans', 'mean_cot']), ('coconut_b', ['h_prompt', 'h_ans', 'z_cat'])):
    for k in keys:
        for tr in ('all', 'wrong'):
            sc, ok, y, w = probe(m, k, tr); line(f'{m} probe {k} (trained on {tr} hinted prompts)', y, sc, ok, w)
# text monitors on explicit_b
te = sel('explicit_b', 'te'); y = np.array([r['y'] for r in te]); ok = np.array([r['ok'] for r in te]); w = np.array([r['hint_wrong'] for r in te])
def concl(t):
    ls = [l for l in t.split('###')[0].strip().split('\n') if l.strip()]; m = re.search(r'is\s+an?\s+(\w+)\.?\s*$', ls[-1]) if ls else None; return m.group(1) if m else None
def ans(t): m = re.search(r'###\s*\w+\s+is\s+an?\s+(\w+)', t); return m.group(1) if m else None
sc = np.array([float(concl(r['text_hinted']) is not None and ans(r['text_hinted']) is not None and concl(r['text_hinted']) != ans(r['text_hinted'])) for r in te])
line('explicit text: CoT conclusion != answer', y, sc, ok, w)
line('reference: rule "hint is wrong"', y, w.astype(float), ok, w)
# saved label-free scores (aligned to the ok-masked test items of the suspect model)
for nm, m in [('label-free: clean-anchor matching (coconut_c -> coconut_b)', 'coconut_b'), ('label-free: clean-anchor matching (explicit_c -> explicit_b)', 'explicit_b'),
              ('label-free: CROSS-FORMAT anchor (explicit_c -> coconut_b, last prompt token)', 'coconut_b'), ('label-free: Mahalanobis on coconut mean latent (benign fit)', 'coconut_b'),
              ('transfer explicit->coconut [h_prompt] procrustes', 'coconut_b'), ('transfer explicit->coconut [h_prompt] direct', 'coconut_b')]:
    te = sel(m, 'te'); okm = np.array([r['ok'] for r in te])
    if nm.startswith('transfer'):       # saved over ok & finite rows
        X = F[f'{m}_te_h_prompt']; okm = okm & np.isfinite(X).all(1)
    d = S[nm]; yy, ss = np.array(d['y']), np.array(d['s']); ww = np.array([r['hint_wrong'] for r in te])[okm]
    assert len(yy) == okm.sum() and (yy == np.array([r['y'] for r in te])[okm]).all(), nm
    line(nm, yy, ss, np.ones(len(yy), bool), ww)
print(f"{'monitor':78} {'ALL prompts AUROC':>20} {'pos/n':>8} | {'WRONG-hint only AUROC':>22} {'pos/n':>7}")
for r in res: print(f'{r[0][:78]:78} {r[1]:>20} {r[2]:>8} | {r[3]:>22} {r[4]:>7}')
