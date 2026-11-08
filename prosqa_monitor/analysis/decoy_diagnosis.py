"""Does the harmless decoy line leak into the monitors? Works on saved features (s{seed}_features.npz) and generations; no model needed.
    python -I prosqa_monitor/analysis/decoy_diagnosis.py <results_dir> [<results_dir> ...]
A: is the decoy tied to the hint type or the label?  B: can the decoy be read from the features?
C: probe AUROC on the [correct-hint] subset overall, inside each decoy stratum, and with the decoy direction projected out."""
import json, sys, glob, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
lin = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000))
auc = lambda y, s: roc_auc_score(y, s) if len(set(y)) == 2 else float('nan')
FEATS = {'explicit_b': ['h_prompt', 'h_ans', 'mean_cot'], 'coconut_b': ['h_prompt', 'h_ans', 'z_mean']}
rows_out = []
for d in sys.argv[1:]:
    for mp in sorted(glob.glob(f'{d}/s*_run_meta.json')):
        seed = json.load(open(mp))['seed']; z = np.load(mp.replace('run_meta.json', 'features.npz'))
        gen = [json.loads(l) for l in open(mp.replace('run_meta.json', 'generations.jsonl'))]
        for name, keys in FEATS.items():
            R = {s: sorted([r for r in gen if r['model'] == name and r['split'] == s], key=lambda r: r['idx']) for s in ('tr', 'te')}
            A = {s: {k: np.array([r[k] for r in R[s]]) for k in ('y', 'ok', 'hint_wrong', 'decoy')} for s in R}
            te = A['te']; cm = te['ok'] & ~te['hint_wrong'].astype(bool)
            print(f"\n=== seed {seed} {name} ===  decoy~hint_wrong corr {np.corrcoef(te['decoy'], te['hint_wrong'])[0,1]:+.2f} | P(y=1|decoy) {te['y'][te['ok'] & te['decoy'].astype(bool)].mean():.2f} vs no-decoy {te['y'][te['ok'] & ~te['decoy'].astype(bool)].mean():.2f} (test), "
                  f"train {A['tr']['y'][A['tr']['ok'] & A['tr']['decoy'].astype(bool)].mean():.2f} vs {A['tr']['y'][A['tr']['ok'] & ~A['tr']['decoy'].astype(bool)].mean():.2f}")
            for k in keys:
                get = lambda s: z[f'{name}_{s}_z_cat'].reshape(-1, 6, 768).mean(1) if k == 'z_mean' else z[f'{name}_{s}_{k}']
                Xtr, Xte = get('tr'), get('te'); mtr = A['tr']['ok'] & np.isfinite(Xtr).all(1); mte = A['te']['ok'] & np.isfinite(Xte).all(1)
                dtr, dte = A['tr']['decoy'][mtr].astype(int), A['te']['decoy'][mte].astype(int)
                dec_auc = auc(dte, lin().fit(Xtr[mtr], dtr).predict_proba(Xte[mte])[:, 1])            # B: decoy readable from features
                P = lin().fit(Xtr[mtr], A['tr']['y'][mtr]); sc = np.full(len(Xte), np.nan); sc[mte] = P.predict_proba(Xte[mte])[:, 1]
                sel = cm & mte; ys, ss, ds = te['y'][sel], sc[sel], te['decoy'][sel].astype(int)
                a_all = auc(ys, ss); a_dec = auc(ys[ds == 1], ss[ds == 1]); a_nod = auc(ys[ds == 0], ss[ds == 0])
                # C2: project the decoy mean-difference direction out of both train and test, refit
                sd = StandardScaler().fit(Xtr[mtr]); Ztr, Zte = sd.transform(Xtr[mtr]), sd.transform(Xte)
                v = Ztr[dtr == 1].mean(0) - Ztr[dtr == 0].mean(0); v /= np.linalg.norm(v) + 1e-9
                Ptr, Pte = Ztr - np.outer(Ztr @ v, v), Zte - np.outer(Zte @ v, v)
                P2 = LogisticRegression(C=0.01, max_iter=3000).fit(Ptr, A['tr']['y'][mtr]); sc2 = np.full(len(Xte), np.nan); sc2[mte] = P2.predict_proba(Pte[mte])[:, 1]
                a_proj = auc(ys, sc2[sel])
                print(f"  {k:9s} decoy readable AUROC {dec_auc:.2f} | [correct-hint] AUROC {a_all:.3f} | within decoy {a_dec:.3f} | within no-decoy {a_nod:.3f} | decoy direction removed {a_proj:.3f}")
                rows_out.append(dict(seed=seed, model=name, feature=k, decoy_readable=dec_auc, auroc=a_all, within_decoy=a_dec, within_nodecoy=a_nod, decoy_removed=a_proj))
import pandas as pd
df = pd.DataFrame(rows_out); print('\n=== mean over seeds ==='); print(df.groupby(['model', 'feature']).mean(numeric_only=True).drop(columns='seed').round(3).to_string())
df.to_csv('decoy_diagnosis.csv', index=False)
