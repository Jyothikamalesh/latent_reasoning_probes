# Causal latent patching, seed 2 (kernel `prosqa-causal-patching` version 5)

- **Code:** tag `prosqa-causal-patching-v5-seed2`, notebook sha1 f31e048ef9, protocol `docs/causal_protocol.md` Amendments 1-5 (Amendment 5 = the third-option guard fix). Run id 1794158736. Kaggle: 1 of 2 Tesla T4 used (seed-2-only notebook), torch 2.11.0+cu130, transformers 5.20.2, batch size 32. Status COMPLETE, 27.3 min, queue wait short (running from about 23:03).
- **Inputs:** dataset `jyothikamaleshs/prosqa-coconut-b-twins`, `s2_coconut_b.pt` (v4 twin, SHA-256 in `docs/provenance_checkpoints.json`) and its saved generations.
- **Guards:** 16 unit tests OK; identity-patch difference 0.0; rebuild agreement 1.000 hinted, 0.9967 unhinted (one harmful-group row disagrees between the unhinted greedy answer and the sign of M; gate is 0.90); 299 rows (105 harmful / 86 resistant / 108 wrong-either), 1 third-option row skipped.
- **Files:** `results/patching_s2/patching_summary.json`, `patching_rows.jsonl`, `results/patching_s2.log`, `results/kernel.log`. The download folder `kaggle_output/prosqa-causal-patching` also holds leftovers of the failed version 3 attempt; only files dated after this run were used (run id checked in the summary).

## Result, harmful switches (n=105), change in M [95% question-clustered CI]
single-step restoration (primary) +0.000 [-0.000, +0.000]; position-matched +0.000; persistent +0.000; all six inputs replaced +0.000; zeros in all six +0.041 [+0.025, +0.056]; another question's six -0.000; answer blocked from hint span and `<bot>` +2.330 [+2.156, +2.507], 99% now prefer the true option; both blocked (matched-attention donors) +2.329; the same blocks on UNHINTED prompts +0.000 [-0.005, +0.005].
See the three-seed table in `docs/HISTORY.md`.
