# Causal latent patching, seeds 0 and 1 (kernel `prosqa-causal-patching` version 4)

- **Code:** tag `prosqa-causal-patching-v4`, notebook sha1 1a1ca815b9, protocol `docs/causal_protocol.md` frozen at Amendment 4. Run id 1794156430. Kaggle: 2 x Tesla T4, torch 2.11.0+cu130, transformers 5.20.2 (local development used 5.22.2; the 16 unit tests and a 6-row driver run were repeated under 5.20.2 before the push). Batch size 32 (speed only).
- **Inputs:** dataset `jyothikamaleshs/prosqa-coconut-b-twins` (`s{0,1,2}_coconut_b.pt`, SHA-256 in `docs/provenance_checkpoints.json`, and the saved generations). Seed 0 is the v3 twin, seeds 1 and 2 the v4 twins.
- **Outcome of the kernel:** status `ERROR` because the seed 2 worker failed (see below); GPU 0 ran seed 0 then seed 2, GPU 1 ran seed 1. Seeds 0 and 1 completed (28.0 and 27.3 min), all guards passed: 16 unit tests OK on Kaggle, identity-patch difference 0.0, rebuild agreement 1.000 (hinted and unhinted, every group), 300 rows each (seed 0: 116 harmful / 78 resistant / 106 wrong-either; seed 1: 120 / 68 / 112).
- **Seed 2 failure:** an input guard asserted that every saved answer is one of the two options; one seed-2 row has a third-option unhinted answer (`horpus`), which is in none of the three groups. The guard was too strict and fired before group assignment. Fixed in the next version (third-option rows are counted and skipped; `third_option_rows_skipped` in the meta). The fix touches no condition or estimand. It was written after the seed-0 PRIMARY line had been seen in the log; seeds 0 and 1 never triggered the guard, so their results are unaffected. Offline check on the real files: seed 2 gives 299 rows (105 / 86 / 108), 1 row skipped; seed 1 gives 300 rows, 0 skipped.
- **Earlier attempts:** versions 1 and 2 of the kernel were cancelled while queued (no log, cause unknown); version 3 started after about 30 min in the queue and failed on the first code cell because Kaggle mounts the dataset under `/kaggle/input/datasets/<user>/<slug>/` (fixed with a recursive search).
- **Files:** `results/patching_s{0,1}/patching_summary.json` (with `condition_descriptions`, length sensitivity), `patching_rows.jsonl`, per-seed logs, `kernel.log`, `patching_s2_FAILED.log`.

## Result, harmful switches, change in M (positive = towards the true option), mean [95% question-clustered CI]
| condition | seed 0 (n=116) | seed 1 (n=120) |
|---|---|---|
| single-step restoration, natural unhinted donor (registered primary) | -0.000 [-0.000, -0.000] | -0.000 [-0.001, -0.000] |
| single-step, position-matched donor | -0.000 | -0.000 |
| persistent restoration (secondary) | -0.001 | -0.002 |
| all six inputs replaced (hinted prompt still accessible) | -0.001 | +0.000 |
| scale controls: zeros in all six slots / another question's six | -0.034 / -0.000 | +0.019 / +0.001 |
| answer blocked from the hint span and `<bot>` | +3.581 [+3.344, +3.823], 99% now prefer the true option | +3.565 [+3.321, +3.821], 97% |
| both pathways blocked (natural / matched-attention donors) | +3.581 / +3.580 | +3.567 / +3.566 |
| disruption baseline: same blocks on UNHINTED prompts | +0.002 | +0.004 |
