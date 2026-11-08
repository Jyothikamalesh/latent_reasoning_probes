# ProsQA seeds experiment v2: seeds 0, 1, 2 (kernels `prosqa-seeds-s0` v2 and `prosqa-seeds-s1-s2` v1)

- **Code:** git tags `prosqa-seeds-v2` (seed 0, hash 7a444ecd3a, notebook sha1 d86ea32abb), `prosqa-seeds-v2c` (seeds 1 and 2 in ONE notebook `prosqa_monitor_seeds_seed1_plus2.ipynb`, hash 371dd42e6d, sha1 fc85a39176). The two hashes differ only by checkpoint saving and the multi-seed cell (no training or evaluation change); merge with `aggregate_seeds.py <dirs> --equivalent=7a444ecd3a,371dd42e6d`.
- **Config (`seeds` mode):** N_TRAIN 3000, 1 epoch per stage (explicit twins 7 epochs), 150 held-out questions per learning check, monitor sets 300/600, suite 80 questions x 4 wordings, fail-fast guard on. Primary outcome: correct-hint counterfactual.
- **Logs:** the Kaggle kernel log of the combined seeds 1+2 run was not saved by the download (only seed 0's log is in `results/`); the metrics and gates for every seed are in `s*_run_meta.json`.
- **Time on a T4:** seed 0 113.6 min, seed 1 114.5 min, seed 2 104.9 min. Peak GPU 8.9 of 14.6 GiB (seed 0 log); host RAM was not logged.
- **Checkpoints:** seeds 1 and 2 saved `s{seed}_{explicit_b,explicit_c,coconut_b,coconut_c}.pt` (fp16, 326 MB each) in `results/` (git-ignored, kept locally). Seed 0 has none (its code predates checkpoint saving).
- **History of getting here:** v1 seed 0 (`2026-11-07_seeds_s0`) valid; v1 seed 1 aborted by the guard (`2026-11-07_seeds_s1_aborted`); v2 raised N_TRAIN to 3000, isolated the smoke output, then the OAuth login expired mid-chain, the separate seed-1 kernel was deleted and seeds 1+2 were re-run in one notebook at the user's request.

## Behaviour twins (per seed: unhinted accuracy behaviour / clean, harmful rate behaviour / clean)
| seed | explicit acc | explicit harmful | coconut acc | coconut harmful |
|---|---|---|---|---|
| 0 | 0.72 / 0.71 | 0.54 / 0.03 | 0.72 / 0.72 | 0.35 / 0.01 |
| 1 | 0.69 / 0.59 | 0.52 / 0.00 | 0.74 / 0.71 | 0.54 / 0.01 |
| 2 | 0.64 / 0.72 | 0.45 / 0.03 | 0.64 / 0.71 | 0.25 / 0.00 |

Exit criteria (accuracy gap <= 0.08 and harmful-rate gap >= 0.25 on every seed): **not met on 3 seeds.** Explicit: accuracy gap 0.01 / 0.10 / 0.08 (seed 1: explicit_c fell to 0.59). COCONUT: harmful-rate gap 0.34 / 0.53 / 0.24 (seed 2 just under 0.25; its coconut_b is the weakest behaviour twin, accuracy 0.64). The in-run gates use their own accuracy measure and also flagged explicit_b in seed 2 (below 0.6). Wrong-hint compliance on deep vs shallow problems: explicit 0.89 vs 0.50, COCONUT 0.66 vs 0.71: the depth condition is learned only partly by the explicit twin and not by COCONUT. Status of every seed: `GATES_FAILED` (see `*_run_meta.json`).

## Monitors, [correct-hint] outcome, AUROC mean [min-max] over 3 seeds
Explicit CoT-conclusion text monitor **0.904** [0.852-0.964]; explicit probe h_ans 0.722 [0.647-0.817] vs its difficulty baseline 0.672 [0.583-0.780]; explicit mean_cot 0.709 vs baseline 0.715; explicit prompt-token probe 0.520; output-only exact match 0.579 (explicit) / 0.550 (COCONUT); COCONUT probes 0.51-0.56 (z_cat 0.563); label-free COCONUT clean-anchor 0.565 [0.550-0.592]; cross-format anchor 0.531; probe transfer 0.48-0.52; shuffled-label controls 0.476-0.509. The decoy control for COCONUT anchor residual was far from 0.5 on one seed (0.005): unresolved.

## Reading
The text monitor is strong and stable. Explicit probes are indistinguishable from difficulty baselines (h_ans +0.05 within seed spread, mean_cot -0.006). Nothing latent beats chance reliably. The twins' depth condition and the explicit accuracy match are not reproducible at this size, so the exit criteria remain open.
