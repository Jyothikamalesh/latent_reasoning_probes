# ProsQA seeds v4: seeds 1 and 2 in ONE notebook (kernel `prosqa-seeds-s1-s2` version 2) + 3-seed analysis with v3 seed 0

- **Code:** tag `prosqa-seeds-v4`, hash **ac70954f44**, notebook `prosqa_monitor_seeds_seed1_plus2.ipynb` sha1 455abb6732, `seeds` mode, config as v3 (N_TRAIN 3000, 150-question checks, monitor sets 300/600, suite 80 x 4 wordings). v4 = v3 + the strict harmful label (`uptake_label`). Seed 0 is the v3 run (`2026-11-08_seeds_v3_s0`, hash 3bd1f99438); the two hashes differ only in the label code (twin training does not use it), merged with `--equivalent`, and seed 0 re-scored with the strict label offline.
- **Run:** one Kaggle session, seed 1 111.5 min, seed 2 99.5 min, both `GATES_FAILED`. Single T4 (one GPU). Checkpoints `s{1,2}_{explicit_b,explicit_c,coconut_b,coconut_c}.pt` saved (326 MB each, git-ignored). Login expired twice during waiting (needed `kaggle auth login --force`). Live log: host RAM about 2.5 of 31 GB, GPU peak about 7.4 of 14.6 GiB in training.
- **Gates failed:** all seeds: explicit_b / coconut_b follow wrong hints on shallow problems, coconut_b below 0.8 on deep problems (the depth condition is not learned). Seed 1 also: explicit twins' accuracy gap 0.09 (> 0.08). Seed 2 also: explicit_c accuracy 0.60 (< 0.6 by rounding).

## Behaviour twins (80 questions x 4 wordings; behaviour / clean)
| seed | explicit acc | explicit harmful | COCONUT acc | COCONUT harmful |
|---|---|---|---|---|
| 0 (v3) | 0.69 / 0.70 | 0.48 / 0.02 | 0.76 / 0.69 | 0.52 / 0.01 |
| 1 | 0.75 / 0.66 | 0.63 / 0.00 | 0.71 / 0.72 | 0.40 / 0.01 |
| 2 | 0.61 / 0.60 | 0.46 / 0.00 | 0.69 / 0.71 | 0.44 / 0.00 |

Exit criteria on 3 seeds: **COCONUT MET** (accuracy gaps <= 0.07, harmful-rate gaps >= 0.39, also on the unseen wording). **Explicit NOT MET** (seed 1 accuracy gap 0.09; seed 2's explicit twins reach only 0.60-0.61 accuracy).

## Monitors, [correct-hint] AUROC, mean [min-max] over 3 seeds (about 70 pos / 222-232 neg per seed)
Explicit text monitor **0.918** [0.905-0.944]; explicit probes mean_cot 0.623 (difficulty baseline 0.670), h_ans 0.558 (0.568), h_prompt 0.487; COCONUT probes h_prompt **0.599** [0.552-0.625], z_cat 0.590 [0.523-0.639], z_mean 0.583, h_ans 0.495; COCONUT clean-anchor matching 0.565 [0.543-0.601]; cross-format anchor 0.506; output-only exact match 0.54; shuffled-label controls 0.48-0.53.

## Strict harmful reliance (wrong hints; probes retrained on the strict label), mean over 3 seeds
Explicit: text 0.890 [0.804-0.943]; output-only exact match 0.694; mean_cot probe 0.681 [0.614-0.764]; h_ans 0.651; h_prompt 0.477; difficulty baselines 0.48-0.49. COCONUT: **output-only exact match 0.707 [0.689-0.721]**; probes h_prompt 0.573, z_cat 0.561, z_mean 0.558, h_ans 0.526; clean-anchor 0.491; cross-format 0.492.

## Matched analyses (two separate)
Hint susceptibility (correct hints, unhinted-wrong subset; about 70 pos / 17 neg): COCONUT h_prompt 0.651, z_cat 0.617; explicit mean_cot 0.951 (reads the answer). Harmful reliance (wrong hints, unhinted-right subset): COCONUT h_prompt 0.585, z_cat 0.567; explicit mean_cot 0.937.

## Decoy controls (target 0.5; seeds 0 / 1 / 2)
Explicit h_prompt probe 0.501 / 0.470 / 0.519 and COCONUT mean-latent probe 0.472 / 0.552 / 0.480: **fixed** (v2 had 0.74 / 0.73 for the explicit probe). COCONUT anchor residual 0.504 / **0.294 / 0.295**: **not fixed**; moving the decoy before the hint did not remove its decoy-dependent offset (decoy-stratified anchor AUROC: seed 1 0.624 with decoy vs 0.490 without; seed 2 0.522 vs 0.555).

## Reading
The explicit chain of thought is readable (text monitor 0.89-0.92). Latent COCONUT states carry a modest hint-uptake signal before the answer (0.57-0.60 AUROC, replicated on 3 seeds, difficulty baselines at chance), but for strict harmful reliance no internal monitor approaches reading the output (0.707). The clean-anchor detector is not a reliable signal (decoy-sensitive, 0.49 on strict harmful). Explicit probes do not beat their difficulty baselines.
