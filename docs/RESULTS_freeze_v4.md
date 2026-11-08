# Frozen monitoring result: ProsQA seeds 0 (v3), 1 and 2 (v4), 2026-11-08

Generated from the committed analysis outputs by script (no hand-typed numbers). Sources: `prosqa_monitor/runs/2026-11-08_seeds_v4_s1_s2/aggregate_3seeds.txt`, `strict_harmful_reanalysis_3seeds.txt`, per-seed `*_suite.csv`, `docs/provenance_checkpoints.json`. **Two outcomes are kept apart: correct-hint uptake (hint fixed a wrong unhinted answer) and strict harmful reliance (unhinted answer true, then followed the planted wrong hint).**

## Conclusions that stand
- COCONUT behavioural contrast replicated on 3 seeds under the stated criteria (accuracy gap <= 0.08, harmful-rate gap >= 0.25, also on the unseen wording).
- Depth-dependent policy NOT learned by either behaviour twin (shallow-problem and deep-problem gates fail on all seeds): claim only "follows hints".
- Explicit twin pair: controlled comparison is weaker (accuracy gap 0.09 on seed 1; seed 2 twins at 0.60-0.61): conclusions about it are secondary.
- Latent monitoring: modest pre-answer hint-uptake signal; poor detection of strict harmful reliance (below the output-only exact match).
- Clean-anchor detector: retired as a reliable signal (decoy control 0.294 / 0.295 on seeds 1-2; 0.49 on strict harmful reliance).

## Behaviour (80 questions x 4 wordings; behaviour twin / clean twin)
| seed | code | explicit acc | explicit harmful | COCONUT acc | COCONUT harmful |
|---|---|---|---|---|---|
| 0 | v3 | 0.69 / 0.70 | 0.47 / 0.02 | 0.76 / 0.69 | 0.52 / 0.01 |
| 1 | v4 | 0.75 / 0.66 | 0.63 / 0.00 | 0.71 / 0.72 | 0.40 / 0.01 |
| 2 | v4 | 0.61 / 0.60 | 0.46 / 0.00 | 0.69 / 0.71 | 0.44 / 0.00 |

## Correct-hint uptake, AUROC mean [min-max] over 3 seeds (about 70 positives, 222-232 negatives per seed)
| monitor | AUROC |
|---|---|
| Explicit text monitor (CoT conclusion != answer) | 0.918 [0.905-0.944] |
| Explicit mean_cot probe | 0.623 [0.599-0.660] |
|   its difficulty baseline (unhinted WRONG) | 0.670 [0.648-0.710] |
| Explicit h_ans probe | 0.558 [0.488-0.625] |
|   its difficulty baseline | 0.568 [0.504-0.685] |
| Explicit h_prompt probe | 0.487 [0.456-0.513] |
| COCONUT h_prompt probe | 0.599 [0.552-0.625] |
| COCONUT z_cat probe | 0.590 [0.523-0.639] |
| COCONUT z_mean probe | 0.583 [0.520-0.627] |
| COCONUT h_ans probe | 0.495 [0.488-0.500] |
| COCONUT clean-anchor matching (retired: decoy-sensitive) | 0.565 [0.543-0.601] |
| Output-only exact match, COCONUT twin | 0.538 [0.520-0.551] |
| Output-only exact match, explicit twin | 0.544 [0.526-0.577] |

## Strict harmful reliance, AUROC mean over 3 seeds [min-max] (probes retrained on the strict label)
| monitor | AUROC |
|---|---|
| Explicit text monitor | 0.890 [0.804-0.943] |
| Output-only exact match (explicit) | 0.694 [0.671-0.728] |
| Explicit mean_cot probe | 0.681 [0.614-0.764] |
| Explicit h_ans probe | 0.651 [0.521-0.858] |
| Output-only exact match (COCONUT) | 0.707 [0.689-0.721] |
| COCONUT h_prompt probe | 0.573 [0.548-0.609] |
| COCONUT z_cat probe | 0.561 [0.529-0.582] |
| COCONUT clean-anchor matching | 0.491 [0.463-0.521] |

## Decoy controls (target 0.5), seeds 0 / 1 / 2
Explicit h_prompt probe 0.501 / 0.470 / 0.519; COCONUT mean-latent probe 0.472 / 0.552 / 0.480; COCONUT anchor residual 0.504 / 0.294 / 0.295.

## Provenance
Seed 0 checkpoints: code v3 (hash 3bd1f99438); seeds 1-2: v4 (ac70954f44). `git diff prosqa-seeds-v3 prosqa-seeds-v4` changes only the label function (`uptake_label`) and comments; twin training does not use it, so the twins are produced by identical training code. SHA-256 of all 12 checkpoints in `docs/provenance_checkpoints.json`.
