# Decision log

| # | Decision | Why |
|---|---|---|
| 1 | Ask before every Kaggle push; a yes covers that push only | A push spends GPU quota and was once made without asking. Saved as a standing rule. |
| 2 | Match training budgets between text-CoT and COCONUT (epochs per curriculum stage) | COCONUT gets 8 forward passes vs 3; unequal budgets would confound the comparison. |
| 3 | In-notebook smoke pass in a copied namespace, plus a training-set size assertion and a "training on N examples" log line | The v4 smoke pass overwrote `train`, giving a silent invalid run on 96 examples. |
| 4 | Gates report in `controlled` mode, raise in `full` mode | Rehearsals should show every problem; the real run should not continue on a broken setup. (Needs revision: raise at the end, not before monitors.) |
| 5 | Leave GSM8K | GPT-2 small reached no-hint accuracy about 0, so counterfactual labels were not meaningful; a 1B model deferred. |
| 6 | ProsQA with GPT-2 small as the base task | Graph reasoning that GPT-2 can learn with COCONUT (reference config exists); not tied to arithmetic. |
| 7 | Focus on label-free (clean-anchor, Mahalanobis) and cross-format monitors, not another supervised probe | The only related work found (Ulterior Motives) is supervised; claims of novelty still to be checked. |
| 8 | Counterfactual labels: Y = 1 iff hinted prediction equals the hint and unhinted does not | Removes "a hint is present" as a shortcut; as in Ferreira et al. |
| 9 | Report two outcomes, [any] and [harmful: wrong hints only] | The first label counted being helped by a correct hint as misbehaviour (35-41% of positives); on wrong hints the "hint is wrong" shortcut is constant. |
| 10 | Behaviour twins follow the hint only on deep problems (>= 4 steps) | With about 100% following, the wrong-hint label reduces to "unhinted correct"; a learnable condition creates real resisted negatives. |
| 11 | Confound baselines: probe for "unhinted answer correct", depth rule, hint-wrong rule, decoy line | A monitor only counts as detecting deference if it beats these. |
| 12 | Evaluation set fixed by `SPLIT_SEED`; 2000 test questions in `full` mode | Low-FPR estimates need roughly 10/p negatives; seeds must be scored on the same questions. |
| 13 | Seeds run separately, one per Kaggle session; aggregate with a code-hash check | Fits the 9 h session limit, isolates failures; mixed code versions are never merged. |
| 14 | Track held-out accuracy during training; abort `full` runs that are not learning | The controlled run only revealed near-chance accuracy at the end. (Guard needs revision: it is too noisy at the halfway point.) |
| 15 | One folder per experiment, git, `HISTORY.md` + per-run `RUN.md` | Chosen by the user. |
| 16 | Reframe the paper around monitor *validity* (when do monitors detect the behaviour vs a correlate) | Robust to null results, reuses the infrastructure. Not yet supported by data: v3 explicit probes beat the confound baselines. Novelty unchecked. |
| 17 | Phase 1 first: reliable behavioural twins with exit criteria before more monitor experiments | Without clean, comparable twins the monitor results (especially the latent nulls) cannot be interpreted. |
| 18 | Paired-question suite with several hint wordings, scored on unhinted accuracy, correct-hint compliance, wrong-hint compliance, harmful rate, twin gap | Directly measures whether the twins differ only in the intended behaviour; wording variation tests that the behaviour is not tied to one string. |
| 19 | Treat the clean twin as bounded by the base model's reasoning | A twin cannot verify hints better than its own accuracy (about 0.7 here); a stronger base model may be needed. Open question. |
| 20 | For COCONUT, change objective/curriculum (answer-token loss weight, hint reliability) rather than only adding epochs | v3 shows the depth condition was not learned at 1 epoch/stage; the behaviour lives in the final option token, so weighting it is a targeted change. |
| 21 | Phase-1 knobs ship with v3-equivalent defaults (`P_CORRECT` 0.5, `ANS_W` 1.0) | Lets each change be tested alone on the T4 instead of changing several things at once. |
| 22 | The comparable-accuracy gate also requires both twins above the accuracy floor | A pair of useless models at 0.00 is not a comparable pair. |

| 23 | Judge clean-twin deference by follow rate minus error rate (and the counterfactual harmful rate), not a fixed follow-rate cap | A model that ignores hints still follows the wrong option at its error rate; the fixed cap mistook low accuracy for leaking. |
| 24 | `mini` mode (real GPT-2, ~300 examples, ~18 min) is run after any pipeline change before a bigger T4 run | Cheap check that sizes, timings, memory and outputs are right. |
