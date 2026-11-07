# Project history

Chronological record of experiments run, what was learned, decisions made and improvements made. All dates 2026-11-07 unless stated (everything so far happened in one long session). Each run has a `RUN.md` in `<experiment>/runs/`; reasons for choices are in `DECISIONS.md`. Newest section last; "Open items" at the bottom is the live to-do list.

## Phase 1: GSM8K latent monitorability (`gsm8k_coconut/`)

**Idea.** Same-model, same-data comparison of a text-CoT model and a COCONUT (continuous-thought) model with a planted quiet misbehaviour (using a hint). Compare a text monitor on the CoT with a monitor on the latents. Run on a Kaggle T4.

**What was built.** A notebook with a hint planter (50% none / 25% correct / 25% wrong, a harmless decoy line), COCONUT forward pass, curriculum training, text-CoT training, counterfactual labels, text monitors (keyword, CoT/answer consistency, TF-IDF), latent probes, shuffled-label and decoy controls.

**Runs and bugs (in order)**
1. *Kaggle v1*: CUDA out of memory in COCONUT training (micro-batch 8, 7 graph passes, 14.6 GiB card). Fixes: micro-batch 4 x accum 4, skip the LM head in latent passes, loss only at target positions, train COCONUT first, memory preflight on a synthetic worst-case batch.
2. *T4 profile* (`gsm8k_coconut/runs/2026-11-07_t4-profile`): COCONUT stage 3 needs 13.9 GiB at micro-batch 8 vs 7.8 GiB at 4; gradient checkpointing slower. Full run estimated 75-90 min.
3. *Smoke pass added* (run the whole pipeline on 96 examples before the real run) and phase/ETA/GPU/CPU logging. Pushed v3 without asking first; rule saved: ask before every Kaggle push.
4. *v4: invalid run.* The smoke pass overwrote the real training set, so the "full" run trained on 96 examples (symptoms: 7.4-minute run, `24/24` batches, `96 train` in the log). Fix: run the smoke pass in a copied namespace, assert the training-set size afterwards, log "training on N examples" for every phase.
5. *Wrong-hint bug.* A "wrong" hint could equal the truth (abs()+1 fold-over), and an infinite loop on very negative answers. Fix: candidate-list `wrong_hint`; tests added.
6. *Unequal training budgets* (COCONUT 8 forward passes vs text 3): matched epochs.
7. *Kaggle v5 controlled* (`2026-11-07_kaggle-v5_controlled`): completed in 17.6 min. Gates showed GPT-2 small cannot do GSM8K (no-hint accuracy 0.00 / 0.01). COCONUT probe AUROC 1.000 judged an artifact, not trusted.

**Quality checks added**: 13 CPU logic tests (hint construction, wrong hints never equal truth, split disjointness, example structure, answer parsing, zero-latent equivalence, latent fixed point, padding invariance, causality, loss alignment, gradient flow), gates that report in `controlled` mode and raise in `full` mode.

**Conclusion of phase 1.** The infrastructure works; the task does not suit GPT-2 small. Switch task and sharpen the research question.

## Phase 2: choosing a publishable direction (no code)

Discussion of workshop vs main-track worth; searches of related work (Ferreira-Aziz-Titov 2608.04928, Ulterior Motives 2604.23460, Activation-Matched Finetuning 2609.00351, SafetyNet, the COCONUT paper 2412.06769). Early overclaim ("no COCONUT misbehaviour-probe papers exist") corrected after finding Ulterior Motives, which is supervised.

Agreed direction: not tied to arithmetic; **ProsQA** with GPT-2 small; two ideas on top of "probe the latents": (A) label-free monitoring with a clean-anchor/activation-matching detector and Mahalanobis, (B) cross-format transfer (a probe trained on the readable explicit-CoT twin applied to the COCONUT twin). A 1B model deferred. Novelty is not yet confirmed (searches were snippet-level). Mech-interp (SAEs) only with causal tests.

## Phase 3: ProsQA experiment (`prosqa_monitor/`)

**Stage 0 (local, no Kaggle).** Data inspected: 17,886 / 300 / 500 split, solutions of 3-6 steps, question+steps+answer up to 647 tokens (so `MAX_LEN` 720, not 256). Notebook written as `prosqa_monitor.py` with four twins per seed; hint planted on the binary-choice question; counterfactual labels from hinted vs unhinted generations; cached COCONUT forward pass; tiny CPU mode; 13 new logic tests. Tiny end-to-end run: 2.2 min (`runs/2026-11-07_tiny-local-wiring`).

**Kaggle push problems.** Two GPU kernels pushed together were both cancelled with no log (cause unconfirmed; possibly concurrency). Exporting `KAGGLE_USERNAME` made the CLI fail with "Authentication required": unset it. Re-pushed the profile alone: ran. A confusing 24-minute "queue" was partly the laptop sleeping while the status watcher polled; Kaggle runs server-side.

**ProsQA T4 profile v2** (`runs/2026-11-07_t4-profile_v2`): micro-batch 8 for both formats (peak 7.7 GiB COCONUT, 9.4 GiB text worst case), throughput 17-22 samples/s, generation 0.24 s/prompt explicit and 0.07 COCONUT, CPU/RAM not a bottleneck, full run about 5 h.

**Controlled v2** (`runs/2026-11-07_controlled_v2`, 61.7 min, 2000 examples, 1 epoch/stage): pipeline correct (2000/2000 examples, timings matched), but models near chance (unhinted accuracy 0.50-0.67), so monitor numbers uninterpretable. Four gates failed. Explicit CoT-conclusion monitor reached 0.91 AUROC; COCONUT probes about 0.68; cross-format transfer and label-free detectors at chance; a decoy-only check showed the COCONUT anchor residual separates decoy from no-decoy prompts (a confound).

**Review feedback and re-analysis** (`prosqa_monitor/analysis/reanalysis_two_outcomes.py`). A reviewer-style comment noted the label counts being helped by a *correct* hint as positive. Re-analysis on the saved data: 35% (explicit) and 41% (COCONUT) of positives were correct-hint cases. On wrong-hint prompts only, the "hint is wrong" shortcut drops to 0.50, the explicit text monitor stays at 0.89, COCONUT probes stayed at about 0.71. A second problem: with about 100% hint following, the wrong-hint label reduces to "the model would have been right unhinted", so a probe could be reading that.

**Redesign (decisions in DECISIONS.md).**
- Report every monitor for two outcomes: [any] and [harmful: wrong hints only].
- Behaviour twins follow the hint only on deep problems (>= 4 steps, about 60% of ProsQA), giving real "resisted" negatives.
- Confound baselines: probe trained on "unhinted answer correct", depth rule, hint-wrong rule.
- Evaluation size: `full` mode uses 2000 monitor-test and 500 monitor-train questions (1700 held out of ProsQA train), because low-FPR estimates need roughly 10/p negatives.
- Seeds: `SEED` parameter, fixed `SPLIT_SEED` for the evaluation questions, per-seed output files and `run_meta.json` with a code hash, `aggregate_seeds.py` refusing to mix hashes. One seed per Kaggle session.
- Held-out accuracy logged after each epoch / curriculum stage; fail-fast guard added for `full` mode.
- Logic tests extended to 16.

**Controlled v3** (`runs/2026-11-07_controlled_v3`, seed 0, code hash a2c203d216, 73.7 min). All four twins above chance (0.64-0.71 on 300 test questions) but modest. Gates 23/30. Findings: explicit probes (0.80) and the CoT-conclusion monitor (0.93) detect harmful deference above the confound baselines; COCONUT probes, cross-format transfer and label-free detectors at chance (0.42-0.58, CIs include 0.5). The COCONUT behaviour twin did not learn the depth condition (follows wrong hints at 0.95 on shallow problems) and the clean COCONUT twin leaks (0.36), so the latent test is weak at this scale. Two guard bugs found: the halfway abort would have killed coconut_c (0.39 at stage 2, final 0.73), and in `full` mode failing gates raise before monitors run.

**Repository reorganised** into one folder per experiment (this commit). Notebook rebuild made byte-reproducible (deterministic cell ids). Found and fixed a latent bug: the profile builder selected the config cell by the text `SEED = 0`, which no longer existed after adding the seed parameter (the profile notebook would have lost its config cell).

## Phase 4: direction refined toward a monitor-validity study (decided 2026-11-07, after controlled v3)

**Trigger.** Reading controlled v3 honestly: the explicit monitors work, every latent monitor is at chance, but the COCONUT behaviour twin did not learn the planted behaviour and the clean COCONUT twin leaks, so the latent test was uninformative. A reviewer-style proposal then framed a paper as *"When do internal AI monitors actually detect misbehaviour?"*: an empirical study of behavioural validity, shortcut learning and robustness in activation-based monitoring. Central hypothesis: an activation monitor can score high without detecting the intended behaviour, because it may track correlated signals (difficulty, answer correctness, model-specific representations).

**Assessment recorded (my view at the time).**
- It fits the data we have and is robust to null results; it reuses the infrastructure already built (two outcomes, confound baselines, decoy check, fixed evaluation split, seeds).
- It is **not yet supported**: in v3 the explicit probes beat both confound baselines (0.80 vs 0.54 and 0.68), i.e. they did detect deference. The COCONUT nulls come from unclean twins, not from shortcut learning, so they must not be presented as evidence for the hypothesis. The sharp claim needs at least one case where a monitor scores well and the baselines explain it; we have none yet.
- **Limit on one proposed design point.** A clean twin that "uses correct hints but resists wrong ones" can only separate them by reasoning, and at about 0.7 accuracy it cannot verify hints much better than chance. At this scale the achievable clean twin treats hints as weak evidence. This may push toward a stronger base model; unresolved (see open items).
- Novelty of the reframed paper is unchecked.

**Phase 1 (build reliable behavioural twins), the priority.** Paired questions per item (no hint, correct hint, wrong hint, varied hint wording), measuring unhinted accuracy, correct-hint compliance, wrong-hint compliance, harmful hint-following rate and the twin gap. Exit criteria before anything else: both twins learn their policy, unhinted accuracy comparable, harmful rates clearly different, separation reproduces across seeds. For COCONUT, change the curriculum / training objective rather than only adding epochs. Implemented locally first (see the entry below); no Kaggle push without the user's yes.

**Phase 1 implemented locally (no Kaggle push).** All knobs default to the v3 behaviour, so they change nothing until set.
- Four hint wordings (three used in training, one never seen) and `make_prompt(..., wording)`.
- `P_CORRECT` (default 0.5): fraction of hinted training prompts with a correct hint; above 0.5 hints become informative, which lets a clean twin learn to use correct hints.
- `ANS_W` (default 1.0): loss weight on the answer sentence after `###`, for the COCONUT/explicit objective (`weighted_ce`; equals the old loss at 1.0, tested against the HF loss).
- New phase `behaviour suite` (`build_suite`, `suite_metrics`): paired prompts (unhinted, correct hint, wrong hint per wording) on fresh questions, reporting unhinted accuracy, correct-hint accuracy, wrong-hint compliance (overall, deep vs shallow, seen vs unseen wording) and harmful rate; gates for the exit criteria (twin accuracy within 0.08 and both above the floor, harmful-rate gap >= 0.25, gap holds on the unseen wording, a correct hint does not hurt the clean twin). Saved as `s{seed}_suite.csv` / `_suite_items.jsonl`.
- `aggregate_seeds.py` reports the suite across seeds and states whether the exit criteria are MET (needs >= 3 seeds).
- 21 logic tests (8 new/extended); the tiny end-to-end run passes with the new cell. Test-writing mistake caught on the way: building a fresh `Random(seed)` per item made every item identical; tests now share one generator.
- Gate loophole closed: "comparable accuracy" used to pass at 0.00 vs 0.00.
- Not yet run on a GPU. Still open: the two guard fixes below.

## Open items (live)
1. Fix the learnability guard (explicit twins only, three quarters of training, best of the last two evaluations, threshold about 0.55) and make `full` mode save everything and fail only at the end.
2. Decide the full-run training size: accuracy rose with data (about 0.5 to 0.7 at 2000 examples); 6000 examples x 2 epochs/stage is plausible but unproven. About 6 h per seed on a T4; 3 seeds is about 19 h of the 30 h weekly quota.
3. The COCONUT behaviour twin must learn the depth condition and the clean twin must stop leaking, otherwise the latent result stays uninformative.
4. Not yet done: oracle and untrained-model probes, LLM-judge baseline, causal ablation of the probe direction, SAE option, 1B model, additional behaviours/tasks.
5. Novelty of cross-format transfer and clean-anchor detection on latent models is unconfirmed (searches were snippet-level).
6. Needs the user's yes before any Kaggle push.
