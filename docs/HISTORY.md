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

**Mini run** (`runs/2026-11-07_mini_v1`, 18.1 min, 300 examples). Confirms on the T4 that the Phase-1 pipeline runs end to end and behaves as intended: sizes, timings (faster than planned), memory (8.4 of 14.6 GiB), learning-curve lines, behaviour suite, per-seed files. At this size the suite already separates the twins: harmful rate 0.42 / 0.45 for the behaviour twins vs 0.00 / 0.04 for the clean twins, also on the wording never trained on; accuracy and the depth condition are not learned yet, as expected.

**Correction to earlier claims.** I said the clean COCONUT twin "leaks" because it followed wrong hints at 0.36 (v3) and 0.47 (mini). That compared the follow rate with a fixed 0.3, ignoring that a model which never looks at the hint still picks the wrong option at its error rate: v3 coconut_c error was 0.29 (excess +0.07, about 1.7 standard errors), mini coconut_c error 0.47 (excess 0). The leak was overstated. The counterfactual harmful rate is the proper statistic, and it is about 0 for the clean twins. The gate now tests follow rate minus error rate <= 0.1.

**Novelty check, abstract level (2026-11-07).** Read the abstracts of the three closest papers (full texts not yet read):
- *Does Out-of-Sight Equal Out-of-Mind in CoT Monitorability?* (Ferreira, Aziz, Titov, 2608.04928): hint reliance as the monitorability target, explicit CoT vs weakly/strongly supervised latent CoT, monitors with different internal access; finding: monitorability depends more on task properties and access level than on reasoning mode. **Closest to our setup**; overlaps with our hint-reliance framing and explicit-vs-latent comparison.
- *Ulterior Motives: Detecting Misaligned Reasoning in Continuous Thought Models* (Ramjee, 2604.23460, ICLR 2026 LIT workshop): backdoor-triggered misaligned latent reasoning, supervised linear probes, probes transfer from armed-with-output to armed-benign, signal in early latents. Different behaviour (backdoor) and supervised.
- *Detecting Hidden Behaviors in LLMs via Activation-matched Finetuning* (Haselhorst, Flek, Mai, 2609.00351): unsupervised anchor-model residual for hidden behaviours in standard LLMs; not applied to latent-reasoning models in the abstract.
Conclusion: our differentiators are (a) the validity framing with confound baselines, two outcomes and exit-criteria twins, (b) adapting label-free anchor detection and cross-format transfer to COCONUT-style models. The hint-reliance counterfactual setup and explicit-vs-latent comparison are not new. Novelty claim is thinner than earlier stated; full texts and other related work (cocoracle, SafetyNet, backdoor-evasion papers, the probing-controls literature) still to be checked.

**Full-paper comparison with Ferreira, Aziz & Titov (2608.04928)** (read from the arXiv HTML; the last ~16k characters, part of Appendix A.2, were not read).
- *Their setup:* Llama-3.2-1B-Instruct for everything; No-CoT, Explicit-CoT, and latent CoT with weak (CODI) and strong (SIM-CoT) supervision, six latents; COCONUT is **not** evaluated. GSM8k-Aug (385k training samples) tested on GSM8k and OOD SVAMP; ECQA tested on ECQA and OOD SIQA. Math latent models are the authors' released checkpoints (not equal budgets). "Intervention-aware fine-tuning" (3 seeds, checkpoints across hint-reliance rates) induces the behaviour.
- *Label:* counterfactual (Y = 1 iff hinted prediction differs from the no-hint one and equals the hint), with a **correct** hint so that a wrong hint would not make Y collapse to "prediction equals hint".
- *Monitors:* output-only Exact Match, prompted LLM (explicit only), fine-tuned LLM on native text, logit lens, SIM-CoT decoder, activation probes; metric AUROC. Finding: monitorability depends more on task properties and access level than on reasoning mode; probes are the most robust (e.g. SVAMP OOD 0.85-0.91), text monitors degrade OOD on QA.
- *Not in their paper:* label-free/anomaly detection, clean-anchor/activation matching, cross-format or cross-model transfer, COCONUT, answer-correctness / difficulty / decoy confound controls, clean (non-reliant) twins.
- *Where we overlap:* counterfactual hint-reliance labels, explicit vs latent monitoring, activation probes. This part is not new.
- *Where we differ:* matched twins (same init, data, budget) with clean twins; COCONUT; label-free clean-anchor detection and cross-format transfer; the confound baselines and the two outcomes. *Where we are weaker:* scale (GPT-2 small, 2k examples, one seed vs their 1B model, 385k samples, 3 seeds, OOD tests, more monitors).

**Correction: an output-only baseline beats our explicit probes.** Their Exact Match baseline was missing from our pipeline. Computed on the saved controlled-v3 data: on wrong-hint prompts the explicit model's exact-match score gives AUROC 0.862 (COCONUT 0.570), above the explicit activation probes (0.80), and only the CoT-conclusion monitor (0.93) beats it. Reason: our harmful label is "followed the wrong hint and the unhinted answer was not the hint", which is almost "followed the hint" when the model is accurate. So the earlier statement that the explicit probes beat the confound baselines was incomplete. The COCONUT probes (0.49-0.58) are not better than the output baseline (0.57). Exact-match baseline rows added to the pipeline.

**Correct-hint label re-analysis of controlled v3** (`prosqa_monitor/analysis/reanalysis_correct_hint.py`; saved data, no GPU). On correct-hint prompts (n about 150, about 40 positives = the hint turned a wrong answer into a right one) the output-only exact-match baseline is weak (explicit 0.57, COCONUT 0.51), so this label does not collapse.
- Explicit CoT-conclusion != answer text monitor: 0.90 [0.84, 0.96].
- Explicit probes: mean-CoT 0.74-0.77, answer token 0.66-0.73. A probe trained to predict "the unhinted answer is WRONG" (a difficulty/uncertainty detector, no hint information) scores 0.73 and 0.65: the explicit probes are not clearly better than a difficulty detector.
- COCONUT probes 0.45-0.57, about equal to their difficulty baselines (0.47-0.57), i.e. chance.
- Label-free: COCONUT clean-anchor matching 0.61 [0.52, 0.71] (the only above-chance latent score; one of about 16 comparisons, so not robust, and the anchor residual was decoy-sensitive earlier); explicit anchor 0.53; Mahalanobis 0.50 / 0.43; cross-format 0.46-0.52.
- Reading: with the primary label of the related paper, only the readable-trace text monitor clearly works; activation probes may largely read difficulty. This is the validity hypothesis showing up weakly, single seed, n about 150.

## Phase 5: seeds experiment (`seeds` mode, 3 seeds at the v3 size), started 2026-11-07
**Design.** Correct-hint counterfactual as the primary outcome ([correct-hint]), wrong-hint [harmful] secondary and always next to the output-only exact-match baseline; difficulty baselines (probes trained on "unhinted answer wrong/correct") on every main probe feature; 600 monitor-test questions; fail-fast guards (explicit twins checked at 3/4 of training on the best of the last two held-out accuracy checks, critical data gates stop at once, other gate failures recorded not fatal); `run_seeds.sh` runs one seed per Kaggle session and stops the chain on ERROR/CANCEL. 23 logic tests incl. guard tests; tiny end-to-end run passed. Tag `prosqa-seeds-v1`, code hash 672cb25746.

**Seed 0** (`runs/2026-11-07_seeds_s0`, 84.2 min): valid, complete. Phase-1 exit criteria met for this seed (twin accuracy comparable, harmful-rate gap 0.47 explicit / 0.41 COCONUT, also on the unseen wording); the depth condition was still not learned. Monitors on the primary outcome: explicit text monitor 0.82; explicit probes 0.61-0.62, **equal to their difficulty baselines (0.61-0.63)**; every COCONUT probe, label-free detector and the cross-format transfer at chance (0.44-0.54). The earlier COCONUT anchor-matching 0.61 (v3) did not reproduce (0.47), which fits it having been noise.

**Seed 1** (`runs/2026-11-07_seeds_s1_aborted`): aborted by the guard at 18.7 min (0.54 vs threshold 0.55, a marginal trip on a 100-question measurement), chain stopped, seed 2 never pushed. **Two findings:** (1) the guard threshold is too tight for a config whose explicit twin sits at about 0.6; (2) the smoke pass writes result files with the real names, so an aborted run leaves smoke output that looks valid (status OK, 128 rows). Mitigation without changing the experiment code: the aggregator now rejects runs whose generation count does not match the configuration. The code fix (separate smoke prefix) waits for the next code version because changing code changes the hash and breaks comparability with seed 0.

### Seeds v2 result (2026-11-08): all three seeds ran
Seed 0 (113.6 min), seeds 1+2 (one combined notebook, 114.5 + 104.9 min), 3000 training questions. `runs/2026-11-08_seeds_v2`. **Exit criteria not met on 3 seeds** (explicit accuracy gap up to 0.10; COCONUT harmful-rate gap 0.24 on seed 2); every seed ended `GATES_FAILED` (the depth condition is not learned; explicit_c/explicit_b accuracy dips). Monitors (correct-hint, mean over 3 seeds): explicit text monitor 0.904; explicit probes 0.71-0.72, **not clearly above their difficulty baselines (0.67-0.72)**; output-only exact match 0.55-0.58; all COCONUT probes, label-free, transfer 0.50-0.57 (clean-anchor 0.565 is the highest). Seeds 1 and 2 saved their twins (`s{1,2}_*.pt`, 2.6 GB, git-ignored). Process notes: the OAuth login expired during the 2-hour wait (re-login needed `kaggle auth login --force`); a Kaggle kernel cannot be cancelled from the CLI, so the redundant seed-1 kernel was deleted; extra seeds can now run in the same notebook (`build_notebook.py --extra-seeds`).

### Decoy diagnosis on the saved seeds-v2 data (2026-11-08)
Triggered by an external review: the decoy control for the COCONUT anchor residual read 0.005 (seed 1; 0.515 / 0.005 / 0.217 over seeds) and the explicit prompt-token probe read 0.74 / 0.73 on seeds 1 and 2 (target 0.5). Scripts `analysis/decoy_diagnosis.py`, `analysis/decoy_stratified_scores.py`; outputs in `runs/2026-11-08_seeds_v2/decoy_*.txt`. Findings:
1. The decoy is perfectly readable from every feature (AUROC 0.92-1.00), by design (it is a prompt line), and the decoy is independent of the hint type (corr 0.00).
2. **Supervised probes are not driven by it:** [correct-hint] AUROC is unchanged when the decoy mean-difference direction is projected out (explicit h_ans 0.722 -> 0.729, mean_cot 0.709 -> 0.709, COCONUT 0.51-0.54 unchanged) and holds inside each decoy stratum.
3. The explicit probe's decoy control (0.73) matches a chance decoy-label correlation in the monitor-train set (train P(y=1): decoy 0.50 vs 0.39 on seed 1, 0.51 vs 0.39 on seed 2, 0.38 vs 0.41 on seed 0 where the control read 0.46): the probe picks up the imbalance. The hypothesis fits all three seeds; not tested directly.
4. **The COCONUT clean-anchor residual has a decoy-dependent offset** (mean score with decoy 0.019 vs 0.053 without on seed 1; scale also differs across seeds, 0.004-0.05). Inside the no-decoy stratum its [correct-hint] AUROC is 0.607 / 0.605 / 0.586; inside the decoy stratum 0.579 / 0.496 / 0.460 (about 35 positives per stratum, CI about +-0.1). The pooled 0.565 therefore mixes a modest signal with a nuisance offset. Likely cause (a hypothesis): the decoy line sits after the hint, directly before the last-token position, so it masks the hint's effect there.
Next code version should: put the decoy before the hint, assign decoy and label jointly balanced in the monitor sets, and report decoy-stratified AUROCs as the headline.

### Seeds v3 code (2026-11-08, built, tested, NOT yet run)
Changes after the decoy diagnosis and the external review: (1) decoy line moved before the hint; (2) probes trained with decoy x label cell weights (a chance decoy-label imbalance cannot be learned); (3) hint-type-specific probes trained and scored on harmful-only (wrong hints) and beneficial-only (correct hints) rows, next to the mixed-label baseline; 27 logic tests (new: prompt order, weight shortcut removal, the new probe block).
Post-hoc on the saved seeds-v2 scores (`analysis/matched_evaluation.py`, `runs/2026-11-08_seeds_v2/matched_evaluation.txt`): inside the difficulty-matched subset the output-only exact match is 1.000 by construction (the label there is 'the output followed the hint'), so matched AUROC only discriminates monitors that cannot see the answer. Pre-answer COCONUT monitors on the matched [correct-hint] subset: z_cat 0.664 [0.580-0.802], clean-anchor 0.605 [0.535-0.697], cross-format anchor 0.593, h_prompt 0.590, versus 0.5 (but only about 24 negatives per seed: wide intervals). Explicit h_prompt 0.533. Explicit h_ans / mean_cot reach 0.96 / 0.92 matched, but they read the answer.

## Open items (live)
1. Fix the learnability guard (explicit twins only, three quarters of training, best of the last two evaluations, threshold about 0.55) and make `full` mode save everything and fail only at the end.
2. Decide the full-run training size: accuracy rose with data (about 0.5 to 0.7 at 2000 examples); 6000 examples x 2 epochs/stage is plausible but unproven. About 6 h per seed on a T4; 3 seeds is about 19 h of the 30 h weekly quota.
3. The COCONUT behaviour twin must learn the depth condition and the clean twin must stop leaking, otherwise the latent result stays uninformative.
4. Not yet done: oracle and untrained-model probes, LLM-judge baseline, causal ablation of the probe direction, SAE option, 1B model, additional behaviours/tasks.
5. Novelty of cross-format transfer and clean-anchor detection on latent models is unconfirmed (searches were snippet-level).
6. Needs the user's yes before any Kaggle push.
