# Longer-trained COCONUT twin: pre-registration (written 2026-11-09, before any training run)

**Why.** The three frozen twins (3000 questions, 1 epoch per stage, about 21k training samples) do not use their latents even on unhinted prompts (Amendment 7, seed 0: zeroing all six latent inputs moves the margin by 0.003 and flips 1.0% of rows). The reference COCONUT/ProsQA recipe sees roughly 30x more samples. The registered patching null therefore cannot yet separate "the hint bypasses working latents" from "the latents were never learned".

**What is trained.** One COCONUT behaviour twin (`coconut_b`: hints planted, P_CORRECT 0.5, same data construction and curriculum as before) per seed, seeds 0 and 1, one per GPU, in `MODE=long`: N_TRAIN 15000, 2 epochs per stage, 7 stages, LR 1e-4, everything else as in `seeds`. No explicit twin, no clean twin, no monitors. Code: `prosqa_monitor/build_long_notebook.py` (reuses the setup and training cells of `prosqa_monitor.py` unchanged). Frozen twins and the frozen causal protocol stay as they are; these are new twins, reported next to them, never pooled with them.

**Outputs per seed.** `s{seed}_coconut_b.pt`, `s{seed}_generations.jsonl` (hinted and unhinted generations on the same 300 + 600 held-out questions, same format as before), `s{seed}_long_meta.json` (learning curve, accuracy, follow rate). Evaluation questions are fixed by SPLIT_SEED as before.

**Gates, fixed now (checked on the 600 held-out monitor-test questions).**
1. Unhinted accuracy >= 0.80 (the old twins were far lower; chance is 0.5). If missed, the twin did not learn the task and nothing below is interpreted.
2. Wrong-hint follow rate and harmful switches: at least 30 harmful switches (unhinted true, hinted wrong hint) among the 300 wrong-hint test prompts. If fewer, there is no behaviour to monitor in that twin.

**Then, only for twins that pass both gates (no further training, no tuning).**
A. The latent-dependence diagnostic of Amendment 7 on unhinted-correct rows. Decision rule unchanged: mean change in M under 0.1 and fewer than 2% flips for both full replacements = latents still inert; mean change above 0.5 or more than 5% flips = latents are used for the plain task; in between = inconclusive.
B. If the latents are used: the frozen causal patching protocol (Amendments 1-5, registered primary = single-step restoration from the natural unhinted donor, interpretation table of Amendment 1), plus the Amendment 6 hint-vs-`<bot>` split. Per-seed estimates, no pooled estimate.

**Claims this can and cannot support.**
- Latents inert after 7x the compute-matched budget: the null on the hint is a statement about this training recipe at this scale, not about COCONUT in general; no claim about latent monitors.
- Latents used and restoration still about 0: the null stands with working latents (strongest version).
- Latents used and restoration works: the first positive causal evidence; reported with the same controls as the null runs and per-seed.
- Nothing here changes the earlier results; they remain valid statements about the frozen twins.

**Budget.** Estimated 3.5-4.5 h wall for both seeds in parallel on T4 x2 (7 stages x 2 epochs x 15000 samples at about 17 samples/s, plus validation). Hard stop: the Kaggle 12 h session limit. Weekly GPU quota remaining: not checked by me.
**Not done in this run:** explicit twin, clean twins, monitors, scale-up beyond GPT-2 small.
