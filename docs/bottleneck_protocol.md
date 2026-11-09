# Forced latent use (information bottleneck): pre-registration (written 2026-11-09, before any bottleneck training)

**Why.** The frozen twins and the longer-trained twins (seeds 0, 1) do not use their latents even without a hint (`docs/long_twin_protocol.md`; zeroing all six latents moves the margin by about 0.05, flips 0.4%). More training did not change it. We need a setting in which the latents are necessarily causal, as a **positive control** for the patching method and as the setting where the monitor-validity question is meaningful.

**What it is (and is not).** In the final curriculum stage (all six latents, no text steps) the answer segment (`<eot>` + answer sentence) cannot attend to the prompt or `<bot>`; it attends only to the six latent positions. The latents themselves still attend to the whole prompt, including the hint line. So everything the answer knows about the question and the hint passes through the latents. This is **forced by construction**: it shows the methods work when the latents matter; it is not evidence that natural latent reasoning is used.

**Training.** Start from the saved longer-trained twins (`s0_coconut_b.pt`, `s1_coconut_b.pt`; same seeds, same training questions, hints planted as before), fine-tune only stage 6 with the bottleneck on: 3 epochs, 15000 questions, LR 1e-4 (fresh AdamW), one seed per GPU. Generation uses the same mask. Code: `BOTTLENECK`/`bneck_att` in `prosqa_monitor.py`, `build_long_notebook.py --bottleneck`. The unmodified long twin's accuracy with the bottleneck on is logged BEFORE fine-tuning (expected to collapse: it answers from the prompt).

**Gates, fixed now (600 held-out monitor-test questions, 300 wrong-hint).**
1. Unhinted accuracy under the bottleneck >= 0.80. If missed, the latents cannot carry the task at this budget: report, do not interpret the rest.
2. At least 30 harmful switches among the 300 wrong-hint prompts. If fewer, no behaviour to study in that twin.
3. Latent use (Amendment 7 rule on unhinted-correct rows): mean change in M when all six latents are zeroed above 0.5 or more than 5% flips. This should hold by construction; if it does not, there is a bug and nothing is interpreted.

**Then, for twins that pass (no further tuning).** The frozen causal patching protocol (Amendments 1-5) with `--bottleneck`, per seed, no pooled estimate; interpretation table of Amendment 1. Note that the conditions that block the answer's attention to the hint tokens are no-ops here by construction (the answer cannot attend to them); the informative conditions are the latent restorations: single-step, persistent, and full six-slot restoration from the natural unhinted donor, and the position-matched donor.

**What each outcome supports.**
- Full restoration removes the hint effect (harmful switches go back to the true option): the patching method detects a causal latent route when one exists; the earlier null is then a property of the unforced twins.
- Restoration still about 0 although latents are necessary: the method or the donor construction misses the route; we cannot conclude anything about monitors until that is resolved.
- Monitors (retrained on these twins) are a later step and are not part of this run.

**Budget.** Estimated 1-1.5 h wall for both seeds in parallel on T4 x2 (3 epochs x 15000 samples at about 17 samples/s, plus validation and generation); then the diagnostic and patching (about 1 h). Two Kaggle pushes, each needing the user's yes.
