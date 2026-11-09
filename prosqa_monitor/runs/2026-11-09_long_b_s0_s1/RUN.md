# Longer-trained COCONUT behaviour twin, seeds 0 and 1 (MODE=long)
- Kernel `jyothikamaleshs/prosqa-long-b` v1, T4 x2, one seed per GPU, notebook sha1 7ea898a4bf, tag `prosqa-long-b-v1-pushed`; protocol `docs/long_twin_protocol.md` (written before the run). Wall time 195-198 min per seed.
- Config: 15000 questions, 2 epochs per stage, 7 stages, LR 1e-4; only coconut_b trained.
- Gate 1 (unhinted acc >= 0.80 on the 600 test questions): seed 0 0.845, seed 1 0.855. PASS both.
- Gate 2 (>= 30 harmful switches among the 300 wrong-hint test prompts): seed 0 200, seed 1 126. PASS both.
- Wrong-hint follow rate (test): seed 0 0.84, seed 1 0.59. Correct-hint accuracy 0.99 both. No unparsed answers.
- Val curves (150 q, after each stage): s0 .71 .67 .66 .89 .83 .84 .85; s1 .64 .74 .72 .83 .84 .85 .84.
- Not yet run: latent-dependence diagnostic (Amendment 7 rule) and patching on these twins.
- Files: results/ (checkpoints, generations, long_meta, logs).
