# Forced latent use (information bottleneck), seeds 0 and 1 (kernel `jyothikamaleshs/prosqa-bneck-b` v2, notebook sha1 4b46231d30, tag `prosqa-bneck-b-v2-pushed`; v1 failed at the unit-test cell, no GPU work)
Pre-registration: `docs/bottleneck_protocol.md`. Training: stage-6 fine-tune of the longer-trained twins, 3 epochs, bottleneck on (answer cannot attend to prompt or <bot>). About 50 min training, then diagnostic and patching in the same session.
- Before fine-tuning, bottleneck on: val accuracy 0.00 (both seeds): the longer-trained twins answer from the prompt.
- **Gate 1** (unhinted acc >= 0.80, 600 test q): s0 0.852, s1 0.843 PASS. **Gate 2** (>= 30 harmful switches): s0 116, s1 170 PASS. **Gate 3** (latents necessary, Amendment 7): zeroing all six latents moves M by -5.9 / -8.3 (s0 harmful / resistant) and -3.6 / -5.3 (s1), flips 7-59% of rows: PASS. Rebuild agreement 1.0.
- Wrong-hint follow rate (test): s0 0.54, s1 ~0.7; correct-hint accuracy 0.94 / 0.97.
- **Patching, harmful switches** (dM = change in M toward the true option; P(true) = share choosing the true option after the intervention; baseline M hinted -2.3 / -2.8, unhinted +5.4 / +5.5):
  - Full six-slot restoration from the natural unhinted donor: +6.4 (P(true) 0.96) / +6.9 (0.94). Position-matched donor the same.
  - Disruption controls: zeroing all six +1.7 (0.34) / +2.1 (0.37); other-question latents +1.8 (0.44) / +2.3 (0.39).
  - Single-step restoration is partial and differs by seed: s0 slot 2 +3.1 (0.65), others 0.0-1.7; s1 slot 0 +4.5 (0.77), slot 3 +2.0 (0.39), others ~0.
  - Blocking the answer from the hint tokens is a no-op here by construction (0.00).
- Not run: monitors on these twins; seed 2.
