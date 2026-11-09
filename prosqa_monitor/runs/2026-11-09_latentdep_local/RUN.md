# Latent-dependence diagnostic, seed 0 (local CPU, Mac M1)

- **Code:** `run_patching.py --latentdep` (Amendment 7, exploratory, decision rule written before the run); commit 8a38cc7 contains the mode. Twin: `s0_coconut_b.pt` (v3, SHA-256 in `docs/provenance_checkpoints.json`), 300 rows (116 harmful / 78 resistant / 106 wrong-either), UNHINTED prompts. transformers 5.22.2, torch 2.9.1, CPU. Run id latentdep_s0. Took **96.5 min** (estimated 25: the three baseline passes plus the eight conditions on the CPU are slower than I assumed).
- **Guards:** identity-patch difference 0.0; rebuild agreement 1.000 (unhinted).
- Only seed 0 was run; seeds 1 and 2 not checked.

## Result (unhinted-correct rows = harmful + resistant, n = 194)
| condition | mean change in M | rows flipping to the wrong option |
|---|---|---|
| all six latent inputs zeroed | +0.003 | 1.0% |
| all six replaced by another question's unhinted states | +0.000 | 0.0% |
Single slots zeroed (t0-t5): |mean change| <= 0.006 in every group. Wrong-either rows (unhinted M -1.78) stay wrong (change -0.014).
Decision rule (Amendment 7): mean change < 0.1 and < 2% flips for both full replacements -> **the latents are inert even without a hint; the twin answers from the prompt.**
