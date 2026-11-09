# Latent-dependence diagnostic + frozen patching on the longer-trained twins, seeds 0 and 1
- Kernel `jyothikamaleshs/prosqa-long-patching` v1 (notebook sha1 e0fd15605b, tag `prosqa-long-patching-v1-pushed`), T4 x2, one seed per GPU; inputs = output of `prosqa-long-b`. Rebuild agreement 1.0 / 0.99 (hinted), 1.0 (unhinted). Rows 300 / 299.
- **Amendment 7 rule, unhinted-correct rows (harmful + resistant), pre-registered in `docs/long_twin_protocol.md`:**
  - Seed 0: zero all six latents dM -0.052 (harmful, n 200), -0.006 (resistant, n 49); flips 0.4%. Other question's latents: +0.000 / -0.001, no flips.
  - Seed 1: zero all six dM -0.064 (n 126), -0.074 (n 122); flips 0.4%. Other question's latents: -0.004 / -0.002, no flips.
  - Both under the thresholds (mean change < 0.1, flips < 2%): **latents still inert** in both longer-trained twins. Single-slot zeros move M by at most ~0.006.
- Patching ran as chained (`patching_s*`); per the protocol it is NOT interpreted as a test of restoration, because the latents are not used for the plain task. Summaries saved only (rows files not copied).
