# ProsQA T4 profile (kernel `prosqa-monitorability-profile`, version 2)

- **Notebook:** `notebooks/prosqa_profile.ipynb` (generator `build_profile.py`). Version 1 was cancelled by Kaggle with no log (cause unconfirmed: two GPU kernels pushed together, or queue/laptop-sleep confusion).
- **Findings (MAX_LEN 720):**
  - COCONUT micro-batch 8 peaks at 7.7 GiB; micro-batch 16 reaches 13.6 GiB at stage 6 (too close to the limit). Throughput plateaus at micro-batch 8 (about 17-22 samples/s).
  - Text training micro-batch 8 peaks at 9.4 GiB on the worst case; 16 runs out of memory; about 20 samples/s.
  - Generation 0.24 s/prompt explicit, 0.07 s/prompt COCONUT (untrained, so an upper bound).
  - CPU about 25%, RAM about 2.3 GB: not a bottleneck.
  - Estimated full run (6000 examples, 2 epochs/stage): about 5 hours, 4 twins.
- **Decision it drove:** BS=8 x accum 2 for both formats; time estimates in the notebook come from these numbers.
- **Files:** `results/profile.csv`, `profile.json`, log.
