# GSM8K T4 memory/throughput profile (kernel `latent-monitorability-profile`)

- **Experiment:** gsm8k_coconut. Notebook `notebooks/profile_t4.ipynb` (generator `src/build_profile.py`).
- **Findings:** COCONUT stage 3 needs 13.9 GiB at micro-batch 8 (out of memory on the 14.6 GiB T4) versus 7.8 GiB at micro-batch 4. Gradient checkpointing is slower and not worth it. Estimated full run about 75-90 minutes.
- **Decision it drove:** BS_COCO=4, ACCUM=4 for GSM8K.
- **Files:** `results/profile.csv`, `profile.json`, log.
