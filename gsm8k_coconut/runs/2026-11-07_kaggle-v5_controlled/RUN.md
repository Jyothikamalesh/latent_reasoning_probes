# GSM8K controlled run (Kaggle kernel `latent-monitorability-coconut`, version 5)

- **Experiment:** gsm8k_coconut. Text-CoT vs COCONUT on GSM8K-style arithmetic with a planted hint.
- **Notebook:** `notebooks/coconut.ipynb` (generator `src/build_notebook.py`). Hash not recorded at the time.
- **Config:** controlled, N_TRAIN=1500, 2 epochs/stage, text epochs 8, monitor sets 150/150, micro-batch COCONUT 4 x accum 4. Duration 17.6 min on a T4.
- **Purpose:** check that the pipeline does what it should after the v4 data-clobber fix, before spending the full budget.
- **Gates:** 16/19 passed. Failed: text no-hint accuracy 0.00 (>= 0.05), text answers parsed 0.94 (>= 0.95), COCONUT no-hint accuracy 0.01 (>= 0.05).
- **Headline numbers:** text CoT/answer-consistency monitor AUROC 0.87; keyword 0.49; TF-IDF on CoT 0.49. COCONUT latent probe AUROC 1.000 (suspicious, see below). Shuffled-label controls 0.41-0.54.
- **Reading:** GPT-2 small cannot do GSM8K (no-hint accuracy ~0), so the counterfactual labels are not meaningful. The perfect COCONUT probe score is not trusted: with the model unable to reason, the label is carried by prompt features (decoy-only probe reached 0.70). This run motivated moving to ProsQA.
- **Files:** `results/` (gitignored: coconut.pt, text_cot.pt, features.npz).
