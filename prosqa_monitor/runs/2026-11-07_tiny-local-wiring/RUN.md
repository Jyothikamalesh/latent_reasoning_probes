# ProsQA tiny local wiring test (CPU, no Kaggle)

- **Mode:** tiny (random 2-layer GPT-2, 48 training examples). 2.2 minutes on the Mac.
- **Purpose:** every cell of the ProsQA notebook runs end to end, including the sandboxed smoke pass, and `train` is restored to its full size afterwards.
- **Result:** wiring OK. Quality gates fail by design on an untrained random model, which shows the gates detect a broken model. Statistics are NaN.
- This test is repeated after every code change; the log here is the last run saved (seed 3 / seed 1 variants).
