# Two-GPU mini test (kernel `prosqa-two-gpu-mini-test`, pushed 2026-11-08 ~19:46 local)

- **Code:** branch `two-gpu-training`, tag `prosqa-twogpu-mini-test` (pushed tag `prosqa-twogpu-mini-test-pushed`), code hash **8a64808c6e**, notebook sha1 c39ceefb89. Mode `mini`, seed 0. NOT the frozen v4 line.
- **Purpose:** verify the two-process training path on a real Kaggle session (COCONUT twins on device 1 in a worker, explicit twins on device 0 in the main process).
- **Result:** the path works. Main process `gpus_visible=2`; worker started on device 1 at 3.6 min; its `nvidia-smi` report showed GPU 0 at 100% (10.9 GB) and GPU 1 at 74% (10 GB), so both were busy at once; the worker exited with code 0 and the main process waited 0.2 min after the explicit twins. Total 13.8 min, status `GATES_FAILED` (behavioural gates only: explicit_b deep/shallow hint following, coconut_b shallow resistance, explicit accuracy gap; expected for a mini run).
- **Not measured:** speed-up versus sequential (the earlier single-GPU mini download had no `run_meta`, and a mini run is too small to compare). The saving shows only on full seeds.
- **Files:** `results/prosqa-two-gpu-mini-test.log` (kernel log), `results/s0_run_meta.json`.
- **Decision:** the two-GPU path stays on its branch; no full-seed run uses it yet. Merge only with a user decision.
