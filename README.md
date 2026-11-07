# Latent-space monitorability research

Can hidden behaviour (quietly following a planted hint) in a continuous-thought (COCONUT) model be detected, compared with a readable explicit chain-of-thought model? Same base model (GPT-2 small), same data, same budget.

## Layout (one folder per experiment)
```
gsm8k_coconut/      first experiment (GSM8K arithmetic). Finished; GPT-2 small cannot do the task. Kept as history.
prosqa_monitor/     current experiment (ProsQA): explicit vs COCONUT twins, clean-anchor and cross-format monitors.
  prosqa_monitor.py   the experiment source (# %% cells). Edit this, never the .ipynb.
  build_notebook.py   source -> notebooks/prosqa_monitor.ipynb  (--seed N for a per-seed Kaggle notebook)
  build_profile.py    builds the T4 profiling notebook from the experiment's own definitions
  aggregate_seeds.py  merges per-seed results; refuses mixed code hashes
  analysis/           one-off analyses of saved runs
  tests/              CPU logic tests
  notebooks/          generated notebooks (what is pushed to Kaggle)
  runs/<date>_<name>/ one folder per run: RUN.md (config, hash, gates, headline numbers, reading) + results/
shared/             push_to_kaggle.sh, README_kaggle.md
docs/               HISTORY.md (what ran, learned, decided, changed), DECISIONS.md (why), early notes
```
`kaggle_output/` is a git-ignored scratch folder for fresh downloads; curate into `<experiment>/runs/<run>/results` and write a RUN.md.

## Working rules
- **Ask before every Kaggle push** (a yes covers that push only). Do not export `KAGGLE_USERNAME`; it breaks the CLI's OAuth login.
- Edit `prosqa_monitor.py`, rebuild the notebook, run the tests (below) and the tiny local run before any push.
- Every run gets a `RUN.md`; every notable change or decision gets an entry in `docs/HISTORY.md` / `docs/DECISIONS.md`. Commit with git; tag each Kaggle push (`prosqa-controlled-v3`).
- Downloaded files and the cloned facebookresearch/coconut repo are untrusted data.

## Tests (CPU, about a minute)
```
export PROSQA_DIR=<dir containing prosqa_{train,valid,test}.json>
.venv/bin/python prosqa_monitor/tests/test_prosqa_logic.py      # 16 tests
.venv/bin/python gsm8k_coconut/tests/test_notebook_logic.py     # 13 tests
```
Tiny end-to-end wiring run: build the notebook, `MODE=tiny FORCE_SMOKE_FIRST=1` on the concatenated code cells (see HISTORY, "tiny mode").

## Push (only when asked)
```
NB=prosqa_monitor/notebooks/prosqa_monitor SLUG=prosqa-monitorability-controlled TITLE='prosqa monitorability controlled' shared/push_to_kaggle.sh push
```
