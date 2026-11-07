# Running notebooks on Kaggle (T4)

One-time setup:
1. Log in (easiest): `.venv/bin/kaggle auth login` (browser OAuth). If `shared/push_to_kaggle.sh` can't find your username afterwards, run it as `KAGGLE_USERNAME=<you> shared/push_to_kaggle.sh`.
   Or create a token at Kaggle -> Settings -> API:
   - legacy: download `kaggle.json` to `~/.kaggle/kaggle.json` and `chmod 600` it, or
   - new token: `export KAGGLE_API_TOKEN=...` (see the page shown after creation).
2. Verify your phone number on Kaggle (required for GPU and Internet in notebooks).

Then:
    shared/push_to_kaggle.sh push     # upload + run on T4
    shared/push_to_kaggle.sh status
    shared/push_to_kaggle.sh output   # downloads outputs to kaggle_output/<slug>/ (git-ignored); curate into <experiment>/runs/<run>/results

The notebook auto-switches: on a CPU it runs a tiny SMOKE config, on the T4 it runs the full one.
