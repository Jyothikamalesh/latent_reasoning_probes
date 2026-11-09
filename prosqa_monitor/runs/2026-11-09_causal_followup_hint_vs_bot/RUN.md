# Causal follow-up: hint tokens vs `<bot>` blocked separately (kernel `prosqa-causal-patching` version 6)

- **Code:** tag `prosqa-causal-followup-v1`, commit be94c5f, notebook sha1 ced0610fa1, `--followup` mode of `run_patching.py`; protocol Amendment 6 (exploratory, outside the frozen protocol). Run id 1794200993. Kaggle 2 x Tesla T4, transformers 5.20.2, batch 32, 17 unit tests OK; each seed 5.7-5.8 min. Same three frozen `coconut_b` twins (seed 0 = v3, seeds 1-2 = v4); rows 300 / 300 / 299.
- **Guards:** identity-patch difference 0.0; rebuild agreement 1.000 hinted, 1.000 / 1.000 / 0.997 unhinted; `f_ans_both` reproduces the registered `block_answer` exactly (+3.581 / +3.565 / +2.330).
- **Disclosure:** the interpretation rule was frozen after a 12-row local smoke had already shown the hint-token route (stated in Amendment 6), so this is a confirmation of that signal, not an independent discovery.
- **Download note:** the first `output` call returned only `coconut_repo`; the second fetched the files into `kaggle_output/prosqa-causal-followup`.

## Harmful switches, change in M (seeds 0 / 1 / 2)
| condition | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| answer blocked from hint tokens only | +3.564 | +3.547 | +2.320 |
| answer blocked from `<bot>` only | -0.030 | -0.011 | -0.034 |
| answer blocked from both (registered) | +3.581 | +3.565 | +2.330 |
| latent steps blocked from hint tokens only / `<bot>` only | -0.001 / +0.000 | -0.004 / +0.001 | -0.000 / +0.000 |
| `<bot>` blocked on UNHINTED prompts (disruption baseline) | +0.002 | +0.004 | +0.000 |
Share of harmful switches choosing the true option after the hint-token block: 100% / 97% / 100%. Resistant and wrong-either rows show the same pattern (hint-token block about +2.0 to +3.0 / +1.8 to +2.6, `<bot>` block about -0.03).
