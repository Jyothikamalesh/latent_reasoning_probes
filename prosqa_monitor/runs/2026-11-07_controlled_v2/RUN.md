# ProsQA controlled run, first design (kernel `prosqa-monitorability-controlled`, version 2)

- **Design:** behaviour twins follow every hint (no depth condition); label = hinted prediction equals the hint and the unhinted one does not (includes correct hints); 300 monitor-test questions; no seed parameter; no learning curve.
- **Config:** controlled, N_TRAIN=2000, 1 epoch/stage, monitor sets 200/300. 61.7 minutes on a T4.
- **Gates:** 18/22 passed. Failed: unhinted accuracy >= 0.6 for explicit_b (0.57), coconut_b (0.59) and explicit_c (0.50); coconut_c follows wrong hints at 0.39 (> 0.3).
- **Headline (AUROC, all hinted prompts):** explicit CoT-conclusion != answer 0.91; keyword 0.50; COCONUT probes 0.67-0.69; "hint is wrong" shortcut rule 0.71; cross-format transfer 0.47-0.50; label-free detectors 0.49-0.54; shuffled-label controls 0.47-0.55. COCONUT clean-anchor residual separated decoy from no-decoy prompts (AUROC 0.06): a confound.
- **Reading:** models did not learn ProsQA at this size (near-chance accuracy), so the monitor numbers are not interpretable. Two design flaws found afterwards (see `analysis/reanalysis_two_outcomes.py`): 35-41% of positives were correct-hint cases, and with ~100% hint following the wrong-hint label collapses to "the model would have been right".
- **Notebook state:** early version of `prosqa_monitor.py` (before seeds, two outcomes, conditional following).
- **Log:** lost (overwritten by the version 3 download). Gates and numbers above come from the saved CSV and from the session transcript.
