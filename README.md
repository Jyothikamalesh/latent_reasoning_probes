# Learning what latent reasoning does, and whether we can monitor it

This repository is a worked investigation of one question:

> Some language models "think" in hidden numbers instead of words. Can we watch those hidden thoughts to catch unwanted behaviour, and when a watcher raises a flag, is it tracking the real cause or just something that goes along with it?

It is written as a learning path. Each section says what we wanted to understand, what we did, and what we learned, including the dead ends. Every claim points to a protocol written before the result and a record of the run that produced it.

**Setting:** GPT-2 small, the ProsQA reasoning task (graph questions with a multi-step answer), and COCONUT, a method that replaces written reasoning steps with continuous "thoughts". We plant a hint in some prompts so that a model can follow it, resist it, or be misled by a wrong one.

## 1. The ideas in two minutes
- **Explicit reasoning:** the model writes its steps in words. Anyone can read them.
- **Latent reasoning (COCONUT):** the model feeds its own hidden state back in as the next "thought". Nothing is written. Six such thoughts replace the six written steps.
- **Monitor:** a small program that reads the model's internals and predicts a behaviour, for example "this answer followed the hint". A probe is a monitor trained on labelled examples.
- **Counterfactual label:** run the same question with and without the hint. If the answer changes to the hinted option, the hint caused it. That is the label the monitors are scored against.
- **Harmful switch:** a wrong hint turns a correct answer into the planted wrong one.
- **Patching:** copy the hidden thoughts from the un-hinted run into the hinted run and see whether the hint effect disappears. If it does, those thoughts carry the cause.
- **Monitor validity:** a monitor can score well because it tracks the cause, or because it tracks a side-effect. Patching is how we tell the two apart.

## 2. What we learned, in order

### Step 1. Can a tiny model do the task? (`gsm8k_coconut/`)
First attempt on GSM8K arithmetic. GPT-2 small could not do the task, so there was nothing to monitor. Kept as history. We moved to ProsQA.

### Step 2. Build models that follow hints, plus clean twins (`prosqa_monitor/`)
We trained pairs of models: a *behaviour* twin that follows a hint on deep problems, and a *clean* twin without that behaviour, one pair in explicit form and one in COCONUT form. Then we built the monitors: text monitors on the explicit twin, supervised probes, probes transferred between formats, and detectors that need no labels.
Lesson: every monitor needs baselines. An output-only check ("did the answer equal the hint?") already scores very high on wrong-hint prompts, and a probe that only measures how hard a question is can look good too. A monitor counts only if it beats both. These are in every table (`docs/DECISIONS.md`, rows 25 to 28).

### Step 3. Do the hidden thoughts matter? (`docs/causal_protocol.md`)
We wrote the patching protocol down before any result and ran it on three seeds.
Finding: restoring the hidden thoughts from the un-hinted run did nothing (change in margin about 0, intervals within ±0.001). Blocking the answer from reading the hint tokens, in contrast, removes the hint effect (97 to 100% of harmful switches go back to the true option). So in these models the answer reads the hint directly from the prompt, and the hidden thoughts are not on the path.

### Step 4. Is it just undertrained? (`docs/long_twin_protocol.md`)
We trained longer twins (15,000 questions, 2 epochs per stage) that reach about 85% accuracy and show many harmful switches. A pre-set rule asks whether zeroing the six thoughts changes the answer. It moves the margin by about 0.05 and flips 0.4% of answers. The thoughts are still inert, so more training of this kind did not change the picture.

### Step 5. Force the thoughts to matter, as a control (`docs/bottleneck_protocol.md`)
If patching is a sound method, it should find a causal route when one exists. So we built one by construction: in the last training stage the answer cannot look at the prompt, only at the six thoughts.
- Unhinted accuracy stays at 0.85 / 0.84 (two seeds), and zeroing the thoughts now flips 7 to 59% of rows.
- Restoring all six thoughts returns 96% / 94% of harmful switches to the true answer, against 34 to 44% for disruption controls (zeroing, or thoughts from another question).
- Restoring one thought at a time is partial and differs by seed, so the route is spread across thoughts.
Conclusion: the method works when a route exists, and the earlier null was a property of the unforced models. This is a control that is positive by construction. It says nothing about whether natural latent reasoning is used.

## 3. What this does and does not show
- **Shown:** hidden thoughts in our unforced COCONUT twins are not on the causal path of hint-following; the same patching method detects a causal route when one is forced; counterfactual labels, baselines and per-seed reporting matter more than raw detection scores.
- **Not shown:** anything about natural latent reasoning in general, other models, other tasks, or whether a monitor on forced thoughts tracks the cause (the next step is pre-registered as a draft, not yet run).
- **Known gap:** our recipe uses about a third of the reference training volume and adds a hint objective; whether either explains the inert thoughts is untested. A clean, hint-free, reference-recipe twin would answer it.
- Closest related work: Ferreira et al. 2026 (arXiv 2608.04928) study how well hint-reliance can be monitored in latent-reasoning models; they do not test whether a monitor tracks the cause.

## 4. Where to look
```
docs/HISTORY.md        what ran, what was learned, in order
docs/DECISIONS.md      why each choice was made
docs/*_protocol.md     the pre-registrations (written before the results)
prosqa_monitor/runs/   one folder per run: RUN.md (config, hash, gates, headline numbers) + results
prosqa_monitor/prosqa_monitor.py        twins, monitors, baselines
prosqa_monitor/analysis/latent_patching.py   the patching code
gsm8k_coconut/         the first, abandoned attempt
shared/                helper for Kaggle runs
```
Model checkpoints (`*.pt`) are not in the repository.

## 5. Reproducing
```
export PROSQA_DIR=<dir containing prosqa_{train,valid,test}.json>
python prosqa_monitor/tests/test_prosqa_logic.py
python prosqa_monitor/tests/test_latent_patching.py
```
Experiments ran on free Kaggle T4 GPUs. The notebooks are generated from `prosqa_monitor.py` (`build_notebook.py`, `build_long_notebook.py`); never edit the `.ipynb` files by hand.

## 6. Status
Work in progress . Results are per seed and are not pooled. Protocol amendments are recorded, not hidden. Treat numbers here as a summary and the `RUN.md` files as the record.
