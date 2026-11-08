# Causal latent-restoration experiment: protocol written BEFORE any patching result (2026-11-08)

Status: design only. Nothing here has been run. Do not edit the primary outcome after the first patching result exists; add amendments below instead.

## Question
When a wrong hint makes the COCONUT behaviour twin (coconut_b) switch from its correct unhinted answer to the planted wrong answer (strict harmful switch), is that switch mediated by particular latent states, or does the hint reach the answer directly?

## Data
Frozen twins only (no retraining): v3 seed 0 now, then v4 seeds 1 and 2. Wrong-hint test rows with `ok`. Groups: harmful switch (unhinted == true, hinted == wrong hint), resistant (unhinted == true, hinted == true), wrong-either-way (unhinted == wrong, hinted == wrong; a comparison group, NOT called susceptibility). Beneficial uptake (correct hint) is a separate later analysis.

## Outcome
M = S(true option) - S(wrong-hint option), S = summed token log-probability of the full option string in the answer template. Length-normalised S as a sensitivity analysis.
**Primary:** mean over the six latent steps of the restoration of M for a single-step same-question unhinted-latent patch, harmful-switch rows, reported per seed with question-clustered bootstrap CIs. Secondary: per-step curves, restoration rate (patched answer correct), full six-step restoration.

## Interventions (all recompute downstream latents and the answer)
1. Identity patch (implementation check: must reproduce the original M exactly).
2. Same-question restoration: hinted z_t <- unhinted z_t (patch the hidden vector only, keep the recipient's position ids).
3. Reverse patch: unhinted z_t <- hinted z_t on rows that did not switch.
4. Matched-donor patch: z_t from another question matched on depth, answer identity and baseline behaviour.
5. Noise patch with the same norm as the donor.
6. Same-condition donor (a hinted state from another hinted run).
7. Pathway decomposition: (a) all six latents from the unhinted run (the hint cannot reach the answer through the latents); (b) hint-line positions masked from the answer's attention (the hint can reach the answer only through the latents). The first latent input is the prompt's final hidden state, which already contains the hint, so masking attention alone is not enough: it must be replaced too.
8. Both pathways blocked as a negative control.
Applied also to resistant rows (does patching disrupt a correct answer?).

## Statistics
Bootstrap over question IDs (not rows). Per-seed estimates first, a pooled estimate only labelled as pooled over the three seeds. One primary outcome (above); everything else is exploratory and labelled so.

## Validity checks before reading any result
Identity patch reproduces M; a patch really changes only the intended latent (check the input embedding of the next step); downstream states are recomputed (changing z_1 changes z_2..z_6); option tokenisation checked; a tiny-model unit test for each control.

## Not in scope
SAEs, neuron attribution, new monitor architectures, retraining twins, larger models.

## Open before claiming novelty
A literature check for same-example counterfactual latent restoration with a direct-access control (Mirtaheri & Belkin, Li et al., Zhang et al., Ferreira et al., Duzan & Stickland, causal-mediation patching work).
