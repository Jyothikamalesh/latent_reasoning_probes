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

## Amendment 1 (2026-11-08, written before any patching result exists)
1. **Single-step restoration stays primary** (as specified above). **Persistent restoration** (replace z_t and also block latent steps t+1..6 from attending to the hint-line positions, replacing the first latent input with its unhinted counterpart where it still carries the hint) is a separately labelled **secondary** analysis. It answers a different question: can removing the hint's latent representation at step t stop its influence from propagating?
2. **Implementation validation on the tiny model before any real run:** identity patch reproduces the output exactly; unhinted-to-unhinted patch changes nothing; only the intended latent position is replaced; later states depend on the patched state (downstream recomputation); evaluation mode and no-grad give identical repeated outputs (no dropout or training-mode differences); hint masking targets exactly the hint-line token span; multi-token option scoring is consistent with a direct full-sequence forward pass.
3. **Interpretation table, fixed in advance:**
| Observation | Interpretation |
|---|---|
| Single-step restoration is selective (harmful switches restored, resistant rows intact, matched/random donors do not) | Evidence that the patched latent state contributes to the harmful switch |
| Restoration only when direct answer access to the hint is blocked | Direct hint access otherwise sustains the wrong answer |
| No single-step restoration, persistent restoration works | Hint influence is distributed over subsequent latent computation |
| No restoration under either | No evidence of mediation through the tested states; NOT proof that the latents are uninvolved (redundant encoding is possible) |
| Resistant rows change as much as harmful rows | Non-specific disruption |
4. **Provenance recorded per run:** checkpoint SHA-256 (`docs/provenance_checkpoints.json`), training code hash and tag, strict-label version, question IDs, tokenizer/model versions, and the commit of the intervention implementation. Seed 0 (v3) and seeds 1-2 (v4) are treated as comparable only because `git diff prosqa-seeds-v3 prosqa-seeds-v4` shows the change is limited to the label function, which twin training does not use.
5. **No additional monitoring experiments to explain the 0.57-0.60 signal until the causal result exists.**

## Amendment 2 (2026-11-08, written before any patching result exists; after review of the first implementation)
1. **Primary unchanged:** single-step same-question restoration with the natural unhinted donor, harmful-switch rows, mean over the six steps.
2. **Position-matched donor (new co-primary sensitivity control, reported next to the primary, never replacing it).** The natural unhinted prompt is shorter than the hinted one, so its `<bot>` and latent states sit at different absolute positions (GPT-2 uses absolute position embeddings), and a restoration effect could partly be a position shift. The position-matched donor is the hinted prompt with the hint-line tokens kept in the sequence (all later position ids unchanged) but removed from attention. Reported as `*_pm` conditions. If primary and position-matched results disagree, the position-matched one is the cleaner statement about the hint.
3. **Latent-pathway blocking replaces slot 0 as well.** Slot 0's input is the `<bot>` hidden state, which was computed while attending to the hint; masking attention alone does not cut that pathway (unit test `test_blocking_needs_slot0_replacement` shows the leak). `block_latents` therefore = slot 0 replaced by the unhinted (or position-matched) state + latent steps blocked from the hint span and `<bot>`; `both_blocked` additionally replaces all six inputs and blocks the answer. Unit test `test_both_blocked_is_hint_independent` checks the output is independent of the hint content in that condition.
4. **Persistent restoration (secondary) is defined exactly as: slots 0 and t replaced by the unhinted states; slots t+1.. cannot attend to the hint span or `<bot>`.** Slot t itself and earlier cached states may already have seen the hint, so this does not remove all hint information from the cache; it is not the primary and is described as such.
5. **What is patched:** the INPUT vector of a latent slot (COCONUT's continuous thought z_t), not the full state of the slot: the slot still attends to the hinted prompt unless blocked, and later states are recomputed.
6. **Run guards:** one row per question (asserted); every answer is one of the two options; hint span non-empty; special-token ids and embedding size equal the training setup; the identity patch must reproduce M; sign(M) must agree with the saved greedy prediction on at least 0.90 of rows (hinted and unhinted; reported per group) or the run stops before any result is read; unit tests run first in the Kaggle kernel too (transformers version).
7. Donor questions are drawn with a seeded random generator (same group, same depth when depth is known).
