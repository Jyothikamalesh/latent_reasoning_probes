"""Counterfactual latent patching for the COCONUT twins (protocol: docs/causal_protocol.md).
Standalone: works on any GPT-2 LM whose vocabulary has the <bot>, <eot>, <latent> tokens; nothing is imported from the notebook.

Sequence layout, as in the notebook: [left pad][prompt][<bot>] [latent slot 0 .. n-1] [<eot> answer sentence (right pad)].
Latent slot i takes as its INPUT embedding the final-layer hidden state of the previous position (slot 0: the <bot> position). A patch replaces that input vector for one slot,
so the slot's cached keys/values, every later slot and the answer all see the patched state (downstream states are recomputed, nothing is cached across interventions).
Blocking the hint (note): masking attention to the hint span and <bot> is NOT enough to cut the latent pathway, because slot 0's INPUT is the <bot> hidden state, which was computed while attending to the hint; replace slot 0 as well (run_patching does).
Blocking the hint: the hint line is the last segment of the prompt and <bot> follows it, so the positions that can carry the hint are the hint-line span AND <bot>."""
import torch
import torch.nn.functional as F

BOTTLENECK = False      # set by run_patching --bottleneck for twins trained with the information bottleneck: the answer segment cannot attend to the prompt or <bot>, only to the latents


def left_pad(seqs, pad_id):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id); att = torch.zeros(len(seqs), L, dtype=torch.long)
    for b, s in enumerate(seqs):
        ids[b, L - len(s):] = torch.tensor(s); att[b, L - len(s):] = 1
    return ids, att


def hint_span_len(enc, hinted_prompt, unhinted_prompt):
    """Number of trailing tokens of the hinted prompt that belong to the hint line. The hint line is appended last, so the unhinted prompt's tokens must be an exact prefix."""
    h, u = enc(hinted_prompt), enc(unhinted_prompt)
    assert h[:len(u)] == u, 'tokenisation of the unhinted prompt is not a prefix of the hinted prompt: the hint-token span is ambiguous'
    return len(h) - len(u)


def make_batch(prompts, hint_lens, tails, n, pad_id, bot, lat, device='cpu', hide_hint=False, block_bot=False):
    """prompts[b]: prompt token ids (no <bot>); hint_lens[b]: trailing prompt tokens that are the hint line (0 if unhinted); tails[b]: [<eot>] + answer-sentence token ids.
    block_bot: also mark the <bot> position as blockable in prompts without a hint (the matched-attention disruption control on unhinted runs).
    hide_hint: the hint-line tokens stay in the sequence (so every later position id equals the hinted run's) but are removed from attention: a position-matched 'unhinted' run."""
    pre_ids, pre_att = left_pad([p + [bot] for p in prompts], pad_id); a = pre_ids.shape[1]; B = len(prompts)
    T = max(len(t) for t in tails)
    tail_ids = torch.full((B, T), pad_id); tail_att = torch.zeros(B, T, dtype=torch.long)
    for b, t in enumerate(tails): tail_ids[b, :len(t)] = torch.tensor(t); tail_att[b, :len(t)] = 1
    ids = torch.cat([pre_ids, torch.full((B, n), lat), tail_ids], 1); att = torch.cat([pre_att, torch.ones(B, n, dtype=torch.long), tail_att], 1)
    pos = (att.cumsum(1) - 1).clamp(min=0)                                       # position ids are computed BEFORE any hiding
    block = torch.zeros(B, ids.shape[1], dtype=torch.bool); block_hint = torch.zeros_like(block); block_botpos = torch.zeros_like(block)
    for b, h in enumerate(hint_lens):
        if h > 0: block[b, a - 1 - h:a] = True; block_hint[b, a - 1 - h:a - 1] = True   # hint-line tokens [a-1-h, a-1) and the <bot> position a-1 (nothing to block in an unhinted prompt)
        if h > 0 or block_bot: block_botpos[b, a - 1] = True
        if block_bot: block[b, a - 1] = True
        if hide_hint and h > 0: att[b, a - 1 - h:a - 1] = 0
    return dict(ids=ids.to(device), att=att.to(device), pos=pos.to(device), block=block.to(device), block_hint=block_hint.to(device), block_botpos=block_botpos.to(device), a=a, n=n, tail_att=tail_att.to(device), tail_ids=tail_ids.to(device))


@torch.no_grad()
def run(model, bt, patch=None, block_latents_from=None, block_answer=False, latent_kind='both'):
    """patch: {slot: tensor (B, d)} replaces that slot's input embedding. block_latents_from=t: from latent slot t on, the hint span and <bot> are removed from attention.
    block_answer: the answer positions cannot attend to the hint span and <bot>; 'hint' = hint tokens only, 'bot' = <bot> only. latent_kind: which of those the blocked latent steps lose ('both' | 'hint' | 'bot'). Returns thoughts (B, n, d) and per-token log-probs of the answer sentence (B, T-1) with their mask."""
    assert not model.training, 'model must be in eval mode (dropout would make interventions non-reproducible)'
    tr = model.transformer; a, n = bt['a'], bt['n']; ids, att, pos = bt['ids'], bt['att'], bt['pos']
    pick = {'both': bt['block'], 'hint': bt['block_hint'], 'bot': bt['block_botpos']}
    blk, ablk = pick[latent_kind], pick[{True: 'both', 'hint': 'hint', 'bot': 'bot'}[block_answer]] if block_answer else None
    emb = model.get_input_embeddings()(ids)
    pre = tr(inputs_embeds=emb[:, :a], attention_mask=att[:, :a], position_ids=pos[:, :a], use_cache=True)
    past, cur, thoughts = pre.past_key_values, pre.last_hidden_state[:, -1:], []
    for i in range(n):
        if patch is not None and i in patch: cur = patch[i][:, None, :].to(cur.dtype)
        thoughts.append(cur)
        m = att[:, :a + i + 1]
        if block_latents_from is not None and i >= block_latents_from: m = m * (~blk[:, :a + i + 1]).long()
        o = tr(inputs_embeds=cur, attention_mask=m, position_ids=pos[:, a + i:a + i + 1], past_key_values=past, use_cache=True)
        past, cur = o.past_key_values, o.last_hidden_state
    m = att * ((~ablk).long() if block_answer else 1)
    if BOTTLENECK: m = m.clone(); m[:, :a] = 0
    o = tr(inputs_embeds=emb[:, a + n:], attention_mask=m, position_ids=pos[:, a + n:], past_key_values=past, use_cache=True)
    logits = model.lm_head(o.last_hidden_state)                     # tail columns: [<eot>, sentence...]
    lp = F.log_softmax(logits[:, :-1].float(), -1).gather(-1, bt['tail_ids'][:, 1:, None])[..., 0]
    return dict(thoughts=torch.cat(thoughts, 1), logp=lp, mask=bt['tail_att'][:, 1:].bool(), logits=logits)


def sentence_logprob(out):
    """Summed log-probability of the full answer sentence (all option tokens and the final period)."""
    return (out['logp'] * out['mask']).sum(1)


def option_logprob(out, start):
    """Summed log-probability and token count of the OPTION WORD tokens only: the shared prefix '### subj is a' and the terminal '.' are excluded.
    start[b] = index in the logp row where the option begins; the last valid token of each row is the final period."""
    idx = torch.arange(out['logp'].shape[1], device=out['logp'].device)[None]
    last = out['mask'].sum(1, keepdim=True) - 1
    m = out['mask'] & (idx >= torch.as_tensor(start, device=idx.device)[:, None]) & (idx < last)
    return (out['logp'] * m).sum(1), m.sum(1)


def margin(model, prompts, hint_lens, tail_true, tail_wrong, n, pad_id, bot, lat, patch=None, block_latents_from=None, block_answer=False, device='cpu', hide_hint=False, block_bot=False, opt_start=None, latent_kind='both'):
    """M = S(true option sentence) - S(wrong option sentence), same context, same intervention. Returns M (B,) and the thoughts of the true-option pass (identical for both passes)."""
    res = []
    for tails in (tail_true, tail_wrong):
        bt = make_batch(prompts, hint_lens, tails, n, pad_id, bot, lat, device, hide_hint, block_bot)
        res.append(run(model, bt, patch, block_latents_from, block_answer, latent_kind))
    M = sentence_logprob(res[0]) - sentence_logprob(res[1])
    if opt_start is None: return M, res[0]['thoughts']
    (st, nt), (sw, nw) = option_logprob(res[0], opt_start), option_logprob(res[1], opt_start)
    return M, res[0]['thoughts'], st / nt - sw / nw, nt - nw                # sensitivity: per-option-token mean log-prob margin, and the option token-count difference
