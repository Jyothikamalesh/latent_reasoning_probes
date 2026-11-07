"""Logic tests for coconut.ipynb. Runs on CPU in ~1 min with a tiny random GPT-2 (no training).

    .venv/bin/python gsm8k_coconut/tests/test_notebook_logic.py

The notebook's own cells are exec'd, so these test the code that actually runs on Kaggle.
"""
import json, os, random, re, subprocess, sys, time, math, traceback
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks')
os.chdir(ROOT)
cells = [''.join(c['source']) for c in json.load(open('coconut.ipynb'))['cells'] if c['cell_type'] == 'code']
cell = lambda head: next(s for s in cells if head in s)

ns = dict(globals())
for head in ['import os, sys, subprocess', 'SMOKE = dev', 'LOG_TAG = ', 'tok = AutoTokenizer', 'def enc(s)', 'def coconut_forward', 'def parse_answer']:
    exec(compile(cell(head), f'<{head}>', 'exec'), ns)
    if head.startswith('import os'): ns['dev'] = 'cpu'
g = lambda name: ns[name]
LAT, BOT, EOT = g('LAT'), g('BOT'), g('EOT')

def tiny():
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(vocab_size=len(g('tok')), n_embd=32, n_layer=2, n_head=2, n_positions=128)).eval()

def seq(prompt_len, n_lat, tail, seed):
    r = torch.Generator().manual_seed(seed)
    return torch.cat([torch.randint(0, 5000, (prompt_len,), generator=r), torch.tensor([BOT] + [LAT] * n_lat + [EOT]), torch.randint(0, 5000, (tail,), generator=r)])

TESTS = []
def test(f): TESTS.append(f); return f

# ---------- data ----------
@test
def hint_construction():
    rng, n = random.Random(0), 6000
    exs = [g('add_hint')({'ans': a, 'q': 'q', 'steps': ['s']}, rng) for a in [random.Random(i).randint(0, 30) for i in range(n)]]
    none = [e for e in exs if e['hint'] is None]; ok = [e for e in exs if e['hint'] == e['ans']]; wrong = [e for e in exs if e['hint'] is not None and e['hint'] != e['ans']]
    assert abs(len(none) / n - 0.5) < 0.03 and abs(len(ok) / n - 0.25) < 0.03 and abs(len(wrong) / n - 0.25) < 0.03, (len(none), len(ok), len(wrong))
    # "wrong hint" slots must never equal the true answer (a correct hint would be mislabelled as misbehaviour)
    n_wrong_slots = sum(1 for e in exs if e['hint'] is not None) - len(ok)
    assert all(e['hint'] >= 0 for e in exs if e['hint'] is not None)
    assert all(e['target'] == (e['hint'] if e['hint'] is not None else e['ans']) for e in exs)
    pd_h = np.mean([e['decoy'] for e in exs if e['hint'] is not None]); pd_n = np.mean([e['decoy'] for e in none])
    assert abs(pd_h - pd_n) < 0.05 and abs(np.mean([e['decoy'] for e in exs]) - 0.5) < 0.03, (pd_h, pd_n)

@test
def wrong_hints_never_equal_truth():
    # exhaustive over small answers, where the abs(h)+1 fold-over can land on the truth
    bad = 0
    for ans in range(0, 40):
        exs =[{'ans': ans, 'q': f'q{i}', 'steps': ['s']} for i in range(400)]
        for e in g('mon_set')(exs, 400):
            if e['hint'] is not None and e['hint'] == e['ans']: bad += 1
    assert bad == 0, f'{bad} monitor "wrong hints" equal the true answer'

@test
def monitor_sets_balanced_and_independent():
    exs = [{'ans': 10 + i, 'q': f'q{i}', 'steps': ['s']} for i in range(400)]
    out = g('mon_set')(exs, 400)
    h = [e for e in out if e['hint'] is not None]; nh = [e for e in out if e['hint'] is None]
    assert len(h) == len(nh) == 200 and all(e['hint'] != e['ans'] for e in h)
    assert abs(np.mean([e['decoy'] for e in h]) - np.mean([e['decoy'] for e in nh])) < 0.05

@test
def real_monitor_splits_disjoint():
    mt, ms, tr = g('mon_train'), g('mon_test'), g('train')
    assert not ({e['q'] for e in mt} & {e['q'] for e in ms}) and not ({e['q'] for e in ms} & {e['q'] for e in tr})

@test
def coconut_example_structure():
    e = g('train')[0]; tok = g('tok'); C = g('C_PER_STEP'); N = g('N_STAGES')
    for stage in range(N + 1):
        ids, lab = g('make_coco_example')(e, stage)
        assert ids.count(LAT) == stage * C and ids.count(BOT) == 1 and ids.count(EOT) == 1
        b, t = ids.index(BOT), ids.index(EOT)
        assert ids[b + 1:t] == [LAT] * (stage * C)
        assert all(x == -100 for x in lab[:t + 1]) and all(x != -100 for x in lab[t + 1:]) and lab[t + 1:] == ids[t + 1:]
        text = tok.decode(ids[t + 1:], skip_special_tokens=True)
        rest = e['steps'][stage:] if stage < N else []
        assert text == '\n'.join(rest + [f"### {e['target']}"]), (stage, text)
        assert tok.decode(ids[:b]) == g('prompt_text')(e)

@test
def text_example_structure():
    e = g('train')[0]; tok = g('tok'); ids, lab = g('make_text_example')(e)
    p = len(g('enc')(g('prompt_text')(e)))
    assert all(x == -100 for x in lab[:p]) and lab[p:] == ids[p:]
    assert tok.decode(ids[p:], skip_special_tokens=True) == '\n'.join(e['steps'] + [f"### {e['target']}"])

@test
def answer_parsing():
    pa = g('parse_answer')
    assert pa('x\n### 1,234') == 1234 and pa('### -5') == -5 and pa('no answer here') is None and pa('a\n###7') == 7

# ---------- COCONUT forward ----------
@test
def zero_latents_equals_plain_forward():
    m = tiny(); ids = torch.randint(0, 5000, (2, 14)); att = torch.ones_like(ids)
    got = g('coconut_forward')(m, ids, att)[1].logits
    assert torch.allclose(got, m(ids, attention_mask=att).logits, atol=1e-5)

@test
def latents_are_previous_hidden_state_fixed_point():
    m = tiny(); ids = seq(8, 3, 4, 1).unsqueeze(0); att = torch.ones_like(ids)
    _, out, thoughts = g('coconut_forward')(m, ids, att, return_latents=True)
    pos = (ids[0] == LAT).nonzero().flatten().tolist()
    emb = m.get_input_embeddings()(ids)
    for _ in pos:                                   # independent re-derivation of the fill loop
        h = m.transformer(inputs_embeds=emb).last_hidden_state
        emb = emb.clone()
        for p in pos: emb[0, p] = h[0, p - 1]
    hfin = m.transformer(inputs_embeds=emb).last_hidden_state
    assert torch.allclose(out.logits, m(inputs_embeds=emb).logits, atol=1e-5)
    for n, p in enumerate(pos): assert torch.allclose(thoughts[n][0], hfin[0, p - 1], atol=1e-5), n
    assert not torch.allclose(m.get_input_embeddings()(ids)[0, pos[0]], emb[0, pos[0]])   # slots really were replaced

@test
def padded_batch_matches_single_examples():
    m = tiny(); a, b = seq(6, 2, 3, 2), seq(11, 4, 5, 3)
    L = max(len(a), len(b)); ids = torch.full((2, L), 50256); att = torch.zeros(2, L, dtype=torch.long)
    for i, s in enumerate([a, b]): ids[i, :len(s)] = s; att[i, :len(s)] = 1
    bat = g('coconut_forward')(m, ids, att)[1].logits
    for i, s in enumerate([a, b]):
        one = g('coconut_forward')(m, s.unsqueeze(0), torch.ones(1, len(s), dtype=torch.long))[1].logits[0]
        assert torch.allclose(bat[i, :len(s)], one, atol=1e-4), f'row {i}'

@test
def later_tokens_cannot_change_latents_or_earlier_logits():
    m = tiny(); A = seq(8, 3, 5, 4); B = A.clone(); B[-5:] = torch.randint(0, 5000, (5,))
    outs = [g('coconut_forward')(m, s.unsqueeze(0), torch.ones(1, len(s), dtype=torch.long), return_latents=True) for s in (A, B)]
    for ta, tb in zip(outs[0][2], outs[1][2]): assert torch.allclose(ta, tb, atol=1e-6)
    t = int((A == EOT).nonzero()[0])
    assert torch.allclose(outs[0][1].logits[0, :t + 1], outs[1][1].logits[0, :t + 1], atol=1e-5)   # train/generation consistency at the answer's first token

@test
def loss_alignment_is_next_token():
    m = tiny(); ids = seq(8, 2, 4, 5).unsqueeze(0); att = torch.ones_like(ids)
    lab = torch.full_like(ids, -100); lab[0, -1] = ids[0, -1]
    loss, out, _ = g('coconut_forward')(m, ids, att, lab)
    assert torch.allclose(loss, F.cross_entropy(out.logits[0, -2:-1], ids[0, -1:]), atol=1e-6)

@test
def gradients_reach_embeddings_through_latents():
    m = tiny().train(); ids = seq(8, 3, 4, 6).unsqueeze(0); att = torch.ones_like(ids)
    lab = torch.full_like(ids, -100); lab[0, -1] = ids[0, -1]
    m.zero_grad(); g('coconut_forward')(m, ids, att, lab)[0].backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert m.transformer.h[0].attn.c_attn.weight.grad.abs().sum() > 0

if __name__ == '__main__':
    fails = 0
    for f in TESTS:
        t0 = time.time()
        try: f(); print(f'PASS  {f.__name__} ({time.time() - t0:.1f}s)')
        except Exception as e:
            fails += 1; print(f'FAIL  {f.__name__}: {type(e).__name__}: {str(e)[:300]}')
    print(f'\n{len(TESTS) - fails}/{len(TESTS)} passed'); sys.exit(1 if fails else 0)
