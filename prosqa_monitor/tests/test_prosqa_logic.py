"""Logic tests for prosqa_monitor.ipynb (CPU, tiny random GPT-2, no training).
    PROSQA_DIR=<dir with prosqa_*.json> .venv/bin/python prosqa_monitor/tests/test_prosqa_logic.py
"""
import json, os, random, sys, time, torch, numpy as np
from transformers import GPT2Config, GPT2LMHeadModel
os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks'))   # the notebook looks for prosqa_monitor.ipynb in the cwd
os.environ['MODE'] = 'tiny'
cells = [''.join(c['source']) for c in json.load(open('prosqa_monitor.ipynb'))['cells'] if c['cell_type'] == 'code']
ns = dict(globals())
for s in cells:
    if s.startswith('# [pipeline]') and 'def train_explicit' in s: break
    if 'RUN_SMOKE_FIRST' in s or s.lstrip().startswith('phase(') or s.lstrip().startswith('# [pipeline]\nphase'): continue
    exec(compile(s, '<cell>', 'exec'), ns)
g = lambda n: ns[n]; LAT, BOT, EOT = g('LAT'), g('BOT'), g('EOT')
TESTS = []
def test(f): TESTS.append(f); return f
def tiny():
    torch.manual_seed(0); return GPT2LMHeadModel(GPT2Config(vocab_size=len(g('tok')), n_embd=32, n_layer=2, n_head=2, n_positions=1024)).eval()

@test
def parse_and_wrong_option():
    for it in g('_items')[:2000]:
        assert it['wrong'] != it['true'] and it['true'] in it['q'] and it['wrong'] in it['q'], it['q'][-80:]

@test
def hint_rates_and_independent_decoy():
    rng = random.Random(0); its = g('_items')[:4000]; ex = [g('plant')(i, rng) for i in its]
    n = len(ex); none = np.mean([e['hint'] is None for e in ex]); ok = np.mean([e['hint'] == e['true'] for e in ex]); wr = np.mean([e['hint'] == e['wrong'] for e in ex])
    assert abs(none - .5) < .03 and abs(ok - .25) < .03 and abs(wr - .25) < .03
    assert abs(np.mean([e['decoy'] for e in ex if e['hint']]) - np.mean([e['decoy'] for e in ex if not e['hint']])) < .05

@test
def targets_follow_hint_only_for_behaviour_twin_on_deep_problems():
    e = dict(g('train')[0]); e['hint'] = e['wrong']
    e['obey'] = True;  assert g('target_option')(e, True) == e['wrong'] and g('target_option')(e, False) == e['true']
    e['obey'] = False; assert g('target_option')(e, True) == e['true']          # shallow problem: behaviour twin resists the hint
    e['hint'] = None;  e['obey'] = True; assert g('target_option')(e, True) == e['true']

@test
def obey_flag_matches_depth_and_has_both_values():
    tr = g('train') if len(g('train')) > 200 else [g('plant')(i, random.Random(0)) for i in g('_items')[:2000]]
    assert all(e['obey'] == (len(e['steps']) >= g('OBEY_MIN_STEPS')) for e in tr)
    r = np.mean([e['obey'] for e in [g('plant')(i, random.Random(0)) for i in g('_items')[:3000]]]); assert 0.5 < r < 0.7, r

@test
def eval_split_is_fixed_across_seeds_and_disjoint_from_training():
    import subprocess
    code = ("import os,json,sys;os.environ['MODE']='tiny';os.environ['SEED']=sys.argv[1];"
            "ns={};src=[''.join(c['source']) for c in json.load(open('prosqa_monitor.ipynb'))['cells'] if c['cell_type']=='code'];"
            "[exec(compile(x,'c','exec'),ns) for x in src[:4]+[x for x in src if x.lstrip().startswith('# [pipeline]') and 'def find_data_dir' in x]];"
            "print(json.dumps(dict(ev=[e['q'] for e in ns['eval_pool'][:50]], tr=[e['q'] for e in ns['train']], seed=ns['SEED'])))")
    outs = [json.loads(subprocess.run([sys.executable, '-c', code, sd], capture_output=True, text=True, env=dict(os.environ, PROSQA_DIR=os.environ['PROSQA_DIR'])).stdout.strip().splitlines()[-1]) for sd in ('0', '1')]
    assert outs[0]['ev'] == outs[1]['ev'], 'evaluation questions must not depend on the seed'
    assert outs[0]['tr'] != outs[1]['tr'], 'training sample should depend on the seed'
    assert outs[0]['seed'] == 0 and outs[1]['seed'] == 1
    assert not (set(outs[0]['ev']) & set(outs[0]['tr'])) and not (set(outs[1]['ev']) & set(outs[1]['tr']))

@test
def held_out_pool_disjoint_when_enlarged():
    # simulate the full-mode pool: 500+2000 eval questions need extra questions held out of train
    items = g('_items'); need = 2500 - len(g('valid_items')) - len(g('test_items')); held, rest = items[:need], items[need:]
    assert need > 0 and not ({i['q'] for i in held} & {i['q'] for i in rest}) and len(rest) > 6000

@test
def splits_disjoint():
    tr = {e['q'] for e in g('train')}; assert not (tr & {i['q'] for i in g('valid_items')}) and not (tr & {i['q'] for i in g('test_items')})

@test
def coconut_example_structure():
    tok, e = g('tok'), g('train')[0]
    for stage in range(g('N_STAGES') + 1):
        x = g('make_coco_example')(e, stage, True)
        assert x['pre'][-1] == BOT and x['post'][0] == EOT and x['n'] == stage
        txt = tok.decode(x['post'][1:], skip_special_tokens=True)
        want = '\n'.join((e['steps'][stage:] if stage < g('N_STAGES') else []) + ['### ' + g('answer_text')(e, g('target_option')(e, True))])
        assert txt == want, (stage, txt, want)
        assert x['plabels'] == [-100] + x['post'][1:]

@test
def text_example_structure():
    e = g('train')[0]; ids, lab = g('make_text_example')(e, False); p = len(g('enc')(g('ex_prompt')(e)))
    assert all(v == -100 for v in lab[:p]) and lab[p:] == ids[p:]
    assert g('tok').decode(ids[p:], skip_special_tokens=True).endswith('### ' + g('answer_text')(e, e['true']))

@test
def collate_latent_slots_aligned():
    es = [g('make_coco_example')(e, 3, True) for e in g('train')[:4]]; ids, att, pos, lab, a, n = g('collate_coco')(es)
    assert (ids[:, a:a + n] == LAT).all() and (att[:, a:a + n] == 1).all() and (ids[:, a - 1] == BOT).all() and (ids[:, a + n] == EOT).all()

@test
def cached_forward_equals_full_forward_with_filled_latents():
    m = tiny(); es = [g('make_coco_example')(e, 3, True) for e in g('train')[:3]]; ids, att, pos, lab, a, n = g('collate_coco')(es)
    out = g('coconut_forward')(m, ids, att, pos, a, n, lab)
    emb = m.get_input_embeddings()(ids).clone()
    for _ in range(n):                                             # independent re-derivation: iterate the fixed point on the full sequence
        h = m.transformer(inputs_embeds=emb, attention_mask=att, position_ids=pos).last_hidden_state
        for i in range(n): emb[:, a + i] = h[:, a + i - 1]
    ref = m(inputs_embeds=emb, attention_mask=att, position_ids=pos).logits[:, a + n:]
    assert torch.allclose(out['logits'], ref, atol=1e-4), (out['logits'] - ref).abs().max()

@test
def padding_invariance():
    m = tiny(); es = [g('make_coco_example')(e, 2, True) for e in g('train')[:3]]; ids, att, pos, lab, a, n = g('collate_coco')(es)
    bat = g('coconut_forward')(m, ids, att, pos, a, n)['thoughts']
    for i, x in enumerate(es):
        i1, a1, p1, _, a_, n_ = g('collate_coco')([x]); one = g('coconut_forward')(m, i1, a1, p1, a_, n_)['thoughts']
        assert torch.allclose(bat[i], one[0], atol=1e-4), i

@test
def encode_matches_training_thoughts():
    m = tiny(); prompts = [g('ex_prompt')(e) for e in g('train')[:3]]
    h, z = g('coconut_encode_batch')(m, prompts, 3)
    es = [g('make_coco_example')(e, 3, True) for e in g('train')[:1]]; ids, att, pos, lab, a, n = g('collate_coco')(es)
    th = g('coconut_forward')(m, ids, att, pos, a, n)['thoughts']
    assert torch.allclose(z[0], th[0], atol=1e-4)

@test
def gradients_flow():
    m = tiny().train(); es = [g('make_coco_example')(e, 3, True) for e in g('train')[:2]]; b = g('collate_coco')(es)
    g('coco_fwd')(m, b).backward(); assert m.transformer.h[0].attn.c_attn.weight.grad.abs().sum() > 0

@test
def counterfactual_label_rule():
    pr = g('pred_option'); assert pr('a\n### Tom is a zhorpus.') == 'zhorpus' and pr('nothing') is None
    assert g('cot_conclusion')('Tom is a x.\nTom is a y.\n### Tom is a z.') == 'y'

@test
def max_len_covers_data():
    L = [len(g('make_text_example')(g('plant')(i, random.Random(1)), True)[0]) for i in g('_items')[:3000]]
    assert max(L) <= g('MAX_LEN'), (max(L), g('MAX_LEN'))

if __name__ == '__main__':
    f = 0
    for t in TESTS:
        t0 = time.time()
        try: t(); print(f'PASS  {t.__name__} ({time.time() - t0:.1f}s)')
        except Exception as e: f += 1; print(f'FAIL  {t.__name__}: {type(e).__name__}: {str(e)[:300]}')
    print(f'\n{len(TESTS) - f}/{len(TESTS)} passed'); sys.exit(1 if f else 0)
