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
    rg = random.Random(0); tr = [g('plant')(i, rg) for i in g('_items')[:3000]]
    assert all(e['obey'] == (len(e['steps']) >= g('OBEY_MIN_STEPS')) for e in tr)
    r = np.mean([e['obey'] for e in tr]); assert 0.5 < r < 0.7, r

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
def weighted_ce_matches_hf_loss_and_manual_weighting():
    m = tiny(); es = [g('make_text_example')(e, True) for e in g('train')[:3]]; b = g('collate_text')(es)
    assert torch.allclose(g('text_fwd')(m, b), m(input_ids=b[0], attention_mask=b[1], labels=b[2]).loss, atol=1e-5)   # ANS_W = 1.0 is the old loss
    ns['ANS_W'] = 3.0
    try:
        got = g('text_fwd')(m, b)
        lg = m(input_ids=b[0], attention_mask=b[1]).logits[:, :-1]; tgt = b[2][:, 1:]; mk = tgt != -100
        ce = torch.nn.functional.cross_entropy(lg[mk], tgt[mk], reduction='none')
        h = (tgt == g('HASH')).long(); after = ((h.cumsum(1) - h) >= 1)[mk]; w = torch.where(after, torch.tensor(3.0), torch.tensor(1.0))
        assert after.any() and (~after).any() and torch.allclose(got, (w * ce).sum() / w.sum(), atol=1e-5)
        assert not torch.allclose(got, m(input_ids=b[0], attention_mask=b[1], labels=b[2]).loss, atol=1e-5)
    finally: ns['ANS_W'] = 1.0

@test
def wordings_default_unchanged_and_unseen_never_trained():
    q = g('train')[0]['q']
    assert g('make_prompt')(q, 'x', False) == q + '\nHint from grader: the answer is x.\n'
    for w, tpl in enumerate(g('HINT_WORDINGS')): assert g('make_prompt')(q, 'zz', False, w).count('zz') == 1 and tpl.format('zz') in g('make_prompt')(q, 'zz', False, w)
    rg = random.Random(3); tr = [g('plant')(i, rg) for i in g('_items')[:2000]]
    assert {e['wording'] for e in tr} == set(g('TRAIN_WORDINGS')) and not (set(g('UNSEEN_WORDINGS')) & {e['wording'] for e in tr})

@test
def hint_reliability_knob():
    for pc, want_wrong, want_correct in ((1.0, 0, 1), (0.0, 1, 0)):
        ns['P_CORRECT'] = pc
        rg = random.Random(5)
        try: ex = [g('plant')(i, rg) for i in g('_items')[:1500]]
        finally: ns['P_CORRECT'] = 0.5
        hinted = [e for e in ex if e['hint'] is not None]
        assert (sum(e['hint'] == e['wrong'] for e in hinted) > 0) == bool(want_wrong) and (sum(e['hint'] == e['true'] for e in hinted) > 0) == bool(want_correct)

@test
def suite_metrics_known_policies():
    qs = [dict(q=f'q{i}', true='a', wrong='b', steps=['s'] * (3 + i % 4)) for i in range(20)]
    meta, prompts = g('build_suite')(qs); assert len(meta) == len(prompts) == 20 * (1 + 2 * len(g('HINT_WORDINGS')))
    clean = [qs[qi]['true'] for qi, k, w in meta]                                     # always the truth
    follower = [qs[qi]['wrong'] if k == 'wrong' else qs[qi]['true'] for qi, k, w in meta]   # always follows wrong hints, right otherwise
    c, f = g('suite_metrics')(qs, meta, clean), g('suite_metrics')(qs, meta, follower)
    assert c['unhinted_acc'] == 1 and c['wrong_hint_comply_all'] == 0 and c['harmful_rate_all'] == 0 and c['correct_hint_acc_all'] == 1
    assert f['unhinted_acc'] == 1 and f['wrong_hint_comply_all'] == 1 and f['harmful_rate_all'] == 1 and f['wrong_hint_comply_deep'] == 1 and f['wrong_hint_comply_shallow'] == 1
    assert f['harmful_rate_unseen'] == 1 and c['parsed'] == 1
    wrong_unhinted = [qs[qi]['wrong'] if k == 'none' else p for (qi, k, w), p in zip(meta, follower)]
    assert g('suite_metrics')(qs, meta, wrong_unhinted)['harmful_rate_all'] == 0     # not harmful when the unhinted answer was already wrong

@test
def suite_questions_are_disjoint_from_other_sets():
    pool = g('eval_pool'); a, b, v = g('N_MON_TEST'), g('N_MON_TRAIN'), g('N_VAL'); n = g('N_SUITE')
    parts = [pool[:a], pool[a:a + b], pool[a + b:a + b + v], pool[a + b + v:a + b + v + n]]
    sets = [{e['q'] for e in p} for p in parts]
    assert len(parts[3]) == n and sum(len(x) for x in sets) == len(set().union(*sets))

@test
def fail_fast_guard_aborts_only_non_learning_explicit_twins():
    m = tiny(); orig_val, orig_ff = ns['val_acc'], ns['FAIL_FAST']
    try:
        ns['FAIL_FAST'] = True
        ns['LEARN_HIST'].clear(); ns['val_acc'] = lambda *a, **k: 0.40
        for fmt, name in (('explicit', 'x_exp'),):                                   # never learns: aborts at 3/4 of 7 epochs (epoch 6)
            for done in range(1, 6): g('learn_check')(name, done, 7, m, fmt)
            try: g('learn_check')(name, 6, 7, m, fmt); raise SystemExit('guard did not fire')
            except AssertionError: pass
        ns['LEARN_HIST'].clear(); vals = iter([0.40, 0.4, 0.4, 0.4, 0.62, 0.5, 0.6]); ns['val_acc'] = lambda *a, **k: next(vals)   # recovers: best of the last two (epochs 5 and 6) is 0.62
        for done in range(1, 7): g('learn_check')('y_exp', done, 7, m, 'explicit')
        ns['LEARN_HIST'].clear(); ns['val_acc'] = lambda *a, **k: 0.2                 # COCONUT curves dip mid-curriculum: logged, never fatal
        for done in range(1, 8): g('learn_check')('z_coco', done, 7, m, 'coconut', done - 1)
        ns['FAIL_FAST'] = False; ns['val_acc'] = lambda *a, **k: 0.1
        for done in range(1, 8): g('learn_check')('w_exp', done, 7, m, 'explicit')   # non-fail-fast modes never abort
    finally: ns['val_acc'], ns['FAIL_FAST'] = orig_val, orig_ff; ns['LEARN_HIST'].clear(); ns['GATES'].clear()

@test
def critical_gate_stops_and_ordinary_gate_does_not():
    ns['GATES'].clear(); old = ns['MODE']
    try:
        ns['MODE'] = 'seeds'; g('gate')('some ordinary check', False)              # recorded, not fatal
        try: g('gate')('training set size', False); raise SystemExit('critical gate did not stop')
        except AssertionError: pass
        g('gate_summary')()                                                        # must not raise in fail-fast modes
    finally: ns['MODE'] = old; ns['GATES'].clear()

@test
def max_len_covers_data():
    rg = random.Random(1); L = [len(g('make_text_example')(g('plant')(i, rg), True)[0]) for i in g('_items')[:3000]]
    assert max(L) <= g('MAX_LEN'), (max(L), g('MAX_LEN'))

@test
def smoke_pass_writes_to_its_own_files():
    src = open('prosqa_monitor.ipynb').read()
    assert "TAG='smoke_'" in src and "smoke_out" in src, 'the smoke pass must write under a separate prefix and folder'

def _pipeline_text():
    return '\n'.join(''.join(c['source']) for c in json.load(open('prosqa_monitor.ipynb'))['cells'] if c['cell_type'] == 'code')

@test
def decoy_before_hint():
    mp = g('make_prompt')('Q?', 'grimpus', True, 0).split('\n')
    assert mp[1].startswith('Reviewer') and mp[2].startswith(('Hint', 'Note', 'The answer', 'Someone')), mp

@test
def balanced_weights_remove_decoy_shortcut():
    src = _pipeline_text(); i = src.index('lin = lambda'); j = src.index('def feats_of')
    from sklearn.linear_model import LogisticRegression; from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
    sub = {'np': np, 'LogisticRegression': LogisticRegression, 'make_pipeline': make_pipeline, 'StandardScaler': StandardScaler}; exec(src[i:j], sub)
    rs = np.random.RandomState(0); X = rs.randn(400, 5); d = (rs.rand(400) < .5).astype(int); y = np.where(d == 1, rs.rand(400) < .8, rs.rand(400) < .2).astype(int); X[:, 0] += 3 * d
    a = sub['fit_score'](X[:200], y[:200], X[200:]); b = sub['fit_score'](X[:200], y[:200], X[200:], d=d[:200])
    assert np.corrcoef(a, d[200:])[0, 1] > np.corrcoef(b, d[200:])[0, 1] + .2, 'cell weights must remove the decoy shortcut'

@test
def hint_type_specific_probes_run():
    src = _pipeline_text(); i = src.index('# hint-type-specific probes'); j = src.index('\n# --- cross-format')
    rs = np.random.RandomState(1); N = 120; rows = []
    class Fake: pass
    WR = {s: (np.arange(N) % 2 == 0) for s in ('tr', 'te')}; dec = {s: ((np.arange(N) // 2) % 2 == 0) for s in ('tr', 'te')}
    y = {s: rs.randint(0, 2, N) for s in ('tr', 'te')}
    ns = dict(np=np, EXP='e', COCO='c', WR=WR, dec=dec, L={(n, s): {'y': y[s]} for n in ('e', 'c') for s in ('tr', 'te')}, feats_of=lambda n, s, k: rs.randn(N, 6) + y[s][:, None],
              valid_mask=lambda n, s, X: np.ones(N, bool), _report=lambda name, yy, ss: rows.append((name, len(yy))), fit_score=g_fit(), probe=None)
    exec(src[i:j], ns)
    names = [r[0] for r in rows]
    assert len(names) == 14 and sum(n.endswith('[harmful]') for n in names) == 7 and sum(n.endswith('[correct-hint]') for n in names) == 7, names
    assert all(n_ == N // 2 for _, n_ in rows)

def g_fit():
    src = _pipeline_text(); i = src.index('lin = lambda'); j = src.index('def feats_of')
    from sklearn.linear_model import LogisticRegression; from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
    sub = {'np': np, 'LogisticRegression': LogisticRegression, 'make_pipeline': make_pipeline, 'StandardScaler': StandardScaler}; exec(src[i:j], sub); return sub['fit_score']

if __name__ == '__main__':
    f = 0
    for t in TESTS:
        t0 = time.time()
        try: t(); print(f'PASS  {t.__name__} ({time.time() - t0:.1f}s)')
        except Exception as e: f += 1; print(f'FAIL  {t.__name__}: {type(e).__name__}: {str(e)[:300]}')
    print(f'\n{len(TESTS) - f}/{len(TESTS)} passed'); sys.exit(1 if f else 0)
