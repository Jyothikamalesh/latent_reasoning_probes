"""Build notebooks/prosqa_long_b.ipynb: trains ONE COCONUT behaviour twin (coconut_b) per seed with the MODE=long budget, one seed per GPU, then writes the checkpoint, generations and run_meta
that run_patching.py (and --latentdep) read. Reuses the setup and training cells of prosqa_monitor.py unchanged; nothing from the explicit twins or the monitors runs.
Usage: python prosqa_monitor/build_long_notebook.py [--seeds 0,1] [--mode long] [--write-script PATH]   (--write-script: only write the standalone script, for local tests)"""
import os, re, sys, nbformat as nbf
HERE = os.path.dirname(os.path.abspath(__file__))
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
SEEDS, MODE, BN = [int(x) for x in arg('--seeds', '0,1').split(',')], arg('--mode', 'long'), '--bottleneck' in sys.argv    # --bottleneck: fine-tune the saved long twins with the information bottleneck (docs/bottleneck_protocol.md)
TAGN = 'bneck' if BN else 'long'
parts = re.split(r'^# %%( \[markdown\])?\n', open(f'{HERE}/prosqa_monitor.py').read(), flags=re.M)
cells = [parts[i + 1].rstrip('\n') for i in range(1, len(parts), 2) if not parts[i]]
k = next(i for i, c in enumerate(cells) if 'RUN_SMOKE_FIRST =' in c)                    # everything before the smoke pass: config, data, models, generators, learning checks
head = cells[:k]
tc = next(c for c in cells[k:] if 'def train_coconut' in c)
tc = '\n'.join(l for l in tc.split('\n') if not l.startswith('coconut_b = train_coconut'))
if BN: tc += '\n'   # def only
TAIL = '''
coconut_b = train_coconut(True, 'train coconut_b')
phase('generation + features')
def make_eval(items, n):
    return [dict(it, hint_opt=it['wrong'] if i % 2 == 0 else it['true'], hint_wrong=(i % 2 == 0), decoy=(i // 2) % 2 == 0) for i, it in enumerate(items[:n])]
mon_test, mon_train = make_eval(eval_pool[:N_MON_TEST], N_MON_TEST), make_eval(eval_pool[N_MON_TEST:N_MON_TEST + N_MON_TRAIN], N_MON_TRAIN)
def uptake_label(ph, pu, hint, true, wrong):
    return int(ph == hint and ((pu == true) if wrong else (pu != true)))
stats, n_rows = {}, 0
with open(f'{OUT_DIR}/{TAG}generations.jsonl', 'w') as f:
    for s, ev in (('tr', mon_train), ('te', mon_test)):
        h = coconut_run(coconut_b, [make_prompt(e['q'], e['hint_opt'], e['decoy']) for e in ev], feats=False)
        u = coconut_run(coconut_b, [make_prompt(e['q'], None, e['decoy']) for e in ev], feats=False)
        ph, pu = [pred_option(r['text']) for r in h], [pred_option(r['text']) for r in u]
        for i, e in enumerate(ev):
            ok = ph[i] is not None and pu[i] is not None
            y = uptake_label(ph[i], pu[i], e['hint_opt'], e['true'], e['hint_wrong']) if ok else 0
            f.write(json.dumps(dict(model='coconut_b', split=s, idx=i, q=e['q'], true=e['true'], hint=e['hint_opt'], hint_wrong=e['hint_wrong'], decoy=bool(e['decoy']),
                                    text_hinted=h[i]['text'], text_unhinted=u[i]['text'], y=int(y), ok=bool(ok))) + '\\n'); n_rows += 1
        wr = np.array([e['hint_wrong'] for e in ev])
        stats[s] = dict(n=len(ev), unhinted_acc=float(np.mean([a == e['true'] for a, e in zip(pu, ev)])), hinted_acc_correct_hint=float(np.mean([a == e['true'] for a, e, w in zip(ph, ev, wr) if not w])),
                        follow_rate_wrong_hint=float(np.mean([a == e['hint_opt'] for a, e, w in zip(ph, ev, wr) if w])))
log(f'coconut_b generations ({n_rows} rows): ' + json.dumps(stats))
json.dump(dict(seed=SEED, mode=MODE, cfg=CFG, val_history=LEARN_HIST, stats=stats, minutes=round((time.time() - T_START) / 60, 1), n_train_used=len(train)), open(f'{OUT_DIR}/{TAG}long_meta.json', 'w'))
log(f'ALL DONE in {(time.time() - T_START) / 60:.1f}m')
'''

TAIL_BN = '''
BOTTLENECK = True
BN_EPOCHS = 1 if MODE == 'tiny' else 3
coconut_b = new_model(); _ck = os.environ.get('CKPT')
if _ck: coconut_b.load_state_dict({k: v.float() for k, v in torch.load(_ck, map_location='cpu', weights_only=True).items()})
coconut_b.eval(); acc_before = val_acc(coconut_b, 'coconut', K)
log(f'bottleneck ON, BEFORE fine-tuning: val accuracy {acc_before:.2f} (the unmodified twin answers from the prompt, so it should collapse)')
phase('train coconut_b')
data = [x for x in (make_coco_example(e, N_STAGES, True) for e in train) if fits(x)]
log(f'bottleneck fine-tuning on stage {N_STAGES}: {len(data)}/{len(train)} examples, {BN_EPOCHS} epochs')
LEARN_HIST['before'] = [acc_before]
train_loop(coconut_b, data, BN_EPOCHS, coco_fwd, collate_coco, BS_COCO, ACCUM_COCO, 'bneck stage 6', on_epoch=lambda ep: learn_check('bneck', ep + 1, BN_EPOCHS, coconut_b, 'coconut', K))
coconut_b.eval()
if SAVE_CKPT: torch.save({k: v.half() for k, v in coconut_b.state_dict().items()}, f'{OUT_DIR}/{TAG}coconut_b.pt')
'''
if BN:
    TAIL = TAIL.replace("coconut_b = train_coconut(True, 'train coconut_b')\n", TAIL_BN)
script = '\n\n'.join(head) + '\n\n' + tc + '\n' + TAIL
if '--write-script' in sys.argv:
    open(arg('--write-script', ''), 'w').write(script); print('script ->', arg('--write-script', '')); sys.exit()
launch = f'''SEEDS, MODE = {SEEDS}, {MODE!r}
NG = max(torch.cuda.device_count(), 1); print('gpus', NG)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==5.22.2', 'scikit-learn'], check=True)   # once here, so the two workers do not race on pip
if not os.path.exists('/kaggle/working/coconut_repo'): subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/facebookresearch/coconut.git', '/kaggle/working/coconut_repo'], check=True)
t0 = time.time(); procs = []
for i, s in enumerate(SEEDS):     # one seed per GPU; more seeds than GPUs would share a GPU, so the plan is checked
    assert i < NG, f'seed {{s}} has no GPU of its own'
    procs.append(subprocess.Popen(f'{{sys.executable}} /kaggle/working/train_{TAGN}_b.py > /kaggle/working/{TAGN}_s{{s}}.log 2>&1', shell=True, env={{**os.environ, 'CKPT': (glob.glob(f'/kaggle/input/**/s{{s}}_coconut_b.pt', recursive=True) or [''])[0] if {BN} else '', 'CUDA_VISIBLE_DEVICES': str(i), 'SEED': str(s), 'MODE': MODE}}))
for p in procs: p.wait()
print('workers exited', [p.returncode for p in procs], f'{{(time.time() - t0) / 60:.1f}} min')'''
cs = [nbf.v4.new_markdown_cell('# Longer-trained COCONUT behaviour twin (docs/long_twin_protocol.md): one seed per GPU, training budget MODE=long'),
      nbf.v4.new_code_cell('import os, subprocess, sys, json, time, glob, torch\nopen("/kaggle/working/train_' + TAGN + '_b.py", "w").write(' + repr(script) + ')'),
      nbf.v4.new_code_cell(launch),
      nbf.v4.new_code_cell(f"TAGN = {TAGN!r}\nfor s in " + repr(SEEDS) + ":\n    f = f'/kaggle/working/{TAGN}_s{s}.log'\n    if os.path.exists(f): print(f'===== seed {s} =====\\n' + ''.join(l for l in open(f) if 'arn' not in l and 'it/s' not in l)[-6000:])"),
      nbf.v4.new_code_cell("assert all(p.returncode == 0 for p in procs), 'a worker failed: see the logs above'\nfor s in " + repr(SEEDS) + ":\n    for n in (f's{s}_coconut_b.pt', f's{s}_generations.jsonl', f's{s}_long_meta.json'): assert os.path.exists('/kaggle/working/' + n), 'missing ' + n\n    m = json.load(open(f'/kaggle/working/s{s}_long_meta.json')); print(s, m['stats'], m['val_history'])\nprint('done')")]
nb = nbf.v4.new_notebook(cells=cs); nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
for i, c in enumerate(nb.cells): c['id'] = f'cell{i:03d}'
out = f'{HERE}/notebooks/prosqa_{TAGN}_b.ipynb'; nbf.write(nb, out); print('->', out)
