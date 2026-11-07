import json, os, nbformat as nbf
HERE = os.path.dirname(os.path.abspath(__file__))
src = [''.join(c['source']) for c in json.load(open(f'{HERE}/notebooks/prosqa_monitor.ipynb'))['cells'] if c['cell_type'] == 'code']
KEEP = ['import os, sys, subprocess', 'SEED_OVERRIDE = None', 'LOG_TAG = ', 'def find_data_dir', 'def enc(s)', 'ANS_RE = re.compile']
cells = [nbf.v4.new_markdown_cell('# ProsQA T4 profile\nPeak GPU memory, step time, CPU/RAM/GPU utilisation per (stage, micro-batch), plus generation speed, at the real ProsQA sequence lengths. Uses the experiment\'s own definitions.')]
for s in src:
    if any(k in s for k in KEEP) and 'RUN_SMOKE_FIRST' not in s: cells.append(nbf.v4.new_code_cell(s))
PROFILE = '''
import pandas as pd
assert dev == 'cuda', 'profile on the T4'
TOTAL = torch.cuda.get_device_properties(0).total_memory / 2**30; BUDGET = 0.90 * TOTAL; sync = torch.cuda.synchronize
model = new_model(); model.train(); opt = torch.optim.AdamW(model.parameters(), lr=1e-5); scaler = torch.amp.GradScaler()
def step(fwd, b):
    with torch.autocast('cuda', dtype=torch.float16): loss = fwd(model, to_dev(b))
    scaler.scale(loss).backward(); scaler.unscale_(opt); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
def util():
    try: return int(subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5).stdout.strip())
    except Exception: return -1
def worst_text(bs):
    ids = torch.randint(0, 50000, (bs, MAX_LEN)); lab = ids.clone(); lab[:, :MAX_LEN - 80] = -100; return ids, torch.ones_like(ids), lab
def worst_coco(bs, n):
    pre, post = MAX_LEN - n - 40, 40
    return collate_coco([dict(pre=torch.randint(0, 5000, (pre,)).tolist(), n=n, post=torch.randint(0, 5000, (post,)).tolist(), plabels=[-100] + torch.randint(0, 5000, (post - 1,)).tolist()) for _ in range(bs)])
def typical(make, collate, bs):
    pool = [x for x in (make(e) for e in random.sample(train, 6 * bs)) if fits(x)]; return collate(pool[:bs])
rows = []
def measure(name, fwd, mem_b, time_b, bs, reps=3):
    gc.collect(); torch.cuda.empty_cache(); row = dict(config=name, bs=bs)
    try:
        torch.cuda.reset_peak_memory_stats(); step(fwd, mem_b); sync(); row['peak_gib'] = torch.cuda.max_memory_allocated() / 2**30
        step(fwd, time_b); sync(); psutil.cpu_percent(); t = time.time(); u = []
        for _ in range(reps): step(fwd, time_b); u.append(util())
        sync(); dt = (time.time() - t) / reps
        row.update(sec_per_step=dt, samples_per_s=bs / dt, gpu_util=float(np.mean(u)), cpu_pct=psutil.cpu_percent(), ram_gib=psutil.virtual_memory().used / 2**30)
    except torch.OutOfMemoryError:
        opt.zero_grad(set_to_none=True); row['note'] = 'OOM'
    rows.append(row); log(str(row)); return 'note' not in row
def sweep(name, fwd, mem_fn, time_fn, bss):
    for bs in bss:
        if not measure(name, fwd, mem_fn(bs), time_fn(bs), bs): break
step(text_fwd, worst_text(2))
sweep('text', text_fwd, worst_text, lambda bs: typical(make_text_example_t, collate_text, bs), (1, 2, 4, 8, 16, 32))
''' 
PROFILE = PROFILE.replace('make_text_example_t', 'lambda e: make_text_example(e, True)')
PROFILE += '''
for stage in range(N_STAGES + 1):
    n = stage * C_PER_STEP
    sweep(f'coconut stage {stage} ({n} latents)', coco_fwd, lambda bs, n=n: worst_coco(bs, n), lambda bs, s=stage: typical(lambda e: make_coco_example(e, s, True), collate_coco, bs), (1, 2, 4, 8, 16))
df = pd.DataFrame(rows); display(df.round(3)); df.to_csv('profile.csv', index=False)

# generation speed (untrained model emits max-length output, so this is an upper bound on time)
del model, opt; gc.collect(); torch.cuda.empty_cache()
pe, pc = new_model().eval(), new_model().eval(); ps = [make_prompt(e['q'], e['hint'], e['decoy']) for e in train[:32]]
sync(); t = time.time(); explicit_run(pe, ps, feats=True, bs=16); sync(); sec_e = (time.time() - t) / 32
t = time.time(); coconut_run(pc, ps, feats=True, bs=16); sync(); sec_c = (time.time() - t) / 32
log(f'generation: explicit {sec_e:.3f}s/prompt, coconut {sec_c:.3f}s/prompt')

def best(prefix):
    d = df[df.config.str.startswith(prefix) & df.peak_gib.notna() & (df.peak_gib <= BUDGET)]
    return None if d.empty else d.sort_values('samples_per_s').iloc[-1]
def minutes(n_train, epochs_stage):
    et = epochs_stage * (N_STAGES + 1); b = best('text'); t_exp = et * n_train / b.samples_per_s / 60
    t_coco = sum(epochs_stage * n_train / best(f'coconut stage {s} (').samples_per_s / 60 for s in range(N_STAGES + 1))
    return t_exp, t_coco
plan = dict(text_bs=int(best('text').bs), coco_bs={s: int(best(f'coconut stage {s} (').bs) for s in range(N_STAGES + 1)}, budget_gib=round(BUDGET, 1))
for nm, nt, ep, nm_, ntest, nb in [('controlled', 2000, 1, 500, 800, 500), ('full', 6000, 2, 800, 1000, 1000)]:
    te, tc = minutes(nt, ep); gen = 2 * (nm_ + ntest) * 2 * (sec_e + sec_c) / 2 / 60
    plan[nm] = dict(explicit_each_min=round(te), coconut_each_min=round(tc), generation_min=round(gen), total_min=round(2 * te + 2 * tc + gen))
print(json.dumps(plan, indent=1)); json.dump(plan, open('profile.json', 'w'), indent=1)
'''
cells.append(nbf.v4.new_code_cell(PROFILE.strip('\n')))
nb = nbf.v4.new_notebook(cells=cells); nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
for i, c in enumerate(nb.cells): c['id'] = f'cell{i:03d}'   # deterministic ids: rebuilding gives a byte-identical notebook
nbf.write(nb, f'{HERE}/notebooks/prosqa_profile.ipynb'); print(len(cells), 'cells')
