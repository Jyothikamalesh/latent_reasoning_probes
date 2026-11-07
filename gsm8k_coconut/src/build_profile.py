import json, nbformat as nbf

import os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks")
src = json.load(open(f"{ROOT}/coconut.ipynb"))["cells"]
KEEP = ["import os, sys, subprocess", "SMOKE = dev", "tok = AutoTokenizer", "def enc(s)", "def coconut_forward", "def parse_answer"]
cells = [nbf.v4.new_markdown_cell("# T4 profile for the COCONUT monitorability run\nReuses the experiment's own definitions, then measures peak memory and step time per (stage, batch size), and extrapolates total runtime.")]
for c in src:
    s = "".join(c["source"])
    if c["cell_type"] == "code" and any(k in s for k in KEEP) and "RUN_SMOKE_FIRST" not in s:
        cells.append(nbf.v4.new_code_cell(s))

PROFILE = '''
import pandas as pd, gc
assert dev == 'cuda', 'profile on the T4'
TOTAL_GIB = torch.cuda.get_device_properties(0).total_memory / 2**30
BUDGET = 0.90 * TOTAL_GIB
sync = torch.cuda.synchronize
model = new_model(); model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
scaler = torch.amp.GradScaler()

def step(fwd, ids, att, lab):
    with torch.autocast('cuda', dtype=torch.float16):
        loss = fwd(model, ids, att, lab)
    scaler.scale(loss).backward()
    scaler.unscale_(opt); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

def measure(fwd, mem_batch, time_batch, reps=3):
    """peak GiB on the worst-case batch, seconds/step on a typical batch. None on OOM."""
    gc.collect(); torch.cuda.empty_cache()
    try:
        torch.cuda.reset_peak_memory_stats()
        step(fwd, *[t.to(dev) for t in mem_batch]); sync()
        peak = torch.cuda.max_memory_allocated() / 2**30
        step(fwd, *[t.to(dev) for t in time_batch]); sync()          # warm-up on typical shapes
        t = time.time()
        for _ in range(reps): step(fwd, *[x.to(dev) for x in time_batch])
        sync(); return peak, (time.time() - t) / reps, ''
    except torch.OutOfMemoryError:
        opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache(); return None, None, 'OOM'
    except Exception as e:
        opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache(); return None, None, repr(e)[:80]

def worst_coco(bs, n_lat):
    ids = torch.randint(0, 50000, (bs, MAX_LEN)); att = torch.ones_like(ids)
    ids[:, 20] = BOT; ids[:, 21:21 + n_lat] = LAT; ids[:, 21 + n_lat] = EOT
    lab = ids.clone(); lab[:, :22 + n_lat] = -100
    return ids, att, lab

def typical(make, bs):
    pool = [make(e) for e in random.sample(train, 4 * bs)]
    return collate([p for p in pool if len(p[0]) <= MAX_LEN][:bs])

def worst_text(bs):
    ids = torch.randint(0, 50000, (bs, MAX_LEN)); lab = ids.clone(); lab[:, :60] = -100
    return ids, torch.ones_like(ids), lab

step(text_fwd, *[t.to(dev) for t in worst_text(2)])      # create optimizer state before measuring
rows = []
def sweep(name, fwd, mem_fn, time_fn, bss=(1, 2, 4, 8, 16, 32, 64)):
    for bs in bss:
        peak, sec, err = measure(fwd, mem_fn(bs), time_fn(bs))
        rows.append(dict(config=name, bs=bs, peak_gib=peak, sec_per_step=sec, samples_per_s=(bs / sec if sec else None), note=err))
        print(rows[-1], flush=True)
        if err: break

sweep('text', text_fwd, worst_text, lambda bs: typical(make_text_example, bs))
for stage in range(N_STAGES + 1):
    n_lat = stage * C_PER_STEP
    sweep(f'coconut stage {stage} ({n_lat} latents)', coco_fwd, lambda bs, n=n_lat: worst_coco(bs, n),
          lambda bs, s=stage: typical(lambda e: make_coco_example(e, s), bs), bss=(1, 2, 4, 8, 16, 32))
model.gradient_checkpointing_enable()
sweep('coconut stage 3 + grad checkpointing', coco_fwd, lambda bs: worst_coco(bs, K),
      lambda bs: typical(lambda e: make_coco_example(e, N_STAGES), bs), bss=(4, 8, 16, 32))
model.gradient_checkpointing_disable()

df = pd.DataFrame(rows); display(df.round(3)); df.to_csv('profile.csv', index=False)
'''

SUMMARY = '''
# generation speed (untrained models emit max-length garbage, so this is an upper bound)
text_model, coco_model = new_model().eval(), new_model().eval()
t = time.time(); gen_text(mon_train[:32]); sync(); sec_text = (time.time() - t) / 32
t = time.time(); gen_coconut(mon_train[:8]); sync(); sec_coco = (time.time() - t) / 8
n_gen = N_MON_TRAIN + N_MON_TEST
gen_min = n_gen * (sec_text + sec_coco) / 60

n_eff = len([e for e in train if len(make_text_example(e)[0]) <= MAX_LEN])
def best(prefix, max_bs=None):
    d = df[df.config.str.startswith(prefix) & df.peak_gib.notna() & (df.peak_gib <= BUDGET)]
    return None if d.empty else d.sort_values('samples_per_s').iloc[-1]
plan, total_min = {}, 0.0
b = best('text'); plan['text_bs'] = int(b.bs); text_min = EPOCHS_TEXT * n_eff / b.samples_per_s / 60; total_min += text_min
coco_min = 0.0
for stage in range(N_STAGES + 1):
    b = best(f'coconut stage {stage} ('); plan[f'coco_stage{stage}_bs'] = int(b.bs)
    coco_min += EPOCHS_PER_STAGE * n_eff / b.samples_per_s / 60
total_min += coco_min + gen_min
plan.update(budget_gib=round(BUDGET, 1), est_minutes=dict(text=round(text_min), coconut=round(coco_min), generation=round(gen_min), total=round(total_min)))
print(json.dumps(plan, indent=1)); json.dump(plan, open('profile.json', 'w'), indent=1)
print(f'\\nEstimated full run on this T4: ~{total_min:.0f} min (+ ~{(total_min * 0.1):.0f} min for the smoke pass, load, monitors)')
'''
cells.append(nbf.v4.new_code_cell(PROFILE.strip("\n")))
cells.append(nbf.v4.new_code_cell(SUMMARY.strip("\n")))
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
nbf.write(nb, f"{ROOT}/profile_t4.ipynb")
print(len(cells), "cells")
