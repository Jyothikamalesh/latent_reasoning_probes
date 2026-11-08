"""Build notebooks/causal_patching.ipynb: embeds latent_patching.py, run_patching.py and the unit tests, runs the tests, then patches seeds 0/1/2 on two GPUs (one worker process per GPU).
Inputs come from the Kaggle dataset DATASET (s{0,1,2}_coconut_b.pt and s{0,1,2}_generations.jsonl). Usage: python build_patch_notebook.py [--dataset user/slug] [--limit N] [--boot N]"""
import os, sys, nbformat as nbf
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
DATASET, LIMIT, BOOT = arg('--dataset', 'jyothikamaleshs/prosqa-coconut-b-twins'), int(arg('--limit', 0)), int(arg('--boot', 2000))
read = lambda p: open(p).read()
w = lambda name, text: f"with open({name!r}, 'w') as f: f.write({text!r})"
cells = [
 nbf.v4.new_markdown_cell('# Causal latent-restoration patching on the frozen COCONUT twins (protocol: docs/causal_protocol.md, Amendments 1-3)'),
 nbf.v4.new_code_cell("import os, subprocess, sys, json, time, glob, torch, transformers\n"
   "print('torch', torch.__version__, 'transformers', transformers.__version__, 'gpus', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])\n"
   "if not os.path.exists('/kaggle/working/coconut_repo'): subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/facebookresearch/coconut.git', '/kaggle/working/coconut_repo'], check=True)\n"
   "DATA = '/kaggle/working/coconut_repo/data'; assert os.path.exists(DATA + '/prosqa_test.json')\n"
   "INP = glob.glob('/kaggle/input/*/s0_coconut_b.pt'); assert INP, os.listdir('/kaggle/input'); INP = os.path.dirname(INP[0]); print('inputs', INP, sorted(os.listdir(INP)))"),
 nbf.v4.new_code_cell(w('/kaggle/working/latent_patching.py', read(f'{HERE}/latent_patching.py')) + '\n' + w('/kaggle/working/run_patching.py', read(f'{HERE}/run_patching.py')) + '\n'
   + "os.makedirs('/kaggle/working/tests', exist_ok=True)\n" + w('/kaggle/working/tests/test_latent_patching.py', read(f'{ROOT}/tests/test_latent_patching.py').replace("os.path.join(os.path.dirname(__file__), '..', 'analysis')", "'/kaggle/working'"))),
 nbf.v4.new_code_cell("r = subprocess.run([sys.executable, '/kaggle/working/tests/test_latent_patching.py'], capture_output=True, text=True); print(r.stdout[-1500:], r.stderr[-1500:])\nassert r.returncode == 0, 'unit tests failed on this transformers/torch version: stop'"),
 nbf.v4.new_code_cell("# two workers, one per GPU: GPU 0 takes seeds 0 and 2, GPU 1 takes seed 1 (seeds are independent; each worker sees a single device)\n"
   f"LIMIT, BOOT = {LIMIT}, {BOOT}\n"
   "PLAN = {0: [0, 2], 1: [1]} if torch.cuda.device_count() >= 2 else {0: [0, 1, 2]}\n"
   "RUN_ID = str(int(time.time()))\n"
   "import shutil\n"
   "for s in (0, 1, 2): shutil.rmtree(f'/kaggle/working/patching_s{s}', ignore_errors=True)\n"
   "def worker(gpu, seeds):\n"
   "    cmds = ' && '.join(f\"{sys.executable} /kaggle/working/run_patching.py --ckpt {INP}/s{s}_coconut_b.pt --gens {INP}/s{s}_generations.jsonl --out /kaggle/working/patching_s{s} --device cuda --seed {s} --boot {BOOT} --bs 32 --run-id {RUN_ID}\" + (f' --limit {LIMIT}' if LIMIT else '') + f' > /kaggle/working/patching_s{s}.log 2>&1' for s in seeds)\n"
   "    return subprocess.Popen(['bash', '-c', cmds], env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PROSQA_DIR': DATA})\n"
   "t0 = time.time(); procs = [worker(g, s) for g, s in PLAN.items()]\n"
   "for p in procs: p.wait()\n"
   "print('workers exited', [p.returncode for p in procs], f'{(time.time() - t0) / 60:.1f} min')"),
 nbf.v4.new_code_cell("for s in (0, 1, 2):\n    f = f'/kaggle/working/patching_s{s}.log'\n    if os.path.exists(f): print(f'===== seed {s} =====\\n' + ''.join(l for l in open(f) if 'arn' not in l)[-5000:])"),
 nbf.v4.new_code_cell("assert all(p.returncode == 0 for p in procs), 'a worker failed: see the logs above'\n"
   "for s in (0, 1, 2):\n"
   "    f = f'/kaggle/working/patching_s{s}/patching_summary.json'; assert os.path.exists(f), f'missing {f}'\n"
   "    assert json.load(open(f))['meta']['run_id'] == RUN_ID, f'seed {s} summary is not from this run'\n"
   "print('all three summaries present and from this run', RUN_ID)"),
]
nb = nbf.v4.new_notebook(cells=cells); nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
for i, c in enumerate(nb.cells): c['id'] = f'cell{i:03d}'
out = f'{ROOT}/notebooks/causal_patching.ipynb'; nbf.write(nb, out); print('->', out)
