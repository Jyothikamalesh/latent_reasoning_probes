"""Convert prosqa_monitor.py (# %% cells) into notebooks/prosqa_monitor.ipynb. Usage: python prosqa_monitor/build_notebook.py [--seed N]"""
import json, os, re, sys, nbformat as nbf
HERE = os.path.dirname(os.path.abspath(__file__))
seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else None
src = open(f'{HERE}/prosqa_monitor.py').read()
parts = re.split(r'^# %%( \[markdown\])?\n', src, flags=re.M)
cells = []
for i in range(1, len(parts), 2):
    md, body = bool(parts[i]), parts[i + 1].rstrip('\n')
    if md: cells.append(nbf.v4.new_markdown_cell('\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.split('\n'))))
    else:
        if seed is not None: body = body.replace('SEED_OVERRIDE = None', f'SEED_OVERRIDE = {seed}', 1)
        cells.append(nbf.v4.new_code_cell(body))
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
out = f'{HERE}/notebooks/' + ('prosqa_monitor.ipynb' if seed is None else f'prosqa_monitor_seed{seed}.ipynb')
for i, c in enumerate(nb.cells): c['id'] = f'cell{i:03d}'   # deterministic ids: rebuilding gives a byte-identical notebook
nbf.write(nb, out); print(len(cells), 'cells ->', out)
