#!/usr/bin/env python3
"""paper_scope_numbers.py (2026-11-10) — the HiGH letter's evaluation scope from numbers/e1_v3.json without the two phone walks
(Paul, draft 36: "we dont use phone walks in this paper"): cells, frames, tree and row heads (census + held-out) and held-out heads
over the eight robot surveys (CitrusFarm 01-05, Klapmuts December and April, Lindendhof), written to numbers/e1_scope.json.
system python3:  paper_scope_numbers.py <paper dir>"""
import json, sys
from pathlib import Path
P = Path(sys.argv[1]); e = json.load(open(P / 'numbers' / 'e1_v3.json')); s = e['surveys']
keep = [k for k in s if not k.startswith('IMG_')]
cells = sum(s[k]['cells'] for k in keep); frames = sum(s[k]['frames'] for k in keep)
heads = sum(s[k][L][sp]['n'] for k in keep for L in ('TREE', 'ROW') if L in s[k] for sp in ('train', 'eval') if sp in s[k][L])
held = sum(s[k][L]['eval']['n'] for k in keep for L in ('TREE', 'ROW') if L in s[k])
out = dict(surveys=keep, n_surveys=len(keep), cells=cells, frames=frames, heads_tree_row=heads, heads_held_out=held, dropped=[k for k in s if k not in keep])
json.dump(out, open(P / 'numbers' / 'e1_scope.json', 'w'), indent=1); print(out)
