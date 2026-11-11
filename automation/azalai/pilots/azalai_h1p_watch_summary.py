#!/usr/bin/env python3
"""azalai_h1p_watch_summary.py (11 Nov 2026, A100 session) - H1 paraphrase test (AZ-3), design Section 3: summary of the watcher
log (ollama_clients.log): every 2 s the pids with an established connection to the ollama port, and every 30 s `ollama ps`.
Reports the samples with no client, one client and more than one client; each pid's first and last sample (to be matched with
the runs in chain.log: one ground process per run); and whether the model was always wholly on the GPU (size_vram == size) with
the v1 digest.
usage: python3 azalai_h1p_watch_summary.py <ollama_clients.log> <chain.log> <out.json>"""
import sys, json, re
from collections import Counter
log, chain, out = sys.argv[1:4]
samples = []; ps = []
for line in open(log):
    line = line.rstrip('\n')
    if ' PS ' in line:
        t, js = line.split(' PS ', 1)
        try:
            ps.append((t, json.loads(js)))
        except ValueError:
            ps.append((t, None))
        continue
    if not re.match(r'^\d\d:\d\d:\d\d', line):
        continue
    t = line[:8]; pids = re.findall(r'pid=(\d+)', line); samples.append((t, sorted(set(pids))))
n = Counter(min(len(p), 2) for _, p in samples)
span = {}
for t, p in samples:
    for x in p:
        span.setdefault(x, [t, t, 0]); span[x][1] = t; span[x][2] += 1
multi = [(t, p) for t, p in samples if len(p) > 1]
models = Counter(); onvram = Counter()
for t, j in ps:
    if not j:
        models['(no reply)'] += 1; continue
    for m in j.get('models', []):
        models[(m.get('name'), m.get('digest', '')[:12])] += 1; onvram[m.get('size_vram') == m.get('size')] += 1
    if not j.get('models'):
        models['(none loaded)'] += 1
runs = [l.strip() for l in open(chain) if l.startswith('[chain]')]
res = dict(samples=len(samples), no_client=n[0], one_client=n[1], more_than_one_client=n[2], multi_client_samples=multi[:50],
           pids={k: dict(first=v[0], last=v[1], samples=v[2]) for k, v in span.items()}, ps_samples=len(ps),
           models={str(k): v for k, v in models.items()}, wholly_on_gpu={str(k): v for k, v in onvram.items()}, chain=runs)
json.dump(res, open(out, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'chain'}, indent=1)); print('\n'.join(runs))
