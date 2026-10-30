#!/usr/bin/env python3
"""azalai_h1_ground.py (29 Oct 2026) — H1 pilot: ground the frozen instruction bank with a local LLM (ollama, Gemma 4 12B) in two
conditions. 'twin': the registry with its attributes (row, order, height, growth, status, fruit counts) is in the prompt, so an
instruction can be resolved by identity, count or change as well as by number. 'enumerated': the same LLM with only the enumerated
scene graph (rows and numbered bags, tree names), the ICRA 2026 baseline's addressing. One call per instruction, JSON out, temperature
0; scored against instruction_bank_v0_gold.json over the resolvable instructions. Usage (box):
  python3 azalai_h1_ground.py <dir with instruction_bank_v0.csv, registry_v0.json, instruction_bank_v0_gold.json> <out.json> [--model M]"""
import sys, json, csv, time, argparse, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument('d'); ap.add_argument('out'); ap.add_argument('--model', default='hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M'); ap.add_argument('--url', default='http://localhost:11434/api/chat'); a = ap.parse_args()
bank = list(csv.DictReader(open(f'{a.d}/instruction_bank_v0.csv'))); R = json.load(open(f'{a.d}/registry_v0.json')); G = json.load(open(f'{a.d}/instruction_bank_v0_gold.json'))
fn = {k: v for k, v in R['fruit']['fruitnerf']['ours'].items() if isinstance(v, (int, float))}
def table(mode):
    L = []
    for b in R['bags']:
        if mode == 'twin': L.append(f"{b['id']} r{b['row']} n{b['n_in_row']} x{b['x']} z{b['z']} h{b['height_dec_m']} dh{b['dh_apr_m']} {b['growth']} {b['status']}")
        else: L.append(f"{b['id']} r{b['row']} n{b['n_in_row']} x{b['x']} z{b['z']}")
    trees = '\n'.join(f"{t} fruit {c}" for t, c in fn.items()) if mode == 'twin' else '\n'.join(fn.keys())
    return '\n'.join(L), trees
SYS = {'twin': "You ground a farm manager's instruction for a field robot against the farm's registry. The registry is a blueberry nursery: rows (hedges) 1..%d, each bag has an integer id, its row r, its order along the row n (n1 is the first bag), position x z in metres (z increases along the row; x across the rows, larger x is the right-hand side), its December plant height h in metres, dh = April height minus December (metres), growth = grown/stable/shrunk/split/None (the April call), status = 'found again' or 'association miss' (both present in April), 'removed' (gone in April), 'added' (new in April), 'unclear', 'december false positive'. Trees of the citrus registry carry a fruit count. Resolve the instruction to the target bag ids (or tree names) in the order they should be visited. If the instruction asks for a number, put it in 'answer'. If it cannot be carried out from this registry (no such thing, an action the robot cannot do, another site, a future state, two robots, a landmark or a previous session that the registry does not hold), set reject true. Reply with JSON only: {\"targets\": [...], \"answer\": number or null, \"reject\": true/false, \"reason\": \"...\"} with the reason in at most twenty words." % R['rows'],
       'enumerated': "You ground a farm manager's instruction for a field robot against the farm's scene graph. The scene graph is a blueberry nursery: rows 1..%d, each bag has an integer id, its row r, its order along the row n (n1 is the first bag) and a position x z in metres (z increases along the row; x across the rows, larger x is the right-hand side). The citrus scene graph lists tree names. Nothing else is known about a bag or a tree. Resolve the instruction to the target bag ids (or tree names) in the order they should be visited. If the instruction asks for a number, put it in 'answer'. If it cannot be carried out from this scene graph (no such thing, an action the robot cannot do, another site, a property the graph does not hold, two robots, a landmark or a previous session), set reject true. Reply with JSON only: {\"targets\": [...], \"answer\": number or null, \"reject\": true/false, \"reason\": \"...\"} with the reason in at most twenty words." % R['rows']}
def ask(mode, ins):
    bags, trees = table(mode)
    msg = [dict(role='system', content=SYS[mode]), dict(role='user', content=f"REGISTRY BAGS (id r n x z{' h dh growth status' if mode == 'twin' else ''}):\n{bags}\n\nCITRUS TREES:\n{trees}\n\nINSTRUCTION: {ins}\n\nJSON:")]
    body = json.dumps(dict(model=a.model, messages=msg, stream=False, format='json', think=False, options=dict(temperature=0, num_ctx=40000, num_predict=900))).encode()   # think=False: Gemma 4 otherwise spends the budget in its thinking channel and returns empty content
    t0 = time.time(); r = urllib.request.urlopen(urllib.request.Request(a.url, body, {'Content-Type': 'application/json'}), timeout=900); j = json.loads(r.read())
    txt = j['message']['content']
    if not txt.strip():   # one retry without the JSON format constraint
        body = json.dumps(dict(model=a.model, messages=msg, stream=False, think=False, options=dict(temperature=0, num_ctx=40000, num_predict=900))).encode()
        r = urllib.request.urlopen(urllib.request.Request(a.url, body, {'Content-Type': 'application/json'}), timeout=900); j = json.loads(r.read()); txt = j['message']['content']
        i0, i1 = txt.find('{'), txt.rfind('}'); txt = txt[i0:i1 + 1] if i0 >= 0 and i1 > i0 else txt
    try: ans = json.loads(txt)
    except Exception: ans = dict(targets=[], answer=None, reject=False, reason='unparsable: ' + txt[:200])
    if not isinstance(ans, dict): ans = dict(targets=[], answer=None, reject=False, reason='not an object')
    return ans, time.time() - t0
def norm(t): return [str(x).strip() for x in (t if isinstance(t, list) else [t]) if x is not None and str(x).strip() != '']
def score(g, ans):
    if g.get('unresolvable'): return None
    if g['reject']: return bool(ans.get('reject'))
    if ans.get('reject'): return False
    gold = [str(x) for x in g['targets']]; got = norm(ans.get('targets', []))
    if g['category'] == 'count' and len(gold) == 1:
        v = ans.get('answer')
        try: return v is not None and abs(float(v) - float(gold[0])) < 0.5
        except Exception: return False
    if g['any']: return len(got) >= 1 and all(x in gold for x in got)
    return got == gold
res = []
for it in bank:
    g = G[it['id']]; row = dict(id=it['id'], category=it['category'], instruction=it['instruction'], gold=g)
    for mode in ('twin', 'enumerated'):
        ans, dt = ask(mode, it['instruction']); row[mode] = dict(answer=ans, seconds=round(dt, 1), correct=score(g, ans))
    res.append(row); print(f"[h1] {it['id']} twin={row['twin']['correct']} enum={row['enumerated']['correct']} ({row['twin']['seconds']}s/{row['enumerated']['seconds']}s)", flush=True)
    json.dump(res, open(a.out, 'w'), indent=1)
from collections import defaultdict
S = defaultdict(lambda: dict(n=0, twin=0, enumerated=0, unresolvable=0))
for r in res:
    c = S[r['category']]
    if r['twin']['correct'] is None: c['unresolvable'] += 1; continue
    c['n'] += 1; c['twin'] += int(bool(r['twin']['correct'])); c['enumerated'] += int(bool(r['enumerated']['correct']))
tot = dict(n=sum(c['n'] for c in S.values()), twin=sum(c['twin'] for c in S.values()), enumerated=sum(c['enumerated'] for c in S.values()), unresolvable=sum(c['unresolvable'] for c in S.values()))
json.dump(dict(model=a.model, per_category=dict(S), total=tot, results=res), open(a.out, 'w'), indent=1); print(json.dumps(dict(per_category=dict(S), total=tot), indent=1))
