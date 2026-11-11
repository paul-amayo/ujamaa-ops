#!/usr/bin/env python3
"""azalai_h1p_failures.py (11 Nov 2026) — H1 paraphrase test (AZ-3), Section 7 of the design: where the twin arm fails.
For every twin-arm record that the frozen gold scores wrong (original wording = v1 seed 0, paraphrase sets A and B) one failure
type is assigned from the stored trace, by the first rule that applies:

  1 no final   2 answered where a refusal was due   3 refusal (with: did a tool result hold the gold)   4 invented id
  5 right lookup, wrong reading   6 right tool, wrong argument   7 wrong tool (with: no tool call at all)

Descriptive, no verdict. REF below (which tools can carry the lookup of each instruction's gold rule) is the author's judgement,
written before any record was read. The trace's tool calls are replayed against a fresh twin registry to recover the whole id set
behind each result_id (the trace stores the first 10 records only).
usage: python3 azalai_h1p_failures.py <Azalai_RSS/numbers> <pilots dir> <out.json> [--dump traces.txt]"""
import sys, os, json, re
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from azalai_h1_tools import Registry

BY_ID = ('get_plant', 'find_plants'); BY_ORDER = ('get_plant_at', 'find_plants'); BY_POS = ('nearest_plants', 'find_plants'); BY_FIELD = ('find_plants',)
REF = {}
for i in ('DIR-01', 'DIR-06', 'DIR-07', 'SPA-04'): REF[i] = BY_ID                                   # the gold is an id named in the instruction
for i in ('DIR-02', 'DIR-03', 'DIR-04', 'DIR-05', 'DIR-08', 'DIR-09', 'DIR-10', 'UNA-01', 'UNA-02', 'UNA-06', 'SPA-03', 'SPA-05', 'SPA-08', 'SPA-10',
          'REG-01', 'REG-03', 'REG-04', 'REG-06', 'REG-07', 'REG-08', 'REG-09'): REF[i] = BY_ORDER   # a row and an order, a neighbour along the row, or whole rows
for i in ('UNA-05', 'UNA-07', 'SPA-01', 'SPA-06', 'SPA-09', 'REG-05', 'REG-10'): REF[i] = BY_POS      # a position (half the lane, the largest z, across the lane)
for i in ('IDE-10', 'COU-06', 'COU-09', 'CHA-01', 'CHA-02', 'CHA-03', 'CHA-04', 'CHA-06', 'CHA-07', 'CHA-08', 'CHA-09'): REF[i] = BY_FIELD   # a twin field of a bag
for i in ('IDE-01', 'IDE-02', 'IDE-03', 'COU-04'): REF[i] = ('find_plants', 'get_plant')              # fruit counts of the three trees
REF['COU-05'] = ('find_plants', 'list_rows'); REF['COU-10'] = ('list_rows',)
GOLD_REJECT = ('FAI-01', 'FAI-02', 'FAI-03', 'FAI-04', 'FAI-05', 'FAI-06', 'FAI-07', 'FAI-08', 'FAI-09', 'FAI-10', 'IDE-07', 'CHA-10')
TYPES = ['no final', 'answered where a refusal was due', 'refusal', 'invented id', 'right lookup, wrong reading', 'right tool, wrong argument', 'wrong tool']
MAX_CALLS = 8


def norm(t): return [str(x).strip() for x in (t if isinstance(t, list) else [t]) if x is not None and str(x).strip() != '']


def score(g, ans):   # v1's rule
    if g['reject']: return bool(ans.get('reject'))
    if ans.get('reject'): return False
    gold = [str(x) for x in g['targets']]; got = norm(ans.get('targets', []))
    if g['category'] == 'count' and len(gold) == 1:
        v = ans.get('answer')
        try: return v is not None and abs(float(v) - float(gold[0])) < 0.5
        except Exception: return False
    if g['any']: return len(got) >= 1 and all(x in gold for x in got)
    return got == gold


def replay(R, rec):
    """-> list of (tool, args, id list or None, numbers dict, error) for the executed tool calls of the trace, in order."""
    reg = Registry(R, rec['arm']); out = []; calls = 0
    for st in rec['steps']:
        if st.get('parsed') != 'tool': continue
        calls += 1
        if calls > MAX_CALLS: break
        res = reg.call(st['tool'], st['args']); assert json.dumps(res) == json.dumps(st['result']), (rec['id'], st['turn'])     # the stored result is the registry's
        ids = None; nums = {}
        if 'error' not in res:
            if 'result_id' in res: ids = [str(x) for x in reg.results[res['result_id']]]; nums = dict(count=res['count'])
            elif 'rows' in res: nums = dict(n_rows=res['n_rows'], n_bags=res['n_bags'], n_trees=res['n_trees'], row_n_bags=[r['n_bags'] for r in res['rows']])
            elif 'id' in res: ids = [str(res['id'])]
        out.append(dict(tool=st['tool'], args=st['args'], ids=ids, nums=nums, error=res.get('error')))
    return out


def held_gold(g, calls):
    gold = [str(x) for x in g['targets']]
    if g['category'] == 'count' and len(gold) == 1:
        n = float(gold[0])
        return any(any((isinstance(v, (int, float)) and v == n) or (isinstance(v, list) and n in v) for v in c['nums'].values()) for c in calls)
    sets = [c['ids'] for c in calls if c['ids'] is not None]
    if g['any']: return any(len(s) > 0 and set(s) <= set(gold) for s in sets)
    if not gold: return any(len(s) == 0 for s in sets)
    singles = {s[0] for s in sets if len(s) == 1}
    return any(set(s) == set(gold) and len(s) == len(set(gold)) for s in sets) or set(gold) <= singles


def classify(R, g, rec):
    f = rec.get('final'); calls = replay(R, rec); tools = [c['tool'] for c in calls]; held = None if g['reject'] else held_gold(g, calls)
    extra = dict(tools=tools, held_gold=held, n_calls=len(calls))
    if not f: return 'no final', extra
    if g['reject']: return 'answered where a refusal was due', extra
    if f.get('reject'): return 'refusal', extra
    seen = set(); [seen.update(c['ids']) for c in calls if c['ids'] is not None]; said = set(re.findall(r'\d+', rec['instruction'])) | set(re.findall(r'tree_\d+', rec['instruction']))
    inv = [t for t in norm(f.get('targets', [])) if t not in seen and t not in said]
    if inv: extra['invented'] = inv; return 'invented id', extra
    if held: return 'right lookup, wrong reading', extra
    if any(t in REF[rec['id']] for t in tools): return 'right tool, wrong argument', extra
    extra['no_tool_call'] = len(calls) == 0; return 'wrong tool', extra


if __name__ == '__main__':
    num, pil, out = sys.argv[1:4]; dump = sys.argv[sys.argv.index('--dump') + 1] if '--dump' in sys.argv else None
    G = json.load(open(f'{num}/instruction_bank_v0_gold.json')); R = json.load(open(f'{num}/registry_v0.json'))
    res = [i for i, g in G.items() if 'unresolvable' not in g]; assert set(REF) | set(GOLD_REJECT) == set(res) and not set(REF) & set(GOLD_REJECT)
    RUNS = {'original (v1 seed 0)': 'h1v1_seed0', 'set A': 'h1p_setA', 'set B': 'h1p_setB'}
    items = []; tab = {w: Counter() for w in RUNS}; bycat = {w: defaultdict(Counter) for w in RUNS}; D = []
    for w, pre in RUNS.items():
        for line in open(f'{pil}/{pre}_twin.jsonl'):
            rec = json.loads(line)
            if rec['id'] not in res: continue
            g = G[rec['id']]
            if rec.get('final') and score(g, rec['final']): continue
            t, extra = classify(R, g, rec); tab[w][t] += 1; bycat[w][g['category']][t] += 1; f = rec.get('final') or {}
            items.append(dict(wording=w, id=rec['id'], category=g['category'], instruction=rec['instruction'], type=t, final_targets=(f.get('targets') if len(f.get('targets', [])) <= 12 else f"{len(f['targets'])} ids"), answer=f.get('answer'), reject=f.get('reject'), reason=f.get('reason'), **extra))
            if dump:
                D.append(f"\n===== {w} | {rec['id']} | {t} | {rec['instruction']}\n  gold: {'REJECT' if g['reject'] else ('any of ' if g['any'] else '') + (str(g['targets']) if len(g['targets']) <= 12 else str(len(g['targets'])) + ' ids')}   held_gold={extra['held_gold']}")
                for st in rec['steps']:
                    if st.get('parsed') == 'tool':
                        r = st['result']; rs = json.dumps(r); D.append(f"  CALL {st['tool']} {json.dumps(st['args'])}\n    -> {rs if len(rs) < 420 else rs[:420] + ' ...'}")
                    else: D.append(f"  REPLY {st['raw'][:400]}")
    n_wrong = {w: sum(tab[w].values()) for w in RUNS}
    json.dump(dict(note='twin arm, records the frozen gold scores wrong; type by the first rule that applies (design Section 7); descriptive, no verdict', types=TYPES, ref_tools=REF,
                   n_wrong=n_wrong, by_type={w: {t: tab[w][t] for t in TYPES} for w in RUNS}, by_category={w: {c: dict(v) for c, v in bycat[w].items()} for w in RUNS}, items=items), open(out, 'w'), indent=1)
    if dump: open(dump, 'w').write('\n'.join(D) + '\n')
    print(f"{'type':36s} " + ' '.join(f'{w:>22s}' for w in RUNS))
    for t in TYPES: print(f'{t:36s} ' + ' '.join(f'{tab[w][t]:22d}' for w in RUNS))
    print(f"{'wrong, total':36s} " + ' '.join(f'{n_wrong[w]:22d}' for w in RUNS))
