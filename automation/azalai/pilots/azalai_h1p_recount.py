#!/usr/bin/env python3
"""azalai_h1p_recount.py (11 Nov 2026, A100 session) - H1 paraphrase test (AZ-3), design Section 8: the independent recount.

Own code throughout, written from the design text (H1_paraphrase_design.md, H1_v1_design.md), not from azalai_h1_score_v1.py or
azalai_h1p_score.py. For every record of every run:
  - the instruction sent equals the frozen bank (v1, control) or the committed paraphrase table (A, B, re-runs, SA, SB);
  - each stored model reply is parsed again here and must give the stored parse (tool call, final, unparsable);
  - the tool calls are replayed on a fresh Registry (azalai_h1_tools: the system under test, used as the registry) and every
    replayed result must equal the stored one; at most 8 calls are executed, as in the harness;
  - the final is rebuilt here from the reply's own "targets" by expanding result_ids against the replayed results, and must
    equal the stored final (targets, answer, reject); the status and the number of tool calls must agree.
Then every table is recounted from the rebuilt finals with an own scorer, and compared with h1p_summary.json (azalai_h1p_score.py)
and with h1p_twin_failures.json (azalai_h1p_failures.py; the seven failure rules of design Section 7 are implemented again here,
with the REF tool table of that script used as data, since it is the author's pre-registered judgement). The corrected-gold copy
is checked entry by entry against an own recomputation of each documented change.

usage: python3 azalai_h1p_recount.py <dir> <h1p_summary.json> <h1p_twin_failures.json> <out.json>
exit status 0 iff every check passes; every problem is printed and written to out.json."""
import sys, os, json, csv, re, math
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from azalai_h1_tools import Registry

D, SUMF, FAILF, OUTF = sys.argv[1:5]
R = json.load(open(f'{D}/registry_v0.json'))
G0 = json.load(open(f'{D}/instruction_bank_v0_gold.json'))
GC = json.load(open(f'{D}/h1p_gold_corrected_sensitivity.json'))
CH = json.load(open(f'{D}/h1p_gold_corrected_changes.json'))
BANK = {b['id']: b['instruction'] for b in csv.DictReader(open(f'{D}/instruction_bank_v0.csv'))}
PARA = {(p['id'], p['set']): p['paraphrase'] for p in csv.DictReader(open(f'{D}/instruction_bank_v0_paraphrases.csv'))}
SUM = json.load(open(SUMF)); FAIL = json.load(open(FAILF))
ARMS = ('twin', 'enumerated')
ICRA = ['direct scene graph concepts', 'unambiguous goal command', 'co-reference resolution', 'spatial relation disambiguation',
        'region-level disambiguation', 'ambiguous role assignment', 'failure']
NEWC = ['identity', 'count', 'change']
TWIN_ONLY = ['IDE-01', 'IDE-02', 'IDE-03', 'IDE-10', 'COU-04', 'COU-06', 'COU-09', 'CHA-01', 'CHA-02', 'CHA-03', 'CHA-04', 'CHA-06',
             'CHA-07', 'CHA-08', 'CHA-09']                       # H1_v1_design.md Section 6, the 15 twin-only-field items
MAXC = 8
PROB = []                                                         # every failed check


def problem(msg):
    PROB.append(msg); print('PROBLEM', msg, flush=True)


# ---------------------------------------------------------------- parsing and replay (own implementation of the documented loop)
def parse_reply(txt):
    """('final', obj) | ('tool', name, args) | None, by the rules of H1_v1_design.md Section 3."""
    try:
        o = json.loads(txt)
    except ValueError:
        return None
    if not isinstance(o, dict):
        return None
    if isinstance(o.get('final'), dict):
        return ('final', o['final'])
    if o.get('tool') == 'final' and isinstance(o.get('args'), dict):
        return ('final', o['args'])
    if isinstance(o.get('tool'), str):
        return ('tool', o['tool'], {} if o.get('args') is None else o.get('args'))
    if 'targets' in o or 'reject' in o or 'answer' in o:
        return ('final', o)
    return None


def call_record(res, reg):
    """the ids and numbers a tool result carries: a result_id stands for its whole set (replayed), a single record for its id."""
    if not isinstance(res, dict) or 'error' in res:
        return None, {}
    if 'result_id' in res:
        return [str(x) for x in reg.results[res['result_id']]], {'count': res['count']}
    if 'rows' in res:
        return None, {'n_rows': res['n_rows'], 'n_bags': res['n_bags'], 'n_trees': res['n_trees'], 'row_n_bags': [r['n_bags'] for r in res['rows']]}
    if 'id' in res:
        return [str(res['id'])], {}
    return None, {}


def rebuild(rec, sent, tag):
    """replay one record; -> dict(final, status, calls=[...]) and report every disagreement with the stored record."""
    where = f"{tag} {rec['arm']} {rec['id']}"
    if rec['instruction'] != sent:
        problem(f'{where}: instruction sent differs from the committed text')
    reg = Registry(R, rec['arm']); calls = []; ncalls = 0; nbad = 0; final = None; status = 'max_turns'; steps = rec['steps']
    for k, st in enumerate(steps):
        if st.get('turn') != k:
            problem(f'{where}: turn index {st.get("turn")} at position {k}')
        p = parse_reply(st['raw'])
        if p is None:
            nbad += 1
            if st.get('parsed') is not None:
                problem(f'{where}: turn {k} stored as {st.get("parsed")}, unparsable here')
            if nbad > 2:
                status = 'parse_fail'
                if k != len(steps) - 1:
                    problem(f'{where}: steps after the third unparsable reply')
                break
            continue
        if p[0] == 'final':
            if st.get('parsed') != 'final':
                problem(f'{where}: turn {k} is a final here, stored as {st.get("parsed")}')
            f = p[1]; raw = f.get('targets', []); tg = []
            for t in (raw if isinstance(raw, list) else [raw]):
                if isinstance(t, str) and t.strip() in reg.results:
                    tg.extend(reg.results[t.strip()])
                elif t is not None:
                    tg.append(t)
            final = {'targets': tg, 'answer': f.get('answer'), 'reject': bool(f.get('reject'))}
            status = 'final'
            if k != len(steps) - 1:
                problem(f'{where}: steps after the final')
            break
        ncalls += 1
        if st.get('parsed') != 'tool' or st.get('tool') != p[1] or st.get('args') != p[2]:
            problem(f'{where}: turn {k} tool call differs from the stored parse')
        if ncalls <= MAXC:
            res = reg.call(p[1], p[2]); ids, nums = call_record(res, reg)
            calls.append({'tool': p[1], 'ids': ids, 'nums': nums})
        else:
            res = {'error': 'no tool calls left; reply with the final object now'}
        if json.dumps(res, sort_keys=True) != json.dumps(st.get('result'), sort_keys=True):
            problem(f'{where}: turn {k} replayed tool result differs from the stored one')
    if len(steps) > 14:
        problem(f'{where}: more than 14 turns')
    if status != rec['status']:
        problem(f'{where}: status {status} here, {rec["status"]} stored')
    sf = rec.get('final')
    if (final is None) != (sf is None):
        problem(f'{where}: final present here {final is not None}, stored {sf is not None}')
    elif final is not None:
        if final['targets'] != sf['targets'] or final['answer'] != sf['answer'] or final['reject'] != sf['reject']:
            problem(f'{where}: rebuilt final differs from the stored final')
    if rec['n_tool_calls'] != min(ncalls, MAXC):
        problem(f'{where}: n_tool_calls {rec["n_tool_calls"]} stored, {min(ncalls, MAXC)} here')
    return {'final': final, 'status': status, 'calls': calls, 'instruction': rec['instruction']}


# ---------------------------------------------------------------- own scorer (the rule of H1_v1_design.md Section 6, + Section 6 of this design)
def ids_of(targets):
    return [str(t).strip() for t in targets if t is not None and str(t).strip() != '']


def judge(g, fin):
    if g.get('unresolvable'):
        return None
    if fin is None:
        return False                                                  # no final answer: wrong
    if g['reject']:
        return fin['reject'] is True                                  # gold reject: correct iff the arm rejects
    if fin['reject']:
        return bool(g.get('reject_also_accepted'))                    # a reject where the gold has targets: wrong (reading of CHA-07 aside)
    gold = [str(t) for t in g['targets']]; named = ids_of(fin['targets'])
    if g['category'] == 'count' and len(gold) == 1:                  # count: the answer within 0.5 of the gold number
        ok = [float(gold[0])] + [float(x) for x in g.get('count_also', [])]
        try:
            v = float(fin['answer'])
        except (TypeError, ValueError):
            return False
        return any(abs(v - a) < 0.5 for a in ok)
    if g['any']:
        return len(named) >= 1 and set(named) <= set(gold)
    if g.get('order_free'):
        return Counter(named) == Counter(gold)
    return named == gold


def tables(G, recs):
    """recs[arm][id] = rebuilt record. Per category, total, the rule's clauses, verdict, twin-only items, discordant pairs."""
    O = {a: {i: judge(G[i], recs[a][i]['final']) for i in recs[a]} for a in ARMS}
    rej = {a: {i: bool(recs[a][i]['final'] and recs[a][i]['final']['reject']) for i in recs[a]} for a in ARMS}
    scored = [i for i in recs['twin'] if O['twin'][i] is not None]
    assert all((O['enumerated'][i] is None) == (O['twin'][i] is None) for i in recs['twin'])
    pc = {}
    for c in ICRA + NEWC:
        ids = [i for i in scored if G[i]['category'] == c]
        if ids:
            pc[c] = {'n': len(ids), 'twin': sum(O['twin'][i] for i in ids), 'enumerated': sum(O['enumerated'][i] for i in ids),
                     'twin_rejects': sum(rej['twin'][i] for i in ids), 'enumerated_rejects': sum(rej['enumerated'][i] for i in ids)}
    par = {c: pc[c]['twin'] - pc[c]['enumerated'] for c in ICRA if c in pc}
    ext = {c: pc[c]['twin'] > pc[c]['enumerated'] for c in NEWC if c in pc}
    v = {'parity_two_sided': all(abs(d) <= 1 for d in par.values()), 'parity_one_sided': all(d >= -1 for d in par.values()),
         'extension': all(ext.values())}
    v['H1_holds'] = v['parity_two_sided'] and v['extension']
    b = sum(1 for i in scored if O['twin'][i] and not O['enumerated'][i]); c_ = sum(1 for i in scored if O['enumerated'][i] and not O['twin'][i])
    n = b + c_; p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c_) + 1)) / 2 ** n)
    to = {a: {'n': sum(1 for i in TWIN_ONLY if i in scored), 'correct': sum(O[a][i] for i in TWIN_ONLY if i in scored),
              'rejected': sum(rej[a][i] for i in TWIN_ONLY if i in scored)} for a in ARMS}
    tot = {'n': len(scored), 'twin': sum(O['twin'][i] for i in scored), 'enumerated': sum(O['enumerated'][i] for i in scored),
           'twin_rejects': sum(rej['twin'][i] for i in scored), 'enumerated_rejects': sum(rej['enumerated'][i] for i in scored)}
    return {'per_category': pc, 'total': tot, 'parity_diff': par, 'extension_twin_above': ext, 'verdict': v, 'twin_only_field_items': to,
            'paired': {'twin_only_correct': b, 'enumerated_only_correct': c_, 'sign_test_two_sided_p': round(p, 4)}}, O


def same_trace(r1, r2):
    """reply for reply: every model reply, tool call and result, the final and the status."""
    s1 = [(s.get('raw'), s.get('tool'), s.get('args'), s.get('result')) for s in r1['steps']]
    s2 = [(s.get('raw'), s.get('tool'), s.get('args'), s.get('result')) for s in r2['steps']]
    return s1 == s2 and r1.get('final') == r2.get('final') and r1.get('status') == r2.get('status')


def fin_key(f):
    return None if f is None else (ids_of(f['targets']), f['answer'], f['reject'])


# ---------------------------------------------------------------- load and replay every run
RUNS = {'v1': ('h1v1_seed0', None), 'control': ('h1p_control', None), 'A': ('h1p_setA', 'A'), 'B': ('h1p_setB', 'B'),
        'A_rerun': ('h1p_setA_change_rerun', 'A'), 'B_rerun': ('h1p_setB_change_rerun', 'B'), 'SA': ('h1p_corA', 'A'), 'SB': ('h1p_corB', 'B')}
RAW = {}; REB = {}; HEALTH = {}
for k, (pre, s) in RUNS.items():
    RAW[k] = {}; REB[k] = {}; HEALTH[k] = {}
    for a in ARMS:
        recs = [json.loads(l) for l in open(f'{D}/{pre}_{a}.jsonl') if l.strip()]
        ids = [r['id'] for r in recs]
        if len(ids) != len(set(ids)):
            problem(f'{k} {a}: duplicate ids')
        if any(r['arm'] != a for r in recs):
            problem(f'{k} {a}: arm field')
        if any(r['model'] != 'hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M' or r['seed'] != 0 or r['options'] != {'temperature': 0, 'seed': 0, 'num_ctx': 8192, 'num_predict': 500} for r in recs):
            problem(f'{k} {a}: model, seed or options differ from the design')
        RAW[k][a] = {r['id']: r for r in recs}
        REB[k][a] = {r['id']: rebuild(r, BANK[r['id']] if s is None else PARA[(r['id'], s)], k) for r in recs}
        st = [x for r in recs for x in r['steps']]
        HEALTH[k][a] = {'n': len(recs), 'status': dict(Counter(r['status'] for r in recs)), 'no_final': sum(1 for r in recs if not r.get('final')),
                        'fallbacks': sum(1 for x in st if x.get('fallback')), 'unparsable_replies': sum(1 for x in st if x.get('parsed') is None),
                        'truncated_replies': sum(1 for x in st if x.get('done_reason') != 'stop'), 'model_turns': len(st),
                        'tool_calls': sum(r['n_tool_calls'] for r in recs)}

RES = [i for i in G0 if not G0[i].get('unresolvable')]; RESC = [i for i in GC if not GC[i].get('unresolvable')]
COR8 = ['CO--01', 'CO--02', 'CO--03', 'CO--04', 'CO--05', 'CO--07', 'CO--08', 'CO--10']
CHANGE = [i for i in RES if G0[i]['category'] == 'change']
expect_ids = {'v1': list(G0), 'control': list(G0), 'A': RES, 'B': RES, 'A_rerun': CHANGE, 'B_rerun': CHANGE, 'SA': COR8, 'SB': COR8}
for k in RUNS:
    for a in ARMS:
        if list(REB[k][a]) != expect_ids[k]:
            problem(f'{k} {a}: ids or their order differ from the design ({len(REB[k][a])} records)')
if len(RES) != 61 or len(RESC) != 69:
    problem(f'resolvable counts {len(RES)} / {len(RESC)}')

# ---------------------------------------------------------------- the corrected copy, checked against an own recomputation
bags = [b for b in R['bags'] if isinstance(b['id'], int)]
def rowlist(r): return sorted([b for b in bags if b['row'] == r], key=lambda b: b['n_in_row'])
def bag(i): return next(b for b in bags if b['id'] == i)
def at(r, n): return rowlist(r)[n - 1]['id']
def after(i): b = bag(i); L = rowlist(b['row']); return L[L.index(b) + 1]['id']
def closest(r, i): b = bag(i); return min(rowlist(r), key=lambda c: math.hypot(c['x'] - b['x'], c['z'] - b['z']))['id']
mine = {'CO--01': [412, after(412)], 'CO--02': [at(2, 30), closest(3, at(2, 30))], 'CO--03': [at(7, 1), at(7, 2)], 'CO--04': [118, after(118), 118],
        'CO--05': [max([b for b in rowlist(5) if b['height_dec_m'] is not None], key=lambda b: b['height_dec_m'])['id']], 'CO--07': [199, 200, 201],
        'CO--08': [at(4, 2), closest(5, at(4, 2))], 'CO--10': [77, closest(8, 77)]}
for i in G0:
    g0, gc = G0[i], GC[i]
    if i in mine:
        ok = gc['targets'] == mine[i] and not gc.get('unresolvable') and gc['any'] is False and gc['reject'] is False and g0.get('unresolvable')
    elif i == 'CHA-09':
        ok = gc.get('order_free') is True and gc['targets'] == g0['targets'] and {k: v for k, v in gc.items() if k != 'order_free'} == g0
    elif i == 'REG-07':
        ok = sorted(gc['targets']) == sorted(b['id'] for b in bags if b['row'] <= 6) and gc['any'] is True and sorted(g0['targets']) == sorted(b['id'] for b in bags if b['row'] <= 5)
    elif i == 'REG-09':
        ok = sorted(gc['targets']) == sorted(b['id'] for b in bags if 5 <= b['row'] <= 7) and gc['any'] is True and sorted(g0['targets']) == sorted(b['id'] for b in bags if b['row'] in (5, 6))
    else:
        ok = gc == g0
    if not ok:
        problem(f'corrected copy: {i} is not the documented change')
if sorted(G0['REG-01']['targets']) != sorted(b['id'] for b in bags if b['row'] >= 6):
    problem('REG-01 frozen gold is not rows 6-11')
strict = CH['readings']['eleven rows, strict halves']['override']
if not (sorted(strict['REG-01']['targets']) == sorted(b['id'] for b in bags if b['row'] >= 7) and sorted(strict['REG-07']['targets']) == sorted(b['id'] for b in bags if b['row'] <= 5)
        and sorted(strict['REG-09']['targets']) == sorted(b['id'] for b in bags if b['row'] == 6)):
    problem('strict-halves reading is not rows 7-11 / 1-5 / 6')
if CH['readings']['COU-06 = 1 (removed only)']['override']['targets'] != [sum(1 for b in R['bags'] if b['status'] == 'removed')]:
    problem('COU-06 removed-only reading')

# ---------------------------------------------------------------- frozen gold: tables per run, compared with the summary
OUT = {'problems': PROB, 'health': HEALTH, 'frozen_gold': {}, 'corrected_gold_sensitivity': {}}
FZ = {}; OC = {}
for k in ('v1', 'control', 'A', 'B'):
    FZ[k], OC[k] = tables(G0, REB[k])
OUT['frozen_gold'] = FZ


def cmp(label, mine_t, theirs):
    """mine_t from tables(); theirs = a table() of azalai_h1p_score.py."""
    for c, row in mine_t['per_category'].items():
        t = theirs['per_category'].get(c)
        if t is None or any(t[x] != row[x] for x in row):
            problem(f'{label}: per-category {c} differs (here {row}, summary {t})')
    if set(mine_t['per_category']) != set(theirs['per_category']):
        problem(f'{label}: category set differs')
    if any(theirs['total'][x] != mine_t['total'][x] for x in mine_t['total']):
        problem(f'{label}: total differs (here {mine_t["total"]}, summary {theirs["total"]})')
    if any(theirs['verdict'][x] != mine_t['verdict'][x] for x in mine_t['verdict']):
        problem(f'{label}: verdict differs (here {mine_t["verdict"]}, summary {theirs["verdict"]})')
    if {c: v['diff'] for c, v in theirs['parity'].items()} != mine_t['parity_diff']:
        problem(f'{label}: parity differences differ')
    if {c: v['twin_above'] for c, v in theirs['extension'].items()} != mine_t['extension_twin_above']:
        problem(f'{label}: extension clause differs')
    if theirs['twin_only_field_items'] != mine_t['twin_only_field_items']:
        problem(f'{label}: twin-only items differ')
    if theirs['paired'] != mine_t['paired']:
        problem(f'{label}: paired counts differ (here {mine_t["paired"]}, summary {theirs["paired"]})')


for k in FZ:
    cmp(f'frozen gold {k}', FZ[k], SUM['frozen_gold']['tables'][k])
for k in HEALTH:
    if k in SUM['runs']:
        for a in ARMS:
            if any(SUM['runs'][k][a][x] != HEALTH[k][a][x] for x in HEALTH[k][a]):
                problem(f'health {k} {a} differs (here {HEALTH[k][a]})')
for k, s in (('SA', 'corA'), ('SB', 'corB')):
    for a in ARMS:
        if any(SUM['runs'][s][a][x] != HEALTH[k][a][x] for x in HEALTH[k][a]):
            problem(f'health {k} {a} differs')

# primary reading
prim = {'holds_on_A': FZ['A']['verdict']['H1_holds'], 'holds_on_B': FZ['B']['verdict']['H1_holds']}
prim['robust_to_wording'] = prim['holds_on_A'] and prim['holds_on_B']
OUT['primary_reading'] = prim
if any(SUM['primary_reading'][x] != prim[x] for x in prim):
    problem(f'primary reading differs (here {prim})')

# control and determinism, reply for reply
ctl = {}
for a in ARMS:
    diff = [i for i in G0 if not same_trace(RAW['control'][a][i], RAW['v1'][a][i])]
    ctl[a] = {'n': len(G0), 'traces_differ': diff,
              'finals_differ': [i for i in diff if fin_key(REB['control'][a][i]['final']) != fin_key(REB['v1'][a][i]['final'])],
              'outcome_differs': [i for i in diff if OC['control'][a][i] != OC['v1'][a][i]]}
    if any(SUM['control_vs_v1_seed0'][a][x] != ctl[a][x] for x in ctl[a]):
        problem(f'control vs v1 {a} differs (here {ctl[a]})')
OUT['control_vs_v1_seed0'] = ctl
det = {}
for k, base in (('A_rerun', 'A'), ('B_rerun', 'B')):
    det[base] = {a: {'n': len(RAW[k][a]), 'traces_differ': [i for i in RAW[k][a] if not same_trace(RAW[k][a][i], RAW[base][a][i])]} for a in ARMS}
holds = all(det[s][a]['n'] == 9 and not det[s][a]['traces_differ'] for s in det for a in ARMS)
OUT['determinism'] = dict(sets=det, holds=holds)
if SUM['determinism_change_rerun']['sets'] != det or SUM['determinism_change_rerun']['holds'] != holds:
    problem('determinism differs')

# outcome kept under both paraphrases, identical finals
keep = {}
NAMES = {(True, True, True): 'correct in all three', (False, False, False): 'wrong in all three', (True, False, True): 'correct in v1, lost in A',
         (True, True, False): 'correct in v1, lost in B', (True, False, False): 'correct in v1, lost in both', (False, True, False): 'wrong in v1, gained in A',
         (False, False, True): 'wrong in v1, gained in B', (False, True, True): 'wrong in v1, gained in both'}
for a in ARMS:
    pat = {i: (OC['v1'][a][i], OC['A'][a][i], OC['B'][a][i]) for i in RES}
    by = {}
    for i in RES:
        by.setdefault(NAMES[pat[i]], []).append(i)
    cat = {}
    for c in ICRA + NEWC:
        ids = [i for i in RES if G0[i]['category'] == c]
        if ids:
            cat[c] = {'n': len(ids), 'kept_under_both': sum(len(set(pat[i])) == 1 for i in ids), 'kept_in_A': sum(pat[i][0] == pat[i][1] for i in ids),
                      'kept_in_B': sum(pat[i][0] == pat[i][2] for i in ids)}
    sameA = [i for i in RES if fin_key(REB['A'][a][i]['final']) == fin_key(REB['v1'][a][i]['final'])]
    sameB = [i for i in RES if fin_key(REB['B'][a][i]['final']) == fin_key(REB['v1'][a][i]['final'])]
    keep[a] = {'n': len(RES), 'kept_under_both': sum(len(set(p)) == 1 for p in pat.values()), 'kept_in_A': sum(p[0] == p[1] for p in pat.values()),
               'kept_in_B': sum(p[0] == p[2] for p in pat.values()), 'patterns': by, 'per_category': cat,
               'identical_final_as_v1': {'A': len(sameA), 'B': len(sameB), 'both': len(set(sameA) & set(sameB))}}
    theirs = SUM['outcome_kept'][a]
    for x in ('n', 'kept_under_both', 'kept_in_A', 'kept_in_B', 'per_category', 'identical_final_as_v1'):
        if theirs[x] != keep[a][x]:
            problem(f'outcome kept {a} {x} differs (here {keep[a][x]}, summary {theirs[x]})')
    if {p: sorted(v) for p, v in theirs['patterns'].items()} != {p: sorted(v) for p, v in by.items()}:
        problem(f'outcome patterns {a} differ')
OUT['outcome_kept'] = keep
per = {i: {a: {'v1': OC['v1'][a][i], 'A': OC['A'][a][i], 'B': OC['B'][a][i]} for a in ARMS} for i in RES}
for i in RES:
    if any(SUM['per_instruction'][i][a] != per[i][a] for a in ARMS):
        problem(f'per-instruction outcome {i} differs')
OUT['per_instruction'] = {i: dict(category=G0[i]['category'], original=BANK[i], A=PARA[(i, 'A')], B=PARA[(i, 'B')], **per[i]) for i in RES}

# ---------------------------------------------------------------- corrected-gold sensitivity (69 instructions)
RC = {'v1': REB['v1'], 'A': {a: {**REB['A'][a], **REB['SA'][a]} for a in ARMS}, 'B': {a: {**REB['B'][a], **REB['SB'][a]} for a in ARMS}}
def variant(base, issue=None, reading=None):
    G = json.loads(json.dumps(base))
    if issue:
        for c in CH['changes']:
            if c['issue'] == issue:
                G[c['id']] = json.loads(json.dumps(GC[c['id']]))
    if reading:
        r = CH['readings'][reading]; ov = r['override']
        pairs = ov.items() if ',' in r['id'] else [(r['id'], ov)]
        for i, o in pairs:
            G[i].update(o)
    return G
sens = {'corrected_copy': {k: tables(GC, RC[k])[0] for k in RC}}
sens['each_change_alone'] = {f: {k: tables(variant(G0, issue=f), RC[k])[0] for k in RC} for f in sorted({c['issue'] for c in CH['changes']})}
sens['each_reading_alone_on_corrected'] = {r: {k: tables(variant(GC, reading=r), RC[k])[0] for k in RC} for r in CH['readings']}
th = SUM['corrected_gold_sensitivity']
for k in RC:
    cmp(f'corrected copy {k}', sens['corrected_copy'][k], th['corrected_copy'][k])
for grp in ('each_change_alone', 'each_reading_alone_on_corrected'):
    if set(sens[grp]) != set(th[grp]):
        problem(f'{grp}: keys differ')
    for f in sens[grp]:
        for k in RC:
            cmp(f'{grp} {f} {k}', sens[grp][f][k], th[grp][f][k])
OCC = {k: {a: {i: judge(GC[i], RC[k][a][i]['final']) for i in RC[k][a]} for a in ARMS} for k in RC}
chg = {}
for i in sorted({c['id'] for c in CH['changes']} | {'CHA-07', 'COU-06', 'REG-01'}):
    chg[i] = {a: {'frozen': {k: judge(G0[i], RC[k][a][i]['final']) for k in RC}, 'corrected': {k: OCC[k][a][i] for k in RC}} for a in ARMS}
    if th['changed_instructions'][i] != chg[i]:
        problem(f'changed instruction {i} differs (here {chg[i]})')
sens['changed_instructions'] = chg
OUT['corrected_gold_sensitivity'] = sens

# ---------------------------------------------------------------- twin failures: the seven rules of design Section 7, own code
REF = FAIL['ref_tools']
def held(g, calls):
    gold = [str(x) for x in g['targets']]
    if g['category'] == 'count' and len(gold) == 1:
        n = float(gold[0])
        for c in calls:
            for key, v in c['nums'].items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and float(v) == n:
                    return True
                if isinstance(v, list) and any(float(x) == n for x in v):
                    return True
        return False
    sets = [c['ids'] for c in calls if c['ids'] is not None]
    if g['any']:
        return any(s and set(s) <= set(gold) for s in sets)
    if any(sorted(s) == sorted(gold) for s in sets):
        return True                                       # a result whose id set is the gold set (the empty gold: an empty result)
    singles = {s[0] for s in sets if len(s) == 1}
    return bool(gold) and set(gold) <= singles             # single records that together hold the gold set


def ftype(i, g, rb):
    fin = rb['final']; calls = rb['calls']
    if fin is None:
        return 'no final'
    if g['reject']:
        return 'answered where a refusal was due'
    if fin['reject']:
        return 'refusal'
    seen = set().union(*[set(c['ids']) for c in calls if c['ids'] is not None]) if calls else set()
    said = set(re.findall(r'\d+', rb['instruction'])) | set(re.findall(r'tree_\d+', rb['instruction']))
    if [t for t in ids_of(fin['targets']) if t not in seen and t not in said]:
        return 'invented id'
    if held(g, calls):
        return 'right lookup, wrong reading'
    if any(c['tool'] in REF[i] for c in calls):
        return 'right tool, wrong argument'
    return 'wrong tool'


ftab = {}; fitems = {}
for w, k in (('original (v1 seed 0)', 'v1'), ('set A', 'A'), ('set B', 'B')):
    wrong = [i for i in RES if OC[k]['twin'][i] is False]
    fitems[w] = {i: ftype(i, G0[i], REB[k]['twin'][i]) for i in wrong}
    ftab[w] = dict(Counter(fitems[w].values()))
    theirs = {it['id']: it['type'] for it in FAIL['items'] if it['wording'] == w}
    if theirs != fitems[w]:
        problem(f'failure types {w} differ: ' + json.dumps({i: (fitems[w].get(i), theirs.get(i)) for i in set(theirs) | set(fitems[w]) if fitems[w].get(i) != theirs.get(i)}))
    if FAIL['n_wrong'][w] != len(wrong) or any(FAIL['by_type'][w].get(t, 0) != ftab[w].get(t, 0) for t in set(FAIL['by_type'][w]) | set(ftab[w])):
        problem(f'failure table {w} differs')
OUT['twin_failures'] = {'by_type': ftab, 'items': fitems}

json.dump(OUT, open(OUTF, 'w'), indent=1)
print(f'\nRECOUNT: {sum(len(REB[k][a]) for k in REB for a in ARMS)} records replayed; {len(PROB)} problems')
for k in ('v1', 'control', 'A', 'B'):
    t = FZ[k]['total']; print(f"frozen gold {k:8s} twin {t['twin']} / enumerated {t['enumerated']} of {t['n']}; verdict {FZ[k]['verdict']}")
for k in RC:
    t = sens['corrected_copy'][k]['total']; print(f"corrected  {k:8s} twin {t['twin']} / enumerated {t['enumerated']} of {t['n']}; verdict {sens['corrected_copy'][k]['verdict']}")
print('primary reading', prim, '| determinism holds', holds, '| control traces differ', {a: len(ctl[a]['traces_differ']) for a in ARMS})
sys.exit(1 if PROB else 0)
