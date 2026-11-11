#!/usr/bin/env python3
"""azalai_h1p_score.py (11 Nov 2026) — H1 paraphrase test (AZ-3): everything the design of record
(Azalai_RSS/numbers/pilots_20261111/H1_paraphrase_design.md, Sections 3, 4 and 6) reports, from the per-instruction records.

  1. run health (finals, fallbacks, truncated or unparsable replies), and the instruction sent against the committed table
  2. reproduction control: the frozen wording rerun tonight against v1 seed 0, reply for reply
  3. determinism: the change re-runs against the first runs, reply for reply
  4. frozen gold: the scorer of record azalai_h1_score_v1.py is run UNCHANGED on each run (gold restricted to the run's ids);
     this file's own scorer (v1's rule + order_free, reject_also_accepted, count_also for the sensitivity) must reproduce it
     instruction for instruction before anything else is computed
  5. per category, the rule's clauses and the verdict per paraphrase set; outcome kept under both paraphrases; identical finals
  6. corrected-gold SENSITIVITY (not adopted): v1 records, set A + SA, set B + SB on the 69 instructions; each change alone;
     each alternative reading alone on top of the corrected copy

usage: python3 azalai_h1p_score.py <Azalai_RSS/numbers> <pilots dir> <h1p_summary.json>   (prints the tables)"""
import sys, os, json, csv, copy, shutil, subprocess, tempfile
from math import comb
from collections import Counter
num, pil, out = sys.argv[1:4]; HERE = os.path.dirname(os.path.abspath(__file__))
G0 = json.load(open(f'{num}/instruction_bank_v0_gold.json')); GC = json.load(open(f'{pil}/h1p_gold_corrected_sensitivity.json')); CH = json.load(open(f'{pil}/h1p_gold_corrected_changes.json'))
PAR = {(p['id'], p['set']): p['paraphrase'] for p in csv.DictReader(open(f'{num}/instruction_bank_v0_paraphrases.csv'))}; BANK = {b['id']: b['instruction'] for b in csv.DictReader(open(f'{num}/instruction_bank_v0.csv'))}
ARMS = ('twin', 'enumerated'); NUMBERED = ['direct scene graph concepts', 'unambiguous goal command', 'co-reference resolution', 'spatial relation disambiguation', 'region-level disambiguation', 'ambiguous role assignment', 'failure']; NEW = ['identity', 'count', 'change']
TWIN_ONLY = ['IDE-01', 'IDE-02', 'IDE-03', 'IDE-10', 'COU-04', 'COU-06', 'COU-09', 'CHA-01', 'CHA-02', 'CHA-03', 'CHA-04', 'CHA-06', 'CHA-07', 'CHA-08', 'CHA-09']
RES = [i for i in G0 if 'unresolvable' not in G0[i]]; RESC = [i for i in GC if 'unresolvable' not in GC[i]]; assert len(RES) == 61 and len(RESC) == 69


def load(pre): return {arm: {r['id']: r for r in map(json.loads, (l for l in open(f'{pil}/{pre}_{arm}.jsonl') if l.strip()))} for arm in ARMS}


def norm(t): return [str(x).strip() for x in (t if isinstance(t, list) else [t]) if x is not None and str(x).strip() != '']


def sc(g, rec):
    """v1's rule (azalai_h1_score_v1.py) plus the three additions of design Section 6. None = not scored."""
    if g.get('unresolvable'): return None
    ans = rec.get('final')
    if not ans: return False
    if g['reject']: return bool(ans.get('reject'))
    if ans.get('reject'): return bool(g.get('reject_also_accepted'))                       # addition (a reading of CHA-07 only)
    gold = [str(x) for x in g['targets']]; got = norm(ans.get('targets', []))
    if g['category'] == 'count' and len(gold) == 1:
        v = ans.get('answer')
        try: return v is not None and any(abs(float(v) - float(x)) < 0.5 for x in gold + [str(x) for x in g.get('count_also', [])])   # count_also: addition (a reading of COU-06 only)
        except Exception: return False
    if g['any']: return len(got) >= 1 and all(x in gold for x in got)
    if g.get('order_free'): return sorted(got) == sorted(gold)                              # addition (CHA-09 in the corrected copy)
    return got == gold


def outcomes(G, recs): return {arm: {i: sc(G[i], r) for i, r in recs[arm].items()} for arm in ARMS}


def table(G, recs, O):
    """per category counts, the rule's clauses and the verdict, over the instructions of recs that G scores."""
    cats = [c for c in NUMBERED + NEW]; T = {}
    for c in cats:
        ids = [i for i in recs['twin'] if G[i]['category'] == c and O['twin'][i] is not None]
        if not ids: continue
        T[c] = dict(n=len(ids), **{arm: sum(O[arm][i] for i in ids) for arm in ARMS}, **{arm + '_rejects': sum(bool((recs[arm][i].get('final') or {}).get('reject')) for i in ids) for arm in ARMS})
    parity = {c: dict(n=T[c]['n'], twin=T[c]['twin'], enumerated=T[c]['enumerated'], diff=T[c]['twin'] - T[c]['enumerated']) for c in NUMBERED if c in T}
    ext = {c: dict(n=T[c]['n'], twin=T[c]['twin'], enumerated=T[c]['enumerated'], twin_above=T[c]['twin'] > T[c]['enumerated']) for c in NEW if c in T}
    v = dict(parity_two_sided=all(abs(p['diff']) <= 1 for p in parity.values()), parity_one_sided=all(p['diff'] >= -1 for p in parity.values()), extension=all(e['twin_above'] for e in ext.values()))
    v['H1_holds'] = v['parity_two_sided'] and v['extension']; ids = [i for i in recs['twin'] if O['twin'][i] is not None]
    tot = dict(n=len(ids), **{arm: sum(O[arm][i] for i in ids) for arm in ARMS}, **{arm + '_rejects': sum(bool((recs[arm][i].get('final') or {}).get('reject')) for i in ids) for arm in ARMS})
    b = sum(O['twin'][i] and not O['enumerated'][i] for i in ids); c_ = sum(O['enumerated'][i] and not O['twin'][i] for i in ids); n = b + c_
    p = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c_) + 1)) / 2 ** n) if n else 1.0
    to = {arm: dict(n=sum(1 for i in TWIN_ONLY if i in ids), correct=sum(O[arm][i] for i in TWIN_ONLY if i in ids), rejected=sum(bool((recs[arm][i].get('final') or {}).get('reject')) for i in TWIN_ONLY if i in ids)) for arm in ARMS}
    return dict(per_category=T, total=tot, parity=parity, extension=ext, verdict=v, twin_only_field_items=to, paired=dict(twin_only_correct=b, enumerated_only_correct=c_, sign_test_two_sided_p=round(p, 4)))


def trace(r): return ([(s.get('raw'), s.get('tool'), s.get('args'), s.get('result')) for s in r['steps']], r.get('final'), r.get('status'))
def fin(r): f = r.get('final') or {}; return (norm(f.get('targets', [])), f.get('answer'), bool(f.get('reject'))) if r.get('final') else None


def health(recs):
    H = {}
    for arm in ARMS:
        R = list(recs[arm].values()); st = [s for r in R for s in r['steps']]
        H[arm] = dict(n=len(R), status=dict(Counter(r['status'] for r in R)), no_final=sum(1 for r in R if not r.get('final')), fallbacks=sum(bool(s.get('fallback')) for s in st), unparsable_replies=sum(1 for s in st if s.get('parsed') is None),
                      truncated_replies=sum(1 for s in st if s.get('done_reason') != 'stop'), model_turns=len(st), tool_calls=sum(r['n_tool_calls'] for r in R), seconds=round(sum(r['seconds'] for r in R), 1),
                      model={r['model'] for r in R}.pop(), options=R[0]['options'], seeds=sorted({r['seed'] for r in R}))
    return H


def score_v1(pre, ids):
    """the scorer of record, unchanged, in a temp dir holding the frozen gold restricted to the run's ids."""
    d = tempfile.mkdtemp(); json.dump({i: G0[i] for i in G0 if i in ids}, open(f'{d}/instruction_bank_v0_gold.json', 'w'), indent=1); shutil.copy(f'{num}/registry_v0.json', d)
    o = f'{pil}/{pre}_scorev1_summary.json'; p = subprocess.run([sys.executable, f'{HERE}/azalai_h1_score_v1.py', d, f'{pil}/{pre}', o], capture_output=True, text=True); shutil.rmtree(d)
    assert p.returncode == 0, p.stderr
    return json.load(open(o))


S = dict(design='H1_paraphrase_design.md', gold_of_record='instruction_bank_v0_gold.json (frozen)', runs={}, frozen_gold={}, corrected_gold_sensitivity={})
REC = {k: load(p) for k, p in dict(v1='h1v1_seed0', control='h1p_control', A='h1p_setA', B='h1p_setB').items()}
for k in ('A', 'B'):
    assert all(list(REC[k][arm]) == RES for arm in ARMS), k
    assert all(REC[k][arm][i]['instruction'] == PAR[(i, k)] for arm in ARMS for i in RES), k            # the paraphrase sent is the committed one
assert all(REC[k][arm][i]['instruction'] == BANK[i] for k in ('v1', 'control') for arm in ARMS for i in G0)
S['runs'] = {k: health(v) for k, v in REC.items()}
# 2. control against v1 seed 0
ctl = {}
for arm in ARMS:
    diff = [i for i in G0 if trace(REC['control'][arm][i]) != trace(REC['v1'][arm][i])]
    ctl[arm] = dict(n=len(G0), traces_differ=diff, finals_differ=[i for i in diff if fin(REC['control'][arm][i]) != fin(REC['v1'][arm][i])],
                    outcome_differs=[i for i in diff if sc(G0[i], REC['control'][arm][i]) != sc(G0[i], REC['v1'][arm][i])])
S['control_vs_v1_seed0'] = ctl
# 3. determinism
det = {}
for k in ('A', 'B'):
    rr = load(f'h1p_set{k}_change_rerun')
    det[k] = {arm: dict(n=len(rr[arm]), traces_differ=[i for i in rr[arm] if trace(rr[arm][i]) != trace(REC[k][arm][i])]) for arm in ARMS}
S['determinism_change_rerun'] = dict(sets=det, holds=all(not det[k][arm]['traces_differ'] and det[k][arm]['n'] == 9 for k in det for arm in ARMS))
# 4. frozen gold: the scorer of record, and this file's scorer against it
OUT = {k: outcomes(G0, REC[k]) for k in REC}; TAB = {}
for k, pre in dict(v1='h1v1_seed0', control='h1p_control', A='h1p_setA', B='h1p_setB').items():
    s1 = score_v1(pre, set(REC[k]['twin']))
    for arm in ARMS:
        assert all(s1['per_instruction'][arm][i]['correct'] == OUT[k][arm][i] for i in REC[k][arm]), (k, arm)
    TAB[k] = table(G0, REC[k], OUT[k]); v1v = s1['verdict']
    assert all(TAB[k]['verdict'][x] == v1v[x] for x in v1v) and all(TAB[k]['total'][arm] == s1['total'][arm]['correct'] for arm in ARMS), k
    assert all(TAB[k]['per_category'][c][arm] == s1['per_category'][c][arm]['correct'] for c in TAB[k]['per_category'] for arm in ARMS), k
S['frozen_gold'] = dict(note='scorer of record azalai_h1_score_v1.py, unchanged (summaries h1p_*_scorev1_summary.json); this file reproduces it instruction for instruction', tables=TAB)
# 5. outcome kept under both paraphrases, per arm
keep = {}
for arm in ARMS:
    pat = {i: (OUT['v1'][arm][i], OUT['A'][arm][i], OUT['B'][arm][i]) for i in RES}; lab = {(True, True, True): 'correct in all three', (False, False, False): 'wrong in all three', (True, False, True): 'correct in v1, lost in A',
           (True, True, False): 'correct in v1, lost in B', (True, False, False): 'correct in v1, lost in both', (False, True, False): 'wrong in v1, gained in A', (False, False, True): 'wrong in v1, gained in B', (False, True, True): 'wrong in v1, gained in both'}
    by = {}
    for i, p in pat.items(): by.setdefault(lab[p], []).append(i)
    cat = {}
    for c in NUMBERED + NEW:
        ids = [i for i in RES if G0[i]['category'] == c]
        if ids: cat[c] = dict(n=len(ids), kept_under_both=sum(pat[i][0] == pat[i][1] == pat[i][2] for i in ids), kept_in_A=sum(pat[i][0] == pat[i][1] for i in ids), kept_in_B=sum(pat[i][0] == pat[i][2] for i in ids))
    same = {k: [i for i in RES if fin(REC[k][arm][i]) == fin(REC['v1'][arm][i])] for k in ('A', 'B')}
    keep[arm] = dict(n=len(RES), kept_under_both=sum(p[0] == p[1] == p[2] for p in pat.values()), kept_in_A=sum(p[0] == p[1] for p in pat.values()), kept_in_B=sum(p[0] == p[2] for p in pat.values()), patterns=by, per_category=cat,
                     identical_final_as_v1=dict(A=len(same['A']), B=len(same['B']), both=len(set(same['A']) & set(same['B']))))
S['outcome_kept'] = keep
S['per_instruction'] = {i: dict(category=G0[i]['category'], original=BANK[i], A=PAR[(i, 'A')], B=PAR[(i, 'B')], **{arm: dict(v1=OUT['v1'][arm][i], A=OUT['A'][arm][i], B=OUT['B'][arm][i]) for arm in ARMS}) for i in RES}
v = {k: TAB[k]['verdict']['H1_holds'] for k in ('A', 'B')}
S['primary_reading'] = dict(rule='H1 is robust to wording iff the v1 rule holds on set A and on set B (frozen gold)', holds_on_A=v['A'], holds_on_B=v['B'], robust_to_wording=v['A'] and v['B'])
# 6. corrected-gold sensitivity
cor = {k: load(f'h1p_cor{k}') for k in ('A', 'B')}
RC = dict(v1=REC['v1'], A={arm: {**REC['A'][arm], **cor['A'][arm]} for arm in ARMS}, B={arm: {**REC['B'][arm], **cor['B'][arm]} for arm in ARMS})
assert all(cor[k][arm][i]['instruction'] == PAR[(i, k)] for k in cor for arm in ARMS for i in cor[k][arm]) and all(set(RC[k]['twin']) >= set(RESC) for k in RC)
S['runs'].update({f'cor{k}': health(cor[k]) for k in cor})


def variant(base, fix=None, reading=None):
    G = copy.deepcopy(base)
    if fix:
        for c in CH['changes']:
            if c['issue'] == fix: G[c['id']] = copy.deepcopy(GC[c['id']])
    if reading:
        r = CH['readings'][reading]
        for i, ov in (r['override'].items() if ',' in r['id'] else [(r['id'], r['override'])]): G[i].update(ov)
    return G


def tab3(G): return {k: table(G, RC[k], outcomes(G, RC[k])) for k in RC}
sens = dict(label='SENSITIVITY, not adopted', corrected_copy=tab3(GC), frozen_gold_same_records={k: TAB[k] for k in ('v1', 'A', 'B')},
            each_change_alone={f: tab3(variant(G0, fix=f)) for f in sorted({c['issue'] for c in CH['changes']})}, each_reading_alone_on_corrected={r: tab3(variant(GC, reading=r)) for r in CH['readings']})
OC = {k: outcomes(GC, RC[k]) for k in RC}
sens['changed_instructions'] = {i: {arm: dict(frozen={k: (sc(G0[i], RC[k][arm][i])) for k in RC}, corrected={k: OC[k][arm][i] for k in RC}) for arm in ARMS} for i in sorted({c['id'] for c in CH['changes']} | {'CHA-07', 'COU-06', 'REG-01'})}
S['corrected_gold_sensitivity'] = sens
json.dump(S, open(out, 'w'), indent=1)

# ---- tables
def line(T, name):
    cats = [c for c in NUMBERED + NEW if c in T['per_category']]
    print(f'\n{name}\n| category | n | twin | enumerated | diff | twin rejects | enumerated rejects |\n|---|---|---|---|---|---|---|')
    for c in cats: t = T['per_category'][c]; print(f"| {c} | {t['n']} | {t['twin']} | {t['enumerated']} | {t['twin'] - t['enumerated']:+d} | {t['twin_rejects']} | {t['enumerated_rejects']} |")
    t = T['total']; print(f"| **total** | **{t['n']}** | **{t['twin']}** | **{t['enumerated']}** | {t['twin'] - t['enumerated']:+d} | {t['twin_rejects']} | {t['enumerated_rejects']} |")
    print('verdict', json.dumps(T['verdict']), '| twin-only items', json.dumps(T['twin_only_field_items']), '| paired', json.dumps(T['paired']))
print('RUN HEALTH'); [print(k, arm, json.dumps({x: y for x, y in h.items() if x not in ('options', 'model')})) for k, hh in S['runs'].items() for arm, h in hh.items()]
print('\nCONTROL vs v1 seed 0', json.dumps(ctl)); print('DETERMINISM', json.dumps(S['determinism_change_rerun']))
for k in ('v1', 'control', 'A', 'B'): line(TAB[k], f'FROZEN GOLD, {k}')
print('\nPRIMARY READING', json.dumps(S['primary_reading']))
for arm in ARMS: print(f'\nOUTCOME KEPT, {arm}:', json.dumps({x: y for x, y in keep[arm].items() if x not in ('patterns',)})); [print('  ', p, len(ids), ids) for p, ids in keep[arm]['patterns'].items()]
for k in RC: line(sens['corrected_copy'][k], f'CORRECTED GOLD (sensitivity), {k}')
print('\nEACH CHANGE ALONE (on the frozen gold) and EACH READING ALONE (on the corrected copy): total twin/enumerated of n, H1 rule')
for grp in ('each_change_alone', 'each_reading_alone_on_corrected'):
    for f, t3 in sens[grp].items(): print(f'  {grp[:12]} | {f:32s} | ' + ' | '.join(f"{k}: {t3[k]['total']['twin']}/{t3[k]['total']['enumerated']} of {t3[k]['total']['n']}, holds={t3[k]['verdict']['H1_holds']} (parity={t3[k]['verdict']['parity_two_sided']}, ext={t3[k]['verdict']['extension']})" for k in t3))
print('\nCHANGED INSTRUCTIONS'); [print(' ', i, json.dumps(x)) for i, x in sens['changed_instructions'].items()]
