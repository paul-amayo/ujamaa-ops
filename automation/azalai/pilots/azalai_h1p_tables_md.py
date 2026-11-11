#!/usr/bin/env python3
"""azalai_h1p_tables_md.py (11 Nov 2026, A100 session) - H1 paraphrase test (AZ-3): the markdown tables of the results, generated
from the recount's output (azalai_h1p_recount.py), so that no number in RESULTS.md is typed by hand.
usage: python3 azalai_h1p_tables_md.py <recount.json> <failures.json> <out.md>"""
import sys, json
X = json.load(open(sys.argv[1])); F = json.load(open(sys.argv[2])); OUT = sys.argv[3]
CATS = ['direct scene graph concepts', 'unambiguous goal command', 'co-reference resolution', 'spatial relation disambiguation',
        'region-level disambiguation', 'ambiguous role assignment', 'failure', 'identity', 'count', 'change']
NEWC = ('identity', 'count', 'change')
L = []


def cat_table(tabs, names, title):
    """one row per category: twin / enumerated for each run, the difference, and the clause."""
    L.append(f'\n{title}\n')
    L.append('| category | n | ' + ' | '.join(f'{nm}: twin / enum (diff)' for nm in names) + ' |')
    L.append('|---|---|' + '---|' * len(names))
    for c in CATS:
        if c not in tabs[0]['per_category']:
            continue
        n = tabs[0]['per_category'][c]['n']; cells = []
        for T in tabs:
            r = T['per_category'][c]; d = r['twin'] - r['enumerated']
            ok = (d > 0) if c in NEWC else (abs(d) <= 1)
            cells.append(f"{r['twin']} / {r['enumerated']} ({d:+d}{'' if ok else ', fails'})")
        L.append(f'| {c} | {n} | ' + ' | '.join(cells) + ' |')
    cells = [f"**{T['total']['twin']} / {T['total']['enumerated']}** ({T['total']['twin'] - T['total']['enumerated']:+d})" for T in tabs]
    L.append(f"| **total** | **{tabs[0]['total']['n']}** | " + ' | '.join(cells) + ' |')
    cells = [f"{T['total']['twin_rejects']} / {T['total']['enumerated_rejects']}" for T in tabs]
    L.append('| rejects (scored items) | | ' + ' | '.join(cells) + ' |')
    cells = [f"{T['twin_only_field_items']['twin']['correct']} / {T['twin_only_field_items']['enumerated']['correct']} (enum rejects {T['twin_only_field_items']['enumerated']['rejected']})" for T in tabs]
    L.append(f"| twin-only-field items ({tabs[0]['twin_only_field_items']['twin']['n']}) | | " + ' | '.join(cells) + ' |')
    cells = [f"{T['paired']['twin_only_correct']} / {T['paired']['enumerated_only_correct']}, p = {T['paired']['sign_test_two_sided_p']}" for T in tabs]
    L.append('| discordant pairs twin-only / enum-only, sign test (description) | | ' + ' | '.join(cells) + ' |')
    def vv(v): return ('holds' if v['H1_holds'] else 'fails') + f" (parity two-sided {'yes' if v['parity_two_sided'] else 'no'}, one-sided {'yes' if v['parity_one_sided'] else 'no'}, extension {'yes' if v['extension'] else 'no'})"
    L.append('| **H1 rule** | | ' + ' | '.join(vv(T['verdict']) for T in tabs) + ' |')


FZ = X['frozen_gold']
cat_table([FZ['v1'], FZ['A'], FZ['B']], ['v1 (original)', 'set A', 'set B'], '### Frozen gold, 61 resolvable instructions')
cat_table([FZ['v1'], FZ['control']], ['v1 seed 0', 'control (tonight, original wording)'], '### Reproduction control, frozen gold')
S = X['corrected_gold_sensitivity']
cat_table([S['corrected_copy']['v1'], S['corrected_copy']['A'], S['corrected_copy']['B']], ['v1 records', 'set A + SA', 'set B + SB'],
          '### SENSITIVITY (not adopted): corrected copy of the gold, 69 instructions')
L.append('\n### SENSITIVITY: each change alone on the frozen gold, and each reading alone on the corrected copy (total twin / enum of n; H1 rule)\n')
L.append('| variant | v1 records | set A (+SA) | set B (+SB) |'); L.append('|---|---|---|---|')
def short(T): v = T['verdict']; return f"{T['total']['twin']} / {T['total']['enumerated']} of {T['total']['n']}; {'holds' if v['H1_holds'] else 'fails'} (parity {'yes' if v['parity_two_sided'] else 'no'}, ext {'yes' if v['extension'] else 'no'})"
for grp, lab in (('each_change_alone', 'change alone: '), ('each_reading_alone_on_corrected', 'reading on corrected: ')):
    for f, t3 in S[grp].items():
        L.append(f'| {lab}{f} | ' + ' | '.join(short(t3[k]) for k in ('v1', 'A', 'B')) + ' |')
L.append('\n### Outcome kept under both paraphrases (frozen gold, 61), per arm\n')
K = X['outcome_kept']
L.append('| | twin | enumerated |'); L.append('|---|---|---|')
for x, lab in (('kept_under_both', 'same outcome in v1, A and B'), ('kept_in_A', 'same in v1 and A'), ('kept_in_B', 'same in v1 and B')):
    L.append(f"| {lab} | {K['twin'][x]} of 61 | {K['enumerated'][x]} of 61 |")
pats = ['correct in all three', 'wrong in all three', 'correct in v1, lost in A', 'correct in v1, lost in B', 'correct in v1, lost in both',
        'wrong in v1, gained in A', 'wrong in v1, gained in B', 'wrong in v1, gained in both']
for p in pats:
    L.append(f"| {p} | {len(K['twin']['patterns'].get(p, []))} {' '.join(K['twin']['patterns'].get(p, [])) if p not in ('correct in all three', 'wrong in all three') else ''} | {len(K['enumerated']['patterns'].get(p, []))} {' '.join(K['enumerated']['patterns'].get(p, [])) if p not in ('correct in all three', 'wrong in all three') else ''} |")
for x in ('A', 'B', 'both'):
    L.append(f"| final answer identical to v1's ({x}) | {K['twin']['identical_final_as_v1'][x]} | {K['enumerated']['identical_final_as_v1'][x]} |")
L.append('\nPer category, outcome kept under both (twin; enumerated):\n')
L.append('| category | n | twin kept both / in A / in B | enumerated kept both / in A / in B |'); L.append('|---|---|---|---|')
for c in CATS:
    if c in K['twin']['per_category']:
        t, e = K['twin']['per_category'][c], K['enumerated']['per_category'][c]
        L.append(f"| {c} | {t['n']} | {t['kept_under_both']} / {t['kept_in_A']} / {t['kept_in_B']} | {e['kept_under_both']} / {e['kept_in_A']} / {e['kept_in_B']} |")
L.append('\n### Twin-arm failure types (frozen gold; mechanical, first rule that applies; descriptive)\n')
W = ['original (v1 seed 0)', 'set A', 'set B']
L.append('| type | ' + ' | '.join(W) + ' |'); L.append('|---|---|---|---|')
for t in F['types']:
    L.append(f'| {t} | ' + ' | '.join(str(F['by_type'][w][t]) for w in W) + ' |')
L.append('| **twin errors** | ' + ' | '.join(f"**{F['n_wrong'][w]}**" for w in W) + ' |')
L.append('\nBy category (type: count):\n')
for w in W:
    L.append(f"- {w}: " + '; '.join(f"{c}: " + ', '.join(f'{t} {n}' for t, n in sorted(v.items())) for c, v in F['by_category'][w].items()))
L.append('\n### Instruction-level outcomes (frozen gold; 1 correct, 0 wrong; v1 / A / B)\n')
L.append('| id | category | twin v1 A B | enum v1 A B |'); L.append('|---|---|---|---|')
b = lambda v: '1' if v else '0'
for i, p in X['per_instruction'].items():
    L.append(f"| {i} | {p['category']} | {b(p['twin']['v1'])} {b(p['twin']['A'])} {b(p['twin']['B'])} | {b(p['enumerated']['v1'])} {b(p['enumerated']['A'])} {b(p['enumerated']['B'])} |")
open(OUT, 'w').write('\n'.join(L) + '\n'); print('wrote', OUT)
