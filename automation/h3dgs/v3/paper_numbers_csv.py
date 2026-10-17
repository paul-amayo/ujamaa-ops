#!/usr/bin/env python3
"""paper_numbers_csv.py — the paper's evidence as flat CSVs with provenance (2026-10-13), from the aggregated protocol JSON (protocol_table.py)
and the fleet results: e1_levels.csv (per survey x level x split), e2_arms.csv (geometry arms), e2b_seed.csv (seeding arms, cells that ran),
fleet_psnr.csv (per-survey v3 reconstruction medians). Every row names its source file. Written to aru_sil_core/src/docs/paper/numbers/.
system python3:  paper_numbers_csv.py [--inventory /home/paperspace/logs/identity_inventory_v2.jsonl]"""
import argparse, csv, json, os, statistics as st, subprocess, sys
ap = argparse.ArgumentParser(); ap.add_argument('--inventory', default='/home/paperspace/logs/identity_inventory_v2.jsonl'); ap.add_argument('--min-frac', default='0.002', help='protocol_table.py --min-frac: the paper protocol counts heads covering >= 0.2 % of the frame (10-17)'); a = ap.parse_args()
OUT = '/home/paperspace/code/aru_sil_core/src/docs/paper/numbers'; V3 = '/home/paperspace/code/automation/h3dgs/v3'; os.makedirs(OUT, exist_ok=True)
subprocess.run([sys.executable, f'{V3}/protocol_table.py', '--tag', 'v3', '--inventory', a.inventory, '--min-frac', a.min_frac, '--compare', 'v3,hyper_own,flat96,euclid', '--json', f'{OUT}/e2_arms.json', '--md', '/home/paperspace/logs/e2_arms_table.md'], check=True, capture_output=True)
subprocess.run([sys.executable, f'{V3}/protocol_table.py', '--tag', 'v3', '--inventory', a.inventory, '--min-frac', a.min_frac, '--compare', 'v3,flat96,flat30', '--json', f'{OUT}/e2_arms_flat30.json', '--md', '/home/paperspace/logs/e2_arms_flat30_table.md'], check=True, capture_output=True)   # the equal-storage arm, on the cells it has reached
subprocess.run([sys.executable, f'{V3}/protocol_table.py', '--tag', 'v3', '--inventory', a.inventory, '--min-frac', a.min_frac, '--compare', 'v3,zero3000,refine3000,projvote', '--json', f'{OUT}/e2b_seed.json', '--md', '/home/paperspace/logs/e2b_seed_table.md'], check=True, capture_output=True)
e1 = json.load(open(f'{OUT}/e2_arms.json'))   # the per-survey E1 block is in every protocol_table JSON
SITE = {'01_13B_Jackal': 'citrus (CitrusFarm 01_13B)', '02_13B_Jackal': 'citrus (CitrusFarm 02_13B)', '03_13B_Jackal': 'citrus (CitrusFarm 03_13B)', '04_13D_Jackal': 'citrus (CitrusFarm 04_13D)', '05_13D_Jackal': 'citrus (CitrusFarm 05_13D)',
        'apr_2026_zed': 'Klapmuts April (plants)', 'dec_2025_ten_rows': 'Klapmuts Dec (grow bags)', 'nov_2025_a300': 'Lindendhof (trunk-anchored trees)', 'IMG_7990_s1': 'image farm pepper (phone)', 'IMG_7993_s0': 'image farm cabbage (phone)'}
STR = {'>=3%': 'near', '1-3%': 'mid', '0.2-1%': 'far', '<0.2%': 'tiny'}
with open(f'{OUT}/e1_levels.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['survey', 'site', 'level', 'split', 'n', 'oracle_med', 'argmax_med', 'argmax_prec_med', 'argmax_rec_med', 'fixed_cell_med', 'fixed_survey_med', 'thr_survey'] + [f'{s}_{k}' for s in STR.values() for k in ('n', 'argmax_med', 'oracle_med')] + ['nesting_mean', 'void_mean', 'cells', 'frames', 'source'])
    for sv, d in e1['surveys'].items():
        for lvl in ('TREE', 'ROW', 'FRUIT'):
            if lvl not in d: continue
            for sp in ('train', 'eval'):
                r = d[lvl][sp]; strata = r.get('strata', {}); row = [sv, SITE.get(sv, sv), lvl, 'census' if sp == 'train' else 'held-out', r['n'], r['oracle_med'], r['argmax_med'], r['argmax_prec_med'], r['argmax_rec_med'], r.get('fixed_cell_med'), r.get('fixed_survey_med'), r.get('thr_survey')]
                for s in STR: row += [strata.get(s, {}).get('n'), strata.get(s, {}).get('argmax_med'), strata.get(s, {}).get('oracle_med')] if sp == 'eval' else [None, None, None]
                row += [d['nesting_mean'], d['void_mean'], d['cells'], d['frames'], 'aru_sil_core/src/docs/paper/numbers/e2_arms.json <- protocol_table.py over <identity>/protocol_v3.jsonl (argmax_eval.py)']; w.writerow([None if (isinstance(x, float) and x != x) else x for x in row])
def arms_csv(js, name, tags, extra=()):
    with open(f'{OUT}/{name}', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['survey', 'site', 'level', 'n'] + tags + ['near_n'] + [f'near_{t}' for t in tags] + [f'{e}_{t}' for e in extra for t in tags] + ['source'])
        for r in js['arms']['rows']:
            w.writerow([r['survey'], SITE.get(r['survey'], r['survey']), r['level'], r['n']] + [r['values'].get(t) for t in tags] + [r.get('near_n')] + [r.get('near_values', {}).get(t) for t in tags] + [r.get(f'{e}_values', {}).get(t) for e in extra for t in tags] + [f"{OUT.split('/code/')[1]}/{name.replace('.csv', '.json')} <- protocol_table.py --compare over <identity>/protocol_<arm>.jsonl"])
arms_csv(e1, 'e2_arms.csv', ['v3', 'hyper_own', 'flat96', 'euclid']); arms_csv(json.load(open(f'{OUT}/e2b_seed.json')), 'e2b_seed.csv', ['v3', 'zero3000', 'refine3000', 'projvote'])
arms_csv(json.load(open(f'{OUT}/e2_arms_flat30.json')), 'e2_arms_flat30.csv', ['v3', 'flat96', 'flat30'])
subprocess.run([sys.executable, f'{V3}/protocol_table.py', '--tag', 'v3', '--inventory', a.inventory, '--min-frac', a.min_frac, '--compare', 'cat_v3,cat_flat96,cat_lite', '--json', f'{OUT}/e3_category.json', '--md', '/home/paperspace/logs/e3_category_table.md'], check=True, capture_output=True)
arms_csv(json.load(open(f'{OUT}/e3_category.json')), 'e3_category.csv', ['cat_v3', 'cat_flat96', 'cat_lite'], extra=('oracle', 'rel05'))
subprocess.run([sys.executable, f'{V3}/protocol_table.py', '--tag', 'v3', '--inventory', a.inventory, '--min-frac', a.min_frac, '--compare', 'v3,seed1,seed2,seed3', '--json', f'{OUT}/e7_seeds.json', '--md', '/home/paperspace/logs/e7_seeds_table.md'], check=True, capture_output=True)
arms_csv(json.load(open(f'{OUT}/e7_seeds.json')), 'e7_seeds.csv', ['v3', 'seed1', 'seed2', 'seed3'])
subprocess.run([sys.executable, f'{V3}/e4_fruit_table.py', '--inventory', a.inventory, '--json', f'{OUT}/e4_fruit.json', '--md', '/home/paperspace/logs/e4_fruit_table.md'], check=True, capture_output=True)
e4 = json.load(open(f'{OUT}/e4_fruit.json'))
with open(f'{OUT}/e4_fruit.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['survey', 'site', 'split', 'fruit_heads', 'cells', 'oracle_med', 'argmax_med', 'argmax_prec_med', 'argmax_rec_med', 'inst_recall_argmax', 'inst_recall_oracle', 'tiny_n', 'tiny_argmax_med', 'small_n', 'small_argmax_med', 'tree_argmax_med_same_frames', 'source'])
    for sv, d in e4.items():
        for sp, r in d.items(): w.writerow([sv, SITE.get(sv, sv), sp, r['n'], r['cells'], r['oracle_med'], r['argmax_med'], r['argmax_prec_med'], r['argmax_rec_med'], r['inst_recall_argmax'], r['inst_recall_oracle'], r['strata']['<0.2%']['n'], r['strata']['<0.2%']['argmax_med'], r['strata']['0.2-1%']['n'], r['strata']['0.2-1%']['argmax_med'], r['tree_argmax_med'], f"{OUT.split('/code/')[1]}/e4_fruit.json <- e4_fruit_table.py over <identity>/protocol_v3_fruit.jsonl (fruit-picked frames)"])
rows = [json.loads(l) for l in open('/home/paperspace/logs/v3_fleet_results.jsonl') if l.strip()]; last = {}
for r in rows: last[(r['survey'], r['chunk'])] = r   # the latest line per cell wins
by = {}
for (sv, ch), r in last.items(): by.setdefault(sv, []).append(r)
with open(f'{OUT}/fleet_psnr.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['survey', 'site', 'cells', 'grid_median_dB', 'held_out_median_dB', 'train_minutes_median', 'gaussians_median', 'train_views_median', 'source'])
    for sv, rs in sorted(by.items()):
        g = [r['grid_median'] for r in rs if r.get('grid_median')]; h = [r['held_out'] for r in rs if r.get('held_out')]
        w.writerow([sv, SITE.get(sv, sv), len(rs), round(st.median(g), 2) if g else None, round(st.median(h), 2) if h else None, round(st.median([r['train_s'] for r in rs if r.get('train_s')]) / 60, 1), int(st.median([r['gaussians'] for r in rs if r.get('gaussians')])), int(st.median([r['train_views'] for r in rs if r.get('train_views')])), '/home/paperspace/logs/v3_fleet_results.jsonl (v3_fleet.sh)'])
print('\n'.join(f'{n}: {sum(1 for _ in open(f"{OUT}/{n}")) - 1} rows' for n in ('e1_levels.csv', 'e2_arms.csv', 'e2_arms_flat30.csv', 'e2b_seed.csv', 'e3_category.csv', 'e4_fruit.csv', 'e7_seeds.csv', 'fleet_psnr.csv')))
