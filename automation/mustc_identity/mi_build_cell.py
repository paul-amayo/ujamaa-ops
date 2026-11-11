#!/usr/bin/env python3
"""MuST-C identity feasibility (JS-3, 2026-11-11), step 4: hierarchy + supervision id maps for one rig stop (design of record:
Sankofa_JIRS/numbers/mustc_identity/DESIGN.md, items 2, 4, 5).

  hierarchy   <identity>/marker_hierarchy.json, the schema of automation/lindendhof/lindendhof_identity_build.py: objects = the
              on-row laser instances of the date (id = laser census id, row_id = laser census row, xyz = field frame), rows, no fruit.
  association a SAM 3 mask is labelled with a plant if it contains the projected point (plants.json, mi_project.py) of exactly one
              laser instance and that instance is an on-row node; two or more points = ambiguous, only off-row = off-row, none =
              no census point: all three left unlabelled. No depth test.
  supervision <cell>/supervision/trees_only/<frame name> = uint16 PNG data (65535 = unlabelled) stored under the workspace frame
              name (the HiGH tools open the map by frame name; the rig frames end .jpeg), manifest.json (supervision_manifest/v1),
              <cell>/split_names.json (train = the workspace's training cameras, eval = its held-out cameras), <cell>/frames.json
              (all views), and the per-mask / per-(plant, view) records masks.csv, plant_views.csv, build_summary.json.
Either box python with numpy + PIL:  mi_build_cell.py --dir <stop dir> --masks <mask dir> --identity <identity dir> --cell <cell dir>
                                                      --split <workspace split.json> [--census-json lmi_<date>.json]"""
import argparse, csv, datetime, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH'); from word_utils import get_word_for_id
ap = argparse.ArgumentParser(); ap.add_argument('--dir', required=True); ap.add_argument('--masks', required=True); ap.add_argument('--identity', required=True)
ap.add_argument('--cell', required=True); ap.add_argument('--split', required=True); a = ap.parse_args()
D, M, ID, CELL = Path(a.dir), Path(a.masks), Path(a.identity), Path(a.cell); SUP = CELL / 'supervision/trees_only'; SUP.mkdir(parents=True, exist_ok=True); ID.mkdir(parents=True, exist_ok=True)
UNLAB = 65535; P = json.load(open(D / 'plants.json')); C = json.load(open(D / 'cameras.json')); split = json.load(open(a.split))
plants = P['plants']; nodes = [p for p in plants if p['on_row']]; rows = sorted({p['row'] for p in nodes})
HJ = ID / 'marker_hierarchy.json'
H = {'objects': [{'id': p['id'], 'name': f"Object Object({p['id']})", 'row_id': p['row'], 'super_row_id': -1, 'section_id': -1, 'xyz': [round(float(v), 4) for v in p['xyz_ref']]} for p in nodes],
     'rows': [{'id': r, 'object_ids': [p['id'] for p in nodes if p['row'] == r]} for r in rows], 'fruits': [],
     'source': f"MuST-C plot 198 laser census {P['laser_census']} (on-row instances under the row prior; positions in the {P['reference']} frame, point rule {P['point_rule']})", 'n_objects': len(nodes)}
if HJ.exists():
    old = json.load(open(HJ)); assert [o['id'] for o in old['objects']] == [o['id'] for o in H['objects']] and [o['row_id'] for o in old['objects']] == [o['row_id'] for o in H['objects']], 'hierarchy on disk differs'
else:
    json.dump(H, open(HJ, 'w'), indent=1)
names = sorted(C['frames'], key=lambda n: int(C['frames'][n]['label'].split('_')[1])); present = set(); mrec, prec = [], []; per_view = {}
for name in names:
    label = C['frames'][name]['label']; lab = np.array(Image.open(M / f'{label}.png')); inst = json.load(open(M / f'{label}.json')); Hh, Ww = lab.shape
    pts = []
    for p in plants:
        v = p['views'][name]
        if v.get('in_image'): pts.append((p, int(v['u']), int(v['v'])))
    hit = {}   # mask idx -> plants whose projected point it contains
    for p, u, v in pts:
        k = int(lab[v, u])
        if k > 0: hit.setdefault(k, []).append(p)
    out = np.full((Hh, Ww), UNLAB, np.uint16); cnt = {'labelled': 0, 'ambiguous': 0, 'off_row_only': 0, 'no_census_point': 0}; mask_of = {}
    for it in inst:
        hp = hit.get(it['idx'], [])
        if len(hp) == 1 and hp[0]['on_row']: oc = 'labelled'; out[lab == it['idx']] = hp[0]['id']; mask_of[hp[0]['id']] = it['idx']
        elif len(hp) >= 2: oc = 'ambiguous'
        elif len(hp) == 1: oc = 'off_row_only'
        else: oc = 'no_census_point'
        cnt[oc] += 1
        mrec.append({'view': label, 'split': C['frames'][name]['split'], 'idx': it['idx'], 'score': it['score'], 'area_px': it['area'], 'n_points': len(hp), 'ids': ' '.join(str(p['id']) for p in hp), 'outcome': oc})
    for p in plants:
        v = p['views'][name]
        if not v.get('in_image'): oc, k = 'not_in_image', 0
        else:
            k = int(lab[int(v['v']), int(v['u'])])
            if k == 0: oc = 'no_mask_at_point'
            elif not p['on_row']: oc = 'off_row_instance'
            elif len(hit[k]) >= 2: oc = 'mask_ambiguous'
            else: oc = 'labelled'
        prec.append({'plant': p['id'], 'row': p['row'], 'on_row': int(p['on_row']), 'in_soil_polygon': int(p['in_soil_polygon']), 'view': label, 'split': C['frames'][name]['split'], 'outcome': oc, 'mask_idx': k,
                     'labelled_px': int((out == p['id']).sum()) if oc == 'labelled' else 0})
    Image.fromarray(out).save(SUP / name, format='PNG'); ids = sorted(int(u) for u in np.unique(out) if u != UNLAB); present |= set(ids)
    per_view[label] = {'split': C['frames'][name]['split'], 'masks': len(inst), **cnt, 'plants_labelled': len(ids), 'labelled_frac': float((out != UNLAB).mean())}
    print(f"[build] {label} ({C['frames'][name]['split']}): {len(inst)} masks -> {cnt}; {len(ids)} plants labelled, {100 * per_view[label]['labelled_frac']:.2f} % of the frame", flush=True)
train = [n for n in names if C['frames'][n]['label'] in split['train']]; ev = [n for n in names if C['frames'][n]['label'] in split['test']]
assert len(train) + len(ev) == len(names) and all(C['frames'][n]['split'] == 'train' for n in train) and all(C['frames'][n]['split'] == 'eval' for n in ev)
man = {'schema': 'supervision_manifest/v1', 'created': datetime.datetime.now().isoformat(timespec='seconds'), 'unlabelled_value': UNLAB, 'fruit_id_base': 10000,
       'filter_spec': {'name': 'mustc_census_point_in_sam3_mask_v1', 'rule': 'SAM 3 "plant" masks (confidence 0.3, min area 150 px) on the pinhole-resampled rig views; a mask is labelled with the on-row laser instance whose projected point (x, y, ground + 0.5 h95) is the only census point it contains; ambiguous / off-row / empty masks unlabelled; nothing outside a mask is labelled'},
       'word_table': {str(k): get_word_for_id(int(k), 'mask') for k in sorted(present)}, 'level_table': {str(k): 'tree' for k in sorted(present)},
       'tree_source': {'census': P['laser_census'], 'registration': 'registration.json T_rig_to_utm_datum_corrected', 'masks': str(M)}, 'hierarchy': {'path': str(HJ)}, 'counts': {'frames': len(names)}}
json.dump(man, open(SUP / 'manifest.json', 'w'), indent=1)
json.dump({'source': f'{a.split} x views with maps', 'train': train, 'eval': ev}, open(CELL / 'split_names.json', 'w'), indent=0)
json.dump({'frames': names, 'info': {n: {'split': C['frames'][n]['split']} for n in names}}, open(CELL / 'frames.json', 'w'), indent=1)
for fn, recs in (('masks.csv', mrec), ('plant_views.csv', prec)):
    with open(CELL / fn, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
lab_tr = {}; lab_ev = {}
for r in prec:
    if r['outcome'] == 'labelled': (lab_tr if r['split'] == 'train' else lab_ev).setdefault(r['plant'], []).append(r['view'])
summ = {'stop': P['stop'], 'laser_instances': len(plants), 'nodes_on_row': len(nodes), 'rows': len(rows), 'views': len(names), 'census_views': len(train), 'held_out_views': [C['frames'][n]['label'] for n in ev],
        'masks_total': len(mrec), 'masks_by_outcome': {k: sum(1 for m in mrec if m['outcome'] == k) for k in ('labelled', 'ambiguous', 'off_row_only', 'no_census_point')},
        'nodes_with_mask_in_census_views': len(lab_tr), 'census_views_per_labelled_node': {str(k): len(v) for k, v in sorted(lab_tr.items())},
        'nodes_with_mask_in_held_out_views': {str(k): v for k, v in sorted(lab_ev.items())}, 'nodes_in_soil_polygon': sorted(p['id'] for p in nodes if p['in_soil_polygon']),
        'plant_view_pairs_by_outcome': {k: sum(1 for r in prec if r['outcome'] == k) for k in ('labelled', 'mask_ambiguous', 'no_mask_at_point', 'off_row_instance', 'not_in_image')},
        'ids_in_manifest': sorted(present), 'per_view': per_view}
json.dump(summ, open(CELL / 'build_summary.json', 'w'), indent=1)
nv = np.array(list(summ['census_views_per_labelled_node'].values()))
print(f"[build] hierarchy {len(nodes)} plants / {len(rows)} rows -> {HJ}; {len(lab_tr)} nodes labelled in >= 1 census view (views per node: median {np.median(nv) if len(nv) else 0}, min {nv.min() if len(nv) else 0}, max {nv.max() if len(nv) else 0}); "
      f"{len(lab_ev)} nodes labelled in a held-out view; masks {summ['masks_by_outcome']}; pairs {summ['plant_view_pairs_by_outcome']}", flush=True)
