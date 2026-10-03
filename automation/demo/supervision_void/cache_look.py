# 3a cache_look.py (system python3): laser points each detection lifted in the cluster_tree_instances collect cache, and which
#   global-id 3D boxes they fall in (the centre-tree masks on kf_003112/3113 lift 5 and 1 points, all inside near tree 148)
import json, numpy as np
G = '/home/paperspace/data/citrus_all/01_13B_Jackal/prod/bateleur/sam3_v2'
z = np.load(f'{G}/_collect_cache_v3_db1.5_md10.0_citrus_jackal.npz'); g = json.load(open(f'{G}/global_ids.json'))
keys = z['det_keys']; src = z['sources']; pts = z['pts']; areas = z['det_areas']; kfs = z['det_kfs']
idx = {(int(a), int(b)): i for i, (a, b) in enumerate(keys)}
st = {int(k): v for k, v in g['stats'].items()}
def inside(p, gid):
    s = st[gid]; lo, hi = np.array(s['world_bbox_min']), np.array(s['world_bbox_max']); return np.all((p >= lo) & (p <= hi), axis=1)
order = np.argsort(src, kind='stable'); bounds = np.searchsorted(src[order], np.arange(len(keys) + 1))
for oid, kf, note in [(1617, 3112, 'centre tree, unassigned'), (1618, 3112, 'small, gid 146'), (1632, 3113, 'centre tree, unassigned'), (1633, 3113, 'small, unassigned'), (1660, 3115, 'centre tree, gid 146'), (1661, 3115, 'small, unassigned')]:
    i = idx.get((15, oid))
    if i is None: print(f'oid {oid} kf {kf} ({note}): NOT in the cache -> no laser point survived the gates'); continue
    p = pts[order[bounds[i]:bounds[i + 1]]]
    near = {gid: int(inside(p, gid).sum()) for gid in st}; top = sorted(near.items(), key=lambda t: -t[1])[:4]
    print(f'oid {oid} kf {kf} ({note}): area {int(areas[i])} px, {len(p)} laser pts ({1000*len(p)/max(int(areas[i]),1):.2f}/kpx); y range {p[:,1].min():.2f}..{p[:,1].max():.2f}; inside gid bboxes {top}')
print('gid 146 stats:', {k: (np.round(v, 2).tolist() if isinstance(v, list) else v) for k, v in st[146].items() if k != 'clips'})
