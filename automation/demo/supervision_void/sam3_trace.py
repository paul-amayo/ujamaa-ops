# 2 sam3_trace.py (nerf_new env, pycocotools): every SAM3 detection on the frame -> its global id (member_to_global), how much of the lit
#   tree it covers, what it owns under the painter's smallest-on-top partition, and the supervision ids under it
import json, numpy as np
from PIL import Image
from pycocotools import mask as mu
from scipy import ndimage
S = '/home/paperspace/data/citrus_all/01_13B_Jackal'; G = f'{S}/prod/bateleur/sam3_v2'; NB = f'{S}/experimental/h3dgs_native/chunk_3_1_sam3'
fe = json.load(open(f'{G}/clip_015/frame_entries.json')); m2g = json.load(open(f'{G}/global_ids.json'))['member_to_global']
row_of = {o['id']: o['row_id'] for o in json.load(open(f'{S}/prod/bateleur/scene_graph/marker_hierarchy.json'))['objects']}
L = np.load('lit_row17.npz')
def largest(m):
    lab, n = ndimage.label(m)
    return m if n <= 1 else lab == (1 + np.argmax(np.bincount(lab.ravel())[1:]))
for k in (3112, 3113, 3115):
    ents = fe['frame_entries'].get(str(k - 3000), []); lit = L[f'lit_{k}']
    sup = np.array(Image.open(f'{NB}/supervision/trees_only/kf_{k:06d}.png'), np.uint16)
    dec = [(int(e['obj_id']), largest(mu.decode(e['rle']).astype(bool)), int(e['area']), float(e['score']), e['bbox_xyxy']) for e in ents]
    own = np.full(lit.shape, -1, np.int32)
    for j in sorted(range(len(dec)), key=lambda j: -dec[j][2]): own[dec[j][1]] = j      # smallest-on-top (painter's rule)
    print(f'\nkf_{k:06d}: {len(dec)} SAM3 detections; field-lit row-17 px {int(lit.sum())}; supervision ids {sorted(int(u) for u in np.unique(sup) if u != 65535)}')
    for j, (oid, m, area, sc, bb) in enumerate(dec):
        o = own == j; gid = m2g.get(f'15:{oid}')
        ov = int((m & lit).sum())
        if ov < 0.02 * lit.sum() and gid is None and area < 20000: continue     # small, off the lit tree, unassigned: skip in print
        lab, _ = ndimage.label(o); sizes = np.bincount(lab.ravel())[1:]; ncomp = int((sizes >= 500).sum())
        sv, sc_ = np.unique(sup[o], return_counts=True); sv = {int(a): int(b) for a, b in zip(sv, sc_) if b > 300}
        print(f'  oid {oid:5d} area {area:6d} score {sc:.2f} bbox {[int(x) for x in bb]} | gid {gid} (row {row_of.get(gid)}) | covers {100*ov/max(lit.sum(),1):5.1f}% of lit, owns {int(o.sum())} px ({100*(o & lit).sum()/max(lit.sum(),1):5.1f}% of lit) in {ncomp} piece(s) | supervision under owned px {sv}')
