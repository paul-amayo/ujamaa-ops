#!/usr/bin/env python3
"""16 sam3_mask_1626.py (nerf_new env): one SAM3 detection, what it is (UJAMAA 2026-10-03; Paul: "show me what you think sam3
mask of 1626 is"). kf_003112 #1626 (33.0k px, score 0.36). Panels: (a) the raw mask on the photo; (b) its outline with the two
smaller masks inside it (#1617 centre tree, #1618 small tree); (c) the depths of every LiDAR return inside the FULL raw mask,
split by the 3D tree box each return falls in (lifting cluster boxes from global_ids.json)."""
from PIL import Image
import json, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pycocotools import mask as mu
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
from rig_calib import l2c_flat, load_rig
from cluster_tree_instances import make_laser_projector
from pathlib import Path
S = Path('/home/paperspace/data/citrus_all/01_13B_Jackal'); G = S / 'prod/bateleur/sam3_v2'; W, H = 1280, 720; K = 3112; OID = 1626
I = load_rig(str(S))['intrinsics']; proj = make_laser_projector(S, {'fl_x': I['fx'], 'fl_y': I['fy'], 'cx': I['cx'], 'cy': I['cy']}, W, H, l2c=l2c_flat(str(S)))
st = {int(k): v for k, v in json.load(open(G / 'global_ids.json'))['stats'].items()}
ents = json.load(open(G / 'clip_015/frame_entries.json'))['frame_entries'][str(K % 200)]
dec = {int(e['obj_id']): (mu.decode(e['rle']).astype(bool), int(e['area']), float(e['score'])) for e in ents}
uv, world, z = proj.project(K); u, v = uv[:, 0].astype(int), uv[:, 1].astype(int)
m = dec[OID][0]; ins = m[v, u]; P = world[ins]; zz = z[ins]
def box(g): s = st[g]; return np.all((P >= np.array(s['world_bbox_min'])) & (P <= np.array(s['world_bbox_max'])), axis=1)
b146, b145, b148 = box(146), box(145), box(148)
other = ~(b146 | b145 | b148)
print(f'#{OID}: {int(m.sum())} px, score {dec[OID][2]:.2f}; {int(ins.sum())} LiDAR returns inside: tree 146 box {int(b146.sum())}, tree 145 box {int(b145.sum())}, tree 148 box {int(b148.sum())}, none of these {int(other.sum())}')
print(f'   depth by box: 146 {np.percentile(zz[b146], [5, 50, 95]).round(1) if b146.any() else "-"}, 145 {np.percentile(zz[b145], [5, 50, 95]).round(1) if b145.any() else "-"}, 148 {np.sort(zz[b148]).round(2) if b148.any() else "-"}')
for o in (1617, 1618): print(f'   {100 * (m & dec[o][0]).sum() / m.sum():.1f}% of #{OID} inside #{o}; #{o} inside #{OID}: {100 * (m & dec[o][0]).sum() / dec[o][0].sum():.1f}%')
ph = np.asarray(Image.open(S / f'prod/scratch_sam3/kf_{K:06d}.png').convert('RGB')).astype(np.float32)
x0, x1, y0, y1 = 330, 700, 160, 400; crop = lambda a: a[y0:y1, x0:x1]
fig, axs = plt.subplots(1, 3, figsize=(20, 5.2), dpi=100, gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
a = ph.copy(); a[m] = a[m] * 0.4 + np.array([232, 123, 164], np.float32) * 0.6
axs[0].imshow(crop(a).astype(np.uint8), extent=(x0, x1, y1, y0)); axs[0].contour(np.arange(x0, x1), np.arange(y0, y1), crop(m), levels=[0.5], colors=['white'], linewidths=1.5)
axs[0].set_title(f'SAM3 #{OID} on kf_003112: {int(m.sum()) / 1000:.1f}k px, score {dec[OID][2]:.2f} (pink)', fontsize=10, loc='left')
axs[1].imshow(crop(ph).astype(np.uint8), extent=(x0, x1, y1, y0))
for o, c, lab in ((OID, 'white', f'#{OID}'), (1617, '#eb6834', '#1617 centre tree'), (1618, '#2a78d6', '#1618 small tree')):
    axs[1].contour(np.arange(x0, x1), np.arange(y0, y1), crop(dec[o][0]), levels=[0.5], colors=[c], linewidths=2.0 if o == OID else 1.6, linestyles='--' if o == OID else '-')
    axs[1].plot([], [], color=c, ls='--' if o == OID else '-', label=lab)
axs[1].legend(fontsize=8, loc='lower left'); axs[1].set_title(f'#{OID} (dashed white) vs the two smaller SAM3 masks inside it', fontsize=10, loc='left')
for ax in axs[:2]: ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_axis_off()
bins = np.arange(0, 20.5, 0.5)
axs[2].hist([zz[b148], zz[b146], zz[b145], zz[other]], bins=bins, stacked=True, color=['#7a7a7a', '#eb6834', '#2a78d6', '#cfcfcf'],
            label=[f'in tree 148 box ({int(b148.sum())})', f'in tree 146 box ({int(b146.sum())})', f'in tree 145 box ({int(b145.sum())})', f'in none of these ({int(other.sum())})'])
axs[2].set_xlabel('depth from the camera (m)'); axs[2].set_ylabel('LiDAR returns'); axs[2].legend(fontsize=8)
axs[2].set_title(f'All {int(ins.sum())} LiDAR returns inside #{OID}, by the 3D tree box they fall in', fontsize=10, loc='left')
fig.tight_layout(); fn = '/home/paperspace/data/demo_video_v2/citrus_a_bar/kf003112_sam3_1626.png'; fig.savefig(fn); print('[fig]', fn)
