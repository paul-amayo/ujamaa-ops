#!/usr/bin/env python3
"""The row of a keyframe, lit by ARGMAX over the row words (UJAMAA 2026-10-03; Paul: "retrain reseed and show me the row of
3112"). 01 chunk 3_1 after the CORAL-strict swap (36 rows, embedder 01_13B_v1g retrained, seed B re-seeded). "This row" = the
row of the nearest scene-graph tree in front (demo_path trees). A pixel is lit for the row when that row's word scores highest
among ALL row words (own point + non-extrapolated walk, max), identity norm >= 0.5 - no split, no threshold, no cut. The live
rule (per-frame Otsu + absence check) lights nothing on these frames: the row fills the frame, p99 - p50 < 0.04. Also scored:
IoU vs the SAM3 supervision of the row's trees. ADD_MASKS="K:path.npy,..." adds SAM3 detections the global-id lifting left without
an id (boolean HxW owned masks, picked by LiDAR depth/3D box, not by the prediction) to that frame's row ground truth and reports
the IoU against it too (dashed green). Usage: row_argmax_frame.py K [K ...]"""
import sys, os, math, json, numpy as np, torch
ks = [int(a) for a in sys.argv[1:]] or [3112]
sys.argv = [sys.argv[0], '18', '0', '0']
exec(open('/home/paperspace/code/automation/demo/row_iou_frames.py').read().split("try: F = ImageFont")[0])   # loader: CH, NI, ims, Cam, row_of (prod hierarchy), NB
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
def heats(feat, words):
    E = NI.E[[NI.widx[w] for w in words]]; h_, w_, D = feat.shape; ft = feat.reshape(-1, D).float()
    out = torch.full((len(words), ft.shape[0]), -1.0, device=ft.device); idx = (ft.norm(dim=-1) >= 0.5).nonzero(as_tuple=True)[0]
    for j in range(0, int(idx.shape[0]), 8192):
        sel = idx[j:j + 8192]; hh = L.exp_map0(ft[sel], curv=NI.curv)
        pth, msk = L.get_interpolated_hyperbolic_features(hh, steps=4, curv=NI.curv, max_dist=11.1, return_mask=True)
        d = NI.hyper.decode_features(pth, project=True); d = d / d.norm(dim=-1, keepdim=True)
        sm = (d @ E.T).view(sel.shape[0], 4, len(words)).masked_fill(msk.to(d.device).bool().unsqueeze(-1), -1.0)
        d0 = NI.hyper.decode_features(hh, project=True); d0 = d0 / d0.norm(dim=-1, keepdim=True)
        out[:, sel] = torch.cat([sm, (d0 @ E.T).view(sel.shape[0], 1, len(words))], 1).max(1).values.T
    return out.view(len(words), h_, w_)
path = json.load(open('/home/paperspace/logs/demo_chunks/01_3_1_720/demo_path.json')); TREES = {int(k): np.array(v) for k, v in path['trees'].items()}
RIDS = sorted(int(r) for r in NI.row_words); RW = [NI.row_words[str(r)] for r in RIDS]
objs = json.load(open(f'{S}/prod/bateleur/scene_graph/marker_hierarchy.json'))['objects']; xz = {o['id']: np.array([o['xyz'][0], o['xyz'][2]]) for o in objs}
fig, axs = plt.subplots(len(ks), 2, figsize=(22, 6.4 * len(ks)), dpi=90, gridspec_kw=dict(width_ratios=[1.8, 1])); axs = np.atleast_2d(axs)
for i, k in enumerate(ks):
    kf = f'kf_{k:06d}.png'; im = ims[kf]; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec; c2w = np.linalg.inv(w2c)
    cam = Cam(c2w, W, H, 2 * math.atan(H / (2 * fy)), cx / W, cy / H); img, _ = CH.render(cam, 3.0)
    if kf in EXPO: E_ = np.array(EXPO[kf], np.float32); img = torch.einsum('ij,jhw->ihw', torch.tensor(E_[:, :3]).cuda(), img) + torch.tensor(E_[:, 3]).cuda()[:, None, None]
    rgb = (img.clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy()
    vis = []
    for t, x in TREES.items():
        p = w2c[:3, :3] @ x + w2c[:3, 3]
        if p[2] > 0.5:
            u_, v_ = fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy
            if 0 <= u_ < W and 0 <= v_ < H: vis.append((float(np.linalg.norm(x - c2w[:3, 3])), t))
    t0 = sorted(vis)[0][1]; r0 = row_of[t0]; members = sorted(t for t in row_of if row_of[t] == r0)
    hm = heats(NI.feature_pass(CH, cam, 3.0), RW); valid = (hm > -1).any(0); am = hm.argmax(0)
    lit = (valid & (am == RIDS.index(r0))).cpu().numpy().astype(np.uint8)
    lit = np.array(Image.fromarray(lit * 255).resize((W, H), Image.NEAREST)) > 0
    sup = np.array(Image.open(f'{NB}/supervision/trees_only/{kf}'), np.uint16); gt = np.isin(sup, members)
    add = np.zeros_like(gt)
    for spec in [x for x in os.environ.get('ADD_MASKS', '').split(',') if x]:
        kk, pth = spec.split(':', 1)
        if int(kk) == k: add |= np.load(pth)
    gtp = gt | add; iou_p = float((lit & gtp).sum() / max((lit | gtp).sum(), 1)) if add.any() else None
    iou = float((lit & gt).sum() / max((lit | gt).sum(), 1)); prec = float((lit & gt).sum() / max(lit.sum(), 1)); rec = float((lit & gt).sum() / max(gt.sum(), 1))
    per = {t: int((sup == t).sum()) for t in sorted(int(u) for u in np.unique(sup) if u < 10000)}
    lit_on = {t: round(float((lit & (sup == t)).sum() / max(per[t], 1)), 2) for t in per if per[t] > 500}
    print(f'{kf}: nearest tree in front {t0} -> row {r0} ("{NI.row_words[str(r0)]}"), members {members}; argmax-lit {int(lit.sum())} px; '
          f'IoU vs SAM3 {iou:.3f} (precision {prec:.2f}, recall {rec:.2f})' + (f'; with the id-less SAM3 masks added {iou_p:.3f} (recall {float((lit & gtp).sum() / max(gtp.sum(), 1)):.2f})' if iou_p is not None else '') + '; fraction of each labelled tree lit: ' + ', '.join(f'{t} (row {row_of.get(t)}) {f}' for t, f in lit_on.items()))
    fr = rgb.astype(np.float32); fr[lit] = fr[lit] * 0.45 + np.array([235, 104, 52], np.float32) * 0.55
    ax = axs[i, 0]; ax.imshow(fr.astype(np.uint8)); ax.contour(gt, levels=[0.5], colors=['#00c853'], linewidths=1.6); ax.set_axis_off()
    if add.any(): ax.contour(add & ~gt, levels=[0.5], colors=['#00c853'], linewidths=1.6, linestyles='--')
    ax.set_title(f'{kf}: "this row" = row {r0} (nearest tree in front: {t0}). Orange = pixels whose best row word is row {r0}; green outline = SAM3 trees of row {r0}.\n'
                 f'Row {r0} trees: {", ".join(map(str, members))}. IoU vs SAM3 {iou:.2f} (precision {prec:.2f}, recall {rec:.2f})'
                 + (f'; {iou_p:.2f} with SAM3\'s id-less mask of the missing tree added (dashed green).' if iou_p is not None else '.'), fontsize=10, loc='left')
    ax = axs[i, 1]; cc = xz[t0]
    for r in sorted({v for v in row_of.values() if v >= 0}):
        Q = np.array([xz[t] for t in row_of if row_of[t] == r]); Q = Q[np.argsort(Q[:, 1])]
        ax.plot(Q[:, 0], Q[:, 1], '-o', color='#eb6834' if r == r0 else '#2a78d6', lw=3 if r == r0 else 1, ms=6 if r == r0 else 3, alpha=1 if r == r0 else 0.5)
    for t in (146, 148, 147, 157): ax.text(xz[t][0] + 0.6, xz[t][1], str(t), fontsize=9)
    ax.set_xlim(cc[0] - 20, cc[0] + 20); ax.set_ylim(cc[1] - 30, cc[1] + 20); ax.set_aspect('equal')
    ax.set_title(f'New scene graph from above (CORAL strict): row {r0} in orange', fontsize=10, loc='left')
fig.tight_layout(); fn = f'/home/paperspace/data/demo_video_v2/citrus_a_bar/row_argmax_{"_".join(str(k) for k in ks)}.png'; fig.savefig(fn); print('[fig]', fn)
