#!/usr/bin/env python3
"""7 row_words_on_trees.py (h3dgs env + CUDA_HOME=_cuda12): which row did the survey embedder learn each tree under?
(UJAMAA 2026-10-03). Native seed B on 01 chunk 3_1: per keyframe, the plain containment heat of every row word (rows 15-21)
averaged over each labelled tree's pixels (chunk supervision), and the best-scoring row word per tree. Tree 147 (scene-graph
row 17) stands across the lane from 146 (row 17); if the row-17 word wins on both, the embedder learned the diagonal row."""
import sys, math, json, numpy as np, torch
sys.argv = [sys.argv[0], '17', '0', '0']
exec(open('/home/paperspace/code/automation/demo/row_iou_frames.py').read().split("try: F = ImageFont")[0])   # loader: CH, NI, ims, Cam, row_of, NB
def heats(feat, words):   # citrus_a_bar_video.heats: own point + non-extrapolated 4-step walk, max; norm check 0.5
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
ROWS = [r for r in range(15, 22) if str(r) in NI.row_words]; words = [NI.row_words[str(r)] for r in ROWS]
print('row words:', dict(zip(ROWS, words)))
for k in [int(a) for a in sys.argv[4:]] if len(sys.argv) > 4 else (3133, 3138, 3141, 3143):
    kf = f'kf_{k:06d}.png'; im = ims[kf]; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec
    cam = Cam(np.linalg.inv(w2c), W, H, 2 * math.atan(H / (2 * fy)), cx / W, cy / H)
    hm = heats(NI.feature_pass(CH, cam, 3.0), words).cpu().numpy()
    sup = np.array(Image.open(f'{NB}/supervision/trees_only/{kf}'), np.uint16)
    if sup.shape != hm.shape[1:]: sup = np.array(Image.fromarray(sup).resize((hm.shape[2], hm.shape[1]), Image.NEAREST))   # heat is at the feature pass's half resolution
    for t in sorted(int(u) for u in np.unique(sup) if u < 10000):
        m = sup == t
        if m.sum() < 500: continue
        s = [float(hm[i][m & (hm[i] > -1)].mean()) if (m & (hm[i] > -1)).any() else float('nan') for i in range(len(ROWS))]
        print(f'{kf} tree {t} (scene-graph row {row_of.get(t)}, {int(m.sum())} px): ' + '  '.join(f'r{r} {v:.3f}' for r, v in zip(ROWS, s)) + f'  -> best row word: {ROWS[int(np.nanargmax(s))]}')
