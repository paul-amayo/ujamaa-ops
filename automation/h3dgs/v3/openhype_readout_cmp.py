#!/usr/bin/env python3
"""openhype_readout_cmp.py (2026-11-10, Paul: "show me classical OpenHype relevancy versus your containment") — the same HiGH feature
map read two ways on the Fig. 1 frames of citrus 05 cell 1_0:
  OpenHype (its released evaluator, openhype_eval/{base,scannetpp}_evaluator.py, reproduced step by step): the pixel's geodesic
    sampled at 50 steps over [0, 11.1] with the steps beyond the feature masked; per step the decoded feature's cosine to the query
    and to the negatives object/things/stuff/texture -> LERF relevancy (min over negatives of softmax(10*[pos, neg])[pos]); steps
    aggregated by a softmax-weighted sum (softmax over the steps, temperature 1); then 0.5 * (30x30 box filter + map); per-frame
    min-max normalisation, x2-1, clip to [0, 1]; mask = > 0.4 (mask_thresh "like langsplat"). Their final morphological smooth()
    of the mask is not reproduced (its source is not in the evaluator).
  containment (the letter's readout, Section III-D): s_q = max over the pixel's own feature and the inner waypoints, argmax over
    the level's addresses present in the cell; the category word "tree" at LERF relevancy >= 0.5 of the same score.
Both read the void-free maps (splat_feature_render.py --void-free 0.5) of /tmp/d36void/maps. nerf_new pixi env.
usage: openhype_readout_cmp.py --out <png> [--json <json>]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, cv2
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
from high_splat_hierarchy_accuracy import build_hyper_embedder
import lorentz as L
import open_clip
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--json', default=''); ap.add_argument('--maps', default='/tmp/d36void/maps'); a = ap.parse_args()
N = Path('/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/v3/chunk_1_0/identity'); dev = 'cuda'
SUP = Path('/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs_native/chunk_1_0_sam3/supervision/trees_only')
H = json.load(open('/home/paperspace/data/citrus_all/05_13D_Jackal/prod/bateleur/scene_graph/marker_hierarchy.json')); row_of = {int(o['id']): int(o['row_id']) for o in H['objects']}; ft = {int(f['id']): int(f['tree_id']) for f in H['fruits']}
hyper = build_hyper_embedder(json.load(open(N / 'features_B.bin.json'))['embedder'], dev); curv = hyper.curv.exp()
bank = np.load(N / 'text_bank.npz', allow_pickle=True); words = list(bank['words']); emb = torch.tensor(bank['emb'], dtype=torch.float32, device=dev); emb = emb / emb.norm(dim=-1, keepdim=True)
meta = json.loads(str(bank['meta'])); wt = {int(k): w for k, w in meta['word_table'].items()}; rw = {int(k): w for k, w in meta['row_words'].items()}
model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k', device=dev); tok = open_clip.get_tokenizer('ViT-B-16')
with torch.no_grad():
    T = model.encode_text(tok(['tree', 'object', 'things', 'stuff', 'texture']).to(dev)).float(); T = T / T.norm(dim=-1, keepdim=True)
e_tree, NEG = T[0], T[1:]
def decode_steps(ft, steps):   # decoded, normalised features along the geodesic: [P, steps, 512], mask of extrapolated steps [P, steps]
    h = L.exp_map0(ft, curv=curv); path, mask = L.get_interpolated_hyperbolic_features(h, steps=steps, curv=curv, max_dist=11.1, return_mask=True)
    d = hyper.decode_features(path, project=True); d = d / d.norm(dim=-1, keepdim=True); d0 = hyper.decode_features(h, project=True); d0 = d0 / d0.norm(dim=-1, keepdim=True)
    return d.view(ft.shape[0], steps, -1), mask.to(dev).bool().view(ft.shape[0], steps), d0
def lerf(cp, cn):   # cp [..], cn [.., 4] -> min over negatives of softmax(10*[pos, neg])[pos]
    return torch.softmax(torch.stack([cp[..., None].expand_as(cn), cn], -1) * 10.0, -1)[..., 0].min(-1).values
def openhype_mask(ft, q, Hh, Ww, void):
    d, ext, _ = decode_steps(ft, 50); rel = lerf(d @ q, d @ NEG.T)                       # [P, 50]
    rel = rel.masked_fill(ext, float('-inf')); w_ = torch.softmax(rel, 1); rel = rel.masked_fill(ext, 0.0); agg = (w_ * rel).sum(1)
    agg[void] = 0.0; m = agg.view(Hh, Ww).cpu().numpy().astype(np.float32)
    m = 0.5 * (cv2.filter2D(m, -1, np.ones((30, 30), np.float32) / 900.0) + m)
    o = (m - m.min()) / (m.max() - m.min() + 1e-9) * 2.0 - 1.0; o = np.clip(o, 0, 1); return o > 0.4, m
def containment(ft, qs):   # [P, len(qs)] containment scores (own feature + inner waypoints of the 4-step walk)
    d, ext, d0 = decode_steps(ft, 4); sm = (d @ qs.T).masked_fill(ext[..., None], -1.0).max(1).values; return torch.maximum(sm, d0 @ qs.T)
def iou(p, t): u = (p | t).sum(); return float((p & t).sum() / u) if u else 0.0
res = {}; panels = []
for fr in ('kf_002411', 'kf_000024'):
    z = np.load(Path(a.maps) / f'{fr}.npz'); feat = torch.tensor(z['features'], dtype=torch.float32, device=dev); Hh, Ww, D = feat.shape; ft = feat.reshape(-1, D); void = ft.norm(dim=-1) < 0.5
    I = np.array(Image.fromarray(np.array(Image.open(SUP / f'{fr}.png'))).resize((Ww, Hh), Image.NEAREST)).astype(np.int64); rgb = z['rgb']
    with torch.no_grad():
        if fr == 'kf_002411':
            tw = [t for t in sorted(wt) if t < 10000 and wt[t] in words]; Qt = emb[[words.index(wt[t]) for t in tw]]; rws = [r for r in sorted(rw) if rw[r] in words]; Qr = emb[[words.index(rw[r]) for r in rws]]
            st = containment(ft, Qt); sr = containment(ft, Qr); amt = np.array(tw)[st.argmax(1).cpu().numpy()]; amr = np.array(rws)[sr.argmax(1).cpu().numpy()]; vv = void.cpu().numpy()
            # category "tree": containment score of the word, LERF relevancy >= 0.5
            d, ext, d0 = decode_steps(ft, 4); ctree = torch.maximum((d @ e_tree).masked_fill(ext, -1).max(1).values, d0 @ e_tree)
            cneg = torch.maximum((d @ NEG.T).masked_fill(ext[..., None], -1).max(1).values, d0 @ NEG.T); crel = lerf(ctree, cneg); crel[void] = 0
            queries = [('"tree"', e_tree, I < 10000, (crel >= 0.5).view(Hh, Ww).cpu().numpy()),
                       ('tree 36', emb[words.index(wt[36])], I == 36, ((amt == 36) & ~vv).reshape(Hh, Ww)),
                       ('row 4', emb[words.index(rw[4])], np.isin(I, [t for t, r in row_of.items() if r == 4]), ((amr == 4) & ~vv).reshape(Hh, Ww))]
        else:
            fw = [k for k in sorted(wt) if wt[k] in words]; Qa = emb[[words.index(wt[k]) for k in fw]]; sa = containment(ft, Qa); ama = np.array(fw)[sa.argmax(1).cpu().numpy()]; vv = void.cpu().numpy()
            f5 = [10000 + f for f, t in ft.items() if t == 5 and 10000 + f in wt]; q5 = emb[[words.index(wt[k]) for k in f5]].mean(0); q5 = q5 / q5.norm()
            queries = [('fruit of tree 5', q5, np.isin(I, f5), (np.isin(ama, f5) & ~vv).reshape(Hh, Ww))]
        for name, q, truth, cmask in queries:
            om, oheat = openhype_mask(ft, q, Hh, Ww, void); res[name] = dict(openhype_iou=iou(om, truth), containment_iou=iou(cmask, truth), frame=fr)
            panels.append((name, rgb, oheat, om, cmask, res[name])); print(f"[cmp] {fr} {name:16s} OpenHype readout IoU {res[name]['openhype_iou']:.3f}   containment IoU {res[name]['containment_iou']:.3f}")
fig, ax = plt.subplots(len(panels), 3, figsize=(7.2, 1.45 * len(panels))); fig.subplots_adjust(0.005, 0.005, 0.995, 0.95, 0.02, 0.18)
for i, (name, rgb, oheat, om, cm, r) in enumerate(panels):
    img = rgb.astype(float) / 255
    hv = (oheat - oheat.min()) / (oheat.max() - oheat.min() + 1e-9); ax[i, 0].imshow(img * 0.4 + 0.6 * plt.get_cmap('turbo')(hv)[..., :3]); ax[i, 0].set_title(f'{name}: OpenHype aggregated relevancy', fontsize=7.5, loc='left')
    for j, (m, t) in enumerate(((om, f'OpenHype mask (>0.4), IoU {r["openhype_iou"]:.2f}'), (cm, f'containment, IoU {r["containment_iou"]:.2f}')), 1):
        o = img * 0.45; o[m] = 0.25 * img[m] + 0.75 * np.array([1.0, 0.78, 0.1]); ax[i, j].imshow(np.clip(o, 0, 1)); ax[i, j].set_title(t, fontsize=7.5, loc='left')
    for a_ in ax[i]: a_.axis('off')
fig.suptitle('The same void-free HiGH features read by OpenHype\'s released readout and by containment (citrus 05 cell 1_0, held-out frames)', fontsize=8)
fig.savefig(a.out, dpi=200); print('[cmp] wrote', a.out)
if a.json: json.dump(res, open(a.json, 'w'), indent=1)
