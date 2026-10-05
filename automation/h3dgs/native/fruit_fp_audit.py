#!/usr/bin/env python3
"""Audit of the fruit-seeded Gaussians that light LEAVES on Citrus B (UJAMAA 2026-10-05; Paul: "do an audit and tell me where these
gaussians on leaves come from and why they don't go to trees. If they aren't supervised how do they end up there").
expo chunk 1_0, cell chunk_1_0_sam3 (census W over 294 training views, seed B). For the 13 training-view fruit frames of
citrus_b_cut_trained, the census machinery (hier_census.py's math, the cell's cut: tau 3 at 1280 px, full resolution) attributes
the red (leaf false positive) and green (on SAM3 fruit) pixels to Gaussians. The fruit-seeded Gaussians carrying 90 % of the red /
green fruit-seeded weight are the FP and TP sets. For each set:
  ORIGIN    hierarchy leaf (a trained Gaussian) or merged interior node, depth, max scale, opacity, colour (SH DC) classified by
            nearest CIELAB centroid against fruit-majority Gaussians (orange) vs tree-5-seeded Gaussians (leaf)
  DIET      census shares (tree 5, other trees, fruit 10001, other fruit, void) and total labelled weight
  VIEWS     per training view (real supervision): views that see it, views giving it fruit-5 weight, views where fruit-5 > tree
  SEED PATH what seed B would have given it without the 0.1 share rule: a tree (above the 1.0 floor, void not >= 2x), unassigned
            (floor / void), or fruit by majority
  PIXELS    what the supervision says under the red pixels in each frame (tree 5 / other tree / fruit / void)
-> <out>/fruit_fp_audit.json + printed summary.  h3dgs env, CUDA_HOME=_cuda12, SCENE args as the cell's census."""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hier_common as hc
import numpy as np
import torch
from PIL import Image
from gsplat import rasterization
import cv2
def rgb2lab(a):   # float RGB in [0, 1], shape (1, n, 3) -> CIELAB (OpenCV; skimage is not in the h3dgs env)
    return cv2.cvtColor(np.ascontiguousarray(a, np.float32), cv2.COLOR_RGB2Lab)
ap = argparse.ArgumentParser(); lp, pp = hc.add_scene_args(ap); ap.add_argument('--out', required=True); args = ap.parse_args()
S = '/home/paperspace/data/citrus_all/05_13D_Jackal'; NB = f'{S}/experimental/h3dgs_native/chunk_1_0_sam3'; SUP = f'{NB}/supervision/trees_only'
VER = '/home/paperspace/data/demo_video_v2/citrus_b_cut_trained/verdicts'; TAU, CUTW, U = 3.0, 1280, 65535; os.makedirs(args.out, exist_ok=True)
dataset, scene, g, cams, K0, bufs = hc.load(lp, args); N = g._xyz.size(0); print(f'[audit] {N} gaussians', flush=True)

def census(name, a_full, uids, scale, sel=None):
    cam = cams[name]; hc.to_cuda(cam); viewmat, Ks, Wr, Hr = hc.gsplat_cam(cam, K0, scale)
    a = a_full if a_full.shape == (Hr, Wr) else np.array(Image.fromarray(a_full).resize((Wr, Hr), Image.NEAREST))
    uids = [u for u in uids if (a == u).any()]
    if not uids: return {}
    lab = torch.from_numpy(a.astype(np.int64)).cuda(); onehot = torch.stack([(lab == u) for u in uids], -1).float()
    means, scales, rots, opac, ri, pi, w = hc.expand(g, cam, hc.cut_threshold(cam, TAU, CUTW), bufs)
    means, scales, rots, opac, sky = hc.with_skybox(g, means, scales, rots, opac)
    with torch.enable_grad():
        col = torch.zeros(ri.numel(), len(uids), device='cuda', requires_grad=True)
        colors = torch.cat([col, torch.zeros(sky, len(uids), device='cuda')]) if sky else col
        out, _, _ = rasterization(means=means, quats=rots, scales=scales, opacities=opac, colors=colors, viewmats=viewmat, Ks=Ks, width=Wr,
                                  height=Hr, sh_degree=None, packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB', rasterize_mode='classic')
        gcol, = torch.autograd.grad((out[0] * onehot).sum(), col)
    wv = w[:, 0]; res = {}
    for k, u in enumerate(uids):
        r = torch.zeros(N, device='cuda'); r.index_add_(0, ri, wv * gcol[:, k]); r.index_add_(0, pi, (1 - wv) * gcol[:, k])
        res[u] = (r if sel is None else r[sel]).cpu().numpy()
    del out, col, colors, gcol; return res

# ---- seed B, recomputed (= the features file), and the path each node would take without the share rule ----
z = np.load(f'{NB}/W.npz', allow_pickle=True); labels = [int(x) for x in z['labels']]; Wc = z['W']
bg = Wc[labels.index(U)]; li = [i for i, l in enumerate(labels) if l != U]; Wl = Wc[li]; lab = [labels[i] for i in li]; del Wc
fr = np.array([10000 <= l < U for l in lab]); i5, it5 = lab.index(10001), lab.index(5); tot = Wl.sum(0); maj = Wl.argmax(0); best = Wl.max(0)
fsum = Wl[fr].sum(0); fshare = np.where(tot > 0, fsum / np.maximum(tot, 1e-12), 0.0); fmaj = np.where(fsum > 0, np.where(fr[:, None], Wl, 0).argmax(0), -1)
floor_ok = tot > np.where(fr[maj], 0.01, 1.0); bg_ok = ~(bg >= 2 * best); assign = floor_ok & bg_ok; promote = (tot > 0) & (fshare > 0.1)
final = np.where(promote, fmaj, maj); assigned = assign | promote; f5 = assigned & (final == i5)
PATHN = {1: 'fruit majority', 2: 'would be a tree', 3: 'unassigned: total weight <= 1.0', 4: 'unassigned: void >= 2x best'}
path = np.zeros(N, np.int8); path[f5 & fr[maj]] = 1; t_ = f5 & ~fr[maj]
path[t_ & floor_ok & bg_ok] = 2; path[t_ & ~floor_ok] = 3; path[t_ & floor_ok & ~bg_ok] = 4
treeseed5 = assigned & (final == it5)
diet = {'tree 5': Wl[it5], 'other trees': Wl[~fr].sum(0) - Wl[it5], 'fruit 10001': Wl[i5], 'other fruit': fsum - Wl[i5]}
print('[audit] seed: tree-5 fruit', int(f5.sum()), '| paths', {PATHN[p]: int((path[f5] == p).sum()) for p in PATHN}, flush=True)

# ---- attribution of red / green pixels over the 13 frames ----
frames = sorted(f[:-4] for f in os.listdir(VER) if f.endswith('.npz')); RED = np.zeros(N, np.float32); GRN = np.zeros(N, np.float32); px = {}
for kf in frames:
    V = np.load(f'{VER}/{kf}.npz'); v = V['v']; amap = np.full(v.shape, U, np.uint16); amap[v == 4] = 1; amap[v == 1] = 2
    r = census(kf, amap, [1, 2], 1.0); RED += r.get(1, 0); GRN += r.get(2, 0)
    sup = np.array(Image.open(f'{SUP}/{kf}'), np.uint16); m = v == 4
    px[kf] = {'red_px': int(m.sum()), 'tree5': int((m & (sup == 5)).sum()), 'other_tree': int((m & (sup < 10000) & (sup != 5)).sum()),
              'fruit': int((m & (sup >= 10000) & (sup < U)).sum()), 'void': int((m & (sup == U)).sum())}
print('[audit] attribution done', flush=True)
def top(weight, mask, q=0.9):
    idx = np.nonzero((weight > 0) & mask)[0]; o = idx[np.argsort(-weight[idx])]; c = np.cumsum(weight[o]) / weight[o].sum(); return o[:int(np.searchsorted(c, q)) + 1]
FP = top(RED, f5); TP = top(GRN, f5)
# ---- origin: hierarchy role, size, opacity, colour ----
nodes = g.nodes.cpu().numpy(); start = nodes[:, 2]; nch = nodes[:, 6]; depth = nodes[:, 0]
is_leaf = np.zeros(N, bool); dep = np.full(N, -1); ok = (start >= 0) & (start < N); is_leaf[start[ok]] = nch[ok] == 0; dep[start[ok]] = depth[ok]
with torch.no_grad():
    scl = g.get_scaling.max(1).values.cpu().numpy(); opa = g.get_opacity[:, 0].cpu().numpy(); xyz = g.get_xyz.cpu().numpy()
    rgb = np.clip(0.5 + 0.28209479 * g._features_dc[:, 0, :].detach().cpu().numpy(), 0, 1)
fm = path == 1; ref_o = rgb2lab(rgb[fm][None])[0][:, 1:].mean(0)
near = treeseed5 & (np.linalg.norm(xyz - xyz[fm].mean(0), axis=1) < 3.0); ref_l = rgb2lab(rgb[near][None])[0][:, 1:].mean(0)
def colour(idx):
    ab = rgb2lab(rgb[idx][None])[0][:, 1:]; return np.linalg.norm(ab - ref_o, axis=1) < np.linalg.norm(ab - ref_l, axis=1)
# ---- views: per-view census of FP and TP nodes with the real supervision ----
both = np.unique(np.concatenate([FP, TP])); bt = torch.from_numpy(both).cuda(); seen = np.zeros(both.size, int); gf = np.zeros(both.size, int); win = np.zeros(both.size, int)
names = json.load(open(f'{NB}/split_names.json'))['train']; t0 = time.time()
for name in names:
    sf = f'{SUP}/{name}'
    if name not in cams or not os.path.exists(sf): continue
    a = np.array(Image.open(sf), np.uint16); r = census(name, a, [int(x) for x in np.unique(a)], 0.5, sel=bt)
    if not r: continue
    f5w = r.get(10001, np.zeros(both.size)); tw = sum((x for u, x in r.items() if u < 10000), np.zeros(both.size)); anyw = sum(r.values())
    seen += anyw > 0; gf += f5w > 0; win += f5w > tw
print(f'[audit] per-view census over {len(names)} views in {time.time() - t0:.0f}s', flush=True)
pos = {n: i for i, n in enumerate(both)}
def summary(nm, idx, wgt):
    w = wgt[idx] / wgt[idx].sum(); k = np.array([pos[i] for i in idx]); T = tot[idx]
    s = {'nodes': int(idx.size),
         'seed_path_weight_share': {PATHN[p]: round(float(w[path[idx] == p].sum()), 3) for p in PATHN},
         'hierarchy_leaf_weight_share': round(float(w[is_leaf[idx]].sum()), 3), 'depth_median': int(np.median(dep[idx])),
         'max_scale_cm_median': round(float(np.median(scl[idx])) * 100, 2), 'opacity_median': round(float(np.median(opa[idx])), 3),
         'orange_coloured_weight_share': round(float(w[colour(idx)].sum()), 3),
         'diet_weighted': {d: round(float((w * (x[idx] / np.maximum(T, 1e-12))).sum()), 3) for d, x in diet.items()},
         'void_share_weighted': round(float((w * bg[idx] / np.maximum(bg[idx] + T, 1e-12)).sum()), 3),
         'total_label_weight_median': round(float(np.median(T)), 3),
         'views_seen_median': int(np.median(seen[k])), 'views_giving_fruit5_median': int(np.median(gf[k])),
         'views_where_fruit5_beats_tree_median': int(np.median(win[k])),
         'fraction_of_seen_views_fruit_wins_weighted': round(float((w * win[k] / np.maximum(seen[k], 1)).sum()), 3)}
    print(f'[audit] {nm}: {json.dumps(s)}', flush=True); return s
rep = {'frames': frames, 'false_positive_nodes': summary('FALSE-POSITIVE (red) fruit Gaussians', FP, RED), 'true_positive_nodes': summary('TRUE-POSITIVE (green) fruit Gaussians', TP, GRN),
       'overlap_fp_tp_nodes': int(np.intersect1d(FP, TP).size), 'red_pixels_supervision': px,
       'red_pixels_supervision_total': {k: int(sum(p[k] for p in px.values())) for k in ('red_px', 'tree5', 'other_tree', 'fruit', 'void')}}
print('[audit] red pixels in the supervision:', rep['red_pixels_supervision_total'], '| FP/TP node overlap', rep['overlap_fp_tp_nodes'], flush=True)
json.dump(rep, open(f'{args.out}/fruit_fp_audit.json', 'w'), indent=1); print(f'[audit] -> {args.out}/fruit_fp_audit.json', flush=True)
