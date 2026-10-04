#!/usr/bin/env python3
"""Why does tree 5's fruit light LEAVES on Citrus B kf_000025? (UJAMAA 2026-10-04; Paul: "focus on kf 25, tell me why we have those,
are the gaussians fruits in previous frames that got densified. I need to understand.")  expo chunk 1_0 (never fruit-densified),
cell chunk_1_0_sam3 (census W over 294 training views, seed B: fruit share > 0.1 -> fruit target).
  1. ATTRIBUTION on kf_000025: the census machinery of hier_census.py (same cut: tau 3 at 1280 px, render_post interpolation) with
     the verdict classes of citrus_b_fruit_fp.py as the labels, at full resolution -> every node's alpha-blend weight into the red
     (leaf false positive), green (on SAM3 fruit) and yellow (orange, no SAM3 mask) pixels; each node's seed (tree-5 fruit by
     majority / by the share rule / another target / none) and census fruit share.
  2. ORIGINS: for the tree-5-fruit-seeded nodes behind the red pixels, the census again over all 294 training views with the real
     supervision, per view -> which views gave them their fruit weight and how much tree weight they had.
  3. FIGURE: the red nodes' centres projected into kf_000025 and into the 3 views that gave them the most fruit weight, over the
     photo with SAM3's tree-5 fruit outlined.
h3dgs env: CUDA_HOME=/home/paperspace/code/_cuda12 PATH=$CUDA_HOME/bin:$PATH python fruit_fp_trace.py <SCENE args> --out <dir>"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hier_common as hc
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from gsplat import rasterization
from scipy import ndimage as nd
ap = argparse.ArgumentParser(); lp, pp = hc.add_scene_args(ap); ap.add_argument('--out', required=True); args = ap.parse_args()
S = '/home/paperspace/data/citrus_all/05_13D_Jackal'; NB = f'{S}/experimental/h3dgs_native/chunk_1_0_sam3'; SUP = f'{NB}/supervision/trees_only'
VER = '/home/paperspace/data/demo_video_v2/citrus_b_cut_trained/verdicts'; PH = f'{S}/experimental/h3dgs_expo/camera_calibration/rectified/images'
KF, TAU, CUTW, UNLAB = 'kf_000025.png', 3.0, 1280, 65535; OUT = args.out; os.makedirs(OUT, exist_ok=True)
dataset, scene, g, cams, K0, bufs = hc.load(lp, args); N = g._xyz.size(0); print(f'[trace] {N} nodes', flush=True)

def census(name, a_full, uids, scale, sel=None):
    """alpha-blend weight of every node (or of nodes `sel`) into each label's pixels of view `name` (hier_census.py's math)."""
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

# ---- seed of every node (recomputed seed-B rule = the features file's 8,981 tree-5 fruit nodes) ----
z = np.load(f'{NB}/W.npz', allow_pickle=True); labels = [int(x) for x in z['labels']]; Wc = z['W']
bg = Wc[labels.index(UNLAB)]; li = [i for i, l in enumerate(labels) if l != UNLAB]; Wl = Wc[li]; lab = [labels[i] for i in li]
fr = np.array([10000 <= l < UNLAB for l in lab]); i5 = lab.index(10001); tot = Wl.sum(0); maj = Wl.argmax(0); fsum = Wl[fr].sum(0)
fshare = np.where(tot > 0, fsum / np.maximum(tot, 1e-12), 0.0); fmaj = np.where(fsum > 0, np.where(fr[:, None], Wl, 0).argmax(0), -1)
assign = (tot > np.where(fr[maj], 0.01, 1.0)) & ~(bg >= 2 * Wl.max(0)); promote = (tot > 0) & (fshare > 0.1)
final = np.where(promote, fmaj, maj); assigned = assign | promote
cat = np.zeros(N, np.int8)                                   # 0 no identity, 1 tree-5 fruit by majority, 2 tree-5 fruit by share rule, 3 other fruit, 4 a tree/row
f5 = assigned & (final == i5); cat[f5 & (maj == i5)] = 1; cat[f5 & (maj != i5)] = 2
cat[assigned & ~f5 & fr[final]] = 3; cat[assigned & ~fr[final]] = 4
del Wc, Wl, z
CATN = {0: 'no identity', 1: 'tree-5 fruit (fruit majority)', 2: 'tree-5 fruit (share rule, a tree is the majority)', 3: 'another fruit', 4: 'a tree / row'}
print('[trace] seed categories:', {CATN[k]: int((cat == k).sum()) for k in range(5)}, flush=True)

# ---- 1. attribution on kf_000025 ----
V = np.load(f'{VER}/{KF}.npz'); v = V['v']; amap = np.full(v.shape, UNLAB, np.uint16)
CLS = {1: ('red: leaf false positive', 4), 2: ('green: on SAM3 fruit', 1), 3: ('yellow: orange, no SAM3 mask', 3)}
for u, (_, k) in CLS.items(): amap[v == k] = u
W1 = census(KF, amap, list(CLS), 1.0)
rep = {'view': KF, 'classes': {}}
for u, (nm, _) in CLS.items():
    w = W1.get(u);
    if w is None: continue
    T = float(w.sum()); rep['classes'][nm] = {'pixels': int((amap == u).sum()), 'total_weight': round(T, 2),
        'weight_share_by_seed': {CATN[k]: round(float(w[cat == k].sum() / max(T, 1e-12)), 3) for k in range(5)},
        'nodes': int((w > 0).sum())}
    print(f'[trace] {nm}: {rep["classes"][nm]}', flush=True)
red = W1[1]; sel_all = np.nonzero((red > 0) & (cat >= 1) & (cat <= 2))[0]
order = sel_all[np.argsort(-red[sel_all])]; cum = np.cumsum(red[order]) / red[order].sum(); sel = order[:max(1, int(np.searchsorted(cum, 0.9)) + 1)][:3000]
fsh = fshare[sel]; wts = red[sel]
rep['red_fruit_nodes'] = {'n_for_90pct': int(sel.size), 'by_majority': int((cat[sel] == 1).sum()), 'by_share_rule': int((cat[sel] == 2).sum()),
    'fruit_share_weighted_quantiles_10_50_90': [round(float(x), 3) for x in np.quantile(np.repeat(fsh, np.maximum(1, (wts / wts.max() * 100).astype(int))), [0.1, 0.5, 0.9])],
    'majority_label_counts': {str(lab[m]): int(c) for m, c in zip(*np.unique(maj[sel], return_counts=True))}}
with torch.no_grad():
    xyz = g.get_xyz.detach().cpu().numpy(); scl = g.get_scaling.detach().cpu().numpy().max(1)
gsel = np.nonzero((W1.get(2, np.zeros(N)) > 0) & (cat >= 1) & (cat <= 2))[0]
from scipy.spatial import cKDTree
d_green = cKDTree(xyz[gsel]).query(xyz[sel])[0] if gsel.size else np.full(sel.size, np.nan)
rep['red_fruit_nodes'].update({'max_scale_m_median': round(float(np.median(scl[sel])), 4), 'green_fruit_nodes_max_scale_m_median': round(float(np.median(scl[gsel])), 4) if gsel.size else None,
    'distance_to_nearest_green_fruit_node_m_quantiles_10_50_90': [round(float(x), 3) for x in np.quantile(d_green, [0.1, 0.5, 0.9])]})
print('[trace] red pixels\' fruit nodes:', rep['red_fruit_nodes'], flush=True)

# ---- 2. origins: per training view, the census of the selected nodes with the real supervision ----
names = json.load(open(f'{NB}/split_names.json'))['train']; selt = torch.from_numpy(sel).cuda(); per = []; t0 = time.time()
node_fruit_by_view = {}
for k, name in enumerate(names):
    sf = f'{SUP}/{name}'
    if name not in cams or not os.path.exists(sf): continue
    a = np.array(Image.open(sf), np.uint16); uids = [int(u) for u in np.unique(a)]
    r = census(name, a, uids, 0.5, sel=selt)
    if not r: continue
    f5w = r.get(10001, np.zeros(sel.size)); trees = sum(x for u, x in r.items() if u < 10000); of = sum(x for u, x in r.items() if 10000 <= u < UNLAB and u != 10001)
    per.append({'view': name, 'fruit5': float(f5w.sum()), 'tree': float(np.sum(trees)), 'other_fruit': float(np.sum(of)), 'void': float(r.get(UNLAB, np.zeros(1)).sum()),
                'nodes_with_fruit5': int((f5w > 0).sum())})
    node_fruit_by_view[name] = f5w
    if k % 50 == 0: print(f'[trace] origins: {k} views ({time.time() - t0:.0f}s)', flush=True)
F5 = sum(p['fruit5'] for p in per); TR = sum(p['tree'] for p in per)
per.sort(key=lambda p: -p['fruit5'])
rep['origins'] = {'views': len(per), 'fruit5_total': round(F5, 2), 'tree_total': round(TR, 2), 'other_fruit_total': round(sum(p['other_fruit'] for p in per), 2),
    'void_total': round(sum(p['void'] for p in per), 2), 'views_giving_fruit5': sum(p['fruit5'] > 0 for p in per),
    'top_views': [{k_: (round(v_, 3) if isinstance(v_, float) else v_) for k_, v_ in p.items()} for p in per[:12]],
    'top3_views_share_of_fruit5': round(sum(p['fruit5'] for p in per[:3]) / max(F5, 1e-12), 3),
    'top10_views_share_of_fruit5': round(sum(p['fruit5'] for p in per[:10]) / max(F5, 1e-12), 3)}
print('[trace] origins:', json.dumps(rep['origins'])[:1500], flush=True)
json.dump(rep, open(f'{OUT}/fruit_fp_trace.json', 'w'), indent=1)

# ---- 3. figure: red-blob fruit nodes projected into kf_000025 and the 3 views that gave them the most fruit weight ----
F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20); tiles = []
for name in [KF] + [p['view'] for p in per[:3]]:
    cam = cams[name]; hc.to_cuda(cam); vm = cam.world_view_transform.transpose(0, 1).cpu().numpy(); fx, fy, cx, cy = K0
    pc = (vm[:3, :3] @ xyz[sel].T + vm[:3, 3:4]).T; ok = pc[:, 2] > 0.05; u = fx * pc[:, 0] / pc[:, 2] + cx; vv = fy * pc[:, 1] / pc[:, 2] + cy
    img = np.array(Image.open(f'{PH}/{name}').convert('RGB')); H, W = img.shape[:2]; sup = np.array(Image.open(f'{SUP}/{name}'), np.uint16)
    s5 = sup == 10001; e = nd.binary_dilation(s5, iterations=2) & ~s5; img[e] = (0, 235, 255)
    if name == KF: img[(v == 4) & ~nd.binary_erosion(v == 4)] = (255, 40, 40)
    im = Image.fromarray(img); d = ImageDraw.Draw(im); wn = node_fruit_by_view.get(name, np.zeros(sel.size)) if name != KF else red[sel]
    for j in np.nonzero(ok & (u >= 0) & (u < W) & (vv >= 0) & (vv < H))[0]:
        if wn[j] <= 0: continue
        r_ = 2 + int(4 * min(1.0, wn[j] / max(wn.max(), 1e-12)) ** 0.5); d.ellipse([u[j] - r_, vv[j] - r_, u[j] + r_, vv[j] + r_], outline=(255, 0, 220), width=2)
    pv = next((p for p in per if p['view'] == name), None)
    t = f'{name}: ' + ('where they light leaves (red outline = false-positive pixels)' if name == KF else f"fruit weight {pv['fruit5']:.2f} vs tree {pv['tree']:.2f} on these nodes")
    d.rounded_rectangle([8, 8, 20 + d.textlength(t, font=F), 40], 6, fill=(11, 11, 11)); d.text((14, 12), t, font=F, fill=(255, 255, 255)); tiles.append(im)
w0, h0 = tiles[0].size; sheet = Image.new('RGB', (2 * w0 + 10, 2 * h0 + 10), (255, 255, 255))
for k, t in enumerate(tiles): sheet.paste(t, ((k % 2) * (w0 + 10), (k // 2) * (h0 + 10)))
sheet.save(f'{OUT}/fruit_fp_trace.jpg', quality=88); print(f'[trace] -> {OUT}/fruit_fp_trace.jpg, {OUT}/fruit_fp_trace.json', flush=True)
# ---- 4. ONE cluster followed across views (2026-10-04 v2: the full-frame sheet was too small to read) ----
np.savez_compressed(f'{OUT}/fruit_fp_trace_nodes.npz', sel=sel, xyz=xyz[sel], red=red[sel], fshare=fshare[sel], cat=cat[sel],
                    views=np.array(list(node_fruit_by_view)), fruit_by_view=np.stack(list(node_fruit_by_view.values())))
kd = cKDTree(xyz[sel]); R_ = 0.3; mass = np.array([red[sel][kd.query_ball_point(xyz[sel][j], R_)].sum() for j in range(sel.size)])
cl = np.array(kd.query_ball_point(xyz[sel][mass.argmax()], R_)); print(f'[trace] cluster: {cl.size} of the {sel.size} red fruit nodes within {R_} m, '
      f'{red[sel][cl].sum() / red[sel].sum():.0%} of their red weight', flush=True)
Fb = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22); Fs = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
views4 = sorted({KF} | {p['view'] for p in per[:3]}, key=lambda n: int(n[3:9])); crops = []; CW, CH_, Z = 200, 120, 4
for name in views4:
    cam = cams[name]; vm = cam.world_view_transform.transpose(0, 1).cpu().numpy(); fx, fy, cx, cy = K0
    P = xyz[sel][cl]; pc = (vm[:3, :3] @ P.T + vm[:3, 3:4]).T; u = fx * pc[:, 0] / pc[:, 2] + cx; vv = fy * pc[:, 1] / pc[:, 2] + cy
    img = np.array(Image.open(f'{PH}/{name}').convert('RGB')).astype(np.float32); H, W = img.shape[:2]
    sup = np.array(Image.open(f'{SUP}/{name}'), np.uint16); s5 = sup == 10001
    img[s5] = 0.45 * img[s5] + 0.55 * np.array([0, 235, 255]); img = img.astype(np.uint8)
    if name == KF:
        fp = v == 4; img[fp] = (255, 40, 40)
    x0 = int(np.clip(np.median(u) - CW / 2, 0, W - CW)); y0 = int(np.clip(np.median(vv) - CH_ / 2, 0, H - CH_))
    im = Image.fromarray(img).crop((x0, y0, x0 + CW, y0 + CH_)).resize((CW * Z, CH_ * Z), Image.NEAREST); d = ImageDraw.Draw(im)
    for j in range(cl.size):
        X, Y = (u[j] - x0) * Z, (vv[j] - y0) * Z
        if 0 <= X < CW * Z and 0 <= Y < CH_ * Z: d.ellipse([X - 5, Y - 5, X + 5, Y + 5], fill=(255, 0, 220), outline=(255, 255, 255))
    pv = next((p for p in per if p['view'] == name), None); wv_ = node_fruit_by_view.get(name, np.zeros(sel.size))[cl].sum()
    t = (f'{name} (the frame in question): red = lit as fruit on leaves' if name == KF else f'{name}: these Gaussians sit inside SAM3 fruit (cyan) -> fruit credit {wv_:.1f}')
    d.rounded_rectangle([6, 6, 18 + d.textlength(t, font=Fb), 38], 6, fill=(11, 11, 11)); d.text((12, 9), t, font=Fb, fill=(255, 255, 255)); crops.append(im)
w0, h0 = crops[0].size; sheet = Image.new('RGB', (2 * w0 + 12, 2 * h0 + 12 + 70), (255, 255, 255)); dd = ImageDraw.Draw(sheet)
dd.text((8, 8), f'The same {cl.size} Gaussians (magenta dots; {R_} m cluster, 4x zoom) in four training views of tree 5', font=Fb, fill=(0, 0, 0))
dd.text((8, 40), 'cyan = SAM3 fruit mask of tree 5 in that view, red = pixels kf_000025 lights as fruit on leaves', font=Fs, fill=(0, 0, 0))
for k, t in enumerate(crops): sheet.paste(t, ((k % 2) * (w0 + 12), 70 + (k // 2) * (h0 + 12)))
sheet.save(f'{OUT}/fruit_fp_cluster.jpg', quality=90); print(f'[trace] -> {OUT}/fruit_fp_cluster.jpg', flush=True)

