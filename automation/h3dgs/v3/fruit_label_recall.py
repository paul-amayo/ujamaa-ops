#!/usr/bin/env python3
"""fruit_label_recall.py — why the majority seeding rule loses an orange even on the LoD splat (2026-10-14): the census gives a gaussian the
word its rendered pixels carry most over ALL census frames, so an orange that SAM3 labels in only some of the frames where it is visible
makes its own carriers 'mostly tree'. For one orange (3D centre = mass-weighted centre of the fruit-word painters from lod_trace_fig.py's json):
in how many census (training) frames is it in view and unoccluded (|rendered depth - centre depth| < --vis-m), and in how many of those is
the pixel labelled fruit / tree / unlabelled; then each fruit-word painter's fruit share of its labelled diet over all frames vs over the frames
where the orange is labelled. Census resolution (render scale 0.5).
nerf_new pixi env:  fruit_label_recall.py --config <config.yml> --trace-json <lod_trace_*.json> --sup <id maps> [--vis-m 0.15] [--out json]"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from PIL import Image
from gsplat import rasterization
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--trace-json', required=True); ap.add_argument('--sup', required=True)
ap.add_argument('--vis-m', type=float, default=0.15); ap.add_argument('--w-npz', default=''); ap.add_argument('--maj-labels', default=''); ap.add_argument('--scale', type=float, default=0.5); ap.add_argument('--out', default=''); a = ap.parse_args()
FRUIT, UNLAB = 10000, 65535
_, pipe, _, step = eval_setup(Path(a.config), test_mode='test'); model = pipe.model; model.eval(); model.step = step; dev = model.device; ds = pipe.datamanager.train_dataset
dps = float(json.load(open(Path(a.config).parent / 'dataparser_transforms.json'))['scale']); g = model.gauss_params
means = g['means'].detach(); quats = g['quats'].detach(); scales = torch.exp(g['scales'].detach()); opac = torch.sigmoid(g['opacities'].detach()).squeeze(-1); N = means.shape[0]
T = json.load(open(a.trace_json)); fid = T['fruit_word']; P = [p for p in T['painters'] if p['word'] == 'fruit']
gi = torch.tensor([p['g'] for p in P], device=dev); wts = torch.tensor([p['mass_share'] for p in P], device=dev); C = (means[gi] * wts[:, None]).sum(0) / wts.sum()
names = [Path(p).name for p in ds.image_filenames]; split_m, nfr_k = {}, {}; rows = []; m_fr = torch.zeros(len(P), device=dev); m_lab = torch.zeros(len(P), device=dev); m_fr_L = torch.zeros(len(P), device=dev); m_lab_L = torch.zeros(len(P), device=dev)
for i, nm in enumerate(names):
    f = Path(a.sup) / nm
    if not f.exists(): continue
    cam = ds.cameras[i:i + 1].to(dev); cam.rescale_output_resolution(a.scale); H, W = int(cam.height.item()), int(cam.width.item())
    vm = get_viewmat(cam.camera_to_worlds); K = cam.get_intrinsics_matrices().to(dev)
    pc = vm[0] @ torch.cat([C, torch.ones(1, device=dev)]); z = float(pc[2])
    if z <= 0: continue
    u = float(K[0, 0, 0] * pc[0] / pc[2] + K[0, 0, 2]); v = float(K[0, 1, 1] * pc[1] / pc[2] + K[0, 1, 2])
    if not (2 <= u < W - 2 and 2 <= v < H - 2): continue
    ids = np.array(Image.fromarray(np.array(Image.open(f), np.uint16)).resize((W, H), Image.NEAREST)); isf = (ids >= FRUIT) & (ids < UNLAB); isl = ids != UNLAB
    dummy = torch.ones(N, 3, device=dev, requires_grad=True)
    with torch.enable_grad():
        feat, _, meta = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=dummy, sh_degree=None, viewmats=vm, Ks=K, width=W, height=H, packed=False,
                                      near_plane=0.01, far_plane=1e10, render_mode='RGB', rasterize_mode=model.config.rasterize_mode)
        S = (feat[0, ..., 0] * torch.from_numpy(isf).to(dev).float()).sum() + (feat[0, ..., 1] * torch.from_numpy(isl).to(dev).float()).sum() + feat[0, ..., 2].sum()
        gr = torch.autograd.grad(S, dummy)[0]
    ui, vi = int(u), int(v)
    with torch.no_grad():   # occlusion test: are the orange's own fruit-word painters what this pixel shows? (painter coverage >= 0.3 in a 3x3 window)
        ind = torch.zeros(N, 1, device=dev); ind[gi] = 1.0
        cov, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=ind.expand(N, 3).contiguous(), sh_degree=None, viewmats=vm, Ks=K, width=W, height=H, packed=False,
                                  near_plane=0.01, far_plane=1e10, render_mode='RGB', rasterize_mode=model.config.rasterize_mode)
    visible = float(cov[0, vi - 1:vi + 2, ui - 1:ui + 2, 0].max()) >= 0.3
    win = ids[vi - 2:vi + 3, ui - 2:ui + 3]; lab_fruit = bool(((win >= FRUIT) & (win < UNLAB)).any()); lab_word = int(np.bincount(win.ravel()).argmax())
    mf, ml, mt = gr[gi, 0].detach(), gr[gi, 1].detach(), gr[gi, 2].detach(); m_fr += mf; m_lab += ml
    k_ = 'vis' if visible else 'occ'; split_m.setdefault(k_, torch.zeros(3, len(P), device=dev)); split_m[k_] += torch.stack([mf, ml - mf, mt - ml]); nfr_k[k_] = nfr_k.get(k_, 0) + 1
    if visible and lab_fruit: m_fr_L += mf; m_lab_L += ml
    rows.append({'frame': nm, 'visible': bool(visible), 'labelled_fruit': lab_fruit, 'label_at_centre': lab_word, 'depth_m': z / dps})
vis = [r for r in rows if r['visible']]; lab = [r for r in vis if r['labelled_fruit']]
from collections import Counter
other = Counter('tree ' + str(r['label_at_centre']) if r['label_at_centre'] < FRUIT else ('unlabelled' if r['label_at_centre'] == UNLAB else 'fruit ' + str(r['label_at_centre'])) for r in vis if not r['labelled_fruit'])
print(f'[recall] fruit word {fid}: centre seen in {len(rows)} census frames, unoccluded in {len(vis)}, labelled as fruit in {len(lab)} ({100 * len(lab) / max(len(vis), 1):.0f}%); '
      f'the other {len(vis) - len(lab)} visible frames label its pixels: ' + ', '.join(f'{k} {v}' for k, v in other.most_common(4)), flush=True)
sh_all = (m_fr / m_lab.clamp(min=1e-6)).cpu().numpy(); sh_L = (m_fr_L / m_lab_L.clamp(min=1e-6)).cpu().numpy()
print('[recall] fruit-word painters, fruit share of the labelled diet — all census frames -> only frames where the orange is labelled: '
      + '; '.join(f"g{p['g']} {p['size_cm']:.1f} cm: {sa:.2f} -> {sl:.2f}" for p, sa, sl in zip(P, sh_all, sh_L)), flush=True)
print(f'[recall] painters with share >= 0.5: {int((sh_all >= 0.5).sum())} of {len(P)} over all frames, {int((sh_L >= 0.5).sum())} of {len(P)} over the labelled frames', flush=True)
for k_ in ('vis', 'occ'):
    if k_ in split_m:
        M = split_m[k_].cpu().numpy(); print(f"[split] {'frames where the orange is visible' if k_ == 'vis' else 'frames where it is hidden / glanced'} ({nfr_k[k_]} frames) — per painter fruit / tree / unlabelled mass: "
                                              + '; '.join(f"g{p['g']}: {M[0, j]:.1f} / {M[1, j]:.1f} / {M[2, j]:.1f}" for j, p in enumerate(P)) + f" | total {M[0].sum():.1f} / {M[1].sum():.1f} / {M[2].sum():.1f}", flush=True)
Z = np.load(a.w_npz) if a.w_npz else None
if Z is not None:
    Wm, wl = Z['W'], Z['labels'].tolist(); gix = [p['g'] for p in P]; tw = T.get('tree_word', -1)
    def row(l): return Wm[wl.index(l)][gix] if l in wl else np.zeros(len(gix))
    wf, wt, wb = row(fid), row(tw), row(UNLAB); wo = Wm[:, gix].sum(0) - wf - wt - wb
    lab_maj = np.load(a.maj_labels)[gix] if a.maj_labels else None
    print(f'[census] W (px x alpha over the census) for the fruit-word painters — fruit {fid} / tree {tw} / unlabelled / other -> majority-rule label: ' + '; '.join(
        f"g{g}: {f_:.1f} / {t_:.1f} / {b_:.1f} / {o_:.1f}" + (f" -> {int(lm)}" if lab_maj is not None else '') for g, f_, t_, b_, o_, lm in zip(gix, wf, wt, wb, wo, lab_maj if lab_maj is not None else [0] * len(gix))), flush=True)
if a.out: json.dump({'fruit_word': fid, 'frames': rows, 'share_all': sh_all.tolist(), 'share_labelled_frames': sh_L.tolist(), 'painters': [p['g'] for p in P]}, open(a.out, 'w'), indent=1)
