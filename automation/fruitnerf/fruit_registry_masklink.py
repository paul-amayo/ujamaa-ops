#!/usr/bin/env python3
"""fruit_registry_masklink.py — registry v5 (2026-10-15; Paul: "the fix is in the registry"). The SAM3 masks define the instances (one mask =
one apple in one view); the splat ties the views together.
  stage 1 (GPU, cached in fruit_registry/masklink_votes_<tag>.npz): census of fruit_masks through the splat -> fruit gaussians (fruit share of
    the labelled mass >= SHARE and fruit mass >= MASS); each fruit gaussian is projected into every training view and, where its centre is in
    frame and VISIBLE (depth within DEPTH_TOL of the rendered depth), votes for the SAM3 mask it lands in: one vote = (view, mask, gaussian).
  stage 2 (CPU): two fruit gaussians are LINKED when they land in the same mask in >= LINK_VIEWS views and in more views than they land in
    different masks of one view (a cannot-link: one view that separates them outweighs one that merges them). Apples = connected components
    seen in >= MIN_VIEWS views, counted inside the tree crop (horizontal distance from the camera look-point <= CROP x the median camera
    distance). Touching apples stay apart because they are different masks in the views that see both.
  Pre-registered 10-15: SHARE 0.5, MASS 1, DEPTH_TOL 0.05, LINK_VIEWS 2, conflict rule on, MIN_VIEWS 3, CROP 0.5. --sens prints the grid.
nerf_new pixi env:  fruit_registry_masklink.py --config <run config.yml> --survey <survey> --gt 179 --tag plain30k [--sens] [--rebuild]"""
import argparse, json, time
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--config', type=Path); ap.add_argument('--survey', type=Path, required=True); ap.add_argument('--gt', type=int, default=0)
ap.add_argument('--share', type=float, default=0.5); ap.add_argument('--mass', type=float, default=1.0); ap.add_argument('--depth-tol', type=float, default=0.05, help='visibility tolerance as a fraction of depth (the 10-15 10:xx rule, 5 % = 15 cm at 3 m: a whole apple behind still voted); 0 = use --depth-tol-r'); ap.add_argument('--depth-tol-r', type=float, default=0.0, help='visibility tolerance in units of the median camera distance (0.015 ~ 1.2 apple radii from the measured component extents): only gaussians within the apple\'s own thickness of the rendered surface vote')
ap.add_argument('--link-views', type=int, default=2); ap.add_argument('--min-views', type=int, default=3); ap.add_argument('--crop', type=float, default=0.5)
ap.add_argument('--no-conflict', action='store_true'); ap.add_argument('--merge-check', default='none', choices=['none', 'giants', 'size', 'evidence'], help='evidence = after the (strict) links, merge two components only when, over ALL their co-visible member pairs, same-mask views outnumber different-mask views (recovers one apple pieced into two masks; two touching apples have ~12 different-mask views per pair and stay apart); none = 10-15 10:xx rules; giants = split only components whose visible members hit >= 3 distinct masks in the median view (k-means, k = that median); size = FruitNeRF-style template split (over-splits elongated apples, kept for the record)'); ap.add_argument('--sens', action='store_true'); ap.add_argument('--rebuild', action='store_true'); ap.add_argument('--dominant', action='store_true', help='count views only where the component is the dominant painter (>= 50 %% of the votes) of some mask — a fringe or a lone gaussian under a mask never is (22:1x: with strict links, singletons seen in 3 views were counted as apples)'); ap.add_argument('--co-frac', type=float, default=0.0, help='> 0: link needs co / (co + conflicts) >= this (0.9 measured: 42 of 450 k different-apple pairs pass, 86 %% of same-apple pairs kept)'); ap.add_argument('--co-ratio', type=float, default=1.0, help='link needs co-views >= co_ratio x conflicts (1 = the 10-15 10:xx rule; 5 = measured on tree_03 #49: same-apple pairs are seen together in ~25 views with ~0 conflicts, different-apple pairs together in ~0 with ~12 — Paul 22:0x: the merging was too permissive)'); ap.add_argument('--no-ground', action='store_true', help='exclude the ground layer: apples on the soil (windfalls, 21:3x) form a separate mode at the bottom of the height histogram with an empty gap above it; the floor = the first empty 0.025-cam_r bin above the lowest mode'); ap.add_argument('--giant-q', type=float, default=0.5, help='giants merge check: k = this quantile of the distinct masks the component occupies per view (0.5 = median; 0.9 = the apples seen together in its best views)'); ap.add_argument('--apple-r', type=float, default=0.0125, help='apple radius in units of the median camera distance (the single-apple mask area expected at a mask\'s lifted depth is pi (f r / z)^2)'); ap.add_argument('--mask-outlier-r', type=float, default=0.0, help='> 0: inside each (view, mask), votes from gaussians farther than this many apple radii from the mask\'s median gaussian position do not link (the minority group of a SAM3 mask that covers two touching apples sits ~2.5 radii from the dominant group — 10-15 19:4x); they still count as presence'); ap.add_argument('--link-max-r', type=float, default=0.0, help='> 0: a link is also refused between gaussians farther apart than this many apple radii (--apple-r x cam_r) — two gaussians of ONE apple are at most ~2 radii apart, the bridging pairs inside a SAM3 mask that covers two touching apples are 2-3 radii apart (10-15 19:5x, tree_03 #49)'); ap.add_argument('--single-mask-max', type=float, default=0.0, help='> 0: links (and conflicts) may only come from masks whose area is <= this x the single-apple area expected at their lifted depth — SAM3 masks covering two touching apples (10-15 19:4x: tree_03 #49, bridge masks 1,884 px vs ~800 for one apple) then cannot join them; 0 = off'); ap.add_argument('--tag', default='splat'); ap.add_argument('--dump-tag', default=None, help='name for the component dump / json / figure (default = --tag); lets a variant be written beside the record'); a = ap.parse_args(); a.dump_tag = a.dump_tag or a.tag
S = a.survey.resolve(); R = S / 'fruit_registry'; R.mkdir(exist_ok=True); cache = R / f'masklink_votes_{a.tag}.npz'

def build_votes():
    import torch
    from PIL import Image
    from gsplat import rasterization
    from pycocotools import mask as mask_utils
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.models.splatfacto import get_viewmat
    t0 = time.time(); _, pipe, _, step = eval_setup(a.config, test_mode='test'); model = pipe.model; model.eval(); dev = model.device; ds = pipe.datamanager.train_dataset
    g = model.gauss_params; N = g['means'].shape[0]; means = g['means'].detach(); quats = g['quats'].detach(); scales = torch.exp(g['scales'].detach()); opac = torch.sigmoid(g['opacities'].detach()).squeeze(-1)
    sam = S / 'prod/bateleur/sam3_v2'
    if not sam.exists(): sam = S / 'sam3_v2'
    labmaps = {}
    for cm in sorted(sam.glob('clip_*/frame_entries.json')):
        meta = json.loads(cm.read_text()); l2i = {f['local_idx']: int(f.get('kf_idx', f.get('image_idx'))) for f in meta['frames']}
        for li, ents in meta['frame_entries'].items():
            k = l2i[int(li)]; Lm = None
            for j, e in enumerate(sorted(ents, key=lambda e: e.get('score', 0))):   # higher score painted last wins overlaps
                m = mask_utils.decode(e['rle']).astype(bool)
                if Lm is None: Lm = np.full(m.shape, -1, np.int16)
                Lm[m] = j
            if Lm is not None: labmaps[f'image_{k}.png'] = torch.from_numpy(Lm)
    Wf = torch.zeros(N, device=dev); Wn = torch.zeros(N, device=dev); views = []
    for i in range(len(ds)):
        nm = Path(ds.image_filenames[i]).name; fm = S / 'fruit_masks' / nm
        if not fm.exists() or nm not in labmaps: continue
        cam = ds.cameras[i:i + 1].to(dev); vm = get_viewmat(cam.camera_to_worlds); K = cam.get_intrinsics_matrices().to(dev); W, H = int(cam.width.item()), int(cam.height.item())
        m = torch.from_numpy(np.array(Image.open(fm))).to(dev); col = torch.zeros(N, 2, device=dev, requires_grad=True)
        out, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=col, viewmats=vm, Ks=K, width=W, height=H, sh_degree=None, packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB+ED', rasterize_mode=model.config.rasterize_mode)
        gf, = torch.autograd.grad((out[0, ..., 0] * (m == 2).float()).sum() + (out[0, ..., 1] * (m == 1).float()).sum(), col)
        Wf += gf[:, 0].clamp(min=0); Wn += gf[:, 1].clamp(min=0); views.append((nm, vm, K, W, H, out[0, ..., 2].detach().half()))
    fruit = ((Wf / (Wf + Wn).clamp(min=1e-9)) >= a.share) & (Wf >= a.mass); fid = torch.where(fruit)[0]; P = means[fid]; M = len(fid)
    print(f'[masklink] {S.name} ({a.tag}): census {time.time() - t0:.0f} s — {N} gaussians, {M} fruit gaussians (share >= {a.share}, mass >= {a.mass}), {len(views)} views', flush=True)
    cams0 = ds.cameras.camera_to_worlds.numpy(); C0 = cams0[:, :3, 3]; F0 = -cams0[:, :3, 2]; F0 = F0 / np.linalg.norm(F0, axis=1, keepdims=True); U0 = cams0[:, :3, 1].mean(0); U0 /= np.linalg.norm(U0)
    A0 = np.zeros((3, 3)); b0 = np.zeros(3)
    for c_, f_ in zip(C0, F0): Pm = np.eye(3) - np.outer(f_, f_); A0 += Pm; b0 += Pm @ c_
    L0 = np.linalg.solve(A0, b0); dd0 = C0 - L0; cam_r0 = float(np.median(np.linalg.norm(dd0 - np.outer(dd0 @ U0, U0), axis=1))); tol_abs = a.depth_tol_r * cam_r0
    print(f'[masklink] visibility tolerance: ' + (f'{a.depth_tol * 100:.0f} % of depth' if a.depth_tol > 0 else f'{a.depth_tol_r} x cam_r = {tol_abs:.4f} scene units (cam_r {cam_r0:.3f})'), flush=True)
    VI, LAB, ROW, UV, nvis = [], [], [], [], []; MK = []   # MK rows: view, mask, x, y, z (lifted centroid), area px
    for vi, (nm, vm, K, W, H, D) in enumerate(views):
        pc = P @ vm[0, :3, :3].T + vm[0, :3, 3]; z = pc[:, 2]; u = (K[0, 0, 0] * pc[:, 0] / z.clamp(min=1e-6) + K[0, 0, 2]).round().long(); v = (K[0, 1, 1] * pc[:, 1] / z.clamp(min=1e-6) + K[0, 1, 2]).round().long()
        ok = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H); uu, vv = u.clamp(0, W - 1), v.clamp(0, H - 1)
        dz = (z - D[vv, uu].float()).abs(); ok = ok & ((dz <= a.depth_tol * z) if a.depth_tol > 0 else (dz <= tol_abs)); nvis.append(int(ok.sum()))   # visible: within the apple's own thickness of the rendered surface
        lab = labmaps[nm].to(dev)[vv, uu].long(); ok = ok & (lab >= 0); r = torch.where(ok)[0]
        Lm = labmaps[nm].to(dev); nmk = int(Lm.max().item()) + 1
        if nmk > 0:   # lift every mask's centroid pixel at the rendered depth -> world (inverse of the projection above)
            yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing='ij'); flat = Lm.flatten().long(); valid = flat >= 0
            cnt = torch.bincount(flat[valid], minlength=nmk).float(); cu = torch.bincount(flat[valid], weights=xx.flatten()[valid], minlength=nmk) / cnt.clamp(min=1); cv = torch.bincount(flat[valid], weights=yy.flatten()[valid], minlength=nmk) / cnt.clamp(min=1)
            zc = D[cv.round().long().clamp(0, H - 1), cu.round().long().clamp(0, W - 1)].float(); xc = (cu - K[0, 0, 2]) / K[0, 0, 0] * zc; yc = (cv - K[0, 1, 2]) / K[0, 1, 1] * zc
            Rw = vm[0, :3, :3]; tw = vm[0, :3, 3]; Pw = (torch.stack([xc, yc, zc], 1) - tw) @ Rw   # world = R^T (cam - t)
            MK.append(np.concatenate([np.full((nmk, 1), vi, np.float32), np.arange(nmk, dtype=np.float32)[:, None], Pw.cpu().numpy().astype(np.float32), cnt.cpu().numpy()[:, None]], 1))
        VI.append(np.full(len(r), vi, np.int32)); LAB.append(lab[r].cpu().numpy().astype(np.int16)); ROW.append(r.cpu().numpy().astype(np.int32)); UV.append(torch.stack([uu[r], vv[r]], 1).cpu().numpy().astype(np.int16))
    cams = ds.cameras.camera_to_worlds.numpy()
    np.savez_compressed(cache, P=P.cpu().numpy(), fid=fid.cpu().numpy(), Wf=Wf[fid].cpu().numpy(), Wn=Wn[fid].cpu().numpy(), vi=np.concatenate(VI), lab=np.concatenate(LAB), row=np.concatenate(ROW), uv=np.concatenate(UV),
                        names=np.array([v[0] for v in views]), nvis=np.array(nvis), masks=np.concatenate(MK) if MK else np.zeros((0, 6), np.float32), C=cams[:, :3, 3], Fd=-cams[:, :3, 2], UPc=cams[:, :3, 1], N=N, M=M, share=a.share, mass=a.mass, depth_tol=a.depth_tol, depth_tol_r=a.depth_tol_r)
    print(f'[masklink] {S.name} ({a.tag}): votes {sum(len(x) for x in ROW)} over {len(views)} views (visible fruit gaussians per view median {np.median(nvis):.0f}) -> {cache.name}, {time.time() - t0:.0f} s', flush=True)

if a.rebuild or not cache.exists(): build_votes()
d = np.load(cache); P = d['P']; M = int(d['M']); vi = d['vi'].astype(np.int64); lab = d['lab'].astype(np.int64); row = d['row'].astype(np.int64); V = len(d['names'])
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
# ---- co-view counts: B (M x groups) indicator, co = B B^T; conflicts via the dense (M x V) label table ----
C0 = d['C']; Fd0 = d['Fd'] / np.linalg.norm(d['Fd'], axis=1, keepdims=True); UP0 = d['UPc'].mean(0); UP0 /= np.linalg.norm(UP0); A00 = np.zeros((3, 3)); b00 = np.zeros(3)
for c_, f_ in zip(C0, Fd0): Pm_ = np.eye(3) - np.outer(f_, f_); A00 += Pm_; b00 += Pm_ @ c_
LOOK0 = np.linalg.solve(A00, b00); dd00 = C0 - LOOK0; cam_r = float(np.median(np.linalg.norm(dd00 - np.outer(dd00 @ UP0, UP0), axis=1)))
single = np.ones(len(row), bool)
if a.single_mask_max > 0 and 'masks' in d.files and len(d['masks']):   # drop votes in masks larger than single_mask_max x the view's median mask area
    mk_ = d['masks']; cams_ = np.load(R / f'cameras_{a.tag}.npz'); VM_, KK_ = cams_['viewmat'], cams_['K']; vi_m = mk_[:, 0].astype(int); Pm = mk_[:, 2:5]
    zm = np.einsum('ni,ni->n', VM_[vi_m, 2, :3], Pm) + VM_[vi_m, 2, 3]; exp_area = np.pi * (KK_[vi_m, 0, 0] * a.apple_r * cam_r / np.maximum(zm, 1e-6)) ** 2; ratio = {(int(r_[0]), int(r_[1])): r_[5] / e_ for r_, e_ in zip(mk_, exp_area)}
    single = np.array([ratio.get((int(x_), int(y_)), 0) <= a.single_mask_max for x_, y_ in zip(vi, lab)])
    print(f'[masklink] single-mask rule: {int((~single).sum())} of {len(row)} votes ({100 * (~single).mean():.0f} %) are in masks > {a.single_mask_max} x the single-apple area at their depth and do not link', flush=True)
if a.mask_outlier_r > 0:   # within-mask spatial consistency: a mask's votes must come from one apple-sized group
    key_ = vi.astype(np.int64) * 100000 + lab.astype(np.int64); order_ = np.argsort(key_, kind='stable'); ks_ = key_[order_]; st_ = np.flatnonzero(np.r_[True, ks_[1:] != ks_[:-1]]); en_ = np.r_[st_[1:], len(ks_)]; inl = np.ones(len(row), bool); nout = 0
    for s0_, e0_ in zip(st_, en_):
        idx_ = order_[s0_:e0_]
        if len(idx_) < 3: continue
        Pm_ = P[row[idx_]]; med_ = np.median(Pm_, 0); far_ = np.linalg.norm(Pm_ - med_, axis=1) > a.mask_outlier_r * a.apple_r * cam_r; inl[idx_[far_]] = False; nout += int(far_.sum())
    single = single & inl; print(f'[masklink] mask-outlier rule: {nout} votes ({100 * nout / len(row):.0f} %) farther than {a.mask_outlier_r} radii from their mask\'s median position do not link', flush=True)
vi_l, lab_l, row_l = vi[single], lab[single], row[single]
grp = np.unique(vi_l * 100000 + lab_l, return_inverse=True)[1]; B = csr_matrix((np.ones(len(row_l), np.float32), (row_l, grp)), shape=(M, grp.max() + 1)); B.data[:] = 1
co = (B @ B.T).tocoo(); keep = co.row < co.col; pa, pb, cov = co.row[keep], co.col[keep], co.data[keep]
A = np.full((M, V), -1, np.int16); A[row_l, vi_l] = lab_l   # conflicts likewise only from single-apple masks
def links(link_views, conflict=True):
    cand = cov >= link_views; ia, ib = pa[cand], pb[cand]; conf = np.zeros(len(ia), np.int32)
    for s in range(0, len(ia), 100000):
        Aa, Ab = A[ia[s:s + 100000]], A[ib[s:s + 100000]]; both = (Aa >= 0) & (Ab >= 0); conf[s:s + 100000] = (both & (Aa != Ab)).sum(1)
    ok = (cov[cand] >= a.co_ratio * np.maximum(conf, 1)) if conflict else np.ones(len(ia), bool)   # co-ratio 1 reproduces co > conf (conf >= 1) and co >= 1 (conf = 0)
    if a.co_frac > 0: ok = ok & (cov[cand] / np.maximum(cov[cand] + conf, 1) >= a.co_frac)
    if a.link_max_r > 0:
        far = np.linalg.norm(P[ia] - P[ib], axis=1) > a.link_max_r * a.apple_r * cam_r; ok = ok & ~far
        print(f'[masklink] distance cap: {int((far & (cov[cand] > conf)).sum())} links longer than {a.link_max_r} apple radii refused', flush=True)
    return ia[ok], ib[ok], int((~ok).sum())
seen_rows = csr_matrix((np.ones(len(row), np.float32), (row, vi)), shape=(M, V)); seen_rows.data[:] = 1
C = d['C']; Fd = d['Fd'] / np.linalg.norm(d['Fd'], axis=1, keepdims=True); UP = d['UPc'].mean(0); UP /= np.linalg.norm(UP)
A_ = np.zeros((3, 3)); b_ = np.zeros(3)
for c, f in zip(C, Fd): Pm = np.eye(3) - np.outer(f, f); A_ += Pm; b_ += Pm @ c
LOOK = np.linalg.solve(A_, b_)
def hd(Y): dd = Y - LOOK; return np.linalg.norm(dd - np.outer(dd @ UP, UP), axis=1)
cam_r = float(np.median(hd(C)))
# ---- component-level merge check, SIZE-BASED (10-15 15:4x). First try (signed graph: split where member pairs are seen in different
# masks more often than together) over-split real apples 2x: under leaf occlusion SAM3 pieces one apple into several masks consistently
# across views, so "different masks" is not "different apples". Apples are spheres of one size, so the check is FruitNeRF's template logic:
# e0 = median RMS extent of the multi-member components seen in >= min_views; a component wider than 1.5 x e0 is k-means-split in two,
# recursively, until every part fits. GT-blind; scored by the mask check. ----
from scipy.cluster.vq import kmeans2
def _ext(idx): Q = P[idx]; return float(np.sqrt(((Q - Q.mean(0)) ** 2).sum(1).mean()))
def _split_size(idx, out, e0, stats):
    if len(idx) < 4 or _ext(idx) <= 1.5 * e0: out.append(idx); return
    lab_, = [kmeans2(P[idx], 2, minit='++', seed=0)[1]]
    if lab_.min() == lab_.max(): out.append(idx); return
    stats[0] += 1; _split_size(idx[lab_ == 0], out, e0, stats); _split_size(idx[lab_ == 1], out, e0, stats)
_refine_cache = {}
def refine(comp, key):
    if key in _refine_cache: return _refine_cache[key]
    if a.merge_check == 'none': _refine_cache[key] = comp; return comp
    if a.merge_check == 'evidence':   # 22:5x: component-pair decision on aggregate evidence (same-mask views vs different-mask views over all co-visible cross pairs)
        k0 = int(comp.max()) + 1; cand_pairs = np.unique(np.stack([np.minimum(comp[pa], comp[pb]), np.maximum(comp[pa], comp[pb])], 1), axis=0); cand_pairs = cand_pairs[cand_pairs[:, 0] != cand_pairs[:, 1]]
        members = {}; 
        for c_ in np.unique(cand_pairs): members[int(c_)] = np.where(comp == c_)[0]
        ev = {}   # aggregate evidence per candidate pair: same-mask views S, different-mask views C over all co-visible cross pairs
        for x_, y_ in cand_pairs:
            X_, Y_ = members[int(x_)], members[int(y_)]
            if len(X_) * len(Y_) > 4_000_000: X_ = X_[:2000]; Y_ = Y_[:2000]
            Ax_, Ay_ = A[X_], A[Y_]; S_ = 0; C_ = 0; step_ = max(1, 20_000_000 // max(1, len(X_) * len(Y_)))
            for v0 in range(0, V, step_):
                ax_ = Ax_[:, v0:v0 + step_]; ay_ = Ay_[:, v0:v0 + step_]; both = (ax_[:, None, :] >= 0) & (ay_[None, :, :] >= 0); same = both & (ax_[:, None, :] == ay_[None, :, :]); S_ += int(same.sum()); C_ += int((both & ~same).sum())
            ev[(int(x_), int(y_))] = (S_, C_)
        # transitive-safe agglomeration (23:3x): merge by descending S - C, only while no pair across the two clusters says "different" (C > S)
        cannot = {k_ for k_, (S_, C_) in ev.items() if C_ > S_}; par = np.arange(k0); clus = {i_: [i_] for i_ in range(k0)}
        def f(x):
            while par[x] != x: par[x] = par[par[x]]; x = par[x]
            return x
        nm_, nblock, nkeep = 0, 0, 0
        for (x_, y_), (S_, C_) in sorted(ev.items(), key=lambda kv: -(kv[1][0] - kv[1][1])):
            if S_ <= C_: nkeep += 1; continue
            rx, ry = f(x_), f(y_)
            if rx == ry: continue
            if any(((min(i_, j_), max(i_, j_)) in cannot) for i_ in clus[rx] for j_ in clus[ry]): nblock += 1; continue
            par[ry] = rx; clus[rx] += clus[ry]; del clus[ry]; nm_ += 1
        new = np.array([f(c_) for c_ in comp]); new = np.unique(new, return_inverse=True)[1]
        print(f'[masklink] evidence merge (transitive-safe): {len(ev)} component pairs examined, {nm_} merged, {nkeep} pairs say different, {nblock} merges blocked by a different-pair elsewhere in the clusters -> {new.max() + 1} components (was {k0})', flush=True)
        _refine_cache[key] = new; return new
    if a.merge_check == 'giants':   # 15:5x: the 1,000-1,500-gaussian components are clusters (their members hit >= 3 masks in 76-79 % of views); a leaf pieces ONE apple into 2 masks at most
        order = np.argsort(comp, kind='stable'); cs = comp[order]; starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]]); ends = np.r_[starts[1:], len(cs)]
        new = np.empty_like(comp); nxt = 0; ng = 0; parts_total = 0
        for s0, e0_ in zip(starts, ends):
            idx = order[s0:e0_]
            if len(idx) < 4: new[idx] = nxt; nxt += 1; continue
            Ac = A[idx]; vis = Ac >= 0; nl = [len(np.unique(Ac[vis[:, v], v])) for v in np.where(vis.sum(0) >= 2)[0]]
            mstar = int(np.quantile(nl, a.giant_q)) if nl else 1
            if mstar < 3: new[idx] = nxt; nxt += 1; continue
            lab_ = kmeans2(P[idx], min(mstar, len(idx)), minit='++', seed=0)[1]; ng += 1
            for l_ in np.unique(lab_): new[idx[lab_ == l_]] = nxt; nxt += 1; parts_total += 1
        print(f'[masklink] giant check: {ng} cluster components (members in >= 3 masks per view) split into {parts_total} parts -> {nxt} components (was {int(comp.max()) + 1})', flush=True)
        _refine_cache[key] = new; return new
    k0 = int(comp.max()) + 1; Cm0 = csr_matrix((np.ones(M, np.float32), (comp, np.arange(M))), shape=(k0, M)); nv0 = np.asarray(((Cm0 @ seen_rows) > 0).sum(1)).ravel(); sz0 = np.bincount(comp, minlength=k0)
    order = np.argsort(comp, kind='stable'); cs = comp[order]; starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]]); ends = np.r_[starts[1:], len(cs)]
    ext0 = np.array([_ext(order[s0:e0]) if e0 - s0 > 1 else 0.0 for s0, e0 in zip(starts, ends)]); ref = (sz0 >= 2) & (nv0 >= a.min_views); e0 = float(np.median(ext0[ref])) if ref.any() else 0.0
    new = np.empty_like(comp); nxt = 0; stats = [0]
    for s0, e0_ in zip(starts, ends):
        idx = order[s0:e0_]; parts = []; _split_size(idx, parts, e0, stats)
        for pt in parts: new[pt] = nxt; nxt += 1
    print(f'[masklink] size check: single-apple extent e0 = {e0 / cam_r:.4f} cam_r (median of {int(ref.sum())} components), {int((ext0[ref] > 1.5 * e0).sum())} wider than 1.5 x e0, {stats[0]} splits -> {nxt} components (was {k0})', flush=True)
    _refine_cache[key] = new; return new
def count(link_views, min_views, crop, conflict=True, dump=False):
    ia, ib, nconf = links(link_views, conflict); G = coo_matrix((np.ones(len(ia)), (ia, ib)), shape=(M, M)); k, comp = connected_components(G, directed=False)
    comp = refine(comp, (link_views, conflict)); k = int(comp.max()) + 1
    Cm = csr_matrix((np.ones(M, np.float32), (comp, np.arange(M))), shape=(k, M)); nviews = np.asarray(((Cm @ seen_rows) > 0).sum(1)).ravel(); size = np.bincount(comp, minlength=k)
    if a.dominant:   # views where the component dominates a mask (>= 50 % of that mask's votes)
        key_ = vi.astype(np.int64) * 100000 + lab.astype(np.int64); gk = np.unique(key_, return_inverse=True)[1]; cm_ = comp[row]
        pair = gk.astype(np.int64) * k + cm_; pv, pc_ = np.unique(pair, return_counts=True); gtot = np.bincount(gk); g_ = pv // k; c_ = pv % k; dom_ = pc_ >= 0.5 * gtot[g_]
        view_of_group = np.zeros(gk.max() + 1, np.int64); view_of_group[gk] = vi; dv = np.unique(np.stack([c_[dom_], view_of_group[g_[dom_]]], 1), axis=0); nviews = np.bincount(dv[:, 0], minlength=k)
    cen = (Cm @ P) / size[:, None]; ext = np.sqrt(np.bincount(comp, weights=((P - cen[comp]) ** 2).sum(1), minlength=k) / size)
    inside = hd(cen) <= crop * cam_r; apple = inside & (nviews >= min_views)
    if a.no_ground and apple.any():   # ground layer: the lowest mode of the counted apples' heights and the first empty bin above it
        hh = (cen - LOOK) @ UP / cam_r; hv = hh[apple]; edges = np.arange(np.floor(hv.min() / 0.025) * 0.025, hv.max() + 0.025, 0.025); hist = np.histogram(hv, bins=edges)[0]
        top = int(np.argmax(hist[: max(1, len(hist) // 3)])); empty = np.where(hist[top:] == 0)[0]; floor = edges[top + int(empty[0])] if len(empty) else hv.min()
        ground = apple & (hh < floor); apple = apple & ~ground; n_ground = int(ground.sum())
        if dump: print(f'[masklink] ground layer: {n_ground} apples below {floor:+.3f} cam_r (lowest mode at {edges[top]:+.3f}) excluded as windfalls', flush=True)
    n = int(apple.sum())
    if dump:
        np.savez_compressed(R / f'masklink_components_{a.dump_tag}.npz', centroid=cen, size=size, nviews=nviews, extent=ext, hd=hd(cen), apple=apple, inside=inside, comp=comp, cam_r=cam_r, LOOK=LOOK, UP=UP, link_views=link_views, min_views=min_views, crop=crop)
        return n, k, len(ia), nconf, size, nviews, ext, cen, apple, inside
    return n
n, k, nl, nconf, size, nviews, ext, cen, apple, inside = count(a.link_views, a.min_views, a.crop, not a.no_conflict, dump=True)
err = f' vs GT {a.gt}: {100 * (n - a.gt) / a.gt:+.1f} %' if a.gt else ''
seen3 = nviews >= a.min_views
print(f"[masklink] {S.name} ({a.tag}): {M} fruit gaussians, {len(pa)} co-landing pairs, {nl} links (>= {a.link_views} co-views{'' if a.no_conflict else ', co-views > conflicts'}; {nconf} pairs dropped by conflicts), {k} components; "
      f"{int(seen3.sum())} seen in >= {a.min_views} views of which {int((seen3 & inside).sum())} inside the crop ({a.crop} x cam_r {cam_r:.3f}) -> apples **{n}**{err}; per apple: fruit gaussians median {np.median(size[apple]) if n else 0:.0f}, "
      f"views median {np.median(nviews[apple]) if n else 0:.0f}, extent median {np.median(ext[apple]) / cam_r if n else 0:.4f} cam_r", flush=True)
hs = np.histogram(hd(cen[seen3]) / cam_r, bins=np.arange(0, 1.61, 0.1))[0]
print('[masklink] hd/cam_r histogram of components seen >= min_views (0.0..1.6 by 0.1): ' + ' '.join(str(x) for x in hs), flush=True)
if a.sens:
    print('[masklink] sensitivity (apples): rows link_views x conflict, cols min_views, per crop', flush=True)
    for crop in (0.4, 0.5, 0.7, 1.0):
        for lv in (1, 2, 3):
            for cf in (True, False):
                print(f'  crop {crop:.1f} link>={lv} conflict={int(cf)}: ' + ' '.join(f'mv{mv}={count(lv, mv, crop, cf)}' for mv in (2, 3, 5)), flush=True)
json.dump({'tag': a.tag, 'apples': n, 'gt': a.gt or None, 'fruit_gaussians': M, 'components': int(k), 'rule': {'share': float(d['share']), 'mass': float(d['mass']), 'depth_tol': float(d['depth_tol']), 'link_views': a.link_views, 'conflict': not a.no_conflict, 'min_views': a.min_views, 'crop': a.crop},
           'centroids': cen[apple].tolist(), 'views': nviews[apple].tolist(), 'sizes': size[apple].tolist()}, open(R / f'registry_masklink_{a.dump_tag}.json', 'w'))
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    e1 = np.cross(UP, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(UP, e1); X = (cen - LOOK) @ e1 / cam_r; Y = (cen - LOOK) @ e2 / cam_r; Xc = (C - LOOK) @ e1 / cam_r; Yc = (C - LOOK) @ e2 / cam_r
    fig, ax = plt.subplots(figsize=(7, 7)); ax.scatter(Xc, Yc, s=4, c='0.6', label='cameras'); r = ~seen3; ax.scatter(X[r], Y[r], s=3, c='0.8', label=f'< {a.min_views} views ({int(r.sum())})')
    r = seen3 & ~inside; ax.scatter(X[r], Y[r], s=8, c='tab:red', label=f'outside crop ({int(r.sum())})'); ax.scatter(X[apple], Y[apple], s=10, c='tab:green', label=f'apples ({n})')
    ax.add_patch(plt.Circle((0, 0), a.crop, fill=False, ls='--')); ax.set_aspect('equal'); ax.legend(fontsize=8); ax.set_title(f'{S.name} {a.tag}: mask-linked registry, top view (units: median camera distance)'); fig.savefig(R / f'registry_masklink_topdown_{a.dump_tag}.png', dpi=110); 
except Exception as ex: print('[masklink] figure skipped:', ex)
