#!/usr/bin/env python3
"""twin_ledger_render.py (11 Nov 2026) — JIRS test JS-1, step 2 of 4: the twin-read POINTS of every plant address, per cell.
No score is computed here. Design of record: Sankofa_JIRS/numbers/twin_ledger/DESIGN.md. For every frame of prep.json (training
frames that are keyframes of the pass's graph):
  (b) rendered depth (primary): the identity map of the cell is rendered and read with the HiGH letter's encoder-path readout
      (splat_feature_render.py --void-free 0.5: per-Gaussian features blended over the seeded Gaussians only, ||F|| > 0.5;
      argmax_eval.py --space hyper_tangent: cosine between the pixel's feature direction and g(t) =
      encode_features(raw CLIP text, project=False), argmax among the cell's plant addresses; void where the seeded weight
      < 0.5 or ||f|| < 0.5), together with the expected depth of the full splat (gsplat RGB+ED). Every second pixel of the
      half-resolution render that reads an address with depth <= 3.0 m is one point;
  (a) Gaussian means: the seeded Gaussians whose own feature reads the address (same cosine, same argmax), per frame those
      projecting inside the image with depth in (0, 3.0] m (no occlusion test, no opacity cut).
A point x seen at keyframe k is written in the pass's graph frame: c_graph(k) + R^T (x - c_cell(k)).
Lane floor: at every fifth frame, rendered-depth points 2-4 m ahead within 0.3 m of the optical axis (opacity >= 0.5): the 90th
percentile of camera height minus point height along the pass's levelling direction (>= 15 points).
Per cell one npz: b_xyz / b_addr / b_k, a_xyz / a_addr / a_k, lane_k / lane_h / lane_n, per-frame cover. Waits while the GPU has
less than --min-free-gb free. nerf_new pixi env:
  twin_ledger_render.py <prep.json> <out dir> [--season dec] [--cells lane6,lane7] [--max-frames 0]"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from gsplat import rasterization
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
from nerfstudio.cameras.cameras import Cameras, CameraType
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
from high_splat_hierarchy_accuracy import build_hyper_embedder
ap = argparse.ArgumentParser(); ap.add_argument('prep'); ap.add_argument('out'); ap.add_argument('--season', default='dec,apr'); ap.add_argument('--cells', default=''); ap.add_argument('--max-frames', type=int, default=0)
ap.add_argument('--render-scale', type=float, default=0.5); ap.add_argument('--stride', type=int, default=2); ap.add_argument('--max-depth', type=float, default=3.0); ap.add_argument('--min-free-gb', type=float, default=12.0); a = ap.parse_args()
torch.set_num_threads(3); torch.set_grad_enabled(False)
P = json.load(open(a.prep)); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); GL = np.diag([1.0, -1.0, -1.0, 1.0]); VOID_NORM, VOID_FREE = 0.5, 0.5; hyp = {}
for season in a.season.split(','):
    S = P[season]; Rw = np.array(S['R']); up = np.array(S['up']); cg = {int(k): np.array(v) for k, v in S['cam_centres_graph'].items()}
    for cell in ([c for c in a.cells.split(',') if c] or sorted(S['cells'])):
        if cell not in S['cells']: continue
        f_out = OUT / f'{season}_{cell}.npz'
        if f_out.exists(): print(f'[render] {season} {cell}: done already', flush=True); continue
        C = S['cells'][cell]; frames = C['frames'][:a.max_frames] if a.max_frames else C['frames']
        while torch.cuda.mem_get_info()[0] / 2**30 < a.min_free_gb: print(f'[render] GPU has {torch.cuda.mem_get_info()[0] / 2**30:.1f} GB free; waiting', flush=True); time.sleep(120)
        t0 = time.time(); _, pipe, _, step = eval_setup(Path(C['config']), test_mode='test'); model = pipe.model; model.eval(); model.step = step; dev = model.device
        g = model.gauss_params; N = g['means'].shape[0]; means, quats = g['means'].detach(), g['quats'].detach(); scales = torch.exp(g['scales'].detach()); opac = torch.sigmoid(g['opacities'].detach()).squeeze(-1)
        dp = json.load(open(Path(C['config']).parent / 'dataparser_transforms.json')); T = np.eye(4); T[:3] = np.array(dp['transform']); s = float(dp['scale']); fx, fy, cx, cy = C['K']; W_, H_ = C['wh']
        def camera(c2w_cv):
            c = T @ (np.asarray(c2w_cv, float) @ GL); c[:3, 3] *= s
            return Cameras(camera_to_worlds=torch.tensor(c[:3], dtype=torch.float32)[None], fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy), width=int(W_), height=int(H_), camera_type=CameraType.PERSPECTIVE).to(dev)
        ds = pipe.datamanager.train_dataset; n0 = Path(ds.image_filenames[0]).name; tj = json.load(open(Path(C['ws']) / 'transforms.json'))
        p0 = next(np.array(f['transform_matrix'], float) @ GL for f in tj['frames'] if Path(f['file_path']).name == n0); d0 = float((ds.cameras[0:1].to(dev).camera_to_worlds - camera(p0).camera_to_worlds).abs().max())
        assert d0 < 1e-3, f'{cell}: camera mapping check failed on {n0}: {d0}'
        Nn = Path(C['identity']); F = torch.from_numpy(np.fromfile(Nn / 'features_B.bin', dtype=np.float32).reshape(N, -1)).to(dev); emb = json.load(open(Nn / 'features_B.bin.json'))['embedder']
        if emb not in hyp: hyp[emb] = build_hyper_embedder(emb, dev)
        bank = np.load(Nn / 'text_bank.npz', allow_pickle=True); words = list(bank['words']); wt = {int(k): w for k, w in json.loads(str(bank['meta']))['word_table'].items()}
        ids = [k for k in sorted(wt) if k < 10000 and wt[k] in words]; assert ids == C['ids'], f'{cell}: vocabulary differs from prep'
        G_ = hyp[emb].encode_features(torch.tensor(bank['emb'], dtype=torch.float32, device=dev)[[words.index(wt[k]) for k in ids]], project=False); G_ = G_ / G_.norm(dim=-1, keepdim=True)
        idt = torch.tensor(ids, dtype=torch.int64, device=dev); nF = F.norm(dim=1); seeded = (nF > 0.5); seedf = seeded.float()[:, None]
        # (a) the address each seeded Gaussian's own feature reads
        gi = torch.nonzero(seeded).squeeze(1); g_addr = torch.empty(len(gi), dtype=torch.int64, device=dev)
        for j in range(0, len(gi), 200000): g_addr[j:j + 200000] = idt[((F[gi[j:j + 200000]] / nF[gi[j:j + 200000], None]) @ G_.T).argmax(-1)]
        g_means = means[gi]; lab_file = Nn / 'features_B.bin.labels.npy'; agree = None
        if lab_file.exists(): lab = torch.from_numpy(np.load(lab_file)).to(dev)[gi]; mk = (lab >= 0) & (lab < 10000); agree = (float((g_addr[mk] == lab[mk]).float().mean()), int(mk.sum()))
        print(f'[render] {season} {cell}: {N} gaussians, {len(gi)} seeded, {len(ids)} addresses, camera check {d0:.1e}, {len(frames)} frames; own-feature readout equals the seed label for {agree}; loaded in {time.time() - t0:.0f}s', flush=True)
        upt = torch.tensor(up, dtype=torch.float32, device=dev); B = dict(xyz=[], addr=[], k=[]); Ag = dict(xyz=[], addr=[], k=[]); lane_k, lane_h, lane_n, cover = [], [], [], []
        for n, fr in enumerate(frames):
            k = int(fr['k']); c2w = np.array(fr['c2w'], float); cam = camera(c2w); cam.rescale_output_resolution(a.render_scale); viewmat = get_viewmat(cam.camera_to_worlds); K = cam.get_intrinsics_matrices().to(dev); w, h = int(cam.width.item()), int(cam.height.item())
            fa, alpha, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=torch.cat([F, seedf], 1), viewmats=viewmat, Ks=K, width=w, height=h, packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB+ED', sh_degree=None, rasterize_mode=model.config.rasterize_mode)
            D = F.shape[1]; a_ = fa[0, ..., D:D + 1]; f = torch.where(a_ >= VOID_FREE, fa[0, ..., :D] / a_.clamp(min=1e-6), torch.zeros_like(fa[0, ..., :D])); z = fa[0, ..., D + 1] / s; al = alpha[0, ..., 0]
            nf = f.norm(dim=-1); am = (f.reshape(-1, D) / nf.reshape(-1, 1).clamp(min=1e-6)) @ G_.T; lab = torch.where(nf.reshape(-1) < VOID_NORM, torch.full((h * w,), -1, dtype=torch.int64, device=dev), idt[am.argmax(-1)]).view(h, w)
            vv, uu = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing='ij'); kx, ky, kcx, kcy = K[0, 0, 0], K[0, 1, 1], K[0, 0, 2], K[0, 1, 2]
            pc = torch.stack([(uu + 0.5 - kcx) / kx * z, (vv + 0.5 - kcy) / ky * z, z], -1)   # camera frame (OpenCV), metres
            M = torch.tensor(Rw.T @ c2w[:3, :3], dtype=torch.float32, device=dev); cgk = torch.tensor(cg[k], dtype=torch.float32, device=dev)   # graph offset = M p_cam
            sel = torch.zeros_like(lab, dtype=torch.bool); sel[::a.stride, ::a.stride] = True; sel &= (lab >= 0) & (z > 0) & (z <= a.max_depth)
            B['xyz'].append((pc[sel] @ M.T + cgk).cpu().numpy().astype(np.float32)); B['addr'].append(lab[sel].cpu().numpy().astype(np.uint16)); B['k'].append(np.full(int(sel.sum()), k, np.int32))
            # (a) Gaussian means in this frame
            q = g_means @ viewmat[0, :3, :3].T + viewmat[0, :3, 3]; zq = q[:, 2] / s; uq = kx * q[:, 0] / q[:, 2] + kcx; vq = ky * q[:, 1] / q[:, 2] + kcy; mg = (zq > 0) & (zq <= a.max_depth) & (uq >= 0) & (uq < w) & (vq >= 0) & (vq < h)
            Ag['xyz'].append(((q[mg] / s) @ M.T + cgk).cpu().numpy().astype(np.float32)); Ag['addr'].append(g_addr[mg].cpu().numpy().astype(np.uint16)); Ag['k'].append(np.full(int(mg.sum()), k, np.int32))
            if n % 5 == 0:   # lane floor
                ln = (z >= 2.0) & (z <= 4.0) & (pc[..., 0].abs() <= 0.3) & (al >= 0.5)
                if int(ln.sum()) >= 15: lane_k.append(k); lane_h.append(float(torch.quantile(-((pc[ln] @ M.T) @ upt), 0.9))); lane_n.append(int(ln.sum()))
            cover.append((k, float((lab >= 0).float().mean()), int(sel.sum()), int(mg.sum())))
            if n % 50 == 0: print(f"[render] {season} {cell} {n}/{len(frames)} k {k}: identity on {cover[-1][1] * 100:.1f}% of px, {cover[-1][2]} depth points, {cover[-1][3]} Gaussian points, {time.time() - t0:.0f}s", flush=True)
        cat = lambda L, dt: (np.concatenate(L) if L else np.zeros((0,), dt))
        np.savez(f_out, b_xyz=(np.concatenate(B['xyz']) if B['xyz'] else np.zeros((0, 3), np.float32)), b_addr=cat(B['addr'], np.uint16), b_k=cat(B['k'], np.int32), a_xyz=(np.concatenate(Ag['xyz']) if Ag['xyz'] else np.zeros((0, 3), np.float32)), a_addr=cat(Ag['addr'], np.uint16), a_k=cat(Ag['k'], np.int32),
                 lane_k=np.array(lane_k, np.int32), lane_h=np.array(lane_h, np.float32), lane_n=np.array(lane_n, np.int32), cover=np.array(cover, np.float64), seeded=int(len(gi)), gaussians=int(N), label_agree=np.array(agree if agree else (np.nan, 0), float))
        print(f"[render] {season} {cell}: wrote {f_out.name}: {sum(len(x) for x in B['addr'])} depth points, {sum(len(x) for x in Ag['addr'])} Gaussian points, lane floor on {len(lane_h)} frames, {time.time() - t0:.0f}s", flush=True)
        del model, pipe, F, means, quats, scales, opac, g_means; torch.cuda.empty_cache()
print('[render] done', flush=True); (OUT / f'DONE_{a.season.replace(",", "_")}{("_" + a.cells.replace(",", "_")) if a.cells else ""}').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))
