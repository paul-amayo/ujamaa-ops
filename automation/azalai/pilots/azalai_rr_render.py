#!/usr/bin/env python3
"""azalai_rr_render.py (11 Nov 2026) — RSS Azalai pilot "row to relevancy without a keyframe", step 2: renders the twin's identity at
every pose of prep.json (azalai_rr_prep.py) and reads it with the HiGH letter's encoder-path readout, exactly as
azalai_h3_twin_render.py (per-Gaussian features through the checkpoint's own geometry, --void-free 0.5, render scale 0.5; cosine
between the feature direction and g(t_q) = encode_features(raw CLIP text, project=False); void where ||f|| < 0.5; argmax among the
plant addresses of the cell). No outcome is computed here. Per cell:
  mt_<cell>.npz        for every job with a target: its mask M_t (pixels whose argmax is the target's address), bit-packed, for
                       the twin of the letter (canon) and the bag-level twin (bag) where the cell has one;
  <job>.npz            for the jobs flagged full / rgb / depth (e = 0 of Part 2, all of Part 1): the label maps (uint16, 65535 =
                       void), the splat's colour render, and the splat's expected depth in metres (float16; the model's own
                       'depth' output divided by the dataparser scale) beside the identity, for Part 1;
  gauss_<cell>.json    (--gauss, Part 1) per plant address: the number and the median position (cell world, metres) of the
                       Gaussians whose own feature reads that address (same readout, per Gaussian).
The cell-world -> model camera mapping is self-checked on a workspace camera of every cell (printed). Waits while the GPU has less
than --min-free-gb free. nerf_new pixi env:
  azalai_rr_render.py <prep.json> <out dir> [--cells lane6,lane7] [--gauss] [--render-scale 0.5]"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from gsplat import rasterization
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
from nerfstudio.cameras.cameras import Cameras, CameraType
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
from high_splat_hierarchy_accuracy import build_hyper_embedder
ap = argparse.ArgumentParser(); ap.add_argument('prep'); ap.add_argument('out'); ap.add_argument('--cells', default=''); ap.add_argument('--render-scale', type=float, default=0.5); ap.add_argument('--gauss', action='store_true')
ap.add_argument('--void-free', type=float, default=0.5); ap.add_argument('--min-free-gb', type=float, default=14.0); ap.add_argument('--tag', default=''); a = ap.parse_args()
P = json.load(open(a.prep)); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); GL = np.diag([1.0, -1.0, -1.0, 1.0]); VOID_NORM, UNLAB = 0.5, 65535
torch.set_grad_enabled(False); hyp = {}; cells = [c for c in a.cells.split(',') if c] or sorted(P['cells'])
for cell in cells:
    Ln = P['cells'][cell]; J = [j for j in P['jobs'] if j['cell'] == cell]; agg = OUT / f'mt_{cell}.npz'; need_agg = any(j['target'] is not None for j in J) and not agg.exists()
    single = [j for j in J if (j['full'] or j['rgb'] or j['depth']) and not (OUT / f"{j['job']}.npz").exists()]; todo = J if need_agg else single; gz = OUT / f'gauss_{cell}.json'
    if not todo and not (a.gauss and not gz.exists()): print(f'[render] {cell}: nothing to do', flush=True); continue
    while torch.cuda.mem_get_info()[0] / 2**30 < a.min_free_gb: print(f'[render] GPU has {torch.cuda.mem_get_info()[0] / 2**30:.1f} GB free; waiting', flush=True); time.sleep(120)
    t0 = time.time(); _, pipe, _, step = eval_setup(Path(Ln['config']), test_mode='test'); model = pipe.model; model.eval(); model.step = step; dev = model.device
    g = model.gauss_params; N = g['means'].shape[0]; means, quats = g['means'].detach(), g['quats'].detach(); scales = torch.exp(g['scales'].detach()); opac = torch.sigmoid(g['opacities'].detach()).squeeze(-1)
    dp = json.load(open(Path(Ln['config']).parent / 'dataparser_transforms.json')); T = np.eye(4); T[:3] = np.array(dp['transform']); s = float(dp['scale']); fx, fy, cx, cy = Ln['K']; W_, H_ = Ln['wh']
    def camera(c2w_cv):
        c = T @ (np.asarray(c2w_cv, float) @ GL); c[:3, 3] *= s
        return Cameras(camera_to_worlds=torch.tensor(c[:3], dtype=torch.float32)[None], fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy), width=int(W_), height=int(H_), camera_type=CameraType.PERSPECTIVE).to(dev)
    # self-check of the cell-world -> model mapping on a workspace camera (the H3 renderer's check)
    ds = pipe.datamanager.train_dataset; n0 = Path(ds.image_filenames[0]).name; tj = json.load(open(Path(Ln['ws']) / 'transforms.json'))
    p0 = next(np.array(f['transform_matrix'], float) @ GL for f in tj['frames'] if Path(f['file_path']).name == n0); d0 = float((ds.cameras[0:1].to(dev).camera_to_worlds - camera(p0).camera_to_worlds).abs().max())
    assert d0 < 1e-3, f'{cell}: camera mapping check failed on {n0}: {d0}'
    twins = {}; gauss = {}
    for tw in Ln['bank']:
        Nn = Path(Ln['bank'][tw]['identity']); F = torch.from_numpy(np.fromfile(Nn / 'features_B.bin', dtype=np.float32).reshape(N, -1)).to(dev); emb = json.load(open(Nn / 'features_B.bin.json'))['embedder']
        if emb not in hyp: hyp[emb] = build_hyper_embedder(emb, dev)
        bank = np.load(Nn / 'text_bank.npz', allow_pickle=True); words = list(bank['words']); wt = {int(k): w for k, w in json.loads(str(bank['meta']))['word_table'].items()}
        ids = [k for k in sorted(wt) if k < 10000 and wt[k] in words]; assert ids == Ln['bank'][tw]['ids'], f'{cell} {tw}: vocabulary differs from prep'
        G_ = hyp[emb].encode_features(torch.tensor(bank['emb'], dtype=torch.float32, device=dev)[[words.index(wt[k]) for k in ids]], project=False); G_ = G_ / G_.norm(dim=-1, keepdim=True)
        twins[tw] = (F, (F.norm(dim=1, keepdim=True) > 0.5).float(), G_, torch.tensor(ids, dtype=torch.int64, device=dev))
        if a.gauss and not gz.exists():   # the Gaussians whose own feature reads each address, and their place (cell world, metres)
            nf = F.norm(dim=1); am = torch.cat([((F[i:i + 400000] / nf[i:i + 400000].clamp(min=1e-6)[:, None]) @ G_.T).argmax(-1) for i in range(0, N, 400000)]); Rw = torch.tensor(T[:3, :3], dtype=torch.float32, device=dev); tw_ = torch.tensor(T[:3, 3], dtype=torch.float32, device=dev)
            gauss[tw] = {}
            for n_i, k in enumerate(ids):
                sel = (am == n_i) & (nf >= VOID_NORM)
                if sel.any(): gauss[tw][str(k)] = dict(n=int(sel.sum()), median_xyz=((means[sel].median(dim=0).values / s - tw_) @ Rw).cpu().numpy().astype(float).tolist())   # R^T (p / s - t)
                else: gauss[tw][str(k)] = dict(n=0, median_xyz=None)
    if a.gauss and not gz.exists(): json.dump(gauss, open(gz, 'w'))
    print(f'[render] {cell}: {N} gaussians, camera check {d0:.1e} on {n0}, {len(todo)} jobs ({len(single)} with files), loaded in {time.time() - t0:.0f}s', flush=True)
    names = []; mt = {tw: [] for tw in twins}
    for n, j in enumerate(todo):
        cam = camera(j['c2w']); cam.rescale_output_resolution(a.render_scale); viewmat = get_viewmat(cam.camera_to_worlds); K = cam.get_intrinsics_matrices().to(dev); w, h = int(cam.width.item()), int(cam.height.item())
        kw = dict(means=means, quats=quats, scales=scales, opacities=opac, viewmats=viewmat, Ks=K, width=w, height=h, packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB', rasterize_mode=model.config.rasterize_mode)
        o = {}
        for tw, (F, seeded, G_, ids) in twins.items():
            fa, _, _ = rasterization(colors=torch.cat([F, seeded], 1), sh_degree=None, **kw); a_ = fa[0, ..., -1:]
            f = torch.where(a_ >= a.void_free, fa[0, ..., :-1] / a_.clamp(min=1e-6), torch.zeros_like(fa[0, ..., :-1])).reshape(-1, F.shape[1])
            nf = f.norm(dim=-1); am = ((f / nf.clamp(min=1e-6)[:, None]) @ G_.T).argmax(-1); lab = torch.where(nf < VOID_NORM, torch.full_like(am, UNLAB), ids[am]).view(h, w)
            if j['target'] is not None: mt[tw].append(np.packbits((lab == int(j['target'])).cpu().numpy()))
            if j['full']: o[tw] = lab.cpu().numpy().astype(np.uint16)
        if j['target'] is not None: names.append(j['job'])
        if j['rgb'] or j['depth']:
            out = model.get_outputs(cam)
            if j['rgb']: o['rgb'] = (out['rgb'].clamp(0, 1) * 255).byte().cpu().numpy()
            if j['depth']: o['depth'] = (out['depth'][..., 0] / s).cpu().numpy().astype(np.float16)
        if o and not (OUT / f"{j['job']}.npz").exists(): np.savez_compressed(OUT / f"{j['job']}.npz", **o)
        if n % 500 == 0: print(f"[render] {cell} {n}/{len(todo)} {j['job']}: {w}x{h}, {time.time() - t0:.0f}s", flush=True)
    if need_agg: np.savez_compressed(agg, jobs=np.array(names), shape=np.array([h, w]), **{tw: np.stack(v) for tw, v in mt.items()})
    print(f'[render] {cell}: done in {time.time() - t0:.0f}s', flush=True); del model, pipe, twins; torch.cuda.empty_cache()
print('[render] done', flush=True); (OUT / f'DONE{a.tag}').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))
