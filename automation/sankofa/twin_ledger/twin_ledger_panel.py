#!/usr/bin/env python3
"""twin_ledger_panel.py (11 Nov 2026, A100 session) - JIRS test JS-1, verification panels (addendum A100-1 item 12): the twelve
bags chosen by rank of |twin - lidar height| BEFORE any was looked at (records/panels_selected.json, written by the scorer).
Each is drawn in its keyframe (the observing keyframe with the most points among those that gave its top), rendered by a cell
that has the address's points there: the training image at the render's half resolution, the pixels whose identity reads the
address (magenta) and the pixels that read any other address (light cyan), the twin top (cyan circle, at the twin-read
position) and the lidar top (orange square, at the census position), each with a thin line down to its lane floor
(twin: camera height - the twin lane constant; lidar: camera height - the lidar mount_lane). The readout is the render's
(seeded blend, void-free 0.5, ||f|| < 0.5 void, cosine to g(t), argmax among the cell's addresses). nerf_new pixi env:
  twin_ledger_panel.py <prep.json> <records dir> <results dir>"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from gsplat import rasterization
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
from nerfstudio.cameras.cameras import Cameras, CameraType
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
from high_splat_hierarchy_accuracy import build_hyper_embedder
torch.set_num_threads(3); torch.set_grad_enabled(False)
P = json.load(open(sys.argv[1])); REC = Path(sys.argv[2]); RES = Path(sys.argv[3]); OUT = REC / 'panels'; OUT.mkdir(parents=True, exist_ok=True)
PAN = json.load(open(REC / 'panels_selected.json')); SUM = json.load(open(RES / 'summary.json')); GL = np.diag([1.0, -1.0, -1.0, 1.0]); SCALE = 0.5; hyp = {}
while torch.cuda.mem_get_info()[0] / 2**30 < 12: print('[panel] GPU short; waiting', flush=True); time.sleep(120)


def at_height(x, z, h, up):
    """the graph-frame point with ground position (x, z) at height h along up"""
    return np.array([x, (h - x * up[0] - z * up[2]) / up[1], z])


done = []
for cell in sorted({(p['season'], p['cell']) for p in PAN}):
    season, cname = cell; S = P[season]; C = S['cells'][cname]; up = np.array(S['up'], float); Rw = np.array(S['R'], float)
    _, pipe, _, step = eval_setup(Path(C['config']), test_mode='test'); model = pipe.model; model.eval(); dev = model.device
    g = model.gauss_params; N = g['means'].shape[0]; means, quats = g['means'].detach(), g['quats'].detach(); scales = torch.exp(g['scales'].detach()); opac = torch.sigmoid(g['opacities'].detach()).squeeze(-1)
    dp = json.load(open(Path(C['config']).parent / 'dataparser_transforms.json')); T = np.eye(4); T[:3] = np.array(dp['transform']); s = float(dp['scale']); fx, fy, cx, cy = C['K']; W_, H_ = C['wh']
    Nn = Path(C['identity']); F = torch.from_numpy(np.fromfile(Nn / 'features_B.bin', dtype=np.float32).reshape(N, -1)).to(dev); emb = json.load(open(Nn / 'features_B.bin.json'))['embedder']
    if emb not in hyp: hyp[emb] = build_hyper_embedder(emb, dev)
    bank = np.load(Nn / 'text_bank.npz', allow_pickle=True); words = list(bank['words']); wt = {int(k): w for k, w in json.loads(str(bank['meta']))['word_table'].items()}
    ids = [k for k in sorted(wt) if k < 10000 and wt[k] in words]; assert ids == C['ids']
    G_ = hyp[emb].encode_features(torch.tensor(bank['emb'], dtype=torch.float32, device=dev)[[words.index(wt[k]) for k in ids]], project=False); G_ = G_ / G_.norm(dim=-1, keepdim=True)
    idt = torch.tensor(ids, dtype=torch.int64, device=dev); seedf = (F.norm(dim=1) > 0.5).float()[:, None]
    for p in [p for p in PAN if (p['season'], p['cell']) == cell]:
        fr = next(f for f in C['frames'] if f['k'] == p['keyframe']); c2w = np.array(fr['c2w'], float)
        c = T @ (c2w @ GL); c[:3, 3] *= s
        cam = Cameras(camera_to_worlds=torch.tensor(c[:3], dtype=torch.float32)[None], fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy), width=int(W_), height=int(H_), camera_type=CameraType.PERSPECTIVE).to(dev)
        cam.rescale_output_resolution(SCALE); viewmat = get_viewmat(cam.camera_to_worlds); K = cam.get_intrinsics_matrices().to(dev); w, h = int(cam.width.item()), int(cam.height.item())
        fa, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=torch.cat([F, seedf], 1), viewmats=viewmat, Ks=K, width=w, height=h, packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB', sh_degree=None, rasterize_mode=model.config.rasterize_mode)
        D = F.shape[1]; a_ = fa[0, ..., D:D + 1]; f = torch.where(a_ >= 0.5, fa[0, ..., :D] / a_.clamp(min=1e-6), torch.zeros_like(fa[0, ..., :D])); nf = f.norm(dim=-1)
        lab = torch.where(nf.reshape(-1) < 0.5, torch.full((h * w,), -1, dtype=torch.int64, device=dev), idt[((f.reshape(-1, D) / nf.reshape(-1, 1).clamp(min=1e-6)) @ G_.T).argmax(-1)]).view(h, w).cpu().numpy()
        img = np.asarray(Image.open(Path(C['ws']) / 'images' / fr['name']).convert('RGB').resize((w, h), Image.BILINEAR)).astype(float) / 255.0
        ov = img.copy(); mine = lab == p['address']; other = (lab >= 0) & ~mine
        ov[other] = 0.75 * ov[other] + 0.25 * np.array([0.3, 0.9, 1.0]); ov[mine] = 0.45 * ov[mine] + 0.55 * np.array([1.0, 0.0, 0.9])
        Kh = np.array([[fx * SCALE, 0, cx * SCALE], [0, fy * SCALE, cy * SCALE], [0, 0, 1]]); cg = np.array(S['cam_centres_graph'][str(p['keyframe'])], float); camh = float(cg @ up)
        def proj(X):   # graph frame -> cell camera at this keyframe (inverse of the transfer c_g + R^T (x - c_cell))
            q = c2w[:3, :3].T @ (Rw @ (X - cg)); return (Kh @ (q / q[2]))[:2] if q[2] > 0.05 else None
        tw = p['twin_pos']; ce = p['census_xyz']; lane_t = SUM['lane_constant'][season]['twin']; lane_l = SUM['lane_constant'][season]['lidar']
        marks = dict(twin_top=proj(at_height(tw[0], tw[2], p['twin_top'], up)), twin_floor=proj(at_height(tw[0], tw[2], camh - lane_t, up)),
                     lidar_top=proj(at_height(ce[0], ce[2], p['lidar_top'], up)), lidar_floor=proj(at_height(ce[0], ce[2], camh - lane_l, up)))
        fig, ax = plt.subplots(figsize=(6.4, 3.9)); ax.imshow(ov); ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_axis_off()
        for (top, flo, col, mk, lb) in (('twin_top', 'twin_floor', '#00E5FF', 'o', 'twin top'), ('lidar_top', 'lidar_floor', '#FF8C00', 's', 'lidar top')):
            if marks[top] is not None:
                ax.plot(*marks[top], marker=mk, ms=9, mfc='none', mec=col, mew=2, label=lb)
                if marks[flo] is not None: ax.plot([marks[top][0], marks[flo][0]], [marks[top][1], marks[flo][1]], color=col, lw=1)
        ax.legend(loc='lower right', fontsize=7, framealpha=0.8)
        ttl = (f"{'December' if season == 'dec' else 'April 10:54'}, rank q={p['rank_q']:.2f} of {p['n']}: bag {p['bag']}, address {p['address']}\n"
               f"twin {p['twin_height']:.3f} m, lidar {p['lidar_height']:.3f} m, |diff| {p['abs_diff']:.3f} m; keyframe {p['keyframe']} ({cname}); address on {mine.sum()} px")
        ax.set_title(ttl, fontsize=7.5); fig.tight_layout(); fn = OUT / f"{season}_q{int(round(100 * p['rank_q'])):03d}_bag{p['bag']}.png"; fig.savefig(fn, dpi=150); plt.close(fig)
        done.append(dict(p, file=str(fn), address_px=int(mine.sum()), other_px=int(other.sum()), marks={k: (None if v is None else [float(v[0]), float(v[1])]) for k, v in marks.items()}, image=fr['name']))
        print(f'[panel] {fn.name}: address on {mine.sum()} px, marks {done[-1]["marks"]}', flush=True)
    del model, pipe, F, means, quats, scales, opac; torch.cuda.empty_cache()
json.dump(done, open(OUT / 'panels.json', 'w'), indent=1)
# contact sheet: December top row, April bottom row, by rank
fs = sorted(done, key=lambda d: (d['season'] != 'dec', d['rank_q'])); fig, axs = plt.subplots(2, 6, figsize=(30, 7.6))
for ax, d in zip(axs.ravel(), fs): ax.imshow(plt.imread(d['file'])); ax.set_axis_off()
fig.tight_layout(); fig.savefig(OUT / 'contact_sheet.png', dpi=110); print('[panel] done', flush=True)
