"""Camera path for the 'ask the orchard' demo on 05 (Paul, 2026-09-27) in the H3DGS z-up frame (OpenCV c2w), two segments:
  drive : the row's keyframe poses in order (every --stride-th); questions and overlay modes keyed to time
  orbit : a lane-side ARC around the tree with the most census-assigned gaussians in --orbit-block (a full circle would
          pass through the hedge and the next row, where nothing was ever observed; the 6 m lift-off was dropped for the
          same reason — the ground-level model fills the air above the canopy with sky gaussians)
Also exports every tree's centre (median of its census-assigned gaussians, per block seed) and the drive track for the
top-down map inset. Writes demo_path.json: {scale, intrinsics, fps, frames: [{i, name, c2w_h, segment, t, mode, question, focus}],
trees: {id: [x, y, z]}, track: [[x, y], ...], orbit_tree, rows: {block: [row words]}}.
Modes: plain | row (row outlines) | tree (one tree tinted: the largest in view when asked) | all (every tree + rows)
       | rows (tint = row colour) | orchard (one colour).
  pixi run python sidecar_demo_path.py --survey 05_13D_Jackal --blocks 018 019 020 021 022 023 --orbit-block 020 --out <dir>"""
import argparse, json
from pathlib import Path
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument('--survey', required=True); ap.add_argument('--blocks', nargs='+', required=True); ap.add_argument('--orbit-block', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--fps', type=int, default=8); ap.add_argument('--stride', type=int, default=1)
ap.add_argument('--orbit-radius', type=float, default=5.0); ap.add_argument('--orbit-seconds', type=float, default=15.0); ap.add_argument('--orbit-arc', type=float, default=120.0, help='degrees of arc, centred on the lane-side normal')
ap.add_argument('--scale', type=float, default=0.5); ap.add_argument('--seed-tag', default='glref_bg_f1.0_r2')
# chunk mode (2026-10-01, the demo chunks): models are chunk side-cars under --sidecar-root (dir names, no block_ prefix), the
# H3DGS project is --proj (h3dgs_expo for 05), and the drive is the side-car dataset's cameras INSIDE the chunk cell, in time
# order, as passes of >= --min-pass frames (a 30 m cell holds ~110-frame row passes; several passes = jump cuts between rows).
ap.add_argument('--sidecar-root', default='experimental/h3dgs_sidecar'); ap.add_argument('--proj', default='experimental/h3dgs')
ap.add_argument('--survey-root', default='', help='absolute survey root (default citrus_all/<survey>)'); ap.add_argument('--chunk-drive', action='store_true'); ap.add_argument('--min-pass', type=int, default=40); ap.add_argument('--chunk', default='', help='chunk name for the cell test in --chunk-drive (default: from the model name chunk_<c>[_tag])')
a = ap.parse_args(); S = Path(a.survey_root) if a.survey_root else Path('/home/paperspace/data/citrus_all') / a.survey; P = S / a.proj; OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
import sys; sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH'); from word_utils import get_word_for_id
def model_dir(b):
    d = S / a.sidecar_root / b
    return d if d.exists() else S / a.sidecar_root / f'block_{b}'
meta = json.load(open(P / 'export_meta.json')); R_W = np.asarray(meta.get('world_rotation_to_zup') or meta['world_rotation_lio_to_h3dgs'], np.float64); GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])
def block_frames(b):
    if a.chunk_drive:   # the chunk side-car's own dataset: every camera the chunk BA used; keep those INSIDE the cell, as passes
        tj = json.load(open(model_dir(b) / 'transforms.json')); assert str(tj.get('pose_convention', '')).startswith('opengl'), b
        import re as _re
        cn = a.chunk or _re.match(r'chunk_([0-9]+_[0-9]+)', b).group(1); ctr = np.loadtxt(P / f'camera_calibration/chunks/{cn}/center.txt'); ext = np.loadtxt(P / f'camera_calibration/chunks/{cn}/extent.txt')
        fr = sorted(tj['frames'], key=lambda f: int(_re.search(r'(\d+)', Path(f['file_path']).stem).group(1))); out = []
        for f in fr:
            M = R_W @ (np.asarray(f['transform_matrix'], np.float64) @ GL2CV); c = M[:3, 3]
            if abs(c[0] - ctr[0]) <= ext[0] / 2 and abs(c[1] - ctr[1]) <= ext[1] / 2: out.append((int(_re.search(r'(\d+)', Path(f['file_path']).stem).group(1)), Path(f['file_path']).name, M))
        runs = []; st = 0
        for i in range(1, len(out) + 1):
            if i == len(out) or out[i][0] - out[i - 1][0] > 3: runs.append(out[st:i]); st = i
        keep = [r for r in runs if len(r) >= a.min_pass]
        print(f"[path] chunk {cn}: {len(out)} in-cell cameras in {len(runs)} passes, keeping {len(keep)} passes of >= {a.min_pass} frames: " + ', '.join(f'kf {r[0][0]}-{r[-1][0]} ({len(r)})' for r in keep), flush=True)
        return tj, [(n, M) for r in keep for _, n, M in r]
    tj = json.load(open(S / 'prod/tassili/blocks_ns/lio_row100' / f'block_{b}' / 'transforms.json')); assert str(tj.get('pose_convention', '')).startswith('opengl'), b
    fr = sorted(tj['frames'], key=lambda f: f['file_path']); return tj, [(Path(f['file_path']).name, R_W @ (np.asarray(f['transform_matrix'], np.float64) @ GL2CV)) for f in fr]
def tree_centres(b):
    """id -> centre (H3DGS frame) from the block's seed: gaussians carrying a feature, majority label from the void-row census"""
    O = model_dir(b); run = sorted((O / 'splat_runs_FEATFIX').glob(f'stage2_censusinit_{a.seed_tag}/high/*'))[-1]
    ck = torch.load(next(run.glob('nerfstudio_models/*.ckpt')), map_location='cpu', weights_only=False)
    means = [v for k, v in ck['pipeline'].items() if k.endswith('gauss_params.means')][0].numpy(); hf = [v for k, v in ck['pipeline'].items() if k.endswith('gauss_params.high_features')][0].numpy()
    z = np.load(O / 'splat_runs_FEATFIX/interaction_W_glref_bg.npz'); labels = [int(u) for u in z['labels']]; keep = [i for i, u in enumerate(labels) if u != 65535]; Wm = z['W'][keep]; labels = [labels[i] for i in keep]
    dp = json.load(open(run / 'dataparser_transforms.json')); T = np.asarray(dp['transform'], np.float64); s = float(dp['scale'])
    assigned = np.abs(hf).sum(1) > 0; maj = Wm.argmax(0); out = {}
    for i, u in enumerate(labels):
        m = (maj == i) & assigned
        if m.sum() < 200: continue
        p_lio = (means[m] / s - T[:, 3]) @ T[:, :3]; p_h = p_lio @ R_W[:3, :3].T; out[u] = (np.median(p_h, 0), int(m.sum()))
    return out
tj0, _ = block_frames(a.blocks[0]); K = {k: float(tj0[k]) for k in ('fl_x', 'fl_y', 'cx', 'cy')}; K.update(w=int(tj0['w']), h=int(tj0['h']))
drive = []; trees = {}; rows = {}
vfile = S / 'prod/tassili/blocks_ns/lio_row100/verdicts_censusinit_glref.json'; verd = dict(json.load(open(vfile))['blocks']) if vfile.exists() else {}
HROW = {t: r['id'] for r in json.load(open(S / 'prod/bateleur/scene_graph/marker_hierarchy.json'))['rows'] for t in r['object_ids']}
for b in a.blocks:
    _, fr = block_frames(b); drive += fr[::a.stride]
    tc = tree_centres(b)
    for u, (c, n) in tc.items():
        if u not in trees or n > trees[u][1]: trees[u] = (c, n)
    if a.chunk_drive:   # row words = the hierarchy rows of the trees this side-car actually carries (the block verdict file has no entry for a chunk)
        rows[b] = [get_word_for_id(r, 'row') for r in sorted({HROW[u] for u in tc if u in HROW})]
    else: rows[b] = list(verd.get(b, {}).get('rows', {}).keys())
def lookat(pos, target, up=np.array([0, 0, 1.0])):
    f = target - pos; f /= np.linalg.norm(f); r = np.cross(f, up); r /= np.linalg.norm(r); d = np.cross(f, r)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = r, d, f, pos; return M      # OpenCV camera: x right, y down, z forward
frames = []; fps = a.fps; T_drive = len(drive) / fps
sched = [(0.0, 'plain', ''), (0.12, 'row', 'show me this row'), (0.30, 'tree', 'which tree is this?'), (0.40, 'all', 'show me every tree'), (0.72, 'rows', 'group them by row'), (0.88, 'orchard', 'the whole orchard')]   # 'tree' kept to ~6 s: the chosen tree leaves the view after ~5 s of driving
for j, (name, M) in enumerate(drive):
    t = j / fps; mode, q = 'plain', ''
    for frac, m_, q_ in sched:
        if t >= frac * T_drive: mode, q = m_, q_
    frames.append(dict(name=f'd_{j:05d}.png', c2w_h=M.tolist(), segment='drive', t=round(t, 3), mode=mode, question=q, src=name))
# orbit: lane-side arc around the focus tree (the block's most-populated tree), at the drive's camera height (skipped when --orbit-seconds 0)
tc = tree_centres(a.orbit_block); focus = max(tc, key=lambda u: tc[u][1]); centre = tc[focus][0]
C = np.array([M[:3, 3] for _, M in block_frames(a.orbit_block)[1]]); zcam = float(np.median(C[:, 2]))
# a circle (even a lane-side arc at 5 m) leaves the surveyed track and ends up inside the opposite hedge; instead a DOLLY:
# the camera slides along the drive track past the tree (+- orbit-radius m along the row) while keeping the tree centred
k0 = int(np.argmin(np.linalg.norm(C[:, :2] - centre[:2], axis=1))); dirv = C[min(k0 + 5, len(C) - 1), :2] - C[max(k0 - 5, 0), :2]; dirv /= np.linalg.norm(dirv)
n_orb = int(a.orbit_seconds * fps)
for j in range(n_orb):   # 0 iterations when --orbit-seconds 0
    u = -1 + 2 * (0.5 - 0.5 * np.cos(np.pi * j / max(n_orb - 1, 1)))                  # -1 .. 1, eased
    pos = np.array([C[k0, 0] + u * a.orbit_radius * dirv[0], C[k0, 1] + u * a.orbit_radius * dirv[1], zcam])
    frames.append(dict(name=f'o_{j:05d}.png', c2w_h=lookat(pos, centre + np.array([0, 0, 0.6])).tolist(), segment='orbit', t=round(j / fps, 3), mode='all', question='', focus=int(focus)))
for i, f in enumerate(frames): f['i'] = i
json.dump(dict(scale=a.scale, fps=fps, intrinsics=K, frames=frames, trees={str(u): [float(x) for x in c] for u, (c, n) in trees.items()}, track=[[float(M[0, 3]), float(M[1, 3])] for _, M in drive],
               orbit_tree=int(focus), orbit_centre_h=[float(x) for x in centre], rows=rows), open(OUT / 'demo_path.json', 'w'))
print(f"[path] drive {len(drive)} frames ({T_drive:.0f} s), orbit {n_orb} ({a.orbit_seconds:.0f} s, arc {a.orbit_arc:.0f} deg, r {a.orbit_radius} m) = {len(frames)} frames; {len(trees)} trees mapped; orbit tree {focus} ({tc[focus][1]} gaussians) centre {centre.round(2).tolist()} cam z {zcam:.2f} -> {OUT / 'demo_path.json'}", flush=True)
