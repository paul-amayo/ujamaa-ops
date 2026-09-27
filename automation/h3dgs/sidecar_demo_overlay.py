"""'Ask the orchard' demo compositor (Paul, 2026-09-27). Pass 1: for every demo-path frame and every side-car model within
--max-dist m, render the 32-d identity field at the frame's camera (converted into that model's nerfstudio frame) and keep
the best tree / row per pixel with its margin over the verdict threshold. Pass 2: composite over the full-H3DGS backdrop
by best margin across models, apply the segment's mode (plain | row | tree | all | rows | orchard), the typed question,
the top-down identity map inset, and the sky mask; then ffmpeg. Words never appear (internal indices).
  HIGH_EMBEDDER_CKPT=<emb> pixi run python sidecar_demo_overlay.py --survey 05_13D_Jackal --path demo_path.json
      --models 018 019 020 021 022 023 --backdrop-dir <dir> --out <dir> [--seed-tag glref_bg_f1.0_r2] [--max-dist 70]"""
import argparse, glob, json, os, re, subprocess, sys, time
from pathlib import Path
import numpy as np, torch, cv2
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH')
from high_splat_hierarchy_accuracy import render_frame, build_hyper_embedder
from word_utils import get_word_for_id
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras, CameraType
import lorentz as L, open_clip
UNLAB = 65535; FRUIT_ID_BASE = 10000; dev = 'cuda'; torch.set_grad_enabled(False)
ap = argparse.ArgumentParser(); ap.add_argument('--survey', required=True); ap.add_argument('--path', required=True); ap.add_argument('--models', nargs='+', required=True)
ap.add_argument('--backdrop-dir', required=True); ap.add_argument('--out', required=True); ap.add_argument('--seed-tag', default='glref_bg_f1.0_r2'); ap.add_argument('--verdict-tag', default='bg_f1.0_r2')
ap.add_argument('--max-dist', type=float, default=70.0); ap.add_argument('--tree-thr', type=float, default=0.5); ap.add_argument('--row-thr', type=float, default=0.8); ap.add_argument('--skip-maps', action='store_true')
ap.add_argument('--row-pick', choices=('raw', 'margin'), default='raw', help="which row word wins a pixel. raw (default): the word the field scores HIGHEST — the two row words compared on one common basis. margin (the old rule, and the row-crossing bug): argmax of score MINUS that word's own per-block verdict threshold, which on block 023 (oak thr 0.70 / pine thr 0.85) hands a pine pixel scoring oak 0.762 / pine 0.871 to oak, because +0.062 > +0.021. Either way the pixel is only claimed where the winning word clears its own threshold.")
ap.add_argument('--min-area', type=int, default=800, help="drop connected components smaller than this from each ROW region before drawing (0 = off). Compositing only, no re-render. On kf_001508: none -> oak 35 / pine 411 pieces and 1902 wrong-row px; 200 -> 2/3 pieces, 417 px; 800 -> 1/1 piece, 0 wrong-row px, for 4.5% of the drawn area.")
ap.add_argument('--tree-min-area', type=int, default=200, help='same for each TREE region, kept lower so distant trees still register')
ap.add_argument('--row-from-tree', action='store_true', help="SUPERSEDED by --row-pick raw, and not needed for a row query: route the row answer through the identified tree's hierarchy row (marker_hierarchy.json) instead of the row decode. Kept only to reproduce demo v3.")
a = ap.parse_args(); S = Path('/home/paperspace/data/citrus_all') / a.survey; P = S / 'experimental/h3dgs'; OUT = Path(a.out); (OUT / 'frames').mkdir(parents=True, exist_ok=True); (OUT / 'maps').mkdir(exist_ok=True)
pj = json.load(open(a.path)); FR = pj['frames']; K = pj['intrinsics']; sc = pj['scale']; fps = pj['fps']; W, H = int(round(K['w'] * sc)), int(round(K['h'] * sc))
meta = json.load(open(P / 'export_meta.json')); R_W = np.asarray(meta.get('world_rotation_to_zup') or meta['world_rotation_lio_to_h3dgs'], np.float64)[:3, :3]; GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])
SKY = S / 'prod/tassili/sky_masks'; VL = f'/home/paperspace/logs/sidecar_{a.survey}_verdicts.log'
def verdict_thr(block):
    out = {}
    if Path(VL).exists():
        for line in open(VL):
            if line.startswith(f'[{block} sidecar {a.verdict_tag} '):
                m = re.search(r'"([a-z]+)": thr ([0-9.]+)', line)
                if m: out[m.group(1)] = float(m.group(2))
    return out
def colour(tid):
    h = (int(tid) * 37) % 180; return tuple(int(v) for v in cv2.cvtColor(np.uint8([[[h, 210, 235]]]), cv2.COLOR_HSV2BGR)[0, 0])
ROWCOL = {}; ORCHARD = (60, 220, 60)
# ---------------- pass 1: identity maps per model
if not a.skip_maps:
    clip, _, _ = open_clip.create_model_and_transforms('ViT-B-16', 'laion2b_s34b_b88k', device=dev); tok = open_clip.get_tokenizer('ViT-B-16')
    EMB = os.environ['HIGH_EMBEDDER_CKPT']; hyper = build_hyper_embedder(EMB, dev); curv = hyper.curv.exp()
    def word_vec(words):
        e = clip.encode_text(tok(words).to(dev)).float(); return e / e.norm(dim=-1, keepdim=True)
    def heats(feat, E):
        Hh, Ww, D = feat.shape; ft = torch.from_numpy(np.asarray(feat).reshape(-1, D)).to(dev); void = ft.norm(dim=-1) < 0.5; out = []
        for j in range(0, ft.shape[0], 8192):
            sl = ft[j:j + 8192]; h = L.exp_map0(sl, curv=curv); path, mask = L.get_interpolated_hyperbolic_features(h, steps=4, curv=curv, max_dist=11.1, return_mask=True)
            d = hyper.decode_features(path, project=True); d = d / d.norm(dim=-1, keepdim=True); sm = (d @ E.T).view(sl.shape[0], 4, -1).masked_fill(mask.to(dev).bool().unsqueeze(-1), -1.0)
            d0 = hyper.decode_features(h, project=True); d0 = d0 / d0.norm(dim=-1, keepdim=True); out.append(torch.maximum(sm.max(1).values, d0 @ E.T))
        hm = torch.cat(out); hm[void] = -1.0; return hm.view(Hh, Ww, -1).cpu().numpy()
    for b in a.models:
        O = S / 'experimental/h3dgs_sidecar' / f'block_{b}'; run = sorted((O / 'splat_runs_FEATFIX').glob(f'stage2_censusinit_{a.seed_tag}/high/*'))[-1]
        z = np.load(O / 'splat_runs_FEATFIX/interaction_W_glref_bg.npz'); labels = [int(u) for u in z['labels'] if int(u) != UNLAB and int(u) < FRUIT_ID_BASE]; twords = [get_word_for_id(u, 'mask') for u in labels]
        rwords = pj['rows'].get(b, []); E = word_vec(twords + rwords); nT = len(twords); vt = verdict_thr(b)
        tthr = np.array([vt.get(w, a.tree_thr) for w in twords], np.float32); rthr = np.array([vt.get(w, a.row_thr) for w in rwords], np.float32)
        dp = json.load(open(run / 'dataparser_transforms.json')); T = np.asarray(dp['transform'], np.float64); s = float(dp['scale'])
        _, pipe, _, step = eval_setup(run / 'config.yml', test_mode='inference'); model = pipe.model; model.eval(); model.step = step
        tj = json.load(open(O / 'transforms.json')); Cb = np.array([np.asarray(f['transform_matrix'])[:3, 3] for f in tj['frames']]) @ R_W.T; cen = Cb.mean(0)
        ds = pipe.datamanager.train_dataset; dsn = {Path(f).name: i for i, f in enumerate(ds.image_filenames)}
        def cam_for(c2w_h):
            c2w_cv = np.eye(4); c2w_cv[:3, :3] = R_W.T @ c2w_h[:3, :3]; c2w_cv[:3, 3] = R_W.T @ c2w_h[:3, 3]; g = c2w_cv @ GL2CV
            M = np.zeros((3, 4)); M[:, :3] = T[:, :3] @ g[:3, :3]; M[:, 3] = s * (T[:, :3] @ g[:3, 3] + T[:, 3])
            return Cameras(camera_to_worlds=torch.tensor(M, dtype=torch.float32)[None], fx=K['fl_x'] * sc, fy=K['fl_y'] * sc, cx=K['cx'] * sc, cy=K['cy'] * sc, width=W, height=H, camera_type=CameraType.PERSPECTIVE).to(model.device)
        chk = [f for f in FR if f.get('src') in dsn]
        if chk:
            f = chk[0]; mine = cam_for(np.asarray(f['c2w_h'])).camera_to_worlds[0].cpu().numpy(); theirs = ds.cameras[dsn[f['src']]].camera_to_worlds.cpu().numpy()
            print(f"[maps] block {b}: camera-frame check on {f['src']}: max |dR| {np.abs(mine[:, :3] - theirs[:, :3]).max():.2e}, |dt| {np.abs(mine[:, 3] - theirs[:, 3]).max():.2e} (ns units)", flush=True)
        (OUT / 'maps' / b).mkdir(exist_ok=True); n = 0; t0 = time.time()
        for f in FR:
            c2w_h = np.asarray(f['c2w_h']);
            if np.linalg.norm(c2w_h[:3, 3] - cen) > a.max_dist: continue
            rgb, alpha, feat = render_frame(model, cam_for(c2w_h), model.config.lang_field_dim); hm = heats(feat, E)
            tm = hm[:, :, :nT] - tthr[None, None]; tid = tm.argmax(-1); tmg = tm.max(-1)
            if rwords:                                                             # both row picks are stored so --row-pick is an A/B on one render pass
                hr_ = hm[:, :, nT:]; rm = hr_ - rthr[None, None]
                rid = rm.argmax(-1); rmg = rm.max(-1)                              # margin pick (old): thresholds enter the comparison between row words
                rid_raw = hr_.argmax(-1)                                           # raw pick: the field's own ranking of the row words, thresholds out of it
                rmg_raw = np.take_along_axis(rm, rid_raw[..., None], -1)[..., 0]    # the winner's own margin — still the gate and the cross-model arbiter
            else: rid = np.zeros_like(tid); rmg = np.full(tid.shape, -9, np.float32); rid_raw, rmg_raw = rid, rmg
            np.savez_compressed(OUT / 'maps' / b / (f['name'] + '.npz'), tid=np.array(labels, np.int32)[tid].astype(np.int32), tmg=tmg.astype(np.float16), rid=rid.astype(np.int8), rmg=rmg.astype(np.float16),
                                rid_raw=rid_raw.astype(np.int8), rmg_raw=rmg_raw.astype(np.float16), rows=np.array(rwords)); n += 1
        print(f'[maps] block {b}: {n} frames in {time.time()-t0:.0f}s ({len(labels)} trees, rows {rwords})', flush=True); del model, pipe; torch.cuda.empty_cache()
# ---------------- pass 2: composite
trees = {int(k): np.array(v) for k, v in pj['trees'].items()}; track = np.array(pj['track']); allrow = sorted({w for v in pj['rows'].values() for w in v}); palette = [(0, 220, 255), (255, 200, 0), (200, 0, 255), (0, 255, 120), (0, 120, 255), (255, 120, 0)]
HROW = {t: r['id'] for r in json.load(open(S / 'prod/bateleur/scene_graph/marker_hierarchy.json'))['rows'] for t in r['object_ids']}   # tree id -> hierarchy row id
for i, w in enumerate(allrow): ROWCOL[w] = palette[i % len(palette)]
seen = {}; tree_row = {}; focus = None; last_mode = None; font = cv2.FONT_HERSHEY_SIMPLEX; t_start = {}
mx = np.concatenate([track, np.array([v[:2] for v in trees.values()])]); lo, hi = mx.min(0) - 3, mx.max(0) + 3; MS = 220; pad = 12
def to_map(p): return int(pad + (p[0] - lo[0]) / (hi[0] - lo[0]) * (MS - 2 * pad)), int(MS - pad - (p[1] - lo[1]) / (hi[1] - lo[1]) * (MS - 2 * pad))
for f in FR:
    name = f['name']; img = cv2.imread(str(Path(a.backdrop_dir) / name)); img = cv2.resize(img, (W, H)) if img.shape[:2] != (H, W) else img; over = img.copy()
    best_t = np.full((H, W), -1, np.int32); best_tm = np.full((H, W), -9.0, np.float32); best_r = np.full((H, W), -1, np.int32); best_rm = np.full((H, W), -9.0, np.float32); rown = {}
    for b in a.models:
        p = OUT / 'maps' / b / (name + '.npz')
        if not p.exists(): continue
        z = np.load(p); tmg = z['tmg'].astype(np.float32); m = tmg > best_tm; best_t[m] = z['tid'][m]; best_tm[m] = tmg[m]
        rws = list(z['rows']); kr = 'rmg_raw' if (a.row_pick == 'raw' and 'rmg_raw' in z.files) else 'rmg'   # old maps carry only the margin pick
        rmg = z[kr].astype(np.float32); m = rmg > best_rm; best_r[m] = np.array([allrow.index(w) for w in rws], np.int32)[z['rid_raw' if kr == 'rmg_raw' else 'rid']][m] if rws else -1; best_rm[m] = rmg[m]
    tree_mask = best_tm >= 0; row_mask = best_rm >= 0; mode = f['mode']
    if a.row_from_tree:   # rows follow the identified tree's hierarchy row (row id r <-> word allrow[r]: containment_eval prints 'ROW 0 "oak"', 'ROW 1 "pine"')
        hr = np.vectorize(lambda t: HROW.get(int(t), -1))(best_t); ok = tree_mask & (hr >= 0) & (hr < len(allrow)); best_r = np.where(ok, hr, -1); row_mask = ok
        for u in np.unique(best_t[tree_mask]):
            if HROW.get(int(u), -1) >= 0: tree_row[int(u)] = HROW[int(u)]
    sm = SKY / f.get('src', '')                 # sky out FIRST: it is masked to black at the end anyway, but while it is still
    sky = cv2.resize(cv2.imread(str(sm), 0), (W, H), interpolation=cv2.INTER_NEAREST) > 127 if sm.exists() else np.zeros((H, W), bool)
    tree_mask &= ~sky; row_mask &= ~sky         # in the mask it bridges the two hedges across the top into ONE component,
    def despeckle(lab_img, valid, ids, min_area):   # a row (or a tree) is ONE object, not a scatter. The wrong-row pixels are
        keep = np.zeros_like(valid)                 # canopy seen through: on kf_001508, 198 blobs of median size 2 px
        for u in ids:
            m = (lab_img == u) & valid
            if not m.any(): continue
            n, lb, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
            big = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area]
            if big: keep |= np.isin(lb, big)
        return keep
    if a.min_area > 0: row_mask = despeckle(best_r, row_mask, range(len(allrow)), a.min_area)
    if a.tree_min_area > 0: tree_mask = despeckle(best_t, tree_mask, [int(u) for u in np.unique(best_t[tree_mask])], a.tree_min_area)
    if mode != last_mode: t_start[mode] = f['t']; last_mode = mode
    counts = {int(u): int(((best_t == u) & tree_mask).sum()) for u in np.unique(best_t[tree_mask])}
    for u, c in counts.items():
        if c >= 300 and u not in seen: seen[u] = f['i']
        if c >= 300:
            rr = best_r[(best_t == u) & tree_mask & row_mask]
            if rr.size: tree_row[u] = int(np.bincount(rr[rr >= 0]).argmax()) if (rr >= 0).any() else tree_row.get(u)
    if mode == 'tree' and focus is None and counts: focus = max(counts, key=counts.get)
    tint = np.zeros_like(over, np.float32); tw = np.zeros((H, W), np.float32)
    if mode in ('all', 'tree'):
        for u in counts:
            if mode == 'tree' and u != focus: continue
            m = (best_t == u) & tree_mask; tint[m] = colour(u); tw[m] = 0.5
    elif mode == 'rows':
        for ri, w in enumerate(allrow):
            m = (best_r == ri) & row_mask; tint[m] = ROWCOL[w]; tw[m] = 0.45
    elif mode == 'orchard':
        m = tree_mask | row_mask; tint[m] = ORCHARD; tw[m] = 0.45
    over = (over * (1 - tw[..., None]) + tint * tw[..., None]).astype(np.uint8)
    if mode in ('row', 'tree', 'all'):
        for ri, w in enumerate(allrow):
            m = ((best_r == ri) & row_mask).astype(np.uint8); cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(over, [c for c in cs if cv2.contourArea(c) > 200], -1, ROWCOL[w], 2)
    sm = SKY / f.get('src', '')
    if sm.exists(): over[cv2.resize(cv2.imread(str(sm), 0), (W, H), interpolation=cv2.INTER_NEAREST) > 127] = 0
    # question, typed over the first second of its mode
    if f['question']:
        q = f['question']; k = int(min(1.0, (f['t'] - t_start[mode]) / 1.0) * len(q)); txt = q[:k] + ('|' if k < len(q) else '')
        cv2.rectangle(over, (10, 10), (24 + 11 * len(q), 44), (0, 0, 0), -1); cv2.putText(over, txt, (18, 34), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    # map inset: track, camera, trees (grey until seen, then their colour / row colour / orchard colour)
    mp = np.full((MS, MS, 3), 20, np.uint8); pts = np.array([to_map(p) for p in track], np.int32); cv2.polylines(mp, [pts], False, (90, 90, 90), 1)
    for u, c in trees.items():
        col = (70, 70, 70)
        if u in seen and seen[u] <= f['i']:
            rw = tree_row.get(u); in_rows = rw is not None and 0 <= rw < len(allrow)
            col = colour(u) if mode in ('plain', 'row', 'tree', 'all') else (ROWCOL[allrow[rw]] if mode == 'rows' and in_rows else ORCHARD if mode == 'orchard' else (70, 70, 70))
            if mode == 'tree' and u != focus: col = (110, 110, 110)
        cv2.circle(mp, to_map(c), 5, col, -1)
    cx_, cy_ = to_map(np.asarray(f['c2w_h'])[:3, 3]); fwd = np.asarray(f['c2w_h'])[:3, 2]; cv2.circle(mp, (cx_, cy_), 4, (255, 255, 255), -1)
    cv2.line(mp, (cx_, cy_), (int(cx_ + 14 * fwd[0] / (np.hypot(fwd[0], fwd[1]) + 1e-9)), int(cy_ - 14 * fwd[1] / (np.hypot(fwd[0], fwd[1]) + 1e-9))), (255, 255, 255), 2)
    over[H - MS - 8:H - 8, W - MS - 8:W - 8] = cv2.addWeighted(over[H - MS - 8:H - 8, W - MS - 8:W - 8], 0.25, mp, 0.75, 0); cv2.rectangle(over, (W - MS - 9, H - MS - 9), (W - 8, H - 8), (160, 160, 160), 1)
    cv2.imwrite(str(OUT / 'frames' / f'{f["i"]:05d}.png'), over)
print(f'[demo] {len(FR)} frames composited; trees seen {len(seen)}; focus tree {focus}', flush=True)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', str(OUT / 'frames' / '%05d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', str(OUT / 'demo.mp4')], check=True)
print(f'[demo] wrote {OUT / "demo.mp4"}', flush=True)
