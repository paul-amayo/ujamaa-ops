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
ap.add_argument('--reuse-maps', action='store_true', help='skip a model whose map directory already holds npz files (render only what is missing, e.g. a newly added fruit side-car)')
ap.add_argument('--max-dist', type=float, default=70.0); ap.add_argument('--tree-thr', type=float, default=0.5); ap.add_argument('--row-thr', type=float, default=0.8); ap.add_argument('--skip-maps', action='store_true')
ap.add_argument('--row-pick', choices=('raw', 'margin'), default='raw', help="which row word wins a pixel. raw (default): the word the field scores HIGHEST — the two row words compared on one common basis. margin (the old rule, and the row-crossing bug): argmax of score MINUS that word's own per-block verdict threshold, which on block 023 (oak thr 0.70 / pine thr 0.85) hands a pine pixel scoring oak 0.762 / pine 0.871 to oak, because +0.062 > +0.021. Either way the pixel is only claimed where the winning word clears its own threshold.")
ap.add_argument('--fruit-models', nargs='*', default=[], help="blocks whose FRUIT side-car (the densified one) should be rendered too, for the 'fruit' mode")
ap.add_argument('--fruit-seed-tag', default='fruitdensify_r2'); ap.add_argument('--fruit-verdict-tag', default='fruitdensify')
ap.add_argument('--script', help='demo_script.json from sidecar_demo_script.py: segments with question, answer, focus tree and hold')
ap.add_argument('--fps', type=int, default=5, help='output framerate (the path was laid out at 8; 5 reads slower)')
ap.add_argument('--outline-close', type=int, default=9, help='morphological close (px) on a row mask before contouring: the canopy silhouette is notched at every leaf gap and the raw contour traces all of it')
ap.add_argument('--outline-simplify', type=float, default=2.5, help='approxPolyDP epsilon on the row contour')
ap.add_argument('--sky-fill-max', type=int, default=20000, help='fill holes in the sky mask up to this area (the sun is ~4500 px); anything enclosed by sky is sky')
ap.add_argument('--fruit-max-dist', type=float, default=15.0, help="a fruit side-car is only trusted near its OWN block: its fruit word decodes confidently on canopy it never saw (block 021's 'flashbacks' landed on the opposite hedge two blocks away, where supervision puts tree 98 on the left)")
ap.add_argument('--fruit-min-area', type=int, default=30, help='fruit are small: a much lower component floor than rows or trees')
ap.add_argument('--min-area', type=int, default=800, help="drop connected components smaller than this from each ROW region before drawing (0 = off). Compositing only, no re-render. On kf_001508: none -> oak 35 / pine 411 pieces and 1902 wrong-row px; 200 -> 2/3 pieces, 417 px; 800 -> 1/1 piece, 0 wrong-row px, for 4.5% of the drawn area.")
ap.add_argument('--tree-min-area', type=int, default=200, help='same for each TREE region, kept lower so distant trees still register')
ap.add_argument('--row-from-tree', action='store_true', help="SUPERSEDED by --row-pick raw, and not needed for a row query: route the row answer through the identified tree's hierarchy row (marker_hierarchy.json) instead of the row decode. Kept only to reproduce demo v3.")
ap.add_argument('--sidecar-root', default='experimental/h3dgs_sidecar', help='chunk side-cars: experimental/h3dgs_sidecar_chunks, with --models as dir names (chunk_1_0_expo)')
ap.add_argument('--proj', default='experimental/h3dgs', help='H3DGS project (export_meta for the frame rotation): h3dgs_expo for the 05 demo chunks')
a = ap.parse_args(); S = Path('/home/paperspace/data/citrus_all') / a.survey; P = S / a.proj; OUT = Path(a.out); (OUT / 'frames').mkdir(parents=True, exist_ok=True); (OUT / 'maps').mkdir(exist_ok=True)
def model_dir(b):
    d = S / a.sidecar_root / b
    return d if d.exists() else S / a.sidecar_root / f'block_{b}'
pj = json.load(open(a.path)); FR = pj['frames']; K = pj['intrinsics']; sc = pj['scale']; fps = pj['fps']; W, H = int(round(K['w'] * sc)), int(round(K['h'] * sc))
meta = json.load(open(P / 'export_meta.json')); R_W = np.asarray(meta.get('world_rotation_to_zup') or meta['world_rotation_lio_to_h3dgs'], np.float64)[:3, :3]; GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])
SKY = S / 'prod/tassili/sky_masks'; VL = f'/home/paperspace/logs/sidecar_{a.survey}_verdicts.log'
FVL = f'/home/paperspace/logs/sidecar_{a.survey}_fruit_verdicts.log'    # the fruit chain writes its own verdict log
TREE_OF_FRUIT = {f['id']: f['tree_id'] for f in json.load(open(S / 'prod/bateleur/scene_graph/marker_hierarchy.json')).get('fruits', [])}
def verdict_thr(block, tag=None):
    out = {}; tag = tag or a.verdict_tag; VL_ = FVL if tag == a.fruit_verdict_tag else VL
    if Path(VL_).exists():
        for line in open(VL_):
            if line.startswith(f'[{block} sidecar {tag} '):
                m = re.search(r'"([a-z]+)": thr ([0-9.]+)', line)
                if m: out[m.group(1)] = float(m.group(2))
    return out
def colour(tid):
    h = (int(tid) * 37) % 180; return tuple(int(v) for v in cv2.cvtColor(np.uint8([[[h, 210, 235]]]), cv2.COLOR_HSV2BGR)[0, 0])
ROWCOL = {}; ORCHARD = (60, 220, 60); FRUITCOL = (40, 120, 255)   # BGR: orange — the one colour no tree hue or row colour uses
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
        if a.reuse_maps and list((OUT / 'maps' / b).glob('*.npz')):
            print(f'[maps] block {b}: reusing {len(list((OUT / "maps" / b).glob("*.npz")))} existing maps', flush=True); continue
        O = model_dir(b); run = sorted((O / 'splat_runs_FEATFIX').glob(f'stage2_censusinit_{a.seed_tag}/high/*'))[-1]
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
    # fruit lives in its own side-cars: the densified geometry (leaves + fruit children), scored on its own verdict lines
    for b in a.fruit_models:
        if a.reuse_maps and list((OUT / 'maps_fruit' / b).glob('*.npz')):
            print(f'[maps-fruit] block {b}: reusing existing maps', flush=True); continue
        O = model_dir(b); runs = sorted((O / 'splat_runs_FEATFIX').glob(f'stage2_censusinit_{a.fruit_seed_tag}/high/*'))
        if not runs: print(f'[maps-fruit] block {b}: no {a.fruit_seed_tag} run, skipping', flush=True); continue
        run = runs[-1]; z = np.load(O / 'splat_runs_FEATFIX/interaction_W_fruitdensify_bg.npz')
        flab = [int(u) for u in z['labels'] if FRUIT_ID_BASE <= int(u) < UNLAB]
        fwords = [get_word_for_id(u - FRUIT_ID_BASE, 'fruit') for u in flab]   # containment_eval's own naming: the FRUIT vocabulary, id minus the base
        if not fwords: print(f'[maps-fruit] block {b}: no fruit labels', flush=True); continue
        vt = verdict_thr(b, a.fruit_verdict_tag); fthr = np.array([vt.get(w, 0.8) for w in fwords], np.float32); E = word_vec(fwords)
        dp = json.load(open(run / 'dataparser_transforms.json')); T = np.asarray(dp['transform'], np.float64); s = float(dp['scale'])
        _, pipe, _, step = eval_setup(run / 'config.yml', test_mode='inference'); model = pipe.model; model.eval(); model.step = step
        tj = json.load(open(O / 'transforms.json')); Cb = np.array([np.asarray(f['transform_matrix'])[:3, 3] for f in tj['frames']]) @ R_W.T; cen = Cb.mean(0)
        (OUT / 'maps_fruit' / b).mkdir(parents=True, exist_ok=True); n = 0; t0 = time.time()
        for f in FR:
            c2w_h = np.asarray(f['c2w_h'])
            if np.linalg.norm(c2w_h[:3, 3] - cen) > a.max_dist: continue
            c2w_cv = np.eye(4); c2w_cv[:3, :3] = R_W.T @ c2w_h[:3, :3]; c2w_cv[:3, 3] = R_W.T @ c2w_h[:3, 3]; g = c2w_cv @ GL2CV
            M = np.zeros((3, 4)); M[:, :3] = T[:, :3] @ g[:3, :3]; M[:, 3] = s * (T[:, :3] @ g[:3, 3] + T[:, 3])
            cam = Cameras(camera_to_worlds=torch.tensor(M, dtype=torch.float32)[None], fx=K['fl_x'] * sc, fy=K['fl_y'] * sc, cx=K['cx'] * sc, cy=K['cy'] * sc,
                          width=W, height=H, camera_type=CameraType.PERSPECTIVE).to(model.device)
            _, _, feat = render_frame(model, cam, model.config.lang_field_dim); hm = heats(feat, E)
            fm = hm - fthr[None, None]; fid = fm.argmax(-1); fmg = fm.max(-1)
            np.savez_compressed(OUT / 'maps_fruit' / b / (f['name'] + '.npz'), fid=fid.astype(np.int8), fmg=fmg.astype(np.float16),
                                fruits=np.array(fwords), ftree=np.array([TREE_OF_FRUIT.get(u - FRUIT_ID_BASE, -1) for u in flab], np.int32)); n += 1
        print(f'[maps-fruit] block {b}: {n} frames in {time.time()-t0:.0f}s (fruits {fwords}, thr {fthr.tolist()})', flush=True); del model, pipe; torch.cuda.empty_cache()
# ---------------- pass 2: composite
# Adinkra palette (ujamaa/project), BGR for cv2: paper, ink, amber, cyan, terracotta, olive, blue, light green, tan, rust
PAPER = (241, 248, 251); INK = (18, 22, 26); AMBER = (79, 178, 254); CYAN = (198, 172, 75); TERRA = (58, 122, 217)
OLIVE = (42, 122, 84); BLUE = (125, 73, 31); LGREEN = (127, 215, 184); TAN = (112, 144, 168); RUST = (48, 100, 181)
ORCHARD = OLIVE; FRUITCOL = TERRA
TREEPAL = [BLUE, LGREEN, TAN, RUST, CYAN, AMBER, OLIVE, (150, 120, 90)]
def colour(tid):
    """tree colours drawn from the palette; brightness varied so neighbours stay distinguishable without leaving it"""
    b, g, r = TREEPAL[int(tid) % len(TREEPAL)]; k = [1.0, 0.72, 1.28][(int(tid) // len(TREEPAL)) % 3]
    return tuple(int(min(255, max(0, v * k))) for v in (b, g, r))
trees = {int(k): np.array(v) for k, v in pj['trees'].items()}; track = np.array(pj['track'])
allrow = sorted({w for v in pj['rows'].values() for w in v}); ROWPAL = [AMBER, CYAN, TERRA, LGREEN, BLUE, RUST]
for i, w in enumerate(allrow): ROWCOL[w] = ROWPAL[i % len(ROWPAL)]
HROW = {t: r['id'] for r in json.load(open(S / 'prod/bateleur/scene_graph/marker_hierarchy.json'))['rows'] for t in r['object_ids']}
# the question script: every frame gets its segment's mode, question, answer and focus tree
FCEN = {}
for _b in a.fruit_models:
    _tj = json.load(open(model_dir(_b) / 'transforms.json'))
    FCEN[_b] = (np.array([np.asarray(x['transform_matrix'], float)[:3, 3] for x in _tj['frames']]) @ R_W.T).mean(0)
SCR = json.load(open(a.script)) if a.script else None
segof = [None] * len(FR); LABEL = {}
if SCR:
    LABEL = {int(k): v for k, v in SCR.get('label', {}).items()}
    for si, sg in enumerate(SCR['segments']):
        for i in range(sg['lo'], min(sg['hi'] + 1, len(FR))): segof[i] = si
    print(f"[demo] script: {len(SCR['segments'])} segments, holds of {SCR['segments'][0].get('hold', 0)} frames", flush=True)
seen = {}; tree_row = {}; font = cv2.FONT_HERSHEY_SIMPLEX
mx = np.concatenate([track, np.array([v[:2] for v in trees.values()])]); lo, hi = mx.min(0) - 3, mx.max(0) + 3; MS = 220; pad = 12
def to_map(p): return int(pad + (p[0] - lo[0]) / (hi[0] - lo[0]) * (MS - 2 * pad)), int(MS - pad - (p[1] - lo[1]) / (hi[1] - lo[1]) * (MS - 2 * pad))
def banner(img, text, y, col, scale=0.62, pad_=9):
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    cv2.rectangle(img, (10, y - th - pad_), (14 + tw + pad_, y + pad_), INK, -1)
    cv2.putText(img, text, (14, y), font, scale, col, 2, cv2.LINE_AA)
outn = 0
for fi, f in enumerate(FR):
    name = f['name']; img = cv2.imread(str(Path(a.backdrop_dir) / name)); img = cv2.resize(img, (W, H)) if img.shape[:2] != (H, W) else img; over = img.copy()
    si = segof[fi]; sg = SCR['segments'][si] if (SCR and si is not None) else None
    mode = sg['mode'] if sg else f['mode']; question = sg['question'] if sg else f.get('question', '')
    answer = sg.get('answer', '') if sg else ''; focus = sg.get('focus') if sg else None
    seg_lo = sg['lo'] if sg else 0
    best_t = np.full((H, W), -1, np.int32); best_tm = np.full((H, W), -9.0, np.float32); best_r = np.full((H, W), -1, np.int32); best_rm = np.full((H, W), -9.0, np.float32)
    for b in a.models:
        p = OUT / 'maps' / b / (name + '.npz')
        if not p.exists(): continue
        z = np.load(p); tmg = z['tmg'].astype(np.float32); m = tmg > best_tm; best_t[m] = z['tid'][m]; best_tm[m] = tmg[m]
        rws = list(z['rows']); kr = 'rmg_raw' if (a.row_pick == 'raw' and 'rmg_raw' in z.files) else 'rmg'
        rmg = z[kr].astype(np.float32); m = rmg > best_rm
        best_r[m] = np.array([allrow.index(w) for w in rws], np.int32)[z['rid_raw' if kr == 'rmg_raw' else 'rid']][m] if rws else -1
        best_rm[m] = rmg[m]
    best_f = np.full((H, W), -1, np.int32); best_fm = np.full((H, W), -9.0, np.float32)
    cam_h = np.asarray(f['c2w_h'])[:3, 3]
    for b in a.fruit_models:
        p = OUT / 'maps_fruit' / b / (name + '.npz')
        if not p.exists() or np.linalg.norm(cam_h - FCEN[b]) > a.fruit_max_dist: continue
        z = np.load(p); fmg = z['fmg'].astype(np.float32); m = fmg > best_fm
        best_f[m] = z['ftree'][z['fid']][m]; best_fm[m] = fmg[m]
    tree_mask = best_tm >= 0; row_mask = best_rm >= 0
    sm = SKY / f.get('src', '')
    sky = cv2.resize(cv2.imread(str(sm), 0), (W, H), interpolation=cv2.INTER_NEAREST) > 127 if sm.exists() else np.zeros((H, W), bool)
    if sky.any():   # SAM3 does not class the blown-out SUN as sky, so the mask has a sun-shaped hole and the sun is left
        ns = (~sky).astype(np.uint8)          # floating in the blacked-out sky. Anything fully enclosed by sky IS sky.
        nlb, lbl, stt, _ = cv2.connectedComponentsWithStats(ns, 8)
        # the sun reaches the TOP edge of the frame, so "fully enclosed" never catches it. The test that does: an island
        # that touches no side or bottom edge and stays above the horizon is sky, whatever SAM3 called it.
        edge = set(lbl[-1, :].tolist()) | set(lbl[:, 0].tolist()) | set(lbl[:, -1].tolist())
        fill = [k for k in range(1, nlb) if k not in edge and stt[k, cv2.CC_STAT_AREA] <= a.sky_fill_max
                and stt[k, cv2.CC_STAT_TOP] + stt[k, cv2.CC_STAT_HEIGHT] < H * 0.55]
        if fill: sky |= np.isin(lbl, fill)
    tree_mask &= ~sky; row_mask &= ~sky
    def despeckle(lab_img, valid, ids, min_area):
        keep = np.zeros_like(valid)
        for u in ids:
            m = (lab_img == u) & valid
            if not m.any(): continue
            n, lb, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
            big = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area]
            if big: keep |= np.isin(lb, big)
        return keep
    if a.min_area > 0: row_mask = despeckle(best_r, row_mask, range(len(allrow)), a.min_area)
    if a.tree_min_area > 0: tree_mask = despeckle(best_t, tree_mask, [int(u) for u in np.unique(best_t[tree_mask])], a.tree_min_area)
    fruit_mask = (best_fm >= 0) & ~sky
    if a.fruit_min_area > 0 and fruit_mask.any():
        fruit_mask = despeckle(best_f, fruit_mask, [int(u) for u in np.unique(best_f[fruit_mask])], a.fruit_min_area)
    counts = {int(u): int(((best_t == u) & tree_mask).sum()) for u in np.unique(best_t[tree_mask])}
    for u, c in counts.items():
        if c >= 300 and u not in seen: seen[u] = fi
        if c >= 300:
            rr = best_r[(best_t == u) & tree_mask & row_mask]
            if rr.size and (rr >= 0).any(): tree_row[u] = int(np.bincount(rr[rr >= 0]).argmax())
    tint = np.zeros_like(over, np.float32); tw = np.zeros((H, W), np.float32)
    if mode == 'fruit':
        # answering "which tree has the most fruit" means showing THAT tree's fruit: ringing every tree's fruit puts the
        # rings on one side of the lane and the highlighted tree on the other, which reads as a contradiction
        if focus is not None: fruit_mask &= (best_f == focus)
        for u in counts:
            m = (best_t == u) & tree_mask; tint[m] = colour(u); tw[m] = 0.15
        if focus is not None:
            m = (best_t == focus) & tree_mask; tint[m] = AMBER; tw[m] = 0.42
        tint[fruit_mask] = FRUITCOL; tw[fruit_mask] = 0.85
    elif mode == 'tree':
        for u in counts:
            m = (best_t == u) & tree_mask
            if focus is not None and u != focus: tint[m] = (78, 74, 68); tw[m] = 0.30
            else: tint[m] = AMBER if focus is not None else colour(u); tw[m] = 0.50   # the answer is always the accent
    elif mode == 'all':
        for u in counts:
            m = (best_t == u) & tree_mask; tint[m] = colour(u); tw[m] = 0.5
    elif mode == 'rows':
        for ri, w in enumerate(allrow):
            m = (best_r == ri) & row_mask; tint[m] = ROWCOL[w]; tw[m] = 0.45
    elif mode == 'orchard':
        m = tree_mask | row_mask; tint[m] = ORCHARD; tw[m] = 0.45
    over = (over * (1 - tw[..., None]) + tint * tw[..., None]).astype(np.uint8)
    if mode in ('row', 'tree', 'all'):
        for ri, w in enumerate(allrow):
            m = ((best_r == ri) & row_mask).astype(np.uint8)
            if a.outline_close > 1: m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.outline_close, a.outline_close)))
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cs = [cv2.approxPolyDP(c, a.outline_simplify, True) for c in cs if cv2.contourArea(c) > 200]
            cv2.drawContours(over, cs, -1, ROWCOL[w], 2, cv2.LINE_AA)
    nfr = 0
    if mode == 'fruit' and fruit_mask.any():
        cs, _ = cv2.findContours(fruit_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            (cx_, cy_), rr_ = cv2.minEnclosingCircle(c); cv2.circle(over, (int(cx_), int(cy_)), int(min(max(rr_, 3) + 3, 12)), FRUITCOL, 1, cv2.LINE_AA)
        nfr = len(cs)
    if sm.exists(): over[sky] = 0
    # map inset
    mp = np.full((MS, MS, 3), INK, np.uint8); pts = np.array([to_map(p) for p in track], np.int32); cv2.polylines(mp, [pts], False, (90, 84, 76), 1)
    for u, c in trees.items():
        col = (70, 66, 60)
        if u in seen and seen[u] <= fi:
            rw = tree_row.get(u); inr = rw is not None and 0 <= rw < len(allrow)
            col = colour(u) if mode in ('plain', 'row', 'tree', 'all', 'fruit') else (ROWCOL[allrow[rw]] if mode == 'rows' and inr else ORCHARD if mode == 'orchard' else (70, 66, 60))
            if mode in ('tree', 'fruit') and focus is not None and u != focus: col = (100, 96, 90)
        cv2.circle(mp, to_map(c), 5, col, -1)
        if focus is not None and u == focus and mode in ('tree', 'fruit'): cv2.circle(mp, to_map(c), 8, AMBER, 2, cv2.LINE_AA)
    cx_, cy_ = to_map(np.asarray(f['c2w_h'])[:3, 3]); fwd = np.asarray(f['c2w_h'])[:3, 2]
    cv2.circle(mp, (cx_, cy_), 4, PAPER, -1)
    cv2.line(mp, (cx_, cy_), (int(cx_ + 14 * fwd[0] / (np.hypot(fwd[0], fwd[1]) + 1e-9)), int(cy_ - 14 * fwd[1] / (np.hypot(fwd[0], fwd[1]) + 1e-9))), PAPER, 2)
    over[H - MS - 8:H - 8, W - MS - 8:W - 8] = cv2.addWeighted(over[H - MS - 8:H - 8, W - MS - 8:W - 8], 0.06, mp, 0.94, 0)
    cv2.rectangle(over, (W - MS - 9, H - MS - 9), (W - 8, H - 8), (120, 114, 104), 1)
    # question types over the segment's first second (8 frames of demo time), then holds
    def draw_text(dst, reveal_q=1.0, reveal_a=0.0):
        if question:
            k = int(min(1.0, reveal_q) * len(question)); txt = question[:k] + ('|' if k < len(question) else '')
            banner(dst, txt, 36, PAPER)
        if answer and reveal_a > 0:
            k = int(min(1.0, reveal_a) * len(answer)); banner(dst, answer[:k], 74, AMBER, 0.70)
        if mode == 'fruit' and nfr: cv2.putText(dst, f'{nfr} fruit clusters', (12, H - 14), font, 0.55, FRUITCOL, 1, cv2.LINE_AA)
    rq = min(1.0, (fi - seg_lo + 1) / 8.0) if (sg is None or sg.get('retype', True)) else 1.0
    base = over.copy(); draw_text(over, rq, 0.0)
    cv2.imwrite(str(OUT / 'frames' / f'{outn:05d}.png'), over); outn += 1
    # hold: freeze this frame and type the answer in
    if sg and answer and fi == sg.get('hold_at'):
        for k in range(sg.get('hold', 10)):
            h = base.copy(); draw_text(h, 1.0, min(1.0, (k + 1) / 4.0))
            cv2.imwrite(str(OUT / 'frames' / f'{outn:05d}.png'), h); outn += 1
print(f'[demo] {outn} frames written ({len(FR)} path frames + holds); trees seen {len(seen)}', flush=True)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(a.fps), '-i', str(OUT / 'frames' / '%05d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-r', '30', str(OUT / 'demo.mp4')], check=True)
print(f'[demo] wrote {OUT / "demo.mp4"} at {a.fps} fps ({outn / a.fps:.0f} s)', flush=True)
