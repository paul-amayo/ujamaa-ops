"""Walkthrough reel down one row of a citrus survey through the H3DGS containment side-cars (Paul, 2026-09-27: "like the
walkthrough demo, moving frame to frame down a single row of 05"). For every keyframe of the given blocks, in row
order: the side-car's own render, each pixel tinted by the TREE whose containment ladder fires highest (the block's
census words; identity colour = fixed per tree id, so a tree keeps its colour across frames and blocks), the ROW words
outlined, tree words written at the mask centroids. Same scorer as containment_eval (hyperbolic ladder, 0.5 identity
gate); thresholds are FIXED here (no ground truth to sweep against) and printed on the frame.
  HIGH_EMBEDDER_CKPT=<emb> pixi run python sidecar_row_reel.py --survey 05_13D_Jackal --blocks 018 019 020 021 022 023
      --row-words oak pine --out <dir> [--seed-tag glref_bg_f1.0_r2] [--tree-thr 0.5] [--row-thr 0.7] [--fps 8]"""
import argparse, glob, os, subprocess, sys, time
from pathlib import Path
import numpy as np, torch, cv2
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH')
from high_splat_hierarchy_accuracy import render_frame, build_hyper_embedder
from word_utils import get_word_for_id
from nerfstudio.utils.eval_utils import eval_setup
import lorentz as L, open_clip
UNLAB = 65535; FRUIT_ID_BASE = 10000; dev = 'cuda'; torch.set_grad_enabled(False)
ap = argparse.ArgumentParser(); ap.add_argument('--survey', required=True); ap.add_argument('--blocks', nargs='+', required=True); ap.add_argument('--row-words', nargs='+', default=[])
ap.add_argument('--out', required=True); ap.add_argument('--seed-tag', default='glref_bg_f1.0_r2'); ap.add_argument('--tree-thr', type=float, default=0.5); ap.add_argument('--row-thr', type=float, default=0.7)
ap.add_argument('--fps', type=int, default=8); ap.add_argument('--scale', type=float, default=0.5)
ap.add_argument('--verdict-log', default='', help="per-block thresholds from this verdict log (default sidecar_<survey>_verdicts.log; words without a line use --tree-thr / --row-thr)")
ap.add_argument('--verdict-tag', default='bg_f1.0_r2')
ap.add_argument('--sky-masks', default='', help="sky-mask dir (255 = sky; default <survey>/prod/tassili/sky_masks): sky pixels are blacked out like the block models' sky loss does (Paul, 2026-09-27)")
ap.add_argument('--backdrop-dir', default='', help="PNGs named by keyframe from sidecar_row_backdrop.py (the FULL H3DGS model at the same pose/scale): used as the base image instead of the side-car's own render, which has no gaussians beyond the block's cut (grey patches)")
a = ap.parse_args()
S = Path('/home/paperspace/data/citrus_all') / a.survey; EMB = os.environ['HIGH_EMBEDDER_CKPT']; OUT = Path(a.out); (OUT / 'frames').mkdir(parents=True, exist_ok=True)
sky_dir = a.sky_masks or str(S / 'prod/tassili/sky_masks'); SKY = Path(sky_dir); print(f'[reel] sky masks: {SKY} ({"found" if SKY.exists() else "MISSING — sky not masked"})', flush=True)
clip, _, _ = open_clip.create_model_and_transforms('ViT-B-16', 'laion2b_s34b_b88k', device=dev); tok = open_clip.get_tokenizer('ViT-B-16')
hyper = build_hyper_embedder(EMB, dev); curv = hyper.curv.exp()
def word_vec(words):
    e = clip.encode_text(tok(words).to(dev)).float(); return e / e.norm(dim=-1, keepdim=True)
def colour(tid):
    h = (int(tid) * 37) % 180   # hues 37 deg apart for consecutive tree ids (neighbours along a row have consecutive ids)
    return tuple(int(v) for v in cv2.cvtColor(np.uint8([[[h, 210, 235]]]), cv2.COLOR_HSV2BGR)[0, 0])
import re
def verdict_thr(log, block, tag):
    """word -> best threshold from the block's containment verdict lines '[NNN sidecar <tag> kf] TREE/ROW n "word": thr t IoU ...'"""
    out = {}
    if not log or not Path(log).exists(): return out
    for line in open(log):
        if line.startswith(f'[{block} sidecar {tag} '):
            m = re.search(r'"([a-z]+)": thr ([0-9.]+)', line)
            if m: out[m.group(1)] = float(m.group(2))
    return out
ROWCOL = [(0, 220, 255), (255, 200, 0), (200, 0, 255), (0, 255, 120)]
def heats(feat, E):
    """containment ladder for every word column of E over every pixel: max over own point + 4 interpolated ancestor steps"""
    H, W_, D = feat.shape; ft = torch.from_numpy(np.asarray(feat).reshape(-1, D)).to(dev); void = (ft.norm(dim=-1) < 0.5); out = []
    for j in range(0, ft.shape[0], 8192):
        sl = ft[j:j + 8192]; h = L.exp_map0(sl, curv=curv)
        path, mask = L.get_interpolated_hyperbolic_features(h, steps=4, curv=curv, max_dist=11.1, return_mask=True)
        d = hyper.decode_features(path, project=True); d = d / d.norm(dim=-1, keepdim=True); sm = (d @ E.T).view(sl.shape[0], 4, -1)
        sm = sm.masked_fill(mask.to(sm.device).bool().unsqueeze(-1), -1.0)
        d0 = hyper.decode_features(h, project=True); d0 = d0 / d0.norm(dim=-1, keepdim=True); own = (d0 @ E.T)
        out.append(torch.maximum(sm.max(1).values, own))
    hm = torch.cat(out); hm[void] = -1.0; return hm.view(H, W_, -1).cpu().numpy()
idx = 0; t0 = time.time(); font = cv2.FONT_HERSHEY_SIMPLEX
for b in a.blocks:
    O = S / 'experimental/h3dgs_sidecar' / f'block_{b}'; cfgs = sorted(glob.glob(str(O / f'splat_runs_FEATFIX/stage2_censusinit_{a.seed_tag}/high/*/config.yml')))
    if not cfgs: print(f'[reel] block {b}: no seed {a.seed_tag} — skipped', flush=True); continue
    z = np.load(O / 'splat_runs_FEATFIX/interaction_W_glref_bg.npz'); labels = [int(u) for u in z['labels'] if int(u) != UNLAB and int(u) < FRUIT_ID_BASE]
    tree_words = [get_word_for_id(u, 'mask') for u in labels]; E = word_vec(tree_words + list(a.row_words)); nT = len(tree_words)
    _, pipe, _, step = eval_setup(Path(cfgs[-1])); model = pipe.model; model.eval(); model.step = step
    cams = []
    for ds in (pipe.datamanager.train_dataset, pipe.datamanager.eval_dataset):
        for i, f in enumerate(ds.image_filenames): cams.append((int(Path(f).name[3:9]), Path(f).name, ds.cameras[i:i + 1]))
    cams.sort(key=lambda c: c[0])
    vt = verdict_thr(a.verdict_log or f'/home/paperspace/logs/sidecar_{a.survey}_verdicts.log', b, a.verdict_tag)
    tthr = np.array([vt.get(w, a.tree_thr) for w in tree_words]); rthr = [vt.get(w, a.row_thr) for w in a.row_words]
    print(f'[reel] block {b}: {len(cams)} keyframes, trees {dict(zip(labels, tree_words))}, thresholds trees {dict(zip(tree_words, tthr.round(2)))} rows {dict(zip(a.row_words, np.round(rthr, 2)))}', flush=True)
    for k, name, cam in cams:
        cam = cam.to(model.device); cam.rescale_output_resolution(a.scale); rgb, alpha, feat = render_frame(model, cam, model.config.lang_field_dim)
        hm = heats(feat, E); img = np.ascontiguousarray(rgb[:, :, ::-1])
        if a.backdrop_dir and (Path(a.backdrop_dir) / name).exists():   # full-model backdrop, pixel-aligned (same pose, intrinsics, scale)
            bd = cv2.imread(str(Path(a.backdrop_dir) / name)); img = bd if bd.shape[:2] == img.shape[:2] else cv2.resize(bd, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        over = img.copy()
        marg = hm[:, :, :nT] - tthr[None, None, :]          # margin over each tree's own threshold; tint = best margin >= 0
        tmax = marg.max(-1); targ = marg.argmax(-1)
        for ti, u in enumerate(labels):
            m = (targ == ti) & (tmax >= 0)
            if m.sum() < 30: continue
            over[m] = (0.5 * over[m] + 0.5 * np.array(colour(u), np.float32)).astype(np.uint8)   # no word labels: the words are internal tree indices, not user-facing (Paul, 2026-09-27)
        for ri, rw in enumerate(a.row_words):
            m = (hm[:, :, nT + ri] >= rthr[ri]).astype(np.uint8); cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(over, [c for c in cs if cv2.contourArea(c) > 200], -1, ROWCOL[ri % len(ROWCOL)], 2)
        sm = SKY / name
        if sky_dir and sm.exists():   # sky masked out, as the block models' sky loss renders it (black background)
            skym = cv2.resize(cv2.imread(str(sm), cv2.IMREAD_GRAYSCALE), (over.shape[1], over.shape[0]), interpolation=cv2.INTER_NEAREST) > 127; over[skym] = 0
        cv2.putText(over, f'{a.survey}  block {b}  {name}   tint = tree identity (containment side-car)   outlines = the two rows', (8, 16), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(OUT / 'frames' / f'{idx:05d}.png'), over); idx += 1
    del model, pipe; torch.cuda.empty_cache()
print(f'[reel] {idx} frames in {time.time() - t0:.0f}s', flush=True)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(a.fps), '-i', str(OUT / 'frames' / '%05d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', str(OUT / 'row_reel.mp4')], check=True)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(a.fps), '-i', str(OUT / 'frames' / '%05d.png'), '-vf', 'fps=4,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse', '-frames:v', '120', str(OUT / 'row_reel_preview.gif')], check=True)
print(f'[reel] wrote {OUT / "row_reel.mp4"} and {OUT / "row_reel_preview.gif"}', flush=True)
