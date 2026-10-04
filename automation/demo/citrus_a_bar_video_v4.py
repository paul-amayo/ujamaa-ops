#!/usr/bin/env python3
"""v4 (2026-10-04, Paul: "no threshold is IoU, I still need best containment"): trees light by BEST CONTAINMENT (argmax over
all tree words), like rows - no per-frame split anywhere. Output citrus_a_bar_v4/.
v3 (2026-10-04, Paul: "remove the minimum iou, show them all"): NO IoU minimum and no SAM3 presence requirement - every
object the query names is lit on every kept frame; the caption carries the measured IoU where SAM3 labels the object (n/a
otherwise). Frames still need a trained view and >= 25 dB. Output citrus_a_bar_v3/.
v2 (2026-10-04, after the 01 swap: CORAL-strict rows, median-anchored lifting, re-seeded chunk 3_1 cell). Changes vs v1:
rows light by ARGMAX over the 36-37 row words (no split, no threshold; the per-frame Otsu/absence rule lit nothing when the
row fills the frame, kf_003112), "this row" is ONE row per row segment (the majority row of the nearest tree in front over
the segment), tree labels come from the current hierarchy (row ids as the registry stores them). Output citrus_a_bar_v2/.
v1 docstring follows.
Citrus A demo video under Paul's bar (UJAMAA, 2026-10-03: "we need to show people 25 dBs and IoUs of over 0.8 … make
the new video and add sky"). 01 chunk 3_1 (H3DGS), the original reel's drive and storyline (demo_path.json), but:
  frames   only the chunk's TRAINED views, rendered with that view's own trained exposure, SKY INCLUDED, and kept only if
           the frame's PSNR against its photo (full frame) is >= 25 dB
  identity native HiGH features in the hierarchy (chunk-supervision seed B), NO cut: per word, containment heat +
           per-frame Otsu/absence guard (native_identity.py); rows and trees by PLAIN containment (no exclusive step).
           "this row" / "this tree" = the nearest scene-graph tree in front of the camera (geometry, not supervision).
           Each lit object is scored against its SAM3 mask on that frame and lit only if IoU >= 0.8; a single-query frame
           whose object misses the bar is dropped.
  caption  the question, the answer, and the frame's own measured PSNR / IoU.
Outputs: <out>/frames/*.png, <out>/metrics.csv, <out>/citrus_a_bar.mp4 (+ phone copy)."""
import csv, json, math, os, subprocess, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/splat_viewer'); sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians/preprocess')
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from hier_compact import CompactHierarchy
from native_identity import NativeIdentity, _otsu
import lorentz as L
from read_write_model import read_images_binary, read_cameras_binary, qvec2rotmat
S = '/home/paperspace/data/citrus_all/01_13B_Jackal'; P = f'{S}/experimental/h3dgs'; CN = '3_1'; NB = f'{S}/experimental/h3dgs_native/chunk_3_1_sam3'
R = Path('/home/paperspace/logs/demo_chunks/01_3_1_720'); OUT = Path('/home/paperspace/data/demo_video_v2/citrus_a_bar_v4'); (OUT / 'frames').mkdir(parents=True, exist_ok=True)
BAR_DB, BAR_IOU = 25.0, None   # no IoU minimum (v3)
torch.set_grad_enabled(False)   # inference only (the embedder decoder has trainable weights)
src = f'{P}/camera_calibration/chunks/{CN}/sparse/0'; cam0 = list(read_cameras_binary(f'{src}/cameras.bin').values())[0]; W, H = int(cam0.width), int(cam0.height); fx, fy, cx, cy = cam0.params[:4]
ims = {v.name: v for v in read_images_binary(f'{src}/images.bin').values()}; TEST = {l.strip() for l in open(f'{src}/test.txt') if l.strip()}
EXPO = json.load(open(f'{P}/output/trained_chunks/{CN}/exposure.json'))
Hj = json.load(open(f'{S}/prod/bateleur/scene_graph/marker_hierarchy.json')); row_of = {o['id']: o['row_id'] for o in Hj['objects']}
script = json.load(open(R / 'demo_script.json'))
_objs = json.load(open(f'{S}/prod/bateleur/scene_graph/marker_hierarchy.json'))['objects']; _z = {o['id']: o['xyz'][2] for o in _objs}
LABEL = {}
for _r in sorted({o['row_id'] for o in _objs if o['row_id'] >= 0}):
    for _k, _t in enumerate(sorted((t for t in row_of if row_of[t] == _r), key=lambda t: _z[t])): LABEL[_t] = f'tree {_k + 1} of row {_r}'
path = json.load(open(R / 'demo_path.json')); TREES = {int(k): np.array(v) for k, v in path['trees'].items()}
class Cam:   # hier_render_service.Cam
    def __init__(self, c2w_h, W, H, fovy, primx=0.5, primy=0.5):
        w2c = np.linalg.inv(c2w_h); R_, T_ = c2w_h[:3, :3], w2c[:3, 3]
        self.image_width, self.image_height = W, H; self.FoVy = fovy; self.FoVx = 2 * math.atan(math.tan(fovy / 2) * W / H); self.primx, self.primy = primx, primy
        self.world_view_transform = torch.tensor(getWorld2View2(R_, T_)).float().transpose(0, 1).cuda()
        self.projection_matrix = torch.tensor(getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy, primx=primx, primy=primy)).float().transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
CH = CompactHierarchy(f'{P}/output/trained_chunks/{CN}/hierarchy.hier_opt', f'{P}/output/scaffold/point_cloud/iteration_30000')
NI = NativeIdentity(f'{NB}/features_B_bg2share.bin', sorted(Path(f'{S}/prod/bateleur/embedder').glob('*/ckpts/model_best.pth'))[-1].as_posix(), f'{NB}/text_bank.npz', CH.N, n_hier=CH.n_hier)
def heats(feat, words):
    """Plain containment heat for several words at once (own point + non-extrapolated walk, max; norm gate) -> [n, h, w]."""
    E = NI.E[[NI.widx[w] for w in words]]; h_, w_, D = feat.shape; ft = feat.reshape(-1, D).float()
    out = torch.full((len(words), ft.shape[0]), -1.0, device=ft.device); idx = (ft.norm(dim=-1) >= 0.5).nonzero(as_tuple=True)[0]
    for j in range(0, int(idx.shape[0]), 8192):
        sel = idx[j:j + 8192]; hh = L.exp_map0(ft[sel], curv=NI.curv)
        pth, msk = L.get_interpolated_hyperbolic_features(hh, steps=4, curv=NI.curv, max_dist=11.1, return_mask=True)
        d = NI.hyper.decode_features(pth, project=True); d = d / d.norm(dim=-1, keepdim=True)
        sm = (d @ E.T).view(sel.shape[0], 4, len(words)).masked_fill(msk.to(d.device).bool().unsqueeze(-1), -1.0)
        d0 = NI.hyper.decode_features(hh, project=True); d0 = d0 / d0.norm(dim=-1, keepdim=True)
        out[:, sel] = torch.cat([sm, (d0 @ E.T).view(sel.shape[0], 1, len(words))], 1).max(1).values.T
    return out.view(len(words), h_, w_)
def lit(score_hw):
    NI.cut = 0.0; a = NI.alpha(torch.nn.functional.interpolate(score_hw[None, None], size=(H, W), mode='nearest')[0, 0])
    return None if a is None else a.float().cpu().numpy()
def iou(m, g): return float((m & g).sum() / max((m | g).sum(), 1)) if g.sum() > 200 else None   # None = SAM3 does not label it here
RIDS = sorted(int(r) for r in NI.row_words); RWORDS = [NI.row_words[str(r)] for r in RIDS]
def row_argmax(feat):
    """best row word per pixel (identity norm >= 0.5, else -1), upsampled to the frame -> int array [H, W] of row ids"""
    hm = heats(feat, RWORDS); valid = (hm > -1).any(0); am = hm.argmax(0)
    out = torch.where(valid, torch.tensor(RIDS, device=am.device)[am], torch.full_like(am, -1)).cpu().numpy().astype(np.int32)
    return np.array(Image.fromarray(out).resize((W, H), Image.NEAREST))
TREE_IDS = sorted(int(k) for k in NI.word_table if int(k) < 10000 and NI.word_table[k] in NI.widx); TWORDS = [NI.word_table[str(t)] for t in TREE_IDS]
ALLW = [w for w in NI.words if w in NI.widx]
def best_of(feat, ids, words):
    """BEST CONTAINMENT (Paul 2026-10-04): per pixel, the id whose word has the highest containment heat among `words`
    (identity norm >= 0.5, else -1). No split, no threshold. -> int map [H, W]"""
    hm = heats(feat, words); valid = (hm > -1).any(0); am = hm.argmax(0)
    out = torch.where(valid, torch.tensor(ids, device=am.device)[am], torch.full_like(am, -1)).cpu().numpy().astype(np.int32)
    return np.array(Image.fromarray(out).resize((W, H), Image.NEAREST))
def front_trees(c2w, w2c):
    cc = c2w[:3, 3]; fwd = c2w[:3, 2]; vis = []
    for t, x in TREES.items():
        v = x - cc
        if v @ fwd <= 0.5: continue
        p = w2c[:3, :3] @ x + w2c[:3, 3]; u, vv = fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy
        if 0 <= u < W and 0 <= vv < H: vis.append((float(np.linalg.norm(v)), t))
    return sorted(vis)
from collections import Counter
_seg = Counter()
for f in path['frames']:
    if f['mode'] == 'row' and f['src'] not in TEST and f['src'] in ims:
        im_ = ims[f['src']]; w2c_ = np.eye(4); w2c_[:3, :3] = qvec2rotmat(im_.qvec); w2c_[:3, 3] = im_.tvec; v_ = front_trees(np.linalg.inv(w2c_), w2c_)
        if v_ and v_[0][1] in row_of: _seg[row_of[v_[0][1]]] += 1
SEG_ROW = _seg.most_common(1)[0][0] if _seg else None
print(f'[bar] "this row" segment row: {SEG_ROW} (nearest-front-tree rows over the segment: {dict(_seg)})', flush=True)
PAL = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0), (232, 123, 164), (0, 131, 0), (74, 58, 167), (227, 73, 72)]
try:
    F1 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 26); F2 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
except Exception: F1 = F2 = ImageFont.load_default()
rows_out, n = [], 0
for f in path['frames']:
    kf = f['src']; mode = f['mode']; q = f['question']
    if kf in TEST or kf not in ims or kf not in EXPO: continue                       # untrained view: never shown
    im = ims[kf]; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec; c2w = np.linalg.inv(w2c)
    cam = Cam(c2w, W, H, 2 * math.atan(H / (2 * fy)), cx / W, cy / H); E_ = np.array(EXPO[kf], np.float32)
    with torch.no_grad():
        img, _ = CH.render(cam, 3.0); img = torch.einsum('ij,jhw->ihw', torch.tensor(E_[:, :3]).cuda(), img) + torch.tensor(E_[:, 3]).cuda()[:, None, None]
        rgb = (img.clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy()
    photo = np.asarray(Image.open(f'{S}/prod/scratch_sam3/{kf}').convert('RGB')).astype(np.float32)
    psnr = float(10 * np.log10(255 ** 2 / max(((rgb.astype(np.float32) - photo) ** 2).mean(), 1e-9)))
    if psnr < BAR_DB: rows_out.append((f['i'], kf, mode, round(psnr, 2), '', '', 'dropped: render < 25 dB')); continue
    sup = np.array(Image.open(f'{NB}/supervision/trees_only/{kf}'), np.uint16)
    if sup.shape != (H, W): sup = np.array(Image.fromarray(sup).resize((W, H), Image.NEAREST))
    # trees in front of the camera, nearest first (scene-graph geometry, not the supervision)
    cc = c2w[:3, 3]; fwd = c2w[:3, 2]; vis = []
    for t, x in TREES.items():
        v = x - cc; z = v @ fwd
        if z <= 0.5: continue
        p = w2c[:3, :3] @ x + w2c[:3, 3]; u, vv = fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy
        if 0 <= u < W and 0 <= vv < H: vis.append((float(np.linalg.norm(v)), t))
    vis.sort(); lights, answer, ious = [], '', []
    if mode in ('row', 'tree', 'all', 'rows') and vis:
        with torch.no_grad(): feat = NI.feature_pass(CH, cam, 3.0)
        if mode == 'row':
            r = SEG_ROW
            if r is not None and str(r) in NI.row_words:
                am = row_argmax(feat); a = (am == r).astype(np.float32) * 0.55; g = np.isin(sup, [k for k, rr in row_of.items() if rr == r])
                lights.append((a, PAL[0])); ious.append(iou(a > 0, g)); answer = f'Row {r}'
        elif mode == 'tree':
            t = vis[0][1] if not script['segments'][4].get('focus') or script['segments'][4]['focus'] not in [x for _, x in vis] else script['segments'][4]['focus']
            if str(t) in NI.word_table:
                bt = best_of(feat, TREE_IDS, TWORDS); a = (bt == t).astype(np.float32) * 0.55; g = sup == t
                lights.append((a, PAL[1])); ious.append(iou(a > 0, g))
                answer = LABEL.get(t, f'tree {t}').capitalize()
        else:
            if mode == 'rows':
                am = row_argmax(feat)
                for h in sorted({row_of[t] for _, t in vis if t in row_of and row_of[t] >= 0}):
                    a = (am == h).astype(np.float32) * 0.55; g = np.isin(sup, [kk for kk, rr in row_of.items() if rr == h])
                    if a.any(): lights.append((a, PAL[h % len(PAL)])); ious.append(iou(a > 0, g))
                answer = f'{len(lights)} rows'
            else:
                bt = best_of(feat, TREE_IDS, TWORDS)
                for h in [t for _, t in vis if str(t) in NI.word_table]:
                    a = (bt == h).astype(np.float32) * 0.55; g = (sup == h)
                    if a.any(): lights.append((a, PAL[h % len(PAL)])); ious.append(iou(a > 0, g))
                answer = f'{len(lights)} trees'
    frame = rgb.astype(np.float32)
    for a, col in lights: frame = frame * (1 - a[..., None]) + np.array(col, np.float32) * a[..., None]
    pim = Image.fromarray(frame.astype(np.uint8)); d = ImageDraw.Draw(pim)
    if q:
        d.rounded_rectangle([16, 16, 30 + d.textlength(q, font=F1), 60], 8, fill=(11, 11, 11)); d.text((24, 22), q, font=F1, fill=(255, 255, 255))
    if answer: d.rounded_rectangle([16, 68, 30 + d.textlength(answer, font=F2), 102], 8, fill=(255, 122, 0)); d.text((24, 73), answer, font=F2, fill=(11, 11, 11))
    mi = [x for x in ious if x is not None]; na = len(ious) - len(mi)
    foot = f'render {psnr:.1f} dB vs the photo' + (f'  ·  IoU {min(mi):.2f}' + (f'-{max(mi):.2f}' if len(mi) > 1 else '') if mi else '') + (f'  ·  {na} not labelled by SAM3' if na else '')
    d.rounded_rectangle([16, H - 50, 30 + d.textlength(foot, font=F2), H - 16], 8, fill=(11, 11, 11)); d.text((24, H - 45), foot, font=F2, fill=(255, 255, 255))
    pim.save(OUT / 'frames' / f'{n:05d}.png'); n += 1
    rows_out.append((f['i'], kf, mode, round(psnr, 2), len(ious), round(min(mi), 3) if mi else '', 'shown'))
    if n % 50 == 0: print(f'[bar] {n} frames written', flush=True)
with open(OUT / 'metrics.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['path_i', 'keyframe', 'mode', 'psnr_db', 'objects_lit', 'min_iou', 'status']); w.writerows(rows_out)
shown = [r for r in rows_out if r[6] == 'shown']
print(f'[bar] shown {len(shown)} of {len(path["frames"])} path frames; dropped: untrained {sum(1 for f in path["frames"] if f["src"] in TEST)}, render < 25 dB {sum(1 for r in rows_out if r[6].startswith("dropped: render"))}, object IoU < 0.8 {sum(1 for r in rows_out if r[6].startswith("dropped: object"))}', flush=True)
for m in ('plain', 'row', 'tree', 'all', 'rows', 'orchard'):
    ss = [r for r in shown if r[2] == m]
    if ss: print(f'[bar]   {m:8s}: {len(ss)} frames, PSNR min {min(r[3] for r in ss):.1f} median {np.median([r[3] for r in ss]):.1f}' + (f', lit objects {sum(r[4] for r in ss if r[4] != "")}, min IoU {min(r[5] for r in ss if r[5] != ""):.2f}' if any(r[5] != '' for r in ss) else ''), flush=True)
mp4 = OUT / 'citrus_a_bar_v4.mp4'
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-framerate', '8', '-i', str(OUT / 'frames' / '%05d.png'), '-vf', 'fps=30,format=yuv420p', '-c:v', 'libx264', '-crf', '20', '-movflags', '+faststart', str(mp4)], check=True)
print(f'[bar] wrote {mp4} ({mp4.stat().st_size / 2**20:.1f} MiB, {n / 8:.1f} s)', flush=True)
