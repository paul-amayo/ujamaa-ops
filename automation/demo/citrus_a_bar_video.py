#!/usr/bin/env python3
"""Citrus A demo video under Paul's bar (UJAMAA, 2026-10-03: "we need to show people 25 dBs and IoUs of over 0.8 … make
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
R = Path('/home/paperspace/logs/demo_chunks/01_3_1_720'); OUT = Path('/home/paperspace/data/demo_video_v2/citrus_a_bar'); (OUT / 'frames').mkdir(parents=True, exist_ok=True)
BAR_DB, BAR_IOU = 25.0, 0.8
torch.set_grad_enabled(False)   # inference only (the embedder decoder has trainable weights)
src = f'{P}/camera_calibration/chunks/{CN}/sparse/0'; cam0 = list(read_cameras_binary(f'{src}/cameras.bin').values())[0]; W, H = int(cam0.width), int(cam0.height); fx, fy, cx, cy = cam0.params[:4]
ims = {v.name: v for v in read_images_binary(f'{src}/images.bin').values()}; TEST = {l.strip() for l in open(f'{src}/test.txt') if l.strip()}
EXPO = json.load(open(f'{P}/output/trained_chunks/{CN}/exposure.json'))
Hj = json.load(open(f'{S}/prod/bateleur/scene_graph/marker_hierarchy.json')); row_of = {o['id']: o['row_id'] for o in Hj['objects']}
script = json.load(open(R / 'demo_script.json')); LABEL = {int(k): v for k, v in script['label'].items()}
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
def iou(m, g): return float((m & g).sum() / max((m | g).sum(), 1))
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
        img, _ = CH.render(cam, 3.0); img = torch.einsum('ji,jhw->ihw', torch.tensor(E_[:, :3]).cuda(), img) + torch.tensor(E_[:, 3]).cuda()[:, None, None]
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
            r = row_of.get(vis[0][1]); word = NI.row_words.get(str(r)) if r is not None else None
            if word:
                a = lit(heats(feat, [word])[0]); g = np.isin(sup, [k for k, rr in row_of.items() if rr == r])
                if a is not None and iou(a > 0, g) >= BAR_IOU: lights.append((a, PAL[0])); ious.append(iou(a > 0, g)); answer = f'Row {r}'
        elif mode == 'tree':
            t = vis[0][1] if not script['segments'][4].get('focus') or script['segments'][4]['focus'] not in [x for _, x in vis] else script['segments'][4]['focus']
            word = NI.word_table.get(str(t))
            if word:
                a = lit(heats(feat, [word])[0]); g = sup == t
                if a is not None and iou(a > 0, g) >= BAR_IOU: lights.append((a, PAL[1])); ious.append(iou(a > 0, g)); answer = LABEL.get(t, f'tree {t}').capitalize()
        else:
            heads = ([t for _, t in vis] if mode == 'all' else sorted({row_of[t] for _, t in vis if t in row_of}))
            words = [(h, NI.word_table.get(str(h)) if mode == 'all' else NI.row_words.get(str(h))) for h in heads]; words = [(h, w) for h, w in words if w]
            if words:
                hm = heats(feat, [w for _, w in words])
                for k, (h, w) in enumerate(words):
                    a = lit(hm[k]); g = (sup == h) if mode == 'all' else np.isin(sup, [kk for kk, rr in row_of.items() if rr == h])
                    if a is not None and g.sum() > 200 and iou(a > 0, g) >= BAR_IOU: lights.append((a, PAL[h % len(PAL)])); ious.append(iou(a > 0, g))
                answer = f'{len(lights)} {"trees" if mode == "all" else "rows"} at IoU >= 0.8'
        if mode in ('row', 'tree') and not lights: rows_out.append((f['i'], kf, mode, round(psnr, 2), '', '', 'dropped: object IoU < 0.8')); continue
    frame = rgb.astype(np.float32)
    for a, col in lights: frame = frame * (1 - a[..., None]) + np.array(col, np.float32) * a[..., None]
    pim = Image.fromarray(frame.astype(np.uint8)); d = ImageDraw.Draw(pim)
    if q:
        d.rounded_rectangle([16, 16, 30 + d.textlength(q, font=F1), 60], 8, fill=(11, 11, 11)); d.text((24, 22), q, font=F1, fill=(255, 255, 255))
    if answer: d.rounded_rectangle([16, 68, 30 + d.textlength(answer, font=F2), 102], 8, fill=(255, 122, 0)); d.text((24, 73), answer, font=F2, fill=(11, 11, 11))
    foot = f'render {psnr:.1f} dB vs the photo' + (f'  ·  IoU {min(ious):.2f}' + (f'-{max(ious):.2f}' if len(ious) > 1 else '') if ious else '')
    d.rounded_rectangle([16, H - 50, 30 + d.textlength(foot, font=F2), H - 16], 8, fill=(11, 11, 11)); d.text((24, H - 45), foot, font=F2, fill=(255, 255, 255))
    pim.save(OUT / 'frames' / f'{n:05d}.png'); n += 1
    rows_out.append((f['i'], kf, mode, round(psnr, 2), len(ious), round(min(ious), 3) if ious else '', 'shown'))
    if n % 50 == 0: print(f'[bar] {n} frames written', flush=True)
with open(OUT / 'metrics.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['path_i', 'keyframe', 'mode', 'psnr_db', 'objects_lit', 'min_iou', 'status']); w.writerows(rows_out)
shown = [r for r in rows_out if r[6] == 'shown']
print(f'[bar] shown {len(shown)} of {len(path["frames"])} path frames; dropped: untrained {sum(1 for f in path["frames"] if f["src"] in TEST)}, render < 25 dB {sum(1 for r in rows_out if r[6].startswith("dropped: render"))}, object IoU < 0.8 {sum(1 for r in rows_out if r[6].startswith("dropped: object"))}', flush=True)
for m in ('plain', 'row', 'tree', 'all', 'rows', 'orchard'):
    ss = [r for r in shown if r[2] == m]
    if ss: print(f'[bar]   {m:8s}: {len(ss)} frames, PSNR min {min(r[3] for r in ss):.1f} median {np.median([r[3] for r in ss]):.1f}' + (f', lit objects {sum(r[4] for r in ss if r[4] != "")}, min IoU {min(r[5] for r in ss if r[5] != ""):.2f}' if any(r[5] != '' for r in ss) else ''), flush=True)
mp4 = OUT / 'citrus_a_bar.mp4'
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-framerate', '8', '-i', str(OUT / 'frames' / '%05d.png'), '-vf', 'fps=30,format=yuv420p', '-c:v', 'libx264', '-crf', '20', '-movflags', '+faststart', str(mp4)], check=True)
print(f'[bar] wrote {mp4} ({mp4.stat().st_size / 2**20:.1f} MiB, {n / 8:.1f} s)', flush=True)
