#!/usr/bin/env python3
"""Chilli cut (UJAMAA 2026-10-05; Paul: "make a chilli cut, where are the red chillies"). Gwakungu IMG_7990_s1 chilli splat (H3DGS
lane chunk, 8 M), class-level red-chilli identity cell_red (tiled SAM3 'red chili pepper' masks, seed share > 0.3: training views
IoU 0.44, held-out 0.33). Frames: the segment's training views image_11 .. image_141 in drive order (the opening frames and the
collapsed-pose tail are left out, as in the chilli crop), cropped to the middle (half width x two-thirds height, Paul's chilli crop).
Lighting: a pixel is a red chilli where its OWN decoded word is the pepper word, not the plant word (argmax of the two; the containment
walk passes the plant node and loses every pod). Magenta + white edge (red on red is invisible). Captions per the final-cut rule: the
question, the answer and the frame's IoU against SAM3 - no dB, no training-view wording.
Outputs data/demo_video_v2/chilli_cut/{frames, metrics.csv, chilli_cut.mp4 (6 fps, 2 s on the first frame), chilli_cut_phone.mp4}."""
import csv, json, math, re, subprocess, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as nd
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/splat_viewer'); sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians/preprocess')
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from hier_compact import CompactHierarchy
from native_identity import NativeIdentity
import lorentz as L
from read_write_model import read_images_binary, read_cameras_binary, qvec2rotmat
torch.set_grad_enabled(False)
S = '/home/paperspace/data/image_farm/gwakungu/2026-05-16/IMG_7990_s1'; P = f'{S}/h3dgs'; NB = f'{S}/experimental/pepper_class/cell_red'; FEAT = f'{NB}/features_s03.bin'; SUP = f'{NB}/supervision/pepper_class'
OUT = Path('/home/paperspace/data/demo_video_v2/chilli_cut'); (OUT / 'frames').mkdir(parents=True, exist_ok=True)
Q = ('Pilipili nyekundu ziko wapi? Nionyeshe.', 'Where are the red chillies? Show me.'); ANSWER = 'The red chillies'
src = f'{P}/camera_calibration/chunks/lane/sparse/0'; c0 = list(read_cameras_binary(f'{src}/cameras.bin').values())[0]; W, H = int(c0.width), int(c0.height); fx, fy, cx, cy = c0.params[:4]
ims = {v.name: v for v in read_images_binary(f'{src}/images.bin').values()}; TEST = {l.strip() for l in open(f'{src}/test.txt') if l.strip()}
EXPO = json.load(open(f'{P}/output/trained_chunks/lane/exposure.json'))
frames = [n for n in sorted(ims, key=lambda n: int(re.sub(r'\D', '', n))) if 11 <= int(re.sub(r'\D', '', n)) <= 141 and n not in TEST and n in EXPO]
x0, x1, y0, y1 = 270, 810, 320, 1600   # half width x two-thirds height (Paul, chilli crop v2)
class Cam:
    def __init__(self, c2w_h):
        w2c = np.linalg.inv(c2w_h); R_, T_ = c2w_h[:3, :3], w2c[:3, 3]
        self.image_width, self.image_height = W, H; self.FoVx = 2 * math.atan(W / (2 * fx)); self.FoVy = 2 * math.atan(H / (2 * fy)); self.primx, self.primy = cx / W, cy / H
        self.world_view_transform = torch.tensor(getWorld2View2(R_, T_)).float().transpose(0, 1).cuda()
        self.projection_matrix = torch.tensor(getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy, primx=self.primx, primy=self.primy)).float().transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0); self.camera_center = self.world_view_transform.inverse()[3, :3]
CH = CompactHierarchy(f'{P}/output/trained_chunks/lane/hierarchy.hier_opt', '')
NI = NativeIdentity(FEAT, json.load(open(FEAT + '.json'))['embedder'], f'{NB}/text_bank.npz', CH.N, n_hier=CH.n_hier); E = NI.E[[NI.widx[w] for w in (NI.word_table['0'], NI.word_table['10000'])]]
Fq = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 23); Fa = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 21)
rows = []
for n, kf in enumerate(frames):
    im = ims[kf]; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec; cam = Cam(np.linalg.inv(w2c)); E_ = np.array(EXPO[kf], np.float32)
    img, _ = CH.render(cam, 3.0); img = torch.einsum('ji,jhw->ihw', torch.tensor(E_[:, :3]).cuda(), img) + torch.tensor(E_[:, 3]).cuda()[:, None, None]
    rgb = (img.clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy()
    feat = NI.feature_pass(CH, cam, 3.0); h_, w_, Dd = feat.shape; ft = feat.reshape(-1, Dd).float()
    d0 = NI.hyper.decode_features(L.exp_map0(ft, curv=NI.curv), project=True); d0 = d0 / d0.norm(dim=-1, keepdim=True)
    pep = (((d0 @ E.T).argmax(1) == 1) & (ft.norm(dim=-1) >= 0.5)).view(h_, w_).cpu().numpy()
    pep = np.array(Image.fromarray(pep.astype(np.uint8)).resize((W, H), Image.NEAREST)) > 0
    sup = np.array(Image.open(f'{SUP}/{kf}'), np.uint16); ok = sup != 65535; g = (sup == 10000) & ok; lit = pep & ok
    iou = float((lit & g).sum() / max((lit | g).sum(), 1)) if g.sum() >= 200 else None
    fr = rgb.astype(np.float32); fr[pep] = 0.1 * fr[pep] + 0.9 * np.array([255, 0, 220]); fr = fr.astype(np.uint8)
    edge = nd.binary_dilation(pep, iterations=2) & ~pep; fr[edge] = (255, 255, 255)
    pim = Image.fromarray(fr[y0:y1, x0:x1]); d = ImageDraw.Draw(pim); cw = x1 - x0
    wq = max(d.textlength(t, font=Fq) for t in Q); d.rounded_rectangle([10, 10, 24 + wq, 78], 8, fill=(11, 11, 11))
    d.text((17, 14), Q[0], font=Fq, fill=(255, 255, 255)); d.text((17, 44), Q[1], font=Fq, fill=(255, 255, 255))
    d.rounded_rectangle([10, 86, 24 + d.textlength(ANSWER, font=Fa), 118], 8, fill=(255, 122, 0)); d.text((17, 90), ANSWER, font=Fa, fill=(11, 11, 11))
    if iou is not None:
        foot = f'IoU {iou:.2f}'; d.rounded_rectangle([10, (y1 - y0) - 44, 24 + d.textlength(foot, font=Fa), (y1 - y0) - 10], 8, fill=(11, 11, 11)); d.text((17, (y1 - y0) - 40), foot, font=Fa, fill=(255, 255, 255))
    pim.save(OUT / 'frames' / f'{n:05d}.png'); rows.append({'path_i': n, 'keyframe': kf, 'mode': 'pepper', 'iou': '' if iou is None else round(iou, 3), 'lit_px': int(lit.sum()), 'sam3_px': int(g.sum()), 'status': 'shown'})
    torch.cuda.empty_cache()
    if n % 25 == 0: print(f'[chilli-cut] {n + 1}/{len(frames)} {kf} IoU {iou}', flush=True)
with open(OUT / 'metrics.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
v = [r['iou'] for r in rows if r['iou'] != '']; print(f'[chilli-cut] {len(rows)} frames, IoU mean {np.mean(v):.3f} median {np.median(v):.3f} on {len(v)} frames with SAM3 chillies', flush=True)
subprocess.run([sys.executable, '/home/paperspace/code/automation/demo/compose_hold.py', '--frames', str(OUT / 'frames'), '--metrics', str(OUT / 'metrics.csv'), '--out', str(OUT / 'chilli_cut.mp4'), '--fps', '6', '--hold', '2'], check=True)
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(OUT / 'chilli_cut.mp4'), '-c:v', 'libx264', '-crf', '27', '-preset', 'slow', '-movflags', '+faststart', str(OUT / 'chilli_cut_phone.mp4')], check=True)
print(f"[chilli-cut] wrote {OUT / 'chilli_cut.mp4'} and the phone copy ({(OUT / 'chilli_cut_phone.mp4').stat().st_size / 2**20:.1f} MiB)", flush=True)
