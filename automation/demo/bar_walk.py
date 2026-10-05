#!/usr/bin/env python3
"""RGB bar walk of one H3DGS chunk (2026-10-05; the dashboard session for Paul's final cut: "a Klapmuts lane bar walk and a Kendu Bay
(IMG_7975_s0) bar walk: trained views only in drive order, each view's own exposure, sky in, keep if >= 25 dB, PSNR caption like
citrus_a_bar; RGB only, 8 fps"). The citrus_a_bar rendering: CompactHierarchy at tau 3 (the serving cut), each trained view's own
exposure from exposure.json, PSNR of the full frame (sky in) against the TRAINING image (camera_calibration/rectified/images).
No identity, no map inset (the citrus_a_bar scripts draw none).
  python bar_walk.py --proj <h3dgs project> --out <dir> --title "<farm caption>" [--chunk lane] [--tau 3] [--bar 25] [--fps 8]
-> <out>/frames/*.png (kept frames, drive order), <out>/metrics.csv (every trained view: psnr, kept), <out>/<name>.mp4 + _phone.mp4"""
import argparse, csv, json, math, re, subprocess, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/splat_viewer'); sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians/preprocess')
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from hier_compact import CompactHierarchy
from read_write_model import read_images_binary, read_cameras_binary, qvec2rotmat
ap = argparse.ArgumentParser(); ap.add_argument('--proj', required=True); ap.add_argument('--out', required=True); ap.add_argument('--title', required=True)
ap.add_argument('--chunk', default='lane'); ap.add_argument('--scaffold', default=''); ap.add_argument('--tau', type=float, default=3.0); ap.add_argument('--bar', type=float, default=25.0); ap.add_argument('--fps', type=int, default=8)
ap.add_argument('--nocap', action='store_true', help='final cut (2026-10-05): no title / PSNR caption, the SAME frames as metrics.csv -> frames_nocap/, metrics_nocap.csv, <name>_nocap.mp4')
a = ap.parse_args(); P = a.proj; OUT = Path(a.out); FR = 'frames_nocap' if a.nocap else 'frames'; (OUT / FR).mkdir(parents=True, exist_ok=True); torch.set_grad_enabled(False)
KEEP = {r['image'] for r in csv.DictReader(open(OUT / 'metrics.csv')) if r['kept'] == '1'} if a.nocap else None
src = f'{P}/camera_calibration/chunks/{a.chunk}/sparse/0'; cam0 = list(read_cameras_binary(f'{src}/cameras.bin').values())[0]; W, H = int(cam0.width), int(cam0.height); fx, fy, cx, cy = cam0.params[:4]
ims = {v.name: v for v in read_images_binary(f'{src}/images.bin').values()}; TEST = {l.strip() for l in open(f'{src}/test.txt') if l.strip()}
EXPO = json.load(open(f'{P}/output/trained_chunks/{a.chunk}/exposure.json'))
trained = sorted((n for n in ims if n not in TEST and n in EXPO), key=lambda n: int(re.sub(r'\D', '', n) or 0))
class Cam:   # hier_render_service.Cam, but FoVx from fx (the evaluator's make_cam): the aspect-derived FoVx assumes fx = fy, and
    # Kendu's fx/fy differ by 0.65 % -> ~3.5 px off at the frame edges, 4-5 dB lost on foliage (2026-10-05)
    def __init__(self, c2w_h, W, H, fovx, fovy, primx=0.5, primy=0.5):
        w2c = np.linalg.inv(c2w_h); R_, T_ = c2w_h[:3, :3], w2c[:3, 3]
        self.image_width, self.image_height = W, H; self.FoVy = fovy; self.FoVx = fovx; self.primx, self.primy = primx, primy
        self.world_view_transform = torch.tensor(getWorld2View2(R_, T_)).float().transpose(0, 1).cuda()
        self.projection_matrix = torch.tensor(getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy, primx=primx, primy=primy)).float().transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
CH = CompactHierarchy(f'{P}/output/trained_chunks/{a.chunk}/hierarchy.hier_opt', a.scaffold)
sc = max(W, H) / 1280; F1 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', int(26 * sc)); F2 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', int(20 * sc))
rows, n = [], 0
for kf in trained:
    im = ims[kf]; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec; c2w = np.linalg.inv(w2c)
    cam = Cam(c2w, W, H, 2 * math.atan(W / (2 * fx)), 2 * math.atan(H / (2 * fy)), cx / W, cy / H); E_ = np.array(EXPO[kf], np.float32)
    img, _ = CH.render(cam, a.tau); img = torch.einsum('ij,jhw->ihw', torch.tensor(E_[:, :3]).cuda(), img) + torch.tensor(E_[:, 3]).cuda()[:, None, None]
    rgb = (img.clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy()
    photo = np.asarray(Image.open(f'{P}/camera_calibration/rectified/images/{kf}').convert('RGB')).astype(np.float32)
    psnr = float(10 * np.log10(255 ** 2 / max(((rgb.astype(np.float32) - photo) ** 2).mean(), 1e-9))); keep = (kf in KEEP) if a.nocap else psnr >= a.bar
    rows.append((kf, round(psnr, 2), int(keep))); torch.cuda.empty_cache()
    if not keep: continue
    pim = Image.fromarray(rgb); d = ImageDraw.Draw(pim); m = int(16 * sc)
    if a.nocap: pim.save(OUT / FR / f'{n:05d}.png'); n += 1; continue
    d.rounded_rectangle([m, m, m + int(14 * sc) + d.textlength(a.title, font=F1), m + int(44 * sc)], int(8 * sc), fill=(11, 11, 11)); d.text((m + int(8 * sc), m + int(6 * sc)), a.title, font=F1, fill=(255, 255, 255))
    foot = f'render {psnr:.1f} dB vs the photo'
    d.rounded_rectangle([m, H - m - int(34 * sc), m + int(14 * sc) + d.textlength(foot, font=F2), H - m], int(8 * sc), fill=(11, 11, 11)); d.text((m + int(8 * sc), H - m - int(29 * sc)), foot, font=F2, fill=(255, 255, 255))
    pim.save(OUT / FR / f'{n:05d}.png'); n += 1
with open(OUT / ('metrics_nocap.csv' if a.nocap else 'metrics.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['image', 'psnr_db', 'kept']); w.writerows(rows)
p = np.array([r[1] for r in rows]); k = p[p >= a.bar]
print(f'[bar] {len(rows)} trained views, PSNR median {np.median(p):.2f} (min {p.min():.2f}, max {p.max():.2f}); kept {n} at >= {a.bar:g} dB' + (f', kept median {np.median(k):.2f}' if k.size else ''), flush=True)
if n:
    name = OUT.name + ('_nocap' if a.nocap else '')
    for suf, crf, scl in (('', 20, 'iw:ih'), ('_phone', 27, '1280:-2' if W >= H else '720:-2')):
        f = OUT / f'{name}{suf}.mp4'
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-framerate', str(a.fps), '-i', str(OUT / FR / '%05d.png'), '-vf', f'scale={scl},fps=24,format=yuv420p', '-c:v', 'libx264', '-crf', str(crf), '-preset', 'slow', '-movflags', '+faststart', str(f)], check=True)
        print(f'[bar] wrote {f} ({f.stat().st_size / 2**20:.1f} MiB, {n / a.fps:.1f} s)', flush=True)
