#!/usr/bin/env python3
"""Chilli splat render (2026-10-04; Paul: "let me see the chili clip" ... "no I'll wait for the renders"): the gwakungu IMG_7990_s1
chilli segment's RGB-only H3DGS (single chunk "lane", 8 M budget, no scaffold, no identity) rendered from every camera of
the segment, side by side with the TRAINING image (rectified, undistorted), each frame captioned with its own PSNR.
Trained views use their trained exposure, held-out views (every 10th frame, test.txt) the mean exposure. tau 0 (full
detail), the evaluator's setting. Outputs <out>/frames/*.png, <out>/metrics.csv, <out>/chilli_render.mp4."""
import csv, json, math, subprocess, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/splat_viewer'); sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians/preprocess')
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from hier_compact import CompactHierarchy
from read_write_model import read_images_binary, read_cameras_binary, qvec2rotmat
P = '/home/paperspace/data/image_farm/gwakungu/2026-05-16/IMG_7990_s1/h3dgs'; CN = 'lane'; TAU = 0.0
OUT = Path('/home/paperspace/data/demo_video_v2/chilli/chilli_render'); (OUT / 'frames').mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)
src = f'{P}/camera_calibration/chunks/{CN}/sparse/0'; cam0 = list(read_cameras_binary(f'{src}/cameras.bin').values())[0]; W, H = int(cam0.width), int(cam0.height); fx, fy, cx, cy = cam0.params[:4]
ims = {v.name: v for v in read_images_binary(f'{src}/images.bin').values()}; TEST = {l.strip() for l in open(f'{src}/test.txt') if l.strip()}
EXPO = json.load(open(f'{P}/output/trained_chunks/{CN}/exposure.json')); MEAN_EXPO = np.mean([np.array(v, np.float32) for v in EXPO.values()], axis=0)
class Cam:   # hier_render_service.Cam
    def __init__(self, c2w_h, W, H, fovy, primx=0.5, primy=0.5):
        w2c = np.linalg.inv(c2w_h); R_, T_ = c2w_h[:3, :3], w2c[:3, 3]
        self.image_width, self.image_height = W, H; self.FoVy = fovy; self.FoVx = 2 * math.atan(math.tan(fovy / 2) * W / H); self.primx, self.primy = primx, primy
        self.world_view_transform = torch.tensor(getWorld2View2(R_, T_)).float().transpose(0, 1).cuda()
        self.projection_matrix = torch.tensor(getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy, primx=primx, primy=primy)).float().transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
CH = CompactHierarchy(f'{P}/output/trained_chunks/{CN}/hierarchy.hier_opt', '')   # image-farm chunks train without a scaffold
try:
    F1 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 30); F2 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 26)
except Exception: F1 = F2 = ImageFont.load_default()
names = sorted(ims, key=lambda n: int(''.join(c for c in n if c.isdigit())))
rows, n = [], 0
for kf in names:
    im = ims[kf]; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec; c2w = np.linalg.inv(w2c)
    cam = Cam(c2w, W, H, 2 * math.atan(H / (2 * fy)), cx / W, cy / H); held = kf in TEST or kf not in EXPO
    E_ = MEAN_EXPO if held else np.array(EXPO[kf], np.float32)
    img, _ = CH.render(cam, TAU); img = torch.einsum('ij,jhw->ihw', torch.tensor(E_[:, :3]).cuda(), img) + torch.tensor(E_[:, 3]).cuda()[:, None, None]
    rgb = (img.clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy()
    photo = np.asarray(Image.open(f'{P}/camera_calibration/rectified/images/{kf}').convert('RGB'))
    psnr = float(10 * np.log10(255 ** 2 / max(((rgb.astype(np.float32) - photo.astype(np.float32)) ** 2).mean(), 1e-9)))
    pim = Image.fromarray(np.concatenate([photo, rgb], 1)); d = ImageDraw.Draw(pim)
    for x0, t in ((0, 'photo (training image)'), (W, 'H3DGS render')):
        d.rounded_rectangle([x0 + 16, 16, x0 + 30 + d.textlength(t, font=F1), 62], 8, fill=(11, 11, 11)); d.text((x0 + 24, 22), t, font=F1, fill=(255, 255, 255))
    foot = f'{kf}  ·  ' + ('held-out view, mean exposure' if held else 'trained view') + f'  ·  render {psnr:.1f} dB vs the photo'
    d.rounded_rectangle([W + 16, H - 56, W + 30 + d.textlength(foot, font=F2), H - 16], 8, fill=(11, 11, 11)); d.text((W + 24, H - 51), foot, font=F2, fill=(255, 255, 255))
    pim.save(OUT / 'frames' / f'{n:05d}.png'); n += 1; rows.append((kf, 'held-out' if held else 'trained', round(psnr, 2)))
    torch.cuda.empty_cache()
    if n % 50 == 0: print(f'[chilli] {n} frames written', flush=True)
with open(OUT / 'metrics.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['image', 'split', 'psnr_db']); w.writerows(rows)
for s in ('trained', 'held-out'):
    v = [r[2] for r in rows if r[1] == s]
    if v: print(f'[chilli] {s}: {len(v)} views, PSNR min {min(v):.2f} median {np.median(v):.2f} max {max(v):.2f}', flush=True)
mp4 = OUT / 'chilli_render.mp4'
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-framerate', '8', '-i', str(OUT / 'frames' / '%05d.png'), '-vf', 'scale=1440:-2,fps=30,format=yuv420p', '-c:v', 'libx264', '-crf', '20', '-movflags', '+faststart', str(mp4)], check=True)
print(f'[chilli] wrote {mp4} ({mp4.stat().st_size / 2**20:.1f} MiB, {n / 8:.1f} s)', flush=True)
