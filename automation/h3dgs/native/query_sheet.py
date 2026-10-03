#!/usr/bin/env python3
"""Contact sheet of one query on keyframes, as the live stage shows it (UJAMAA, 2026-10-03): the hierarchy's own colour
render at the keyframe pose with the mean trained exposure (HIER_EXPOSURE=mean), lit in the stage's orange by
(1) the supervision mask (reference), (2) today's side-car field, (3) native under the live client's rule (no cut),
(4) native with a fixed cut. Side-car and native go through the identical heat + alpha code (native_identity.py).
h3dgs env: python query_sheet.py --proj P --chunk C --features F --embedder E --bank B --sup SUP --hj HJ
             --sidecar-dir D --frames kf_a.png ... --query row:7 --cut 0.8 --out DIR [--scaffold S]"""
import argparse, json, math, os, sys
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/splat_viewer')
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians/preprocess')
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from hier_compact import CompactHierarchy
from native_identity import NativeIdentity
from read_write_model import read_images_binary, read_cameras_binary, qvec2rotmat
ap = argparse.ArgumentParser()
for k in ('--proj', '--chunk', '--features', '--embedder', '--bank', '--sup', '--hj', '--sidecar-dir', '--query', '--out'): ap.add_argument(k, required=True)
ap.add_argument('--frames', nargs='+', required=True); ap.add_argument('--cut', type=float, default=0.8); ap.add_argument('--scaffold', default='')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
src = f'{a.proj}/camera_calibration/chunks/{a.chunk}/sparse/0'
cam0 = list(read_cameras_binary(f'{src}/cameras.bin').values())[0]; W, H = int(cam0.width), int(cam0.height); fx, fy, cx, cy = cam0.params[:4]
ims = {v.name: v for v in read_images_binary(f'{src}/images.bin').values()}
class Cam:   # hier_render_service.Cam
    def __init__(self, c2w_h, W, H, fovy, primx=0.5, primy=0.5):
        w2c = np.linalg.inv(c2w_h); R, T = c2w_h[:3, :3], w2c[:3, 3]
        self.image_width, self.image_height = W, H; self.FoVy = fovy; self.FoVx = 2 * math.atan(math.tan(fovy / 2) * W / H); self.primx, self.primy = primx, primy
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).float().transpose(0, 1).cuda()
        self.projection_matrix = torch.tensor(getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy, primx=primx, primy=primy)).float().transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
CH = CompactHierarchy(f'{a.proj}/output/trained_chunks/{a.chunk}/hierarchy.hier_opt', a.scaffold)
NI = NativeIdentity(a.features, a.embedder, a.bank, CH.N, n_hier=CH.n_hier)
E = np.array(list(json.load(open(f'{a.proj}/output/trained_chunks/{a.chunk}/exposure.json')).values()), np.float32)
A, b = torch.tensor(E[:, :, :3].mean(0)).cuda(), torch.tensor(E[:, :, 3].mean(0)).cuda()
Hj = json.load(open(a.hj)); row_of = {o['id']: o['row_id'] for o in Hj['objects']}; tree_of_fruit = {10000 + f['id']: f['tree_id'] for f in Hj.get('fruits', [])}
kind, oid = a.query.split(':'); oid = int(oid)
HL = np.array([255, 122, 0], np.float32)
try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
except Exception: font = ImageFont.load_default()
def tint(rgb, al): return (rgb.astype(np.float32) * (1 - al[..., None]) + HL * al[..., None]).astype(np.uint8)
def stats(al, gt):
    if al is None: return 'nothing lit'
    lit = al > 0; return f'{lit.sum() / max(gt.sum(), 1):.0%} of GT area, {(lit & gt).sum() / max(gt.sum(), 1):.0%} of GT covered, {(lit & gt).sum() / max(lit.sum(), 1):.0%} on target'
for fr in a.frames:
    im = ims[fr]; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec
    cam = Cam(np.linalg.inv(w2c), W, H, 2 * math.atan(H / (2 * fy)), cx / W, cy / H)
    with torch.no_grad():
        img, _ = CH.render(cam, 3.0); img = torch.einsum('ij,jhw->ihw', A, img) + b[:, None, None]
        rgb = (img.clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy()
        sup = np.array(Image.open(os.path.join(a.sup, fr)), np.uint16); sup = np.array(Image.fromarray(sup).resize((W, H), Image.NEAREST))
        if kind == 'row':
            trees = [k for k, r in row_of.items() if r == oid]; gt = np.isin(sup, trees + [f for f, t in tree_of_fruit.items() if t in trees])
        elif kind == 'tree': gt = (sup == oid) | np.isin(sup, [f for f, t in tree_of_fruit.items() if t == oid])
        else: gt = sup == oid
        NI.set_query({kind: oid}); al_n = NI.mask(CH, cam, 3.0)
        sc = torch.from_numpy(np.load(os.path.join(a.sidecar_dir, fr[:-4] + '_sidecar.npz'))['features']).float().cuda()
        al_s = NI.alpha(NI.heat(sc, (H, W))); al_s = None if al_s is None else al_s.float().cpu().numpy()
        NI.set_query({kind: oid, 'cut': a.cut}); al_c = NI.mask(CH, cam, 3.0)
    panels = [(tint(rgb, gt.astype(np.float32) * 0.6), f'supervision: {a.query} ({int(gt.sum())} px)'),
              (tint(rgb, al_s) if al_s is not None else rgb, f"today's side-car, no cut: {stats(al_s, gt)}"),
              (tint(rgb, al_n) if al_n is not None else rgb, f'native, no cut (live rule): {stats(al_n, gt)}'),
              (tint(rgb, al_c) if al_c is not None else rgb, f'native, fixed cut {a.cut:g}: {stats(al_c, gt)}')]
    tiles = []
    for arr, label in panels:
        t = Image.fromarray(arr).resize((960, 540), Image.LANCZOS); d = ImageDraw.Draw(t)
        d.rectangle([0, 0, 960, 34], fill=(0, 0, 0)); d.text((10, 5), label, fill=(255, 255, 255), font=font); tiles.append(t)
    sheet = Image.new('RGB', (1920, 1080)); [sheet.paste(t, ((i % 2) * 960, (i // 2) * 540)) for i, t in enumerate(tiles)]
    fn = os.path.join(a.out, f'{a.query.replace(":", "")}_{fr[:-4]}.jpg'); sheet.save(fn, quality=88)
    print(f'[sheet] {fr}: side-car {stats(al_s, gt)} | native {stats(al_n, gt)} | native cut {a.cut:g} {stats(al_c, gt)} -> {fn}', flush=True)
