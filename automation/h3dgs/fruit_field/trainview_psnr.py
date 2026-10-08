"""Training-view PSNR the way the walkthroughs render (own exposure from exposure.json, CompactHierarchy at tau 3, full frame, sky in)
for one or more H3DGS projects' chunk: 60 evenly spaced trained views. usage: trainview_psnr.py <chunk> <label=project> [label=project ...]"""
import json, math, os, re, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/splat_viewer'); sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians/preprocess')
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from hier_compact import CompactHierarchy
from read_write_model import read_images_binary, read_cameras_binary, qvec2rotmat
CN = sys.argv[1]; runs = [a.split('=', 1) for a in sys.argv[2:]]
for lab, P in runs:
    src = f'{P}/camera_calibration/chunks/{CN}/sparse/0'; c0 = list(read_cameras_binary(f'{src}/cameras.bin').values())[0]; W, H = int(c0.width), int(c0.height); fx, fy, cx, cy = c0.params[:4]
    ims = {v.name: v for v in read_images_binary(f'{src}/images.bin').values()}; TEST = {l.strip() for l in open(f'{src}/test.txt') if l.strip()}; EXPO = json.load(open(f'{P}/output/trained_chunks/{CN}/exposure.json'))
    trained = sorted((n for n in ims if n not in TEST and n in EXPO), key=lambda n: int(re.sub(r'\D', '', n) or 0)); pick = trained[::max(1, len(trained) // 60)][:60]
    if os.environ.get('VIEWS'): pick = [l.strip() for l in open(os.environ['VIEWS']) if l.strip() in ims and l.strip() in EXPO]   # VIEWS=<file of kf names>: score exactly these views
    TAUV = float(os.environ.get('TAU', '3'))
    import os; hp = f"{P}/output/trained_chunks/{CN}/{os.environ.get('HIER', 'hierarchy.hier_opt')}"; CH = CompactHierarchy(hp, f'{P}/output/scaffold/point_cloud/iteration_30000'); ps = []
    class Cam:
        def __init__(self, c2w_h):
            w2c = np.linalg.inv(c2w_h); R_, T_ = c2w_h[:3, :3], w2c[:3, 3]; self.image_width, self.image_height = W, H; self.FoVx = 2 * math.atan(W / (2 * fx)); self.FoVy = 2 * math.atan(H / (2 * fy)); self.primx, self.primy = cx / W, cy / H
            self.world_view_transform = torch.tensor(getWorld2View2(R_, T_)).float().transpose(0, 1).cuda(); self.projection_matrix = torch.tensor(getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy, primx=self.primx, primy=self.primy)).float().transpose(0, 1).cuda()
            self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0); self.camera_center = self.world_view_transform.inverse()[3, :3]
    with torch.no_grad():
        for kf in pick:
            im = ims[kf]; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec; cam = Cam(np.linalg.inv(w2c)); E_ = np.array(EXPO[kf], np.float32)
            img, _ = CH.render(cam, TAUV); img = torch.einsum('ji,jhw->ihw', torch.tensor(E_[:, :3]).cuda(), img) + torch.tensor(E_[:, 3]).cuda()[:, None, None]   # H3DGS convention: pixel-row x E (gaussian_renderer use_trained_exp), i.e. E^T on column vectors
            gt = torch.from_numpy(np.array(Image.open(f'{P}/camera_calibration/rectified/images/{kf}').convert('RGB'))).float().permute(2, 0, 1).cuda() / 255
            mse = float(((img.clamp(0, 1) - gt) ** 2).mean()); ps.append(10 * math.log10(1 / max(mse, 1e-10)))
    print(f'[trainview] {lab:28s} chunk {CN}: {len(ps)} trained views, own exposure, tau {TAUV:g}: PSNR mean {np.mean(ps):.2f} median {np.median(ps):.2f} (nodes {CH.N:,})', flush=True); del CH; torch.cuda.empty_cache()
