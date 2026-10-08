#!/usr/bin/env python3
"""Fruit by BEST CONTAINMENT on the Citrus B fruit clip (05 chunk 1_0 drive frames 13-52, 346-360), for any H3DGS project +
native cell (UJAMAA 2026-10-04): per frame, render (own exposure, mean for held-out views), PSNR vs the photo, the native
feature pass, the pixels where fruit 10001's word beats every other bank word, IoU vs the cell's SAM3 fruit 10001.
usage: fruit_bc_score.py <project dir> <native cell dir> <tag>   (h3dgs env, CUDA_HOME=_cuda12)"""
import json, math, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/splat_viewer'); sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians/preprocess')
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from hier_compact import CompactHierarchy
from native_identity import NativeIdentity
import lorentz as L
from read_write_model import read_images_binary, read_cameras_binary, qvec2rotmat
torch.set_grad_enabled(False)
import os; P, NB, TAG = sys.argv[1], sys.argv[2], sys.argv[3]; TAU = float(os.environ.get('TAU', '3.0')); FEAT_OVERRIDE = os.environ.get('FEAT'); S = '/home/paperspace/data/citrus_all/05_13D_Jackal'; CN = '1_0'; FID = 10001
src = f'{P}/camera_calibration/chunks/{CN}/sparse/0'; cam0 = list(read_cameras_binary(f'{src}/cameras.bin').values())[0]; W, H = int(cam0.width), int(cam0.height); fx, fy, cx, cy = cam0.params[:4]
ims = {v.name: v for v in read_images_binary(f'{src}/images.bin').values()}; TEST = {l.strip() for l in open(f'{src}/test.txt') if l.strip()}
EXPO = json.load(open(f'{P}/output/trained_chunks/{CN}/exposure.json')); MEAN = np.mean([np.array(v, np.float32) for v in EXPO.values()], axis=0)
path = json.load(open('/home/paperspace/logs/demo_chunks/05_1_0_720/demo_path.json'))['frames']; clip = [path[i] for i in list(range(13, 53)) + list(range(346, 361))]
class Cam:
    def __init__(self, c2w_h, W, H, fovy, primx=0.5, primy=0.5):
        w2c = np.linalg.inv(c2w_h); R_, T_ = c2w_h[:3, :3], w2c[:3, 3]
        self.image_width, self.image_height = W, H; self.FoVy = fovy; self.FoVx = 2 * math.atan(math.tan(fovy / 2) * W / H); self.primx, self.primy = primx, primy
        self.world_view_transform = torch.tensor(getWorld2View2(R_, T_)).float().transpose(0, 1).cuda()
        self.projection_matrix = torch.tensor(getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy, primx=primx, primy=primy)).float().transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
CH = CompactHierarchy(f'{P}/output/trained_chunks/{CN}/hierarchy.hier_opt', f'{P}/output/scaffold/point_cloud/iteration_30000')
NI = NativeIdentity(FEAT_OVERRIDE or f'{NB}/features_B_bg2share.bin', sorted(Path(f'{S}/prod/bateleur/embedder').glob('*/ckpts/model_best.pth'))[-1].as_posix(), f'{NB}/text_bank.npz', CH.N, n_hier=CH.n_hier)
ALLW = [w for w in NI.words if w in NI.widx]; FW = NI.word_table.get(str(FID)); assert FW in ALLW, 'fruit word missing'
def heats(feat, words):
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
res = []
for f in clip:
    kf = f['src']
    if kf not in ims: continue
    im = ims[kf]; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec
    cam = Cam(np.linalg.inv(w2c), W, H, 2 * math.atan(H / (2 * fy)), cx / W, cy / H); E_ = np.array(EXPO[kf], np.float32) if kf in EXPO else MEAN
    img, _ = CH.render(cam, TAU); img = torch.einsum('ij,jhw->ihw', torch.tensor(E_[:, :3]).cuda(), img) + torch.tensor(E_[:, 3]).cuda()[:, None, None]
    rgb = (img.clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy().astype(np.float32)
    photo = np.asarray(Image.open(f'{S}/prod/scratch_sam3/{kf}').convert('RGB')).astype(np.float32); psnr = float(10 * np.log10(255 ** 2 / max(((rgb - photo) ** 2).mean(), 1e-9)))
    hm = heats(NI.feature_pass(CH, cam, TAU), ALLW); valid = (hm > -1).any(0); am = hm.argmax(0)
    lit = (valid & (am == ALLW.index(FW))).cpu().numpy().astype(np.uint8); lit = np.array(Image.fromarray(lit * 255).resize((W, H), Image.NEAREST)) > 0
    sup = np.array(Image.open(f'{NB}/supervision/trees_only/{kf}'), np.uint16); gt = sup == FID
    iou = float((lit & gt).sum() / max((lit | gt).sum(), 1)) if gt.sum() > 0 else None
    res.append({'kf': kf, 'held_out': kf in TEST, 'psnr': round(psnr, 2), 'gt': int(gt.sum()), 'lit': int(lit.sum()), 'iou': iou}); torch.cuda.empty_cache()
out = Path(NB) / f'fruit_bc_{TAG}.json'; json.dump(res, open(out, 'w'), indent=1)
g = [r for r in res if r['iou'] is not None]
for name, sel in (('all', g), ('held-out', [r for r in g if r['held_out']]), ('trained', [r for r in g if not r['held_out']])):
    if sel: print(f'[fruit-bc {TAG}] {name:9s}: {len(sel)} frames with SAM3 fruit, mean IoU {np.mean([r["iou"] for r in sel]):.3f}, mean PSNR {np.mean([r["psnr"] for r in sel]):.2f}', flush=True)
print(f'[fruit-bc {TAG}] all clip frames mean PSNR {np.mean([r["psnr"] for r in res]):.2f} (trained {np.mean([r["psnr"] for r in res if not r["held_out"]]):.2f}, held-out {np.mean([r["psnr"] for r in res if r["held_out"]]):.2f}) -> {out}', flush=True)
