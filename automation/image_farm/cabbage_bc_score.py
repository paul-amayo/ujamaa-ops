#!/usr/bin/env python3
"""Cabbage identity by BEST CONTAINMENT on all 72 lane frames (UJAMAA 2026-10-04): per frame, render (own exposure; held-out views
without), PSNR vs the TRAINING (undistorted) image, native feature pass, each pixel -> its best cabbage word (identity norm >= 0.5),
IoU per cabbage labelled >= 200 px in the UNDISTORTED SAM3 maps. usage: cabbage_bc_score.py <features.bin> <tag>  (h3dgs env)"""
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
FEAT, TAG = sys.argv[1], sys.argv[2]
D = '/home/paperspace/data/image_farm/gwakungu/2026-05-16/IMG_7993_s0_cabbage/demo_root'; P = '/home/paperspace/data/image_farm/gwakungu/2026-05-16/IMG_7993_s0/h3dgs_8m'; CN = 'lane'
NB = f'{D}/experimental/h3dgs_native/chunk_lane_ud'; SUP = f'{NB}/supervision/trees_only'
src = f'{P}/camera_calibration/chunks/{CN}/sparse/0'; cam0 = list(read_cameras_binary(f'{src}/cameras.bin').values())[0]; W, H = int(cam0.width), int(cam0.height); fx, fy, cx, cy = cam0.params[:4]
ims = {v.name: v for v in read_images_binary(f'{src}/images.bin').values()}; T = {l.strip() for l in open(f'{src}/test.txt') if l.strip()}
EX = json.load(open(f'{P}/output/trained_chunks/{CN}/exposure.json'))
class Cam:
    def __init__(self, c2w_h, W, H, fovy, primx=0.5, primy=0.5):
        w2c = np.linalg.inv(c2w_h); R_, T_ = c2w_h[:3, :3], w2c[:3, 3]
        self.image_width, self.image_height = W, H; self.FoVy = fovy; self.FoVx = 2 * math.atan(math.tan(fovy / 2) * W / H); self.primx, self.primy = primx, primy
        self.world_view_transform = torch.tensor(getWorld2View2(R_, T_)).float().transpose(0, 1).cuda()
        self.projection_matrix = torch.tensor(getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy, primx=primx, primy=primy)).float().transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
CH = CompactHierarchy(f'{P}/output/trained_chunks/{CN}/hierarchy.hier_opt', '')
NI = NativeIdentity(FEAT, json.load(open(FEAT + '.json'))['embedder'], f'{NB}/text_bank.npz', CH.N, n_hier=CH.n_hier)
IDS = sorted(int(k) for k in NI.word_table if int(k) < 10000 and NI.word_table[k] in NI.widx); WORDS = [NI.word_table[str(t)] for t in IDS]
def heats(feat, words):
    E = NI.E[[NI.widx[w] for w in words]]; h_, w_, Dd = feat.shape; ft = feat.reshape(-1, Dd).float()
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
for kf in sorted(ims, key=lambda n: int(''.join(c for c in n if c.isdigit()))):
    im = ims[kf]; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec
    cam = Cam(np.linalg.inv(w2c), W, H, 2 * math.atan(H / (2 * fy)), cx / W, cy / H)
    img, _ = CH.render(cam, 3.0)
    if kf in EX: E_ = np.array(EX[kf], np.float32); img = torch.einsum('ij,jhw->ihw', torch.tensor(E_[:, :3]).cuda(), img) + torch.tensor(E_[:, 3]).cuda()[:, None, None]
    r = (img.clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy().astype(np.float32)
    gt_img = np.asarray(Image.open(f'{P}/camera_calibration/rectified/images/{kf}').convert('RGB')).astype(np.float32)
    psnr = float(10 * np.log10(255 ** 2 / max(((r - gt_img) ** 2).mean(), 1e-9)))
    hm = heats(NI.feature_pass(CH, cam, 3.0), WORDS); valid = (hm > -1).any(0); am = hm.argmax(0)
    best = torch.where(valid, torch.tensor(IDS, device=am.device)[am], torch.full_like(am, -1)).cpu().numpy().astype(np.int32)
    best = np.array(Image.fromarray(best).resize((W, H), Image.NEAREST))
    sup = np.array(Image.open(f'{SUP}/{kf}'), np.uint16); ious = {}
    for t in [int(u) for u in np.unique(sup) if u < 10000]:
        g = sup == t
        if g.sum() >= 200: lit = best == t; ious[t] = float((lit & g).sum() / max((lit | g).sum(), 1))
    res.append({'kf': kf, 'held_out': kf in T, 'psnr': round(psnr, 2), 'ious': ious}); torch.cuda.empty_cache()
json.dump(res, open(Path(NB) / f'cabbage_bc_{TAG}.json', 'w'), indent=1)
for name, sel in (('all', res), ('trained', [r for r in res if not r['held_out']]), ('held-out', [r for r in res if r['held_out']])):
    v = [x for r in sel for x in r['ious'].values()]
    print(f'[cabbage-bc {TAG}] {name:8s}: {len(sel)} frames, {len(v)} cabbage-frame pairs, mean IoU {np.mean(v):.3f}, median {np.median(v):.3f}, >=0.5 on {np.mean(np.array(v) >= 0.5):.0%} | mean PSNR vs the training image {np.mean([r["psnr"] for r in sel]):.2f}', flush=True)
