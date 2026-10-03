#!/usr/bin/env python3
"""Fruit containment on the demo's fruit clip (UJAMAA, 2026-10-03; Paul: "and the fruit containment, measure this").
Frames = the clip's 55 keyframes (05 chunk 1_0 drive frames 13-52, 346-360). Ground truth = tree 5's SAM3 fruit
(id 10001) in the chunk-specific supervision. Two lightings:
  clip    the video as made: the densified v3 fruit side-car's maps, lit where the fruit heat clears that fruit's
          VERDICT threshold (fitted against the SAM3 mask on its verdict frame) — the fitted-cut mechanism
  native  seed B (chunk supervision, no side-car) through the live serving path with NO cut: exclusive fruit scoring +
          per-frame Otsu / absence guard (native_identity.py)
Per frame: lit px, IoU, precision, recall (fruit px within 1 px of a lit px), oranges lit, and where off-fruit light lands."""
import json, math, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy import ndimage
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/splat_viewer'); sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians/preprocess')
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from hier_compact import CompactHierarchy
from native_identity import NativeIdentity
from read_write_model import read_images_binary, read_cameras_binary, qvec2rotmat
S = '/home/paperspace/data/citrus_all/05_13D_Jackal'; P = f'{S}/experimental/h3dgs_expo'; N = f'{S}/experimental/h3dgs_native/chunk_1_0_sam3'
R = Path('/home/paperspace/logs/demo_chunks/05_1_0_720'); FID, TREE = 10001, 5
path = json.load(open(R / 'demo_path.json'))['frames']; clip = [path[i] for i in list(range(13, 53)) + list(range(346, 361))]
src = f'{P}/camera_calibration/chunks/1_0/sparse/0'; cam0 = list(read_cameras_binary(f'{src}/cameras.bin').values())[0]; W, H = int(cam0.width), int(cam0.height); fx, fy, cx, cy = cam0.params[:4]
ims = {v.name: v for v in read_images_binary(f'{src}/images.bin').values()}; T = {l.strip() for l in open(f'{src}/test.txt') if l.strip()}
class Cam:
    def __init__(self, c2w_h, W, H, fovy, primx=0.5, primy=0.5):
        w2c = np.linalg.inv(c2w_h); R_, T_ = c2w_h[:3, :3], w2c[:3, 3]
        self.image_width, self.image_height = W, H; self.FoVy = fovy; self.FoVx = 2 * math.atan(math.tan(fovy / 2) * W / H); self.primx, self.primy = primx, primy
        self.world_view_transform = torch.tensor(getWorld2View2(R_, T_)).float().transpose(0, 1).cuda()
        self.projection_matrix = torch.tensor(getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy, primx=primx, primy=primy)).float().transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
CH = CompactHierarchy(f'{P}/output/trained_chunks/1_0/hierarchy.hier_opt', f'{P}/output/scaffold/point_cloud/iteration_30000')
NI = NativeIdentity(f'{N}/features_B_bg2share.bin', f'{S}/prod/bateleur/embedder/05_13D_v1g/ckpts/model_best.pth', f'{N}/text_bank.npz', CH.N, n_hier=CH.n_hier)
def stats(lit, sup):
    gt = sup == FID; d = ndimage.binary_dilation(lit, iterations=1); n = int(lit.sum())
    lab, k = ndimage.label(gt, structure=np.ones((3, 3))); hit = sum(1 for i in range(1, k + 1) if (d & (lab == i)).any())
    off = lit & ~ndimage.binary_dilation(gt, iterations=1)
    return dict(gt=int(gt.sum()), lit=n, iou=float((lit & gt).sum() / max((lit | gt).sum(), 1)), prec=float((lit & gt).sum() / max(n, 1)),
                rec1=float((gt & d).sum() / max(gt.sum(), 1)), oranges=k, hit=hit, off_tree5=int((off & (sup == TREE)).sum()),
                off_other_tree=int((off & (sup < 10000) & (sup != TREE)).sum()), off_unlabelled=int((off & (sup == 65535)).sum()))
res = {'clip': [], 'native': []}
for f in clip:
    kf = f['src']; sup = np.array(Image.open(f'{N}/supervision/trees_only/{kf}'), np.uint16)
    if (sup == FID).sum() == 0: continue
    z = np.load(R / 'maps_fruit/chunk_1_0_expo' / (f['name'] + '.npz')); ft = z['ftree'][z['fid']]
    lit_c = (z['fmg'].astype(np.float32) > 0) & (ft == TREE)
    s2 = np.array(Image.fromarray(sup).resize((lit_c.shape[1], lit_c.shape[0]), Image.NEAREST)) if sup.shape != lit_c.shape else sup
    res['clip'].append(dict(kf=kf, held_out=kf in T, **stats(lit_c, s2)))
    im = ims[kf]; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec
    cam = Cam(np.linalg.inv(w2c), W, H, 2 * math.atan(H / (2 * fy)), cx / W, cy / H); NI.set_query({'fruit': FID})
    with torch.no_grad(): al = NI.mask(CH, cam, 3.0)
    lit_n = (al > 0) if al is not None else np.zeros(sup.shape, bool)
    res['native'].append(dict(kf=kf, held_out=kf in T, **stats(lit_n, sup)))
json.dump(res, open('/home/paperspace/data/demo_video_v2/psnr/fruit_clip_containment.json', 'w'), indent=1)
for k, v in res.items():
    a = lambda key, sel=None: np.mean([x[key] for x in v if sel is None or x['held_out'] == sel])
    tot = lambda key: sum(x[key] for x in v)
    print(f'[fruit] {k:6s}: {len(v)} frames with tree-5 fruit | IoU mean {a("iou"):.3f} (trained {a("iou", False):.3f}, held out {a("iou", True):.3f}) | precision {a("prec"):.3f} | fruit px within 1 px of light {a("rec1"):.3f} | '
          f'oranges lit {tot("hit")}/{tot("oranges")} | frames lighting nothing {sum(1 for x in v if x["lit"] == 0)} | off-fruit light: tree 5 canopy {tot("off_tree5")} px, other trees {tot("off_other_tree")} px, unlabelled {tot("off_unlabelled")} px', flush=True)
