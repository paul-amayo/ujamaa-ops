#!/usr/bin/env python3
"""Offline check of native identity serving (UJAMAA, 2026-10-03): build the service's objects (CompactHierarchy + the
scaffold tail, NativeIdentity), render a keyframe at its COLMAP pose like hier_render_service does, fire tree / row
queries through the exact serving path (half-res feature pass -> containment heat -> Otsu + absence guard), and report
the fired mask's IoU against the supervision, the time per query, and an overlay PNG per query.
h3dgs env (CUDA_HOME=/home/paperspace/code/_cuda12):
  python native_identity_check.py --proj <h3dgs project> --chunk 1_0 --features <bin> --embedder <pth> --bank <npz>
     --sup <supervision dir> --frame kf_002411.png --queries tree:36 tree:60 row:3 --out <dir> [--tau 3]"""
import argparse, json, math, os, struct, sys, time
import numpy as np, torch, cv2
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/splat_viewer')
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from hier_compact import CompactHierarchy
from native_identity import NativeIdentity
from PIL import Image
ap = argparse.ArgumentParser()
for k in ('--proj', '--chunk', '--features', '--embedder', '--bank', '--sup', '--frame', '--out'): ap.add_argument(k, required=True)
ap.add_argument('--queries', nargs='+', required=True, help='kind:id[@cut], e.g. tree:36 row:3@0.88')
ap.add_argument('--tau', type=float, default=3.0); ap.add_argument('--scaffold', default='')
ap.add_argument('--compare-npz', default='', help='side-car feature map of the same frame (sidecar_feature_dump.py): pushed through the identical heat + alpha code')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
src = f'{a.proj}/camera_calibration/chunks/{a.chunk}/sparse/0'
def read_cams(p):
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]; cid, model, w, h = struct.unpack('<iiQQ', f.read(24)); npar = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}[model]
        return w, h, struct.unpack('<' + 'd' * npar, f.read(8 * npar))
def read_pose(p, name):
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            iid = struct.unpack('<i', f.read(4)); q = struct.unpack('<4d', f.read(32)); t = struct.unpack('<3d', f.read(24)); f.read(4)
            nm = b''
            while (c := f.read(1)) != b'\x00': nm += c
            k = struct.unpack('<Q', f.read(8))[0]; f.read(24 * k)
            if nm.decode() == name: return np.array(q), np.array(t)
    raise SystemExit(f'{name} not in {p}')
W, H, prm = read_cams(f'{src}/cameras.bin'); fx, fy, cx, cy = prm[:4]
q, t = read_pose(f'{src}/images.bin', a.frame)
w_, x, y, z = q; R = np.array([[1 - 2*(y*y+z*z), 2*(x*y-z*w_), 2*(x*z+y*w_)], [2*(x*y+z*w_), 1 - 2*(x*x+z*z), 2*(y*z-x*w_)], [2*(x*z-y*w_), 2*(y*z+x*w_), 1 - 2*(x*x+y*y)]])
w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = t; c2w_h = np.linalg.inv(w2c)
class Cam:   # hier_render_service.Cam, verbatim
    def __init__(self, c2w_h, W, H, fovy, primx=0.5, primy=0.5):
        w2c = np.linalg.inv(c2w_h); R, T = c2w_h[:3, :3], w2c[:3, 3]
        self.image_width, self.image_height = W, H; self.FoVy = fovy; self.FoVx = 2 * math.atan(math.tan(fovy / 2) * W / H)
        self.primx, self.primy = primx, primy
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).float().transpose(0, 1).cuda()
        self.projection_matrix = torch.tensor(getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy, primx=primx, primy=primy)).float().transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
cam = Cam(c2w_h, W, H, 2 * math.atan(H / (2 * fy)), cx / W, cy / H)
t0 = time.time(); CH = CompactHierarchy(f'{a.proj}/output/trained_chunks/{a.chunk}/hierarchy.hier_opt', a.scaffold)
NI = NativeIdentity(a.features, a.embedder, a.bank, CH.N); print(f'[check] loaded in {time.time() - t0:.0f}s; CH {CH.N} gaussians (tail {CH.tail})', flush=True)
with torch.no_grad():
    im, n = CH.render(cam, a.tau); frame = (im.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
sup = np.array(Image.open(os.path.join(a.sup, a.frame)), np.uint16); sup = np.array(Image.fromarray(sup).resize((W, H), Image.NEAREST))
Hj = json.load(open(json.loads(str(np.load(a.bank)['meta']))['hierarchy_json'])); row_of = {o['id']: o['row_id'] for o in Hj['objects']}
res = []
for qs in a.queries:
    kind, rest = qs.split(':'); oid, cut = (rest.split('@') + [''])[:2]; oid = int(oid)
    ack = NI.set_query({kind: oid, 'cut': float(cut) if cut else 0}); torch.cuda.synchronize(); t1 = time.time()
    al = NI.mask(CH, cam, a.tau); torch.cuda.synchronize(); ms = (time.time() - t1) * 1000
    gt = (sup == oid) if kind == 'tree' else np.isin(sup, [k for k, r in row_of.items() if r == oid])
    fired = (al > 0) if al is not None else np.zeros_like(gt)
    iou = (fired & gt).sum() / max((fired | gt).sum(), 1); prec = (fired & gt).sum() / max(fired.sum(), 1); rec = (fired & gt).sum() / max(gt.sum(), 1)
    sc = ''
    if a.compare_npz:
        zf = torch.from_numpy(np.load(a.compare_npz)['features']).float().cuda()
        al_s = NI.alpha(NI.heat(zf, (H, W))); al_s = al_s.float().cpu().numpy() if al_s is not None else None
        fs = (al_s > 0) if al_s is not None else np.zeros_like(gt)
        sc = f' || side-car same path: IoU {(fs & gt).sum() / max((fs | gt).sum(), 1):.3f} prec {(fs & gt).sum() / max(fs.sum(), 1):.3f} rec {(fs & gt).sum() / max(gt.sum(), 1):.3f}'
    res.append({'query': qs, 'word': NI.word, 'iou': round(float(iou), 3), 'prec': round(float(prec), 3), 'rec': round(float(rec), 3), 'ms': round(ms, 1), 'fired_px': int(fired.sum()), 'gt_px': int(gt.sum()), 'sidecar': sc})
    print(f'[check] {a.frame} {qs} "{NI.word}": fired IoU {iou:.3f} prec {prec:.3f} rec {rec:.3f} | {fired.sum()} px fired vs {gt.sum()} GT | {ms:.0f} ms{sc}', flush=True)
    ov = frame.astype(np.float32)
    if al is not None: ov = ov * (1 - al[..., None]) + np.array([255, 122, 0], np.float32) * al[..., None]
    cv2.imwrite(os.path.join(a.out, f'{a.frame[:-4]}_{kind}{oid}.jpg'), ov.astype(np.uint8)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
json.dump(res, open(os.path.join(a.out, f'{a.frame[:-4]}_check.json'), 'w'), indent=1)
