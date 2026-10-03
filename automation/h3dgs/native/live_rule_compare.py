#!/usr/bin/env python3
"""Native vs the live side-car under the LIVE query rule (no client cut -> per-frame Otsu + absence guard), the
mechanism the stage actually runs (UJAMAA, 2026-10-03). For each query: where the lit pixels land, by supervision class
(the object asked for, other labelled objects, unlabelled), for the native hierarchy render and for the side-car's
own half-res map pushed through the identical heat + alpha code.
h3dgs env: python live_rule_compare.py --proj P --chunk C --features F --embedder E --bank B --sup SUP --hj HJ
             --frame kf.png --sidecar-npz Z --queries tree:5 fruit:10001 row:7 [--scaffold S]"""
import argparse, json, math, os, struct, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/splat_viewer')
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from hier_compact import CompactHierarchy
from native_identity import NativeIdentity
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians/preprocess')
from read_write_model import read_images_binary, read_cameras_binary, qvec2rotmat
ap = argparse.ArgumentParser()
for k in ('--proj', '--chunk', '--features', '--embedder', '--bank', '--sup', '--hj', '--frame', '--sidecar-npz'): ap.add_argument(k, required=True)
ap.add_argument('--queries', nargs='+', required=True); ap.add_argument('--scaffold', default=''); ap.add_argument('--tau', type=float, default=3.0)
a = ap.parse_args()
src = f'{a.proj}/camera_calibration/chunks/{a.chunk}/sparse/0'
cam0 = list(read_cameras_binary(f'{src}/cameras.bin').values())[0]; W, H = int(cam0.width), int(cam0.height); fx, fy, cx, cy = cam0.params[:4]
im = next(v for v in read_images_binary(f'{src}/images.bin').values() if v.name == a.frame)
w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec; c2w_h = np.linalg.inv(w2c)
class Cam:   # hier_render_service.Cam
    def __init__(self, c2w_h, W, H, fovy, primx=0.5, primy=0.5):
        w2c = np.linalg.inv(c2w_h); R, T = c2w_h[:3, :3], w2c[:3, 3]
        self.image_width, self.image_height = W, H; self.FoVy = fovy; self.FoVx = 2 * math.atan(math.tan(fovy / 2) * W / H); self.primx, self.primy = primx, primy
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).float().transpose(0, 1).cuda()
        self.projection_matrix = torch.tensor(getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy, primx=primx, primy=primy)).float().transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
cam = Cam(c2w_h, W, H, 2 * math.atan(H / (2 * fy)), cx / W, cy / H)
CH = CompactHierarchy(f'{a.proj}/output/trained_chunks/{a.chunk}/hierarchy.hier_opt', a.scaffold)
NI = NativeIdentity(a.features, a.embedder, a.bank, CH.N, n_hier=CH.n_hier)
sup = np.array(Image.open(os.path.join(a.sup, a.frame)), np.uint16); sup = np.array(Image.fromarray(sup).resize((W, H), Image.NEAREST))
Hj = json.load(open(a.hj)); row_of = {o['id']: o['row_id'] for o in Hj['objects']}; tree_of_fruit = {10000 + f['id']: f['tree_id'] for f in Hj.get('fruits', [])}
sc = torch.from_numpy(np.load(a.sidecar_npz)['features']).float().cuda()
def target(kind, oid):
    if kind == 'tree': return (sup == oid) | np.isin(sup, [f for f, t in tree_of_fruit.items() if t == oid])
    if kind == 'fruit': return sup == oid
    trees = [k for k, r in row_of.items() if r == oid]
    return np.isin(sup, trees + [f for f, t in tree_of_fruit.items() if t in trees])
def breakdown(al, tg):
    if al is None: return 'nothing lit (absence guard)'
    lit = al > 0; n = int(lit.sum()); labelled = sup != 65535
    on_t = int((lit & tg).sum()); other = int((lit & labelled & ~tg).sum()); unl = int((lit & ~labelled).sum())
    per = {int(u): int((lit & (sup == u)).sum()) for u in np.unique(sup[lit]) if u != 65535}
    top = ', '.join(f'{u}: {c}' for u, c in sorted(per.items(), key=lambda kv: -kv[1])[:4])
    return f'lit {n:7d} px [{top}] | on the object {on_t / max(n, 1):5.1%} | other objects {other / max(n, 1):5.1%} | unlabelled {unl / max(n, 1):5.1%} | object covered {on_t / max(tg.sum(), 1):5.1%}'
print(f'[live-rule] {a.frame}: labels present {sorted(int(u) for u in np.unique(sup) if u != 65535)}')
for qs in a.queries:
    excl = qs.endswith('!x'); q0 = qs[:-2] if excl else qs
    kind, rest = q0.split(':'); oid, cut = (rest.split('@') + [''])[:2]; oid = int(oid)
    ack = NI.set_query({kind: oid, 'cut': float(cut) if cut else 0})
    if excl and NI.word: NI.competitors = [w for w in NI.words if w != NI.word]   # exclusive against every other word in the bank
    tg = target(kind, oid)
    with torch.no_grad():
        al_n = NI.mask(CH, cam, a.tau)
        al_s = NI.alpha(NI.heat(sc, (H, W))) if NI.word else None
        if al_s is not None: al_s = al_s.float().cpu().numpy()
    print(f'[live-rule] {qs:14s} word "{NI.word}"\n      native   {breakdown(al_n, tg)}\n      side-car {breakdown(al_s, tg)}', flush=True)
