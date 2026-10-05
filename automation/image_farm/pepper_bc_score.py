#!/usr/bin/env python3
"""Class-level pepper identity scored by BEST CONTAINMENT (UJAMAA 2026-10-05): chilli segment IMG_7990_s1, cell
experimental/pepper_class/cell. Per frame: native feature pass at tau 3, each pixel -> the better of the two words (asus = plant /
other, bumper = pepper; identity norm >= 0.5), pepper pixels vs the UNDISTORTED tiled-SAM3 pepper map (65535 ignored). IoU,
precision, recall per frame; pooled precision / recall; split into training views 0-141, held-out views 0-141 and frames 142-240
(collapsed SfM poses: reported, not trusted). FoVx from fx (bar_walk.py's fix). usage: pepper_bc_score.py <features.bin> <tag> (h3dgs env)"""
import json, math, re, sys
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
S = '/home/paperspace/data/image_farm/gwakungu/2026-05-16/IMG_7990_s1'; P = f'{S}/h3dgs'; CN = 'lane'; NB = f'{S}/experimental/pepper_class/' + __import__('os').environ.get('CELL', 'cell'); SUP = f'{NB}/supervision/pepper_class'
src = f'{P}/camera_calibration/chunks/{CN}/sparse/0'; cam0 = list(read_cameras_binary(f'{src}/cameras.bin').values())[0]; W, H = int(cam0.width), int(cam0.height); fx, fy, cx, cy = cam0.params[:4]
ims = {v.name: v for v in read_images_binary(f'{src}/images.bin').values()}; T = {l.strip() for l in open(f'{src}/test.txt') if l.strip()}
class Cam:
    def __init__(self, c2w_h):
        w2c = np.linalg.inv(c2w_h); R_, T_ = c2w_h[:3, :3], w2c[:3, 3]
        self.image_width, self.image_height = W, H; self.FoVx = 2 * math.atan(W / (2 * fx)); self.FoVy = 2 * math.atan(H / (2 * fy)); self.primx, self.primy = cx / W, cy / H
        self.world_view_transform = torch.tensor(getWorld2View2(R_, T_)).float().transpose(0, 1).cuda()
        self.projection_matrix = torch.tensor(getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy, primx=self.primx, primy=self.primy)).float().transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
CH = CompactHierarchy(f'{P}/output/trained_chunks/{CN}/hierarchy.hier_opt', '')
NI = NativeIdentity(FEAT, json.load(open(FEAT + '.json'))['embedder'], f'{NB}/text_bank.npz', CH.N, n_hier=CH.n_hier)
WORDS = [NI.word_table['0'], NI.word_table['10000']]
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
for kf in sorted(ims, key=lambda n: int(re.sub(r'\D', '', n))):
    im = ims[kf]; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec; cam = Cam(np.linalg.inv(w2c))
    feat = NI.feature_pass(CH, cam, 3.0); hm = heats(feat, WORDS); valid = (hm > -1).any(0)
    # two decision rules (2026-10-05): 'walk' = best containment (own point + 4-step walk toward the root, max) - the walk passes the
    # plant (parent) node, so plant wins on pods; 'own' = the pixel's own decoded word only (argmax of the two words)
    h_, w_, Dd = feat.shape; ft = feat.reshape(-1, Dd).float(); E = NI.E[[NI.widx[w] for w in WORDS]]
    d0 = NI.hyper.decode_features(L.exp_map0(ft, curv=NI.curv), project=True); d0 = d0 / d0.norm(dim=-1, keepdim=True)
    own = ((d0 @ E.T).argmax(1) == 1).view(h_, w_) & (ft.norm(dim=-1) >= 0.5).view(h_, w_)
    k = int(re.sub(r'\D', '', kf)); grp = 'broken poses 142-240' if k > 141 else ('held-out 0-141' if kf in T else 'trained 0-141')
    sup = np.array(Image.open(f'{SUP}/{kf}'), np.uint16); ok = sup != 65535; g = (sup == 10000) & ok; row = {'kf': kf, 'group': grp, 'gt': int(g.sum())}
    for rule, m in (('walk', valid & (hm.argmax(0) == 1)), ('own', own)):
        pep = np.array(Image.fromarray(m.cpu().numpy().astype(np.uint8)).resize((W, H), Image.NEAREST)) > 0; lit = pep & ok
        row[rule] = {'lit': int(lit.sum()), 'tp': int((lit & g).sum()), 'iou': float((lit & g).sum() / max((lit | g).sum(), 1)) if g.sum() >= 200 else None}
    res.append(row)
    torch.cuda.empty_cache()
json.dump(res, open(f'{NB}/pepper_bc_{TAG}.json', 'w'), indent=1)
for rule in ('walk', 'own'):
    for grp in ('trained 0-141', 'held-out 0-141', 'broken poses 142-240'):
        r = [x for x in res if x['group'] == grp]; s = [x[rule]['iou'] for x in r if x[rule]['iou'] is not None]
        tp, lit, gt = sum(x[rule]['tp'] for x in r), sum(x[rule]['lit'] for x in r), sum(x['gt'] for x in r)
        print(f"[pepper {TAG} {rule}] {grp:22s}: {len(r)} frames, {len(s)} with >= 200 px SAM3 pepper: mean IoU {np.mean(s) if s else float('nan'):.3f} "
              f"(median {np.median(s) if s else float('nan'):.3f}); pooled precision {tp / max(lit, 1):.3f}, recall {tp / max(gt, 1):.3f}; lit {lit:,} px vs SAM3 {gt:,} px", flush=True)
