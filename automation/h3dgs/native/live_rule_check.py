#!/usr/bin/env python3
"""The stage's LIVE identity rule vs SAM3 on specific frames (UJAMAA 2026-10-05; dashboard: "is 25 k px for tree 164 at 9.6 m the
honest extent (the side-car over-lit?)" and "can fruit light at all under the live Otsu/exclusive rule ... at close range").
Per frame: NativeIdentity.set_query + NativeIdentity.mask exactly as hier_render_service calls them (no cut = per-frame Otsu,
exclusive scoring for fruit), lit = alpha > 0, against the cell's SAM3 supervision for that id; plus the cuts' best-containment rule
for comparison. usage: live_rule_check.py <survey> <chunk> <cell> <tree|fruit> <id> <kf,kf,...>   (h3dgs env, CUDA_HOME)"""
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
SV, CN, CELL, KIND, OID, KFS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6].split(',')
S = f'/home/paperspace/data/citrus_all/{SV}'; P = f'{S}/experimental/{"h3dgs" if SV.startswith("01") else "h3dgs_expo"}'; NB = f'{S}/experimental/h3dgs_native/{CELL}'
FEAT = f'{NB}/features_B_bg2share.bin'; SUP = f'{NB}/supervision/trees_only'
src = f'{P}/camera_calibration/chunks/{CN}/sparse/0'; c0 = list(read_cameras_binary(f'{src}/cameras.bin').values())[0]; W, H = int(c0.width), int(c0.height); fx, fy, cx, cy = c0.params[:4]
ims = {v.name: v for v in read_images_binary(f'{src}/images.bin').values()}
class Cam:
    def __init__(self, c2w_h):
        w2c = np.linalg.inv(c2w_h); R_, T_ = c2w_h[:3, :3], w2c[:3, 3]
        self.image_width, self.image_height = W, H; self.FoVx = 2 * math.atan(W / (2 * fx)); self.FoVy = 2 * math.atan(H / (2 * fy)); self.primx, self.primy = cx / W, cy / H
        self.world_view_transform = torch.tensor(getWorld2View2(R_, T_)).float().transpose(0, 1).cuda()
        self.projection_matrix = torch.tensor(getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy, primx=self.primx, primy=self.primy)).float().transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0); self.camera_center = self.world_view_transform.inverse()[3, :3]
CH = CompactHierarchy(f'{P}/output/trained_chunks/{CN}/hierarchy.hier_opt', f'{P}/output/scaffold/point_cloud/iteration_30000')
NI = NativeIdentity(FEAT, json.load(open(FEAT + '.json'))['embedder'], f'{NB}/text_bank.npz', CH.N, n_hier=CH.n_hier)
ack = NI.set_query({KIND: OID}); print('[live] query ack:', {k: ack[k] for k in ack if k in ('ok', 'word', 'reason', 'cut')}, flush=True)
word = NI.word_table[str(OID)]; ALLW = [w for w in NI.words if w in NI.widx]
TREES = [w for k, w in NI.word_table.items() if int(k) < 10000 and w in NI.widx]
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
for kf in KFS:
    im = ims[kf]; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec; cam = Cam(np.linalg.inv(w2c))
    a = NI.mask(CH, cam, 3.0); live = (np.asarray(a) > 0) if a is not None else np.zeros((H, W), bool)
    if live.shape != (H, W): live = np.array(Image.fromarray(live.astype(np.uint8)).resize((W, H), Image.NEAREST)) > 0
    words = ALLW if KIND == 'fruit' else TREES; hm = heats(NI.feature_pass(CH, cam, 3.0), words); valid = (hm > -1).any(0)
    bc = (valid & (hm.argmax(0) == words.index(word))).cpu().numpy(); bc = np.array(Image.fromarray(bc.astype(np.uint8)).resize((W, H), Image.NEAREST)) > 0
    sup = np.array(Image.open(f'{SUP}/{kf}'), np.uint16); g = sup == OID
    iou = lambda m: float((m & g).sum() / max((m | g).sum(), 1)) if g.sum() else float('nan')
    print(f'[live] {kf}: SAM3 {int(g.sum()):,} px | live rule lit {int(live.sum()):,} px (IoU {iou(live):.2f}, precision {float((live & g).sum() / max(live.sum(), 1)):.2f}) '
          f'| best containment lit {int(bc.sum()):,} px (IoU {iou(bc):.2f})', flush=True)
    torch.cuda.empty_cache()
