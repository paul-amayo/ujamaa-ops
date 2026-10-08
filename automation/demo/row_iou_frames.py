#!/usr/bin/env python3
"""One row, consecutive keyframes: why does the no-cut IoU dip below 0.8 between passing frames? (UJAMAA, 2026-10-03)
For each keyframe: native seed B, plain containment heat for the row word, the live rule's per-frame split (Otsu +
absence guard, no cut) -> lit mask; against the SAM3 row mask (the row's trees in the chunk supervision): IoU, precision,
recall, the split value, where false light lands (other rows' trees / unlabelled), where misses sit (by tree of the row).
Writes a contact sheet (correct = green, false light = orange, missed = red) and a CSV."""
import csv, json, math, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/splat_viewer'); sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians/preprocess')
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from hier_compact import CompactHierarchy
from native_identity import NativeIdentity, _otsu
import lorentz as L
from read_write_model import read_images_binary, read_cameras_binary, qvec2rotmat
torch.set_grad_enabled(False)
ROW = int(sys.argv[1]); K0, K1 = int(sys.argv[2]), int(sys.argv[3])
S = '/home/paperspace/data/citrus_all/01_13B_Jackal'; P = f'{S}/experimental/h3dgs'; CN = '3_1'; NB = f'{S}/experimental/h3dgs_native/chunk_3_1_sam3'
OUT = Path(f'/home/paperspace/data/demo_video_v2/citrus_a_bar/row{ROW}_kf{K0}-{K1}'); OUT.mkdir(parents=True, exist_ok=True)
src = f'{P}/camera_calibration/chunks/{CN}/sparse/0'; cam0 = list(read_cameras_binary(f'{src}/cameras.bin').values())[0]; W, H = int(cam0.width), int(cam0.height); fx, fy, cx, cy = cam0.params[:4]
ims = {v.name: v for v in read_images_binary(f'{src}/images.bin').values()}; TEST = {l.strip() for l in open(f'{src}/test.txt') if l.strip()}
EXPO = json.load(open(f'{P}/output/trained_chunks/{CN}/exposure.json'))
row_of = {o['id']: o['row_id'] for o in json.load(open(f'{S}/prod/bateleur/scene_graph/marker_hierarchy.json'))['objects']}
ROW_TREES = [t for t, r in row_of.items() if r == ROW]
class Cam:
    def __init__(self, c2w_h, W, H, fovy, primx=0.5, primy=0.5):
        w2c = np.linalg.inv(c2w_h); R_, T_ = c2w_h[:3, :3], w2c[:3, 3]
        self.image_width, self.image_height = W, H; self.FoVy = fovy; self.FoVx = 2 * math.atan(math.tan(fovy / 2) * W / H); self.primx, self.primy = primx, primy
        self.world_view_transform = torch.tensor(getWorld2View2(R_, T_)).float().transpose(0, 1).cuda()
        self.projection_matrix = torch.tensor(getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy, primx=primx, primy=primy)).float().transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
CH = CompactHierarchy(f'{P}/output/trained_chunks/{CN}/hierarchy.hier_opt', f'{P}/output/scaffold/point_cloud/iteration_30000')
NI = NativeIdentity(f'{NB}/features_B_bg2share.bin', sorted(Path(f'{S}/prod/bateleur/embedder').glob('*/ckpts/model_best.pth'))[-1].as_posix(), f'{NB}/text_bank.npz', CH.N, n_hier=CH.n_hier)
word = NI.row_words[str(ROW)]; E = NI.E[[NI.widx[word]]]
def heat(feat):
    h_, w_, D = feat.shape; ft = feat.reshape(-1, D).float(); out = torch.full((ft.shape[0],), -1.0, device=ft.device); idx = (ft.norm(dim=-1) >= 0.5).nonzero(as_tuple=True)[0]
    for j in range(0, int(idx.shape[0]), 8192):
        sel = idx[j:j + 8192]; hh = L.exp_map0(ft[sel], curv=NI.curv)
        pth, msk = L.get_interpolated_hyperbolic_features(hh, steps=4, curv=NI.curv, max_dist=11.1, return_mask=True)
        d = NI.hyper.decode_features(pth, project=True); d = d / d.norm(dim=-1, keepdim=True)
        sm = (d @ E.T).view(sel.shape[0], 4).masked_fill(msk.to(d.device).bool(), -1.0)
        d0 = NI.hyper.decode_features(hh, project=True); d0 = d0 / d0.norm(dim=-1, keepdim=True)
        out[sel] = torch.maximum(sm.max(1).values, (d0 @ E.T).squeeze(-1))
    return torch.nn.functional.interpolate(out.view(1, 1, h_, w_), size=(H, W), mode='nearest')[0, 0]
try: F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 30)
except Exception: F = ImageFont.load_default()
rows, tiles = [], []
for k in range(K0, K1 + 1):
    kf = f'kf_{k:06d}.png'
    if kf not in ims: continue
    im = ims[kf]; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec
    cam = Cam(np.linalg.inv(w2c), W, H, 2 * math.atan(H / (2 * fy)), cx / W, cy / H)
    img, _ = CH.render(cam, 3.0)
    if kf in EXPO:
        E_ = np.array(EXPO[kf], np.float32); img = torch.einsum('ji,jhw->ihw', torch.tensor(E_[:, :3]).cuda(), img) + torch.tensor(E_[:, 3]).cuda()[:, None, None]
    rgb = (img.clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy()
    hm = heat(NI.feature_pass(CH, cam, 3.0)); valid = hm > -1; v = hm[valid]
    p50, p99 = float(torch.quantile(v, 0.5)), float(torch.quantile(v, 0.99)); ot = _otsu(v); thr = max(ot, p50); absent = (p99 - p50) < 0.04
    a = None if absent else NI.alpha(hm); lit = (a.float().cpu().numpy() > 0) if a is not None else np.zeros((H, W), bool)
    sup = np.array(Image.open(f'{NB}/supervision/trees_only/{kf}'), np.uint16)
    gt = np.isin(sup, ROW_TREES); tp = lit & gt; fp = lit & ~gt; fn = gt & ~lit
    iou = tp.sum() / max((lit | gt).sum(), 1); prec = tp.sum() / max(lit.sum(), 1); rec = tp.sum() / max(gt.sum(), 1)
    fp_other = int((fp & (sup < 10000) & ~np.isin(sup, ROW_TREES)).sum()); fp_unl = int((fp & (sup == 65535)).sum())
    miss_by_tree = {int(t): int((fn & (sup == t)).sum()) for t in ROW_TREES if (gt & (sup == t)).any()}
    hm_np = hm.cpu().numpy(); gt_heat = float(np.median(hm_np[gt & (hm_np > -1)])) if (gt & (hm_np > -1)).any() else float('nan')
    other = (sup < 10000) & ~np.isin(sup, ROW_TREES) & (hm_np > -1); other_heat = float(np.median(hm_np[other])) if other.any() else float('nan')
    rows.append(dict(kf=kf, view='held out' if kf in TEST else 'trained', gt_px=int(gt.sum()), lit_px=int(lit.sum()), iou=round(float(iou), 3), prec=round(float(prec), 3), rec=round(float(rec), 3),
                     split=round(thr, 3), otsu=round(ot, 3), p50=round(p50, 3), p99=round(p99, 3), absent=absent, row_heat_median=round(gt_heat, 3), other_rows_heat_median=round(other_heat, 3),
                     false_light_other_rows=fp_other, false_light_unlabelled=fp_unl, missed_by_tree=json.dumps(miss_by_tree)))
    ov = rgb.astype(np.float32); ov[tp] = 0.35 * ov[tp] + 0.65 * np.array([0, 200, 90]); ov[fp] = 0.35 * ov[fp] + 0.65 * np.array([255, 140, 0]); ov[fn] = 0.35 * ov[fn] + 0.65 * np.array([230, 40, 40])
    t = Image.fromarray(ov.astype(np.uint8)); d = ImageDraw.Draw(t); lab = f'{kf[3:9]}  IoU {iou:.2f}  P {prec:.2f}  R {rec:.2f}' + ('  (held out)' if kf in TEST else '')
    d.rectangle([0, 0, W, 44], fill=(11, 11, 11) if iou >= 0.8 else (150, 20, 20)); d.text((10, 6), lab, font=F, fill=(255, 255, 255)); tiles.append(t.resize((640, 360)))
    print(f'[row] {kf} {rows[-1]["view"]:8s} IoU {iou:.3f} P {prec:.3f} R {rec:.3f} | split {thr:.3f} (otsu {ot:.3f}, p50 {p50:.3f}) | row heat {gt_heat:.3f} vs other rows {other_heat:.3f} | false light: other rows {fp_other}, unlabelled {fp_unl} | missed {miss_by_tree}', flush=True)
with open(OUT / 'frames.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
cols = 4; sheet = Image.new('RGB', (640 * cols, 360 * ((len(tiles) + cols - 1) // cols)), (252, 252, 251))
for i, t in enumerate(tiles): sheet.paste(t, ((i % cols) * 640, (i // cols) * 360))
sheet.save(OUT / 'sheet.jpg', quality=85); print(f'[row] sheet -> {OUT / "sheet.jpg"}')
