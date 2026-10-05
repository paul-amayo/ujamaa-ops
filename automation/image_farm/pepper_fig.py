#!/usr/bin/env python3
"""Pepper identity vs SAM3 on chilli training views (UJAMAA 2026-10-05). Seed share > 0.3 (pepper_class cell), decision = the pixel's
OWN decoded word, argmax of plant vs pepper (the containment walk passes the plant parent node and loses every pod - pepper_bc_score.py).
Four training frames picked from the score json at the 90th / 60th / 40th / 10th percentile of IoU. Panels: photo | photo with the
field's pepper (green = on a SAM3 pepper mask, red = off it) and SAM3's masks outlined in cyan. -> cell/pepper_vs_sam3.jpg"""
import json, math, re, sys
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as nd
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/splat_viewer'); sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians/preprocess')
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from hier_compact import CompactHierarchy
from native_identity import NativeIdentity
import lorentz as L
from read_write_model import read_images_binary, read_cameras_binary, qvec2rotmat
torch.set_grad_enabled(False)
S = '/home/paperspace/data/image_farm/gwakungu/2026-05-16/IMG_7990_s1'; P = f'{S}/h3dgs'; NB = f'{S}/experimental/pepper_class/' + __import__('os').environ.get('CELL', 'cell'); FEAT = f'{NB}/features_s03.bin'; SUP = f'{NB}/supervision/pepper_class'
src = f'{P}/camera_calibration/chunks/lane/sparse/0'; c0 = list(read_cameras_binary(f'{src}/cameras.bin').values())[0]; W, H = int(c0.width), int(c0.height); fx, fy, cx, cy = c0.params[:4]
ims = {v.name: v for v in read_images_binary(f'{src}/images.bin').values()}
class Cam:
    def __init__(self, c2w_h):
        w2c = np.linalg.inv(c2w_h); R_, T_ = c2w_h[:3, :3], w2c[:3, 3]
        self.image_width, self.image_height = W, H; self.FoVx = 2 * math.atan(W / (2 * fx)); self.FoVy = 2 * math.atan(H / (2 * fy)); self.primx, self.primy = cx / W, cy / H
        self.world_view_transform = torch.tensor(getWorld2View2(R_, T_)).float().transpose(0, 1).cuda()
        self.projection_matrix = torch.tensor(getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy, primx=self.primx, primy=self.primy)).float().transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0); self.camera_center = self.world_view_transform.inverse()[3, :3]
CH = CompactHierarchy(f'{P}/output/trained_chunks/lane/hierarchy.hier_opt', '')
NI = NativeIdentity(FEAT, json.load(open(FEAT + '.json'))['embedder'], f'{NB}/text_bank.npz', CH.N, n_hier=CH.n_hier); E = NI.E[[NI.widx[w] for w in (NI.word_table['0'], NI.word_table['10000'])]]
sc = [x for x in json.load(open(f'{NB}/pepper_bc_s03.json')) if x['group'] == 'trained 0-141' and x['own']['iou'] is not None]; sc.sort(key=lambda x: x['own']['iou'])
pick = [sc[int(q * (len(sc) - 1))] for q in (0.9, 0.6, 0.4, 0.1)]
F1 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 34); F2 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 30); tiles = []
for x in pick:
    kf = x['kf']; im = ims[kf]; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec; cam = Cam(np.linalg.inv(w2c))
    feat = NI.feature_pass(CH, cam, 3.0); h_, w_, Dd = feat.shape; ft = feat.reshape(-1, Dd).float()
    d0 = NI.hyper.decode_features(L.exp_map0(ft, curv=NI.curv), project=True); d0 = d0 / d0.norm(dim=-1, keepdim=True)
    pep = (((d0 @ E.T).argmax(1) == 1) & (ft.norm(dim=-1) >= 0.5)).view(h_, w_).cpu().numpy()
    pep = np.array(Image.fromarray(pep.astype(np.uint8)).resize((W, H), Image.NEAREST)) > 0
    sup = np.array(Image.open(f'{SUP}/{kf}'), np.uint16); g = sup == 10000
    photo = np.array(Image.open(f'{P}/camera_calibration/rectified/images/{kf}').convert('RGB')); ov = photo.astype(np.float32)
    for m, col in ((pep & g, (40, 210, 40)), (pep & ~g, (235, 30, 30))): ov[m] = 0.35 * ov[m] + 0.65 * np.array(col)
    ov = ov.astype(np.uint8); e = nd.binary_dilation(g, iterations=2) & ~nd.binary_erosion(g, iterations=1); ov[e] = (0, 235, 255)
    crop = (slice(H // 3, H), slice(0, W))   # the plants fill the lower two thirds of these portrait frames
    pair = np.concatenate([photo[crop], np.full((photo[crop].shape[0], 12, 3), 255, np.uint8), ov[crop]], 1); t = Image.fromarray(pair); d = ImageDraw.Draw(t)
    cap = f"{kf}: IoU {x['own']['iou']:.2f}"; d.rounded_rectangle([10, 10, 24 + d.textlength(cap, font=F1), 58], 8, fill=(11, 11, 11)); d.text((17, 14), cap, font=F1, fill=(255, 255, 255))
    tiles.append(t.resize((t.width // 2, t.height // 2)))
w0, h0 = tiles[0].size; sheet = Image.new('RGB', (2 * w0 + 16, 2 * h0 + 16 + 90), (255, 255, 255)); dd = ImageDraw.Draw(sheet)
dd.text((8, 6), 'Where are the peppers? Chilli splat, class-level pepper identity on training views (left: photo, right: what the field lights)', font=F2, fill=(0, 0, 0))
tr = [x for x in json.load(open(f'{NB}/pepper_bc_s03.json')) if x['group'] == 'trained 0-141']; tp = sum(x['own']['tp'] for x in tr); lit = sum(x['own']['lit'] for x in tr); gt = sum(x['gt'] for x in tr)
dd.text((8, 46), f"green = lit on a SAM3 pepper, red = lit off SAM3 pepper, cyan outline = SAM3 pepper mask. Training-view mean IoU {np.mean([x['own']['iou'] for x in tr if x['own']['iou'] is not None]):.2f} (precision {tp / max(lit, 1):.2f}, recall {tp / max(gt, 1):.2f})", font=F2, fill=(0, 0, 0))
for k, t in enumerate(tiles): sheet.paste(t, ((k % 2) * (w0 + 16), 90 + (k // 2) * (h0 + 16)))
sheet.save(f'{NB}/pepper_vs_sam3.jpg', quality=88); print('[fig]', [(x['kf'], round(x['own']['iou'], 3)) for x in pick], '->', f'{NB}/pepper_vs_sam3.jpg', sheet.size)
