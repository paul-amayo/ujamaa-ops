#!/usr/bin/env python3
"""Fruit field on a native H3DGS chunk, CountingFruit-style (Paul 2026-10-08: "try this approach"): per-node feature vectors TRAINED with
a rendered per-pixel loss instead of assigned by the census. Geometry frozen (the served hierarchy, rendered at the serving cut tau 3 through
the same gsplat path native_identity.feature_pass uses), features F[N, D] trainable; targets = class codes on labelled pixels of the cell's
supervision (fruit >= 10000 -> e_fruit, tree < 10000 -> e_tree, 65535 ignored); loss = L1 + (1 - cosine) on labelled pixels (the paper's
semantic loss). Query = dual threshold (close to fruit AND far from leaves) or the parameter-free argmax. Scored on the same 28 Citrus B
fruit-clip frames as fruit_bc_score.py (class-level fruit vs all SAM3 fruit pixels, and fruit-of-tree-5 = fruit mask inside the tree-5
supervision region, which the baseline scorer effectively measures).
  h3dgs env, CUDA_HOME: fruit_field_train.py <project> <cell> <out dir> train [iters=3000] | score"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/splat_viewer'); sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians/preprocess')
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from hier_compact import CompactHierarchy
from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights
from gsplat import rasterization
from read_write_model import read_images_binary, read_cameras_binary, qvec2rotmat
P, NB, OUT, STAGE = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]; ITERS = int(sys.argv[5]) if len(sys.argv) > 5 else 3000; OUT.mkdir(parents=True, exist_ok=True)
CN = os.environ.get('CN', '1_0'); TAU = float(os.environ.get('TAU', '3')); D = 3; SCALE = float(os.environ.get('SCALE', '0.5')); LR = float(os.environ.get('LR', '0.02')); TAU_POS, TAU_NEG = 0.7, 0.5; BALANCE = os.environ.get('BALANCE', '0') == '1'   # v2: SCALE=1 full-res render, BALANCE=1 per-class means (fruit pixels are ~0.3 % of labelled pixels; the plain mean is a tree loss)
E_FRUIT = torch.tensor([1.0, 0.0, 0.0], device='cuda'); E_TREE = torch.tensor([0.0, 1.0, 0.0], device='cuda')
src = f'{P}/camera_calibration/chunks/{CN}/sparse/0'; cam0 = list(read_cameras_binary(f'{src}/cameras.bin').values())[0]; W, H = int(cam0.width), int(cam0.height); fx, fy, cx, cy = cam0.params[:4]
ims = {v.name: v for v in read_images_binary(f'{src}/images.bin').values()}; TEST = {l.strip() for l in open(f'{src}/test.txt') if l.strip()}; SUP = f'{NB}/supervision/trees_only'
class Cam:
    def __init__(self, c2w_h):
        w2c = np.linalg.inv(c2w_h); R_, T_ = c2w_h[:3, :3], w2c[:3, 3]
        self.image_width, self.image_height = W, H; self.FoVx = 2 * math.atan(W / (2 * fx)); self.FoVy = 2 * math.atan(H / (2 * fy)); self.primx, self.primy = cx / W, cy / H
        self.world_view_transform = torch.tensor(getWorld2View2(R_, T_)).float().transpose(0, 1).cuda(); self.camera_center = self.world_view_transform.inverse()[3, :3]
def cam_of(kf): im = ims[kf]; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec; return Cam(np.linalg.inv(w2c))
CH = CompactHierarchy(f'{P}/output/trained_chunks/{CN}/hierarchy.hier_opt', f'{P}/output/scaffold/point_cloud/iteration_30000')
def render_feat(F, cam):   # native_identity.feature_pass with trainable F (gather is differentiable; the cut is not)
    w_h, h_h = int(W * SCALE), int(H * SCALE); thr = (2 * (TAU + 0.5)) * math.tan(cam.FoVx * 0.5) / (0.5 * W)
    with torch.no_grad():
        n = expand_to_size(CH.nodes, CH.boxes, thr, cam.camera_center, torch.zeros(3), CH.ri, CH.pi, CH.nri)
        get_interpolation_weights(CH.nri[:n].contiguous(), thr, CH.nodes, CH.boxes, cam.camera_center.cpu(), torch.zeros(3), CH.iw, CH.ns)
        c = CH.ri[:n].long(); p = CH.pi[:n].long(); w = CH.iw[:n].unsqueeze(1); wi = 1.0 - w
        means = w * CH.xyz[c] + wi * CH.xyz[p]; scales = w * torch.exp(CH.logscale[c].float()) + wi * torch.exp(CH.logscale[p].float())
        rc = torch.nn.functional.normalize(CH.rot[c].float()); rp = torch.nn.functional.normalize(CH.rot[p].float()); rp[(rc * rp).sum(1) < 0] *= -1; rot = w * rc + wi * rp
        op = (w * CH.alpha[c].float().unsqueeze(1) + wi * CH.alpha[p].float().unsqueeze(1)).squeeze(1); t = CH.tail_idx
        if t.numel(): means = torch.cat([means, CH.xyz[t]]); scales = torch.cat([scales, torch.exp(CH.logscale[t].float())]); rot = torch.cat([rot, torch.nn.functional.normalize(CH.rot[t].float())]); op = torch.cat([op, CH.alpha[t].float()])
        K = torch.tensor([[fx * SCALE, 0, cx * SCALE], [0, fy * SCALE, cy * SCALE], [0, 0, 1]], dtype=torch.float32, device='cuda')[None]; viewmat = cam.world_view_transform.transpose(0, 1).contiguous().float()[None]
    f = w * F[c] + wi * F[p]
    if t.numel(): f = torch.cat([f, torch.zeros(t.numel(), D, device='cuda')])
    out, _, _ = rasterization(means=means.contiguous(), quats=rot.contiguous(), scales=scales.contiguous(), opacities=op.contiguous(), colors=f.contiguous(), viewmats=viewmat, Ks=K, width=w_h, height=h_h, sh_degree=None, packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB', rasterize_mode='classic')
    return out[0]   # [h, w, D]
def targets(kf):
    m = np.array(Image.open(f'{SUP}/{kf}'), np.uint16); m = np.array(Image.fromarray(m).resize((int(W * SCALE), int(H * SCALE)), Image.NEAREST)); lab = m != 65535
    T = torch.zeros((m.shape[0], m.shape[1], D), device='cuda'); fr = torch.from_numpy((m >= 10000) & lab).cuda(); tr = torch.from_numpy((m < 10000) & lab).cuda()
    T[fr] = E_FRUIT; T[tr] = E_TREE; return T, torch.from_numpy(lab).cuda()
if STAGE == 'train':
    train = sorted(n for n in ims if n not in TEST and os.path.exists(f'{SUP}/{n}')); F = torch.nn.Parameter(torch.randn(CH.N, D, device='cuda') * 0.01); opt = torch.optim.Adam([F], lr=LR)
    rng = np.random.default_rng(0); t0 = time.time(); run = []
    print(f'[fruit-field] {len(train)} training views with supervision; {CH.N:,} nodes x {D}; tau {TAU}, lr {LR}, {ITERS} its', flush=True)
    for it in range(1, ITERS + 1):
        kf = train[int(rng.integers(len(train)))]; cam = cam_of(kf); T, lab = targets(kf); r = render_feat(F, cam)
        rl, tl = r[lab], T[lab]
        if rl.shape[0] == 0: continue
        per = (rl - tl).abs().mean(-1) + (1 - torch.nn.functional.cosine_similarity(rl, tl, dim=-1, eps=1e-6))
        if BALANCE:
            isf = tl[:, 0] > 0.5; loss = (per[isf].mean() if isf.any() else 0.0) + per[~isf].mean()
        else: loss = per.mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); run.append(float(loss))
        if it % 100 == 0: print(f'[fruit-field] it {it}: loss {np.mean(run[-100:]):.4f} ({(time.time() - t0) / it:.2f} s/it)', flush=True)
    torch.save(F.detach().half().cpu(), OUT / 'F.pt'); json.dump({'D': D, 'tau': TAU, 'lr': LR, 'iters': ITERS, 'codes': {'fruit': E_FRUIT.tolist(), 'tree': E_TREE.tolist()}, 'loss_last100': float(np.mean(run[-100:]))}, open(OUT / 'F.json', 'w'), indent=1)
    print(f'[fruit-field] saved {OUT}/F.pt in {(time.time() - t0) / 60:.1f} min', flush=True)
else:
    F = torch.load(OUT / 'F.pt').float().cuda(); path = json.load(open('/home/paperspace/logs/demo_chunks/05_1_0_720/demo_path.json'))['frames']; clip = [path[i]['src'] for i in list(range(13, 53)) + list(range(346, 361))]
    rows = []
    with torch.no_grad():
        for kf in clip:
            if kf not in ims: continue
            sup = np.array(Image.open(f'{SUP}/{kf}'), np.uint16); g_all = (sup >= 10000) & (sup != 65535); g5 = sup == 10001; t5 = (sup == 5) | g5
            if g_all.sum() == 0: continue
            r = render_feat(F, cam_of(kf)); r = torch.nn.functional.interpolate(r.permute(2, 0, 1)[None], size=(H, W), mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
            nrm = r.norm(dim=-1); cf = torch.nn.functional.cosine_similarity(r, E_FRUIT.expand_as(r), dim=-1); ct = torch.nn.functional.cosine_similarity(r, E_TREE.expand_as(r), dim=-1)
            dual = ((cf > TAU_POS) & (ct < TAU_NEG) & (nrm > 0.3)).cpu().numpy(); amax = ((cf > ct) & (nrm > 0.3)).cpu().numpy()
            iou = lambda m, g: float((m & g).sum() / max((m | g).sum(), 1))
            rows.append({'kf': kf, 'held_out': kf in TEST, 'sam3_fruit_px': int(g_all.sum()), 'sam3_tree5_fruit_px': int(g5.sum()), 'class_iou_dual': iou(dual, g_all), 'class_iou_argmax': iou(amax, g_all), 'tree5_iou_dual': iou(dual & t5, g5) if g5.sum() else None, 'tree5_iou_argmax': iou(amax & t5, g5) if g5.sum() else None, 'lit_px_dual': int(dual.sum()), 'lit_px_argmax': int(amax.sum())})
    json.dump(rows, open(OUT / 'score_clip.json', 'w'), indent=1)
    for key in ('class_iou_dual', 'class_iou_argmax', 'tree5_iou_dual', 'tree5_iou_argmax'):
        for lab, sel in (('all', rows), ('held-out', [x for x in rows if x['held_out']]), ('trained', [x for x in rows if not x['held_out']])):
            v = [x[key] for x in sel if x[key] is not None]; print(f'[fruit-field score] {key:16s} {lab:8s}: {len(v):2d} frames, mean IoU {np.mean(v) if v else float("nan"):.3f}', flush=True)
