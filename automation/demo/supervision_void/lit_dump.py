# Why is a tree the field lights blank in the chunk supervision? (UJAMAA 2026-10-03, row 17 at kf_003112/3113). Run the four scripts from ONE working dir:
# 1 lit_dump.py (h3dgs env + CUDA_HOME=_cuda12): the live-rule lit mask for the row word + scene-graph tree projections -> lit_row17.npz
import sys, math, json, numpy as np, torch
sys.argv = [sys.argv[0], '17', '0', '0']
exec(open('/home/paperspace/code/automation/demo/row_iou_frames.py').read().split("try: F = ImageFont")[0])   # same loader + heat() as the panels
path = json.load(open('/home/paperspace/logs/demo_chunks/01_3_1_720/demo_path.json')); TREES = {int(k): np.array(v) for k, v in path['trees'].items()}
out = {}
for k in (3112, 3113, 3115):
    kf = f'kf_{k:06d}.png'; im = ims[kf]; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec; c2w = np.linalg.inv(w2c)
    cam = Cam(c2w, W, H, 2 * math.atan(H / (2 * fy)), cx / W, cy / H)
    hm = heat(NI.feature_pass(CH, cam, 3.0)); a = NI.alpha(hm); out[f'lit_{k}'] = (a.float().cpu().numpy() > 0) if a is not None else np.zeros((H, W), bool)
    proj = []
    for t, x in TREES.items():
        p = w2c[:3, :3] @ x + w2c[:3, 3]
        if p[2] > 0.3: proj.append((t, row_of.get(t), float(p[2]), float(fx * p[0] / p[2] + cx), float(fy * p[1] / p[2] + cy), float(np.linalg.norm(x - c2w[:3, 3]))))
    out[f'proj_{k}'] = np.array(proj)
    print(kf, 'lit px', int(out[f'lit_{k}'].sum()), '| trees in view (id, row, depth m, u, v, dist m):', [(int(t), r, round(z, 1), int(u), int(v), round(d, 1)) for t, r, z, u, v, d in proj if 0 <= u < W and 0 <= v < H])
np.savez_compressed('lit_row17.npz', **out); print('W,H', W, H)
