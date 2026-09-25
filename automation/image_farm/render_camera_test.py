"""How much does the render service's camera model cost on a block? Inside nerfstudio, render one training view with
  A. the model's own trained camera (datamanager camera: undistorted K, true principal point)
  B. the service's fovy camera: fx = fy = (H/2)/tan(fovy/2) with fovy from the block's fl_y, principal point centred
  C. the service's RENDER_NATIVE_INTRINSICS camera: transforms.json K after nerfstudio's undistortion math, height-scaled
and score each against the cached training image. Run with the nerf_new pixi python.
  python render_camera_test.py <config.yml> [train index]"""
import sys, math, json, torch, numpy as np, cv2
from pathlib import Path
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras, CameraType
cfg = Path(sys.argv[1]); IDX = int(sys.argv[2]) if len(sys.argv) > 2 else 40
config, pipeline, ckpt_path, step = eval_setup(cfg, test_mode="test")
dm = pipeline.datamanager; ds = dm.train_dataset; cam = ds.cameras[IDX]
gt = dm.cached_train[IDX]["image"]; H, W = int(gt.shape[0]), int(gt.shape[1])
tj = None
for d in [cfg.parents[i] for i in range(1, 7)]:
    if (d / "transforms.json").exists(): tj = json.load(open(d / "transforms.json")); break
def psnr(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    if a.shape != b.shape: return float("nan")
    return 10 * math.log10(1.0 / max(float(((a - b) ** 2).mean()), 1e-12))
def render(fx, fy, cx, cy):
    c = Cameras(camera_to_worlds=cam.camera_to_worlds[None], fx=torch.tensor([[float(fx)]]), fy=torch.tensor([[float(fy)]]), cx=torch.tensor([[float(cx)]]), cy=torch.tensor([[float(cy)]]),
                width=torch.tensor([[W]]), height=torch.tensor([[H]]), camera_type=CameraType.PERSPECTIVE).to(pipeline.device)
    with torch.no_grad(): return pipeline.model.get_outputs_for_camera(c)["rgb"].cpu()
name = Path(ds.image_filenames[IDX]).name
print(f"{cfg.parents[4].name if len(cfg.parents) > 4 else cfg} view {name}: trained camera fx {float(cam.fx):.1f} fy {float(cam.fy):.1f} cx {float(cam.cx):.1f} cy {float(cam.cy):.1f} size {W}x{H}")
A = render(cam.fx, cam.fy, cam.cx, cam.cy); print(f"  A trained camera                : {psnr(A, gt):.2f} dB")
if tj:
    f = (H / 2) / math.tan(math.atan(tj["h"] / (2 * tj["fl_y"])))   # fovy from fl_y -> f = fl_y * H/h
    B = render(f, f, W / 2, H / 2); print(f"  B service fovy camera (centred) : {psnr(B, gt):.2f} dB   (fx=fy={f:.1f}, cx={W/2}, cy={H/2})")
    K0 = np.array([[tj["fl_x"], 0, tj["cx"] - 0.5], [0, tj["fl_y"], tj["cy"] - 0.5], [0, 0, 1.0]]); dist = np.array([tj.get(k, 0.0) for k in ("k1", "k2", "p1", "p2")] + [0.0] * 4)
    if np.any(dist):
        newK, (rx, ry, rw, rh) = cv2.getOptimalNewCameraMatrix(K0, dist, (int(tj["w"]), int(tj["h"])), 0); nat = dict(fx=newK[0, 0], fy=newK[1, 1], cx=newK[0, 2] - rx + 0.5, cy=newK[1, 2] - ry + 0.5, w=rw, h=rh)
    else: nat = dict(fx=tj["fl_x"], fy=tj["fl_y"], cx=tj["cx"], cy=tj["cy"], w=tj["w"], h=tj["h"])
    s = H / nat["h"]; C = render(nat["fx"] * s, nat["fy"] * s, W / 2 + (nat["cx"] - nat["w"] / 2) * s, nat["cy"] * s)
    print(f"  C native (trained K reproduced) : {psnr(C, gt):.2f} dB   (fx={nat['fx']*s:.1f}, fy={nat['fy']*s:.1f}, cx={W/2 + (nat['cx'] - nat['w']/2)*s:.1f}, cy={nat['cy']*s:.1f})")
