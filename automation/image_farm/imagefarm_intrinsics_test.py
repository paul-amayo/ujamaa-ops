"""Where does the viewer lose 5-6 dB? Render the same training view three ways inside nerfstudio itself:
  A. the model's own trained camera (datamanager camera after undistortion: fx, fy, cx, cy as trained)
  B. a service-style camera: fx = fy = (H/2)/tan(fovy/2) with fovy from the ORIGINAL fl_y, principal point centred
  C. B but with the trained fx/fy ratio kept
and score each against the raw photo and the undistorted training image. Run with the nerf_new pixi python.
  python imagefarm_intrinsics_test.py <segment_dir> [frame index]"""
import sys, math, json, torch, numpy as np
from pathlib import Path
from PIL import Image
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras, CameraType
SEG = Path(sys.argv[1]); IDX = int(sys.argv[2]) if len(sys.argv) > 2 else 95
cfg = sorted((SEG / "blocks_ns/lio_arc_size15.0_ov0.10_kf20cm_dedup/block_000/splat_runs_high").glob("*/*/*/config.yml"))[-1]
config, pipeline, ckpt_path, step = eval_setup(cfg, test_mode="inference")
dm = pipeline.datamanager; ds = dm.train_dataset
names = [Path(ds.image_filenames[i]).name for i in range(len(ds))]; k = names.index(f"image_{IDX}.png")
cam = ds.cameras[k]; print(f"trained camera for image_{IDX}: fx {float(cam.fx):.1f} fy {float(cam.fy):.1f} cx {float(cam.cx):.1f} cy {float(cam.cy):.1f} size {int(cam.width)}x{int(cam.height)} type {cam.camera_type.item() if hasattr(cam.camera_type,'item') else cam.camera_type} distortion {None if cam.distortion_params is None else cam.distortion_params.numpy().round(4).tolist()}")
t = json.load(open(SEG / "transforms.json")); print(f"transforms.json: fl_x {t['fl_x']:.1f} fl_y {t['fl_y']:.1f} cx {t['cx']} cy {t['cy']} k1 {t['k1']:.4f} k2 {t['k2']:.4f}")
photo = torch.from_numpy(np.asarray(Image.open(SEG / "images" / f"image_{IDX}.png").convert("RGB"), np.float32) / 255)
gt_train = dm.cached_train[k]["image"] if hasattr(dm, "cached_train") else None
def psnr(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    if a.shape != b.shape: return float("nan")
    return 10 * math.log10(1.0 / max(float(((a - b) ** 2).mean()), 1e-12))
def render(c):
    c = Cameras(camera_to_worlds=c.camera_to_worlds[None] if c.camera_to_worlds.dim() == 2 else c.camera_to_worlds, fx=c.fx.reshape(1, 1), fy=c.fy.reshape(1, 1), cx=c.cx.reshape(1, 1), cy=c.cy.reshape(1, 1), width=c.width.reshape(1, 1), height=c.height.reshape(1, 1), camera_type=CameraType.PERSPECTIVE, distortion_params=None).to(pipeline.device)
    with torch.no_grad(): return pipeline.model.get_outputs_for_camera(c)["rgb"].cpu()
c2w = cam.camera_to_worlds; W, H = int(cam.width), int(cam.height)
A = render(cam)
fov_orig = 2 * math.atan(t["h"] / (2 * t["fl_y"])); f = (H / 2) / math.tan(fov_orig / 2)
B = render(Cameras(camera_to_worlds=c2w, fx=torch.tensor(f), fy=torch.tensor(f), cx=torch.tensor(W / 2), cy=torch.tensor(H / 2), width=torch.tensor(W), height=torch.tensor(H)))
C = render(Cameras(camera_to_worlds=c2w, fx=torch.tensor(f * float(cam.fx) / float(cam.fy)), fy=torch.tensor(f), cx=torch.tensor(W / 2), cy=torch.tensor(H / 2), width=torch.tensor(W), height=torch.tensor(H)))
print("shapes: photo", tuple(photo.shape), "gt_train", None if gt_train is None else tuple(gt_train.shape), "A", tuple(A.shape), "B", tuple(B.shape), "C", tuple(C.shape))
print(f"A vs C (should be identical cameras): {psnr(A, C):.2f} dB; A vs B: {psnr(A, B):.2f} dB")
print("camera A c2w[:3,3]", cam.camera_to_worlds[:3, 3].tolist(), "fx/fy/cx/cy", float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy), "| C fx", f * float(cam.fx) / float(cam.fy), "fy", f)
print(f"A trained camera        : PSNR vs raw photo {psnr(A, photo):.2f}" + (f", vs undistorted train image {psnr(A, gt_train):.2f}" if gt_train is not None else ""))
print(f"B service camera (fovy) : PSNR vs raw photo {psnr(B, photo):.2f}" + (f", vs undistorted train image {psnr(B, gt_train):.2f}" if gt_train is not None else ""))
print(f"C service + fx/fy ratio : PSNR vs raw photo {psnr(C, photo):.2f}" + (f", vs undistorted train image {psnr(C, gt_train):.2f}" if gt_train is not None else ""))
import torch.nn.functional as F
def fit(x): return F.interpolate(x.permute(2, 0, 1)[None], size=photo.shape[:2], mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
out = np.concatenate([(photo.numpy() * 255).astype(np.uint8), (fit(A).numpy() * 255).astype(np.uint8), (fit(B).numpy() * 255).astype(np.uint8)], 1)
Image.fromarray(out).resize((out.shape[1] // 2, out.shape[0] // 2)).save(f"/home/paperspace/logs/{SEG.name}_intrinsics_test.jpg", quality=88); print("wrote photo | A | B")
