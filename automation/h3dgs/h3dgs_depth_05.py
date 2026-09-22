"""Monocular inverse-depth maps for the H3DGS project of 05_13D, in the format make_depth_scale.py /
cameras.py consume: one 16-bit PNG per image, inverse depth normalised to [0, 65535] per image (the
per-image scale/offset is refitted against the sparse points downstream, so only the ordering matters).
Model: Depth-Anything-V2-Metric-Outdoor-Large (HF, already cached on the box). Env: ~/envs/hfeval_ft."""
import os, sys, numpy as np, torch
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
PROJ = Path(os.environ.get("H3DGS_PROJ", "/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs")) / "camera_calibration/rectified"
SRC, DST = PROJ / "images", PROJ / "depths"
DST.mkdir(parents=True, exist_ok=True)
name = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
proc = AutoImageProcessor.from_pretrained(name)
model = AutoModelForDepthEstimation.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
files = sorted(p for p in SRC.glob("*.png") if not (DST / p.name).exists())
print(f"{len(files)} images to do", flush=True)
B = 8
for i in range(0, len(files), B):
    batch = files[i:i + B]
    ims = [Image.open(p).convert("RGB") for p in batch]
    inputs = proc(images=ims, return_tensors="pt")
    with torch.no_grad():
        pred = model(pixel_values=inputs["pixel_values"].cuda().half()).predicted_depth   # metric depth (m), (B,h,w)
    pred = torch.nn.functional.interpolate(pred[:, None].float(), size=ims[0].size[::-1], mode="bilinear", align_corners=False)[:, 0]
    for p, d in zip(batch, pred):
        inv = 1.0 / d.clamp(min=0.05)
        inv = (inv - inv.min()) / (inv.max() - inv.min() + 1e-9)
        Image.fromarray((inv * 65535).round().cpu().numpy().astype(np.uint16)).save(DST / p.name)
    if (i // B) % 50 == 0:
        print(f"[depth] {i + len(batch)}/{len(files)}", flush=True)
print(f"DEPTH DONE {len(list(DST.glob('*.png')))} maps")
