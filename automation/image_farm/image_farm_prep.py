#!/usr/bin/env python
"""Phone clip -> working frames + GPU-stack SfM, so prod_image_recipe.sh finds sparse/0 and skips its CPU-BA SfM.
  python image_farm_prep.py <clip_dir>      (clip_dir holds one .MOV/.mov/.mp4)
Frames: ffmpeg at fps = clamp(CAP/duration, FPS_MIN, FPS_MAX) (env IF_CAP=400, IF_FPS_MIN=3, IF_FPS_MAX=6), scaled to
WIDTH 1080 for portrait / HEIGHT 1080 for landscape (the recipe's validated working-set rule), named image_<0..N-1>.png.
SfM: matched colmap 3.14-dev (OPENCV, single camera, GPU SIFT, exhaustive) + GPU-BA glomap (cuDSS stack), see the
gpu-glomap-a100 note. Sky flag (heuristic, for the recipe's --sky): sky-like pixels in the top 30 % of sampled frames.
Writes <clip>/capture_meta.json. Idempotent: skips extraction when images/ is non-empty, SfM when sparse/0/points3D.bin exists."""
import json, os, subprocess, sys, time, glob
from pathlib import Path
import numpy as np, cv2
CLIP = Path(sys.argv[1]).resolve(); NAME = CLIP.name
CAP = int(os.environ.get("IF_CAP", 400)); FMIN = float(os.environ.get("IF_FPS_MIN", 3)); FMAX = float(os.environ.get("IF_FPS_MAX", 6))
CC = "/home/paperspace/code/glomap/build_gpu/_deps/colmap-build/src/colmap/exe/colmap"
GB = "/home/paperspace/code/glomap/build_gpu/glomap/glomap"
GLD = "/home/paperspace/code/_cuda12/lib:/home/paperspace/code/_deps/libcudss-linux-x86_64-0.4.0.2_cuda12-archive/lib:/home/paperspace/code/_deps/absl-install/lib:/usr/local/lib"
LOG = Path("/home/paperspace/logs") / f"prep_{NAME}_sfm.log"
def say(*a): print(f"[{time.strftime('%m-%d %H:%M:%S')}] prep({NAME})", *a, flush=True)
vids = sorted(p for p in CLIP.iterdir() if p.suffix.lower() in (".mov", ".mp4", ".m4v"))
assert vids, f"no video in {CLIP}"; MOV = vids[0]
p = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,codec_name,pix_fmt:format=duration", "-of", "json", str(MOV)], capture_output=True, text=True).stdout)
W, H, DUR = p["streams"][0]["width"], p["streams"][0]["height"], float(p["format"]["duration"]); codec = p["streams"][0]["codec_name"]; pix = p["streams"][0]["pix_fmt"]
fps = float(np.clip(CAP / DUR, FMIN, FMAX)); portrait = H >= W
IM = CLIP / "images"; IM.mkdir(exist_ok=True)
if not any(IM.iterdir()):
    vf = f"fps={fps:.4f}," + ("scale=1080:-2" if portrait else "scale=-2:1080")
    say(f"{MOV.name}: {W}x{H} {codec}/{pix} {DUR:.1f}s -> ffmpeg {vf}")
    t0 = time.time()
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(MOV), "-vf", vf, "-start_number", "0", str(IM / "image_%d.png")], capture_output=True, text=True)
    if r.returncode: say("ffmpeg failed:", r.stderr[-500:]); sys.exit(2)
    say(f"extracted {len(list(IM.glob('image_*.png')))} frames in {time.time()-t0:.0f}s")
frames = sorted(IM.glob("image_*.png"), key=lambda q: int(q.stem.split("_")[1])); N = len(frames)
iw, ih = cv2.imread(str(frames[0])).shape[1::-1]
# sky heuristic on ≤12 sampled frames, top 30 % rows: pale/low-saturation bright OR blue-hue pixels
fr = []
for q in frames[::max(1, N // 12)][:12]:
    im = cv2.imread(str(q)); top = im[: int(im.shape[0] * 0.3)]; hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV); h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    fr.append(float((((s < 60) & (v > 170)) | ((h >= 90) & (h <= 130) & (s > 50) & (v > 120))).mean()))
sky_frac = float(np.median(fr)); sky = int(sky_frac > 0.03 or max(fr) > 0.15)
meta = {"video": MOV.name, "src_w": W, "src_h": H, "codec": codec, "pix_fmt": pix, "duration_s": round(DUR, 2), "fps": round(fps, 4), "n_frames": N,
        "work_w": iw, "work_h": ih, "sky_frac_median": round(sky_frac, 4), "sky_frac_max": round(max(fr), 4), "sky": sky}
say(f"{N} frames @ {iw}x{ih}, fps {fps:.3f}, sky heuristic median {sky_frac:.3f} max {max(fr):.3f} -> sky={sky}")
SP = CLIP / "sparse"; DB = CLIP / "database_gpu.db"
if not (SP / "0/points3D.bin").exists():
    env = dict(os.environ, LD_LIBRARY_PATH=GLD); env.pop("LD_PRELOAD", None)
    if DB.exists(): DB.unlink()
    SP.mkdir(exist_ok=True); t0 = time.time()
    with open(LOG, "w") as lf:
        for step, cmd in (("feature_extractor", [CC, "feature_extractor", "--database_path", str(DB), "--image_path", str(IM), "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "OPENCV", "--FeatureExtraction.use_gpu", "1"]),
                          ("exhaustive_matcher", [CC, "exhaustive_matcher", "--database_path", str(DB), "--FeatureMatching.use_gpu", "1"]),
                          ("glomap mapper", [GB, "mapper", "--database_path", str(DB), "--image_path", str(IM), "--output_path", str(SP)])):
            say(f"{step} ({time.time()-t0:.0f}s so far)")
            if subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode: say(f"{step} FAILED — see {LOG}"); sys.exit(3)
    say(f"SfM done in {time.time()-t0:.0f}s; CPU-fallback lines: {open(LOG, errors='ignore').read().count('Falling back to CPU')}")
sys.path.insert(0, "/home/paperspace/code/colmap/scripts/python"); from read_write_model import read_images_binary, read_cameras_binary
models = sorted(d for d in SP.iterdir() if d.is_dir() and (d / "images.bin").exists())
reg = {d.name: len(read_images_binary(str(d / "images.bin"))) for d in models}
cam = list(read_cameras_binary(str(SP / "0/cameras.bin")).values())[0]
meta.update({"registered": reg.get("0", 0), "models": reg, "camera": {"model": cam.model, "params": [round(float(x), 4) for x in cam.params]}})
say(f"registered {reg.get('0', 0)}/{N} in sparse/0 (models: {reg}); camera {cam.model} {[round(float(x), 1) for x in cam.params[:4]]}")
json.dump(meta, open(CLIP / "capture_meta.json", "w"), indent=1)
