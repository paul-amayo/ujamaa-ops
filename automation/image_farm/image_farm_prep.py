#!/usr/bin/env python
"""Phone clip -> forward-walk SEGMENTS, each a survey dir with working frames + GPU-stack SfM + a per-frame track gate,
ready for image_farm_recipe.sh (which skips its own SfM when sparse/0 exists).
  python image_farm_prep.py <clip_dir>      (clip_dir = image_farm/IMG_NNNN, holds one .MOV/.mov/.mp4)
Why segments (2026-09-24, IMG_7999): a down-and-back walk turns over featureless grass; GLOMAP keeps all frames in one
"connected component" but leaves the whole return pass without 3D points, and those arbitrary poses turn the splat to
mush (13 dB). Frame-to-frame SIFT inliers dip 3-10x in the turn, so the clip is cut there and each pass gets its own SfM.
Stage A (clip dir): ffmpeg at fps = clamp(CAP/duration, FPS_MIN, FPS_MAX) (env IF_CAP=400, IF_FPS_MIN=3, IF_FPS_MAX=6)
  to width 1080 (portrait) / height 1080 (landscape) as image_<0..N-1>.png; SIFT + exhaustive matching on all frames
  (matched colmap 3.14-dev, OPENCV single camera, GPU); consecutive-pair inliers -> segments.json.
Stage B (per segment, sibling dir <clip>_s<k>): hardlinked frames renamed image_<0..M-1>.png, fresh SIFT + exhaustive +
  GPU-BA glomap, then frames with < IF_MIN_OBS (50) track observations are moved to images_dropped/ and removed from
  sparse/0 (so the recipe's ingest never renames over them); capture_meta.json carries fps, dims, sky flag, counts.
Idempotent per stage. Prints one summary line per segment: '<seg_dir> kept K/M registered R obs_p50 ...'."""
import json, os, re, shutil, sqlite3, subprocess, sys, time
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, "/home/paperspace/code/colmap/scripts/python")
from read_write_model import read_model, write_model, read_images_binary, read_cameras_binary
CLIP = Path(sys.argv[1]).resolve(); NAME = CLIP.name; ROOT = CLIP.parent
CAP = int(os.environ.get("IF_CAP", 400)); FMIN = float(os.environ.get("IF_FPS_MIN", 3)); FMAX = float(os.environ.get("IF_FPS_MAX", 6))
MIN_SEG = int(os.environ.get("IF_MIN_SEG", 60)); MIN_OBS = int(os.environ.get("IF_MIN_OBS", 50)); DIP = float(os.environ.get("IF_DIP", 0.35))
CC = "/home/paperspace/code/glomap/build_gpu/_deps/colmap-build/src/colmap/exe/colmap"
GB = "/home/paperspace/code/glomap/build_gpu/glomap/glomap"
GLD = "/home/paperspace/code/_cuda12/lib:/home/paperspace/code/_deps/libcudss-linux-x86_64-0.4.0.2_cuda12-archive/lib:/home/paperspace/code/_deps/absl-install/lib:/usr/local/lib"
LOGS = Path("/home/paperspace/logs")
def say(*a): print(f"[{time.strftime('%m-%d %H:%M:%S')}] prep({NAME})", *a, flush=True)
def frame_no(p): return int(re.search(r"(\d+)", Path(p).stem).group(1))
def run_sfm(d: Path, log: Path, glomap: bool):
    """SIFT + exhaustive on d/images -> d/database_gpu.db; glomap -> d/sparse/0. Skips finished stages."""
    env = dict(os.environ, LD_LIBRARY_PATH=GLD); env.pop("LD_PRELOAD", None); db = d / "database_gpu.db"; t0 = time.time()
    steps = []
    if not db.exists():
        steps += [("feature_extractor", [CC, "feature_extractor", "--database_path", str(db), "--image_path", str(d / "images"), "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "OPENCV", "--FeatureExtraction.use_gpu", "1"]),
                  ("exhaustive_matcher", [CC, "exhaustive_matcher", "--database_path", str(db), "--FeatureMatching.use_gpu", "1"])]
    if glomap and not (d / "sparse/0/points3D.bin").exists():
        (d / "sparse").mkdir(exist_ok=True); steps.append(("glomap mapper", [GB, "mapper", "--database_path", str(db), "--image_path", str(d / "images"), "--output_path", str(d / "sparse")]))
    with open(log, "a") as lf:
        for step, cmd in steps:
            say(f"{d.name}: {step} ({time.time()-t0:.0f}s so far)")
            if subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode:
                if step != "feature_extractor" and db.exists() and not steps[0][0].startswith("feature"): pass
                say(f"{d.name}: {step} FAILED — see {log}"); return False
    return True
def consecutive_inliers(db: Path):
    con = sqlite3.connect(str(db)); num = {i: frame_no(n) for i, n in con.execute("select image_id, name from images")}
    inl = {}
    for pid, rows in con.execute("select pair_id, rows from two_view_geometries"):
        a, b = pid // 2147483647, pid % 2147483647; inl[(num[a], num[b])] = rows; inl[(num[b], num[a])] = rows
    n = max(num.values()) + 1; return np.array([inl.get((k, k + 1), 0) for k in range(n - 1)]), n
# ---------------- Stage A: frames + all-frame matching + segmentation ----------------
vids = sorted(p for p in CLIP.iterdir() if p.suffix.lower() in (".mov", ".mp4", ".m4v")); assert vids, f"no video in {CLIP}"; MOV = vids[0]
p = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,codec_name,pix_fmt:format=duration", "-of", "json", str(MOV)], capture_output=True, text=True).stdout)
W, H, DUR = p["streams"][0]["width"], p["streams"][0]["height"], float(p["format"]["duration"]); codec, pix = p["streams"][0]["codec_name"], p["streams"][0]["pix_fmt"]
fps = float(np.clip(CAP / DUR, FMIN, FMAX)); portrait = H >= W; IM = CLIP / "images"; IM.mkdir(exist_ok=True)
if not any(IM.iterdir()):
    vf = f"fps={fps:.4f}," + ("scale=1080:-2" if portrait else "scale=-2:1080"); say(f"{MOV.name}: {W}x{H} {codec}/{pix} {DUR:.1f}s -> ffmpeg {vf}"); t0 = time.time()
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(MOV), "-vf", vf, "-start_number", "0", str(IM / "image_%d.png")], capture_output=True, text=True)
    if r.returncode: say("ffmpeg failed:", r.stderr[-500:]); sys.exit(2)
    say(f"extracted {len(list(IM.glob('image_*.png')))} frames in {time.time()-t0:.0f}s")
frames = sorted(IM.glob("image_*.png"), key=frame_no); N = len(frames); iw, ih = cv2.imread(str(frames[0])).shape[1::-1]
if not run_sfm(CLIP, LOGS / f"prep_{NAME}_sfm.log", glomap=False): sys.exit(3)
segf = CLIP / "segments.json"
if not segf.exists():
    inl, n = consecutive_inliers(CLIP / "database_gpu.db"); med = float(np.median(inl)); thr = max(300.0, DIP * med)
    good = inl >= thr; segs, k = [], 0
    while k < n - 1:
        if good[k]:
            s = k
            while k < n - 1 and good[k]: k += 1
            if k - s + 1 >= MIN_SEG: segs.append([s, k])       # frames s..k inclusive
        else: k += 1
    if not segs and n >= MIN_SEG: segs = [[0, n - 1]]
    json.dump({"n_frames": n, "median_consecutive_inliers": med, "threshold": thr, "min_seg": MIN_SEG, "segments": segs,
               "dropped_turn_frames": int(n - sum(e - s + 1 for s, e in segs))}, open(segf, "w"), indent=1)
    say(f"consecutive inliers median {med:.0f}, threshold {thr:.0f} -> {len(segs)} segment(s) {segs}, turn frames dropped {n - sum(e - s + 1 for s, e in segs)}")
segs = json.load(open(segf))["segments"]
# ---------------- Stage B: per-segment survey dirs ----------------
summary = []
for si, (s, e) in enumerate(segs):
    SD = ROOT / f"{NAME}_s{si}"; SIM = SD / "images"; SIM.mkdir(parents=True, exist_ok=True)
    if not any(SIM.iterdir()):
        for j, k in enumerate(range(s, e + 1)): os.link(IM / f"image_{k}.png", SIM / f"image_{j}.png")
        json.dump({"clip": NAME, "segment": si, "clip_frames": [s, e]}, open(SD / "segment.json", "w"))
    M = len(list(SIM.glob("image_*.png")))
    if not run_sfm(SD, LOGS / f"prep_{SD.name}_sfm.log", glomap=True): summary.append(f"{SD.name}: SFM FAILED"); continue
    sp = SD / "sparse/0"
    if not (SD / "gate.json").exists():
        cams, ims, pts = read_model(str(sp), ".bin"); obs = {im.name: int((im.point3D_ids >= 0).sum()) for im in ims.values()}
        keep = {n for n, o in obs.items() if o >= MIN_OBS}; dropped = sorted((set(f.name for f in SIM.glob("image_*.png")) - keep), key=frame_no)
        if dropped:
            DD = SD / "images_dropped"; DD.mkdir(exist_ok=True)
            for n in dropped: shutil.move(str(SIM / n), str(DD / n))
            ims2 = {i: im for i, im in ims.items() if im.name in keep}
            gone = {i for i, im in ims.items() if im.name not in keep}
            for pid, pt in list(pts.items()):
                m = ~np.isin(pt.image_ids, list(gone))
                if m.sum() < 2: del pts[pid]
                else: pts[pid] = pt._replace(image_ids=pt.image_ids[m], point2D_idxs=pt.point2D_idxs[m])
            bak = SD / "sparse/0_all"; shutil.move(str(sp), str(bak)); sp.mkdir(parents=True); write_model(cams, ims2, pts, str(sp), ".bin")
        json.dump({"min_obs": MIN_OBS, "frames": M, "registered": len(ims), "kept": len(keep), "dropped": dropped,
                   "obs_p10_p50_p90": [int(x) for x in np.percentile(list(obs.values()), [10, 50, 90])] if obs else []}, open(SD / "gate.json", "w"), indent=1)
    g = json.load(open(SD / "gate.json")); kept = sorted(SIM.glob("image_*.png"), key=frame_no)
    # sky heuristic on the kept frames (top 30 % rows: pale/low-saturation bright OR blue-hue pixels)
    fr = []
    for q in kept[::max(1, len(kept) // 12)][:12]:
        im = cv2.imread(str(q)); top = im[: int(im.shape[0] * 0.3)]; hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV); h_, s_, v_ = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        fr.append(float((((s_ < 60) & (v_ > 170)) | ((h_ >= 90) & (h_ <= 130) & (s_ > 50) & (v_ > 120))).mean()))
    sky_frac = float(np.median(fr)) if fr else 0.0; sky = int(sky_frac > 0.03 or (max(fr) if fr else 0) > 0.15)
    cam = list(read_cameras_binary(str(sp / "cameras.bin")).values())[0]
    meta = {"clip": NAME, "video": MOV.name, "segment": si, "clip_frames": [s, e], "src_w": W, "src_h": H, "codec": codec, "duration_s": round(DUR, 2), "fps": round(fps, 4),
            "n_frames": M, "kept": g["kept"], "dropped": len(g["dropped"]), "registered": g["registered"], "obs_p10_p50_p90": g["obs_p10_p50_p90"], "work_w": iw, "work_h": ih,
            "sky_frac_median": round(sky_frac, 4), "sky": sky, "camera": {"model": cam.model, "params": [round(float(x), 4) for x in cam.params]},
            "ok": bool(g["kept"] >= MIN_SEG and g["kept"] >= 0.8 * M)}
    json.dump(meta, open(SD / "capture_meta.json", "w"), indent=1)
    summary.append(f"{SD.name}: kept {g['kept']}/{M} (registered {g['registered']}, obs p10/p50/p90 {g['obs_p10_p50_p90']}), sky={sky} ({sky_frac:.3f}), fx {cam.params[0]:.0f}, ok={meta['ok']}")
for line in summary: say(line)
