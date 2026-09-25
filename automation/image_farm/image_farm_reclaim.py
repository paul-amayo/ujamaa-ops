"""Lossless disk reclaim for /home/paperspace/data/image_farm (Paul, 2026-09-25: "do all of those reclaims, leave citrus and
klapmuts"). Dry run by default; --apply performs. Steps:
  1. checkpoints: keep only {step, pipeline} (drop optimizers/schedulers/scalers) — the render service and ns-eval load
     only `pipeline`; resuming those runs is what is lost. Atomic (tmp + os.replace). Skips files younger than 10 min.
  2. prod/scratch_sam3/kf_%06d.png -> hardlink to images/image_<i>.png when byte-identical.
  3. COLMAP databases (database_gpu.db / database.db) — SfM done, sparse/ kept.
  4. IMG_7999_s0/blocks_ns_5k_recipe (schedule tests) and IMG_7999/_scrambled_run_2046 (quarantine) — recorded in the notebook.
  5. tensorboard event files: rewrite with scalars only (drop the logged eval images).
Run with the nerf_new pixi python (torch + tensorboard)."""
import os, sys, glob, time, shutil, filecmp, re
from pathlib import Path
APPLY = "--apply" in sys.argv; ONLY_CKPT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--only-ckpt=")), None)
ROOT = Path("/home/paperspace/data/image_farm"); now = time.time(); saved = {}
def log(step, msg): print(f"[reclaim{'' if APPLY else ' DRY'}] {step}: {msg}", flush=True)
# ---- 1. checkpoints
import torch
ckpts = [Path(p) for p in glob.glob(str(ROOT / "**/*.ckpt"), recursive=True)]
if ONLY_CKPT: ckpts = [p for p in ckpts if ONLY_CKPT in str(p)]
tot = 0
def live_run(p: Path) -> bool:
    """A run whose directory has anything written in the last 10 min (nerfstudio rotates intermediate checkpoints)."""
    run = p.parent.parent
    try: return any(now - f.stat().st_mtime < 600 for f in run.rglob("*") if f.is_file())
    except FileNotFoundError: return True
for p in sorted(ckpts):
    try:
        if live_run(p): log("ckpt", f"skip (live run): {p}"); continue
        before = p.stat().st_size
        if APPLY:
            ck = torch.load(str(p), map_location="cpu", weights_only=False)
            if set(ck.keys()) <= {"step", "pipeline"}: continue
            slim = {"step": ck["step"], "pipeline": ck["pipeline"]}; tmp = p.with_suffix(".ckpt.tmp"); torch.save(slim, str(tmp)); os.replace(tmp, p); del ck, slim
            after = p.stat().st_size; tot += before - after; log("ckpt", f"{before/1e9:.2f} -> {after/1e9:.2f} GB  {str(p).split('image_farm/')[1][:60]}")
        else: tot += before * 2 / 3
    except FileNotFoundError: log("ckpt", f"vanished (rotated by its trainer): {p.name}")
saved["1 checkpoints (optimizer state)"] = tot
if ONLY_CKPT: print("only-ckpt done"); sys.exit(0)
# ---- 2. scratch_sam3 -> hardlinks
tot = 0; n = 0
for sd in sorted(ROOT.glob("IMG_*_s[0-9]")):
    sc = sd / "prod/scratch_sam3"
    if not sc.exists(): continue
    for kf in sc.glob("kf_*.png"):
        if kf.stat().st_nlink > 1: continue
        i = int(kf.stem.split("_")[1]); src = sd / "images" / f"image_{i}.png"
        if not src.exists() or not filecmp.cmp(src, kf, shallow=False): continue
        if APPLY:
            tmp = kf.with_suffix(".png.lnk"); os.link(src, tmp); os.replace(tmp, kf)
        tot += kf.stat().st_size; n += 1
saved["2 scratch_sam3 copies -> hardlinks"] = tot; log("scratch", f"{n} files, {tot/1e9:.1f} GB")
# ---- 3. COLMAP databases
tot = 0; n = 0
for p in list(ROOT.glob("*/database_gpu.db")) + list(ROOT.glob("*/database.db")):
    tot += p.stat().st_size; n += 1
    if APPLY: p.unlink()
saved["3 COLMAP databases"] = tot; log("db", f"{n} files, {tot/1e9:.1f} GB")
# ---- 4. test runs + quarantine
tot = 0
for d in (ROOT / "IMG_7999_s0/blocks_ns_5k_recipe", ROOT / "IMG_7999/_scrambled_run_2046"):
    if not d.exists(): continue
    sz = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()); tot += sz; log("dirs", f"{sz/1e9:.1f} GB  {d}")
    if APPLY: shutil.rmtree(d)
saved["4 IMG_7999 test runs + quarantine"] = tot
# ---- 5. tensorboard events: scalars only
tot = 0; n = 0
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
for ev in sorted(ROOT.glob("**/events.out.tfevents.*")):
    if "image_farm_reclaim" in str(ev) or now - ev.stat().st_mtime < 600: continue
    before = ev.stat().st_size
    if before < 20e6: continue
    if APPLY:
        ea = EventAccumulator(str(ev), size_guidance={"scalars": 0, "images": 1}); ea.Reload()
        w = SummaryWriter(log_dir=str(ev.parent), filename_suffix=".scalars")
        for tag in ea.Tags()["scalars"]:
            for x in ea.Scalars(tag): w.add_scalar(tag, x.value, x.step, walltime=x.wall_time)
        w.close(); ev.unlink(); after = sum(f.stat().st_size for f in ev.parent.glob("events.out.tfevents.*")); tot += before - after
    else: tot += before * 0.9
    n += 1
saved["5 tensorboard images"] = tot; log("events", f"{n} files, ~{tot/1e9:.1f} GB")
print("\n".join(f"  {k:40s} {v/1e9:7.1f} GB" for k, v in saved.items())); print(f"  {'TOTAL':40s} {sum(saved.values())/1e9:7.1f} GB")
os.system("df -h /home/paperspace/data | tail -1")
