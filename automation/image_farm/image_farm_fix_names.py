"""Make a segment's frame names and its sparse model names sequential and identical after the track gate dropped frames,
so the recipe's ingest rename is an identity on every run (re-running the recipe after a first rename otherwise shifts
poses onto the wrong files — IMG_7961_s1 second pass, 2026-09-25: FileNotFoundError image_67.png).
  python image_farm_fix_names.py <segment_dir>
Rebuilds images/ from the clip's frames (segment.json range minus gate.json's dropped frames, hardlinks) as image_0..K-1
in capture order and rewrites sparse/0/images.bin names to match; removes stale transforms.json / lio_image_poses /
prod/scratch_sam3 so the recipe regenerates them."""
import json, os, re, shutil, sys
from pathlib import Path
sys.path.insert(0, "/home/paperspace/code/colmap/scripts/python"); from read_write_model import read_model, write_model
SD = Path(sys.argv[1]).resolve(); seg = json.load(open(SD / "segment.json")); gate = json.load(open(SD / "gate.json"))
clip = SD.parent / seg["clip"]; s, e = seg["clip_frames"]
dropped = {int(re.search(r"(\d+)", n).group(1)) for n in gate["dropped"]}   # segment-local indices before any rename
keep = [j for j in range(e - s + 1) if j not in dropped]                     # segment-local index -> clip frame s+j
cams, ims, pts = read_model(str(SD / "sparse/0"), ".bin")
by_name = {im.name: im for im in ims.values()}
old_names = sorted(by_name, key=lambda n: int(re.search(r"(\d+)", n).group(1)))
assert len(old_names) == len(keep), f"sparse has {len(old_names)} images, gate keeps {len(keep)}"
# sparse names are the ORIGINAL segment-local names (image_<j>) if never renamed, or already-sequential ones; map by rank
new_names = [f"image_{k}.png" for k in range(len(keep))]
ims2 = {}
for rank, old in enumerate(old_names):
    im = by_name[old]; ims2[im.id] = im._replace(name=new_names[rank])
tmp = SD / "sparse/0_renamed"; tmp.mkdir(parents=True, exist_ok=True); write_model(cams, ims2, pts, str(tmp), ".bin")
shutil.move(str(SD / "sparse/0"), str(SD / "sparse/0_before_rename")); shutil.move(str(tmp), str(SD / "sparse/0"))
old_images = SD / "images_before_rename"
if old_images.exists(): shutil.rmtree(old_images)
shutil.move(str(SD / "images"), str(old_images)); (SD / "images").mkdir()
for k, j in enumerate(keep): os.link(clip / "images" / f"image_{s + j}.png", SD / "images" / new_names[k])
for stale in ("transforms.json", "lio_image_poses.json", "lio_image_poses_kf20cm.json", "init.ply", "init_da3_fused.ply"):
    p = SD / stale
    if p.exists(): p.unlink()
for d in ("prod/scratch_sam3", "prod/monos", "depth_png", "prod/bateleur", "blocks_ns"):
    p = SD / d
    if p.exists(): shutil.rmtree(p)
shutil.rmtree(old_images)
print(f"[fix-names] {SD.name}: {len(keep)} frames renumbered image_0..{len(keep)-1}, sparse names rewritten, derived outputs cleared")
