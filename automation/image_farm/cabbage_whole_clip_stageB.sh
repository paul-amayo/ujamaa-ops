#!/bin/bash
# Whole-clip cabbage, stage B on the CPU (Paul 2026-10-04: "re-segment and reconstruct the whole cabbage video").
# Stage A showed IMG_7993 is ONE forward walk: consecutive-pair inliers never fall below 511 (median 2091); the prep's
# relative threshold (0.35 x median = 742) flagged pairs 51-53 as a turn and the 51-frame opening fell under MIN_SEG 60;
# frames 0-40 share <= 41 inliers with frames 54-125 (no revisit). So all 126 frames become one segment, IMG_7993_sall:
#   1. image_farm_prep.py stage B on [[0, 125]] (frozen copy: segment override, dir <clip>_sall, CPU glomap), reusing the
#      stage-A database (same image names), then its per-frame track check (frames with < 50 observations moved out);
#   2. pairing test: transforms.json centres vs sparse/0 centres by name after a similarity fit;
#   3. image_farm_recipe.sh steps 0-2 (frozen copy cut before SAM3/DA3): transforms.json + init.ply + monos + scratch_sam3.
# GPU stages (H3DGS 8 M, SAM3 cabbage maps) wait for the 05 retrain chain.
set -uo pipefail
R=/home/paperspace/data/image_farm/gwakungu/2026-05-16; C=$R/IMG_7993; S=$R/IMG_7993_sall; L=/home/paperspace/logs/cabbage_whole_clip.log
IF=/home/paperspace/code/automation/image_farm; PRE=/home/paperspace/logs/cabbage_whole_prep.run.py; REC=/home/paperspace/logs/cabbage_whole_recipe02.run.sh
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
STAGES=${*:-prep recipe pairing}
for st in $STAGES; do case $st in
prep)
  python3 - $IF/image_farm_prep.py $PRE <<'PY' || { say "prep copy FAILED"; exit 1; }
import sys
src = open(sys.argv[1]).read()
edits = [('segs = json.load(open(segf))["segments"]', 'segs = [[0, N - 1]]   # whole clip (cabbage_whole_clip_stageB.sh)'),
         ('SD = ROOT / f"{NAME}_s{si}"', 'SD = ROOT / f"{NAME}_sall"'),
         ('json.dump({"clip": NAME, "segment": si, "clip_frames"', 'json.dump({"clip": NAME, "segment": "all", "clip_frames"'),
         ('meta = {"clip": NAME, "video": MOV.name, "segment": si,', 'meta = {"clip": NAME, "video": MOV.name, "segment": "all",'),
         ('"--output_path", str(d / "sparse")]', '"--output_path", str(d / "sparse"), "--GlobalPositioning.use_gpu", "0", "--BundleAdjustment.use_gpu", "0"]')]
for a, b in edits:
    assert src.count(a) == 1, a
    src = src.replace(a, b)
open(sys.argv[2], "w").write(src)
PY
  mkdir -p $S; [ -e $S/database_gpu.db ] || cp $C/database_gpu.db $S/database_gpu.db
  say "prep stage B (CPU glomap) on all 126 frames -> $S"
  /home/paperspace/miniconda3/envs/h3dgs/bin/python $PRE $C 2>&1 | tee -a $L
  [ -s $S/sparse/0/points3D.bin ] && [ -e $S/gate.json ] || { say "prep FAILED"; exit 1; }
  python3 - $S <<'PY' | tee -a $L
import json, sys; g = json.load(open(sys.argv[1] + "/gate.json"))
print(f"[stageB] track check: registered {g['registered']}/{g['frames']}, kept {g['kept']}, dropped {g['dropped']}, obs p10/p50/p90 {g['obs_p10_p50_p90']}")
PY
  ;;
recipe)
  # steps 0-2 of the recipe of record, cut before step 3 (SAM3 + DA3 on the GPU)
  CUT=$(grep -n '^# ---- 3\. SAM3' $IF/image_farm_recipe.sh | cut -d: -f1); [ -n "$CUT" ] || { say "recipe cut line not found"; exit 1; }
  { head -n $((CUT - 1)) $IF/image_farm_recipe.sh; echo 'say "steps 0-2 done (cabbage_whole_clip_stageB.sh cut)"; exit 0'; } > $REC
  say "recipe steps 0-2 on $S"
  bash $REC $S 2>&1 | tee -a $L
  [ -s $S/transforms.json ] && [ -s $S/init.ply ] || { say "recipe 0-2 FAILED"; exit 1; }
  ;;
pairing)
  /home/paperspace/miniconda3/envs/h3dgs/bin/python - $S <<'PY' | tee -a $L
import json, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/paperspace/code/colmap/scripts/python")
from read_write_model import read_images_binary, qvec2rotmat
S = Path(sys.argv[1]); tj = S / "transforms.json"
if not tj.exists(): print("[stageB] pairing: no transforms.json yet (run after recipe)"); sys.exit(0)
ims = read_images_binary(str(S / "sparse/0/images.bin"))
sp = {im.name: -qvec2rotmat(im.qvec).T @ im.tvec for im in ims.values()}
tr = {Path(f["file_path"]).name: np.array(f["transform_matrix"])[:3, 3] for f in json.load(open(tj))["frames"]}
names = sorted(set(sp) & set(tr), key=lambda n: int(re.search(r"(\d+)", n).group(1)))
A = np.array([sp[n] for n in names]); B = np.array([tr[n] for n in names])
ma, mb = A.mean(0), B.mean(0); U, D, Vt = np.linalg.svd((B - mb).T @ (A - ma)); E = np.eye(3); E[2, 2] = np.sign(np.linalg.det(U @ Vt))
Rm = U @ E @ Vt; s = (D * np.diag(E)).sum() / ((A - ma) ** 2).sum(); res = np.linalg.norm(B - (s * (A - ma) @ Rm.T + mb), axis=1)
step = np.median(np.linalg.norm(np.diff(B, axis=0), axis=1))
print(f"[stageB] pairing: {len(names)} names in both, residual median {np.median(res):.4f} max {res.max():.4f} (median frame step {step:.4f}); "
      f"frames off by > 1 step: {int((res > step).sum())}")
PY
  ;;
esac; done
say "stage B [$STAGES] done"
