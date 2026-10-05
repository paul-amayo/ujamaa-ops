#!/bin/bash
# Chilli H3DGS trainings (UJAMAA 2026-10-05; Paul: "go class-level pepper after the renders then and then kick off the chili
# trainings"). Runs after the class-level pepper chain. Gwakungu clip IMG_7990 (394 frames, 2 prep segments):
#   s1a  clip frames 153-294 = IMG_7990_s1 frames 0-141, the part BEFORE the 17 deg turn where s1's GLOMAP poses collapse
#        (141 -> 142 jumps 7.3x the median step, then 98 steps < 0.1x): fresh SfM on its own frames, checks, recipe 0-2, H3DGS 8 M
#   s0   clip frames 0-150 (prep segment 0, path healthy: max step ratio 1.7): H3DGS 8 M (never H3DGS-trained)
#   s1b  clip frames 295-393 = the tail: fresh SfM; trained only if its camera path passes the step check
# Checks after SfM: per-frame track observations >= 50 (the prep's), and the camera-step check (max consecutive step <= 5x the
# median, < 5 % of steps below 0.1x the median) - the failure IMG_7990_s1 had that the track check cannot see.
set -uo pipefail
G=/home/paperspace/data/image_farm/gwakungu/2026-05-16; IF=/home/paperspace/code/automation/image_farm; L=/home/paperspace/logs/chilli_trainings.log
PYH=/home/paperspace/miniconda3/envs/h3dgs/bin/python; say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
until grep -q "=== done" /home/paperspace/logs/pepper_class_chain.log 2>/dev/null; do sleep 60; done
gpu_free() { while [ "$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '$1 > 2048' | wc -l)" -gt 0 ]; do sleep 60; done; }
step_check() {   # <segment dir> -> prints the verdict, exit 0 when healthy
  $PYH - $1 <<'PY'
import re, sys, numpy as np
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians/preprocess')
from read_write_model import read_images_binary, qvec2rotmat
ims = read_images_binary(sys.argv[1] + '/sparse/0/images.bin')
d = sorted(((int(re.sub(r'\D', '', im.name)), -qvec2rotmat(im.qvec).T @ im.tvec) for im in ims.values()), key=lambda t: t[0])
C = np.array([c for _, c in d]); st = np.linalg.norm(np.diff(C, axis=0), axis=1); r = st / np.median(st)
ok = r.max() <= 5 and (r < 0.1).mean() < 0.05
print(f'[step-check] {len(d)} cameras: max step {r.max():.1f}x median at frame {d[int(r.argmax())][0]}, {int((r < 0.1).sum())} steps < 0.1x -> {"OK" if ok else "FAIL"}')
sys.exit(0 if ok else 1)
PY
}
prep_segment() {   # <name> <clip start> <clip end>: frozen image_farm_prep.py copy with the segment override, then recipe steps 0-2
  local NAME=$1 S0=$2 S1=$3 PRE=/home/paperspace/logs/prep_override_$1.run.py REC=/home/paperspace/logs/recipe02_$1.run.sh
  python3 - $IF/image_farm_prep.py $PRE $NAME $S0 $S1 <<'PY' || return 1
import sys
src, dst, name, s0, s1 = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
s = open(src).read()
for a, b in (('segs = json.load(open(segf))["segments"]', f'segs = [[{s0}, {s1}]]   # chilli_trainings_queue.sh override'),
             ('SD = ROOT / f"{NAME}_s{si}"', f'SD = ROOT / "{name}"'),
             ('if not run_sfm(CLIP, LOGS / f"prep_{NAME}_sfm.log", glomap=False): sys.exit(3)', 'pass   # the clip database was reclaimed; segments come from the override'),
             ('json.dump({"clip": NAME, "segment": si, "clip_frames"', f'json.dump({{"clip": NAME, "segment": "{name}", "clip_frames"'),
             ('meta = {"clip": NAME, "video": MOV.name, "segment": si,', f'meta = {{"clip": NAME, "video": MOV.name, "segment": "{name}",')):
    assert s.count(a) == 1, a; s = s.replace(a, b)
open(dst, 'w').write(s)
PY
  $PYH $PRE $G/IMG_7990 2>&1 | tail -3 | tee -a $L
  [ -s $G/$NAME/sparse/0/points3D.bin ] && [ -e $G/$NAME/gate.json ] || { say "$NAME: SfM FAILED"; return 1; }
  CUT=$(grep -n '^# ---- 3\. SAM3' $IF/image_farm_recipe.sh | cut -d: -f1)
  { head -n $((CUT - 1)) $IF/image_farm_recipe.sh; echo 'say "steps 0-2 done (chilli_trainings_queue.sh cut)"; exit 0'; } > $REC
  bash $REC $G/$NAME 2>&1 | tail -2 | tee -a $L
  [ -s $G/$NAME/transforms.json ] || { say "$NAME: recipe 0-2 FAILED"; return 1; }
}
h3dgs() {   # <segment dir>
  gpu_free; cp $IF/image_farm_h3dgs.sh /home/paperspace/logs/if_h3dgs_$(basename $1).run.sh
  say "H3DGS 8 M on $(basename $1)"; bash /home/paperspace/logs/if_h3dgs_$(basename $1).run.sh $1 60000 45000 0.0075 8000000
  say "$(basename $1): $(grep -a 'HELD-OUT\|TRAINING' /home/paperspace/logs/if_h3dgs_$(basename $1).log | tail -2 | tr '\n' ' ')"
}
gpu_free; say "=== chilli trainings start"
prep_segment IMG_7990_s1a 153 294 && { say "$(step_check $G/IMG_7990_s1a)"; step_check $G/IMG_7990_s1a > /dev/null && h3dgs $G/IMG_7990_s1a; }
say "$(step_check $G/IMG_7990_s0)"; h3dgs $G/IMG_7990_s0
prep_segment IMG_7990_s1b 295 393 && { say "$(step_check $G/IMG_7990_s1b)"; step_check $G/IMG_7990_s1b > /dev/null && h3dgs $G/IMG_7990_s1b || say "IMG_7990_s1b not trained (step check)"; }
say "=== chilli trainings done"
