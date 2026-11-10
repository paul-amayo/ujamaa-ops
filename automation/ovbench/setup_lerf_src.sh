#!/bin/bash
# setup_lerf_src.sh — ovbench (2026-11-10): published LERF source for the benchmark, outside every git checkout.
# The pixi env's editable `lerf` is the RSojen/lerf-scene-graph fork (sky loss, modified negatives, fused bounding-box embeddings,
# plus uncommitted edits) — not LERF as published. Its git history contains kerrj/lerf up to db08d57 (2024-03-13, "fix camera
# optimization bug", the last upstream commit), so `git archive db08d57` reproduces the authors' release without any network.
# One change, forced by the box: dino_extractor.create_model gets DINO ViT-S/8 from dino_timm.DinoViTS8 (same official weights via
# timm, see dino_timm.py) instead of torch.hub.load('facebookresearch/dino:main', ...), which fetches GitHub. PYTHONPATH=$DST makes
# this copy shadow the fork for ns-train's `lerf` entry point (checked: lerf.__file__).
set -euo pipefail
DST=/home/paperspace/logs/ovbench/src/lerf_kerrj_db08d57; BIN=/home/paperspace/logs/ovbench/bin
rm -rf $DST; mkdir -p $DST
git -C /home/paperspace/code/lerf-scene-graph archive db08d57 | tar -x -C $DST
cp $BIN/dino_timm.py $DST/lerf/data/utils/dino_timm.py
python3 - $DST/lerf/data/utils/dino_extractor.py <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
old = "        if 'dino' in model_type:\n            model = torch.hub.load('facebookresearch/dino:main', model_type)\n"
new = ("        if 'dino' in model_type:\n            # ovbench: torch.hub fetches GitHub (blocked on the A100); same network + official weights through timm\n"
       "            assert model_type == 'dino_vits8', model_type\n            from lerf.data.utils.dino_timm import DinoViTS8\n            model = DinoViTS8()\n")
assert s.count(old) == 1, 'patch anchor not found'; open(p, 'w').write(s.replace(old, new)); print('[setup] dino_extractor.create_model patched')
EOF
diff -u <(git -C /home/paperspace/code/lerf-scene-graph show db08d57:lerf/data/utils/dino_extractor.py) $DST/lerf/data/utils/dino_extractor.py > $DST/OVBENCH_PATCH.diff || true
echo "[setup] $DST ready; patch recorded in $DST/OVBENCH_PATCH.diff"
