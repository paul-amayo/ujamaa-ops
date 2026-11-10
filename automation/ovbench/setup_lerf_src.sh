#!/bin/bash
# setup_lerf_src.sh — ovbench (2026-11-10): published LERF source for the benchmark, outside every git checkout.
# The pixi env's editable `lerf` is the RSojen/lerf-scene-graph fork (sky loss, modified negatives, fused bounding-box embeddings,
# plus uncommitted edits) — not LERF as published. Its git history contains kerrj/lerf up to db08d57 (2024-03-13, "fix camera
# optimization bug", the last upstream commit), so `git archive db08d57` reproduces the authors' release without any network.
# One change, forced by the box: dino_extractor.create_model gets DINO ViT-S/8 from dino_timm.DinoViTS8 (same official weights via
# timm, see dino_timm.py) instead of torch.hub.load('facebookresearch/dino:main', ...), which fetches GitHub. PYTHONPATH=$DST makes
# this copy shadow the fork for ns-train's `lerf` entry point (checked: lerf.__file__).
# Second change, type annotations only: upstream config fields carry annotations that do not match their defaults
# (patch_tile_size_range: Tuple[int, int] = (0.05, 0.5); hashgrid_layers: Tuple[int] = (12, 12); negatives: Tuple[str] = 4 words, ...);
# tyro in nerf_new refuses them when building the ns-train CLI. They become variadic / float tuples with the SAME default values —
# no behaviour change (the same edit the fork made; the fork's other edits are not taken).
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
python3 - $DST <<'EOF'
import sys
R = {'lerf/data/lerf_datamanager.py': [('patch_tile_size_range: Tuple[int, int] = (0.05, 0.5)', 'patch_tile_size_range: Tuple[float, float] = (0.05, 0.5)')],
     'lerf/lerf.py': [('hashgrid_layers: Tuple[int] = ', 'hashgrid_layers: Tuple[int, ...] = '), ('hashgrid_resolutions: Tuple[Tuple[int]] = ', 'hashgrid_resolutions: Tuple[Tuple[int, ...], ...] = '),
                      ('hashgrid_sizes: Tuple[int] = ', 'hashgrid_sizes: Tuple[int, ...] = ')],
     'lerf/encoders/openclip_encoder.py': [('negatives: Tuple[str] = ', 'negatives: Tuple[str, ...] = ')], 'lerf/encoders/clip_encoder.py': [('negatives: Tuple[str] = ', 'negatives: Tuple[str, ...] = ')]}
for f, subs in R.items():
    p = sys.argv[1] + '/' + f; s = open(p).read()
    for old, new in subs:
        assert s.count(old) == 1, (f, old); s = s.replace(old, new)
    open(p, 'w').write(s)
print('[setup] annotations patched (types only, defaults unchanged)')
EOF
: > $DST/OVBENCH_PATCH.diff
for f in lerf/data/utils/dino_extractor.py lerf/data/lerf_datamanager.py lerf/lerf.py lerf/encoders/openclip_encoder.py lerf/encoders/clip_encoder.py; do
  diff -u --strip-trailing-cr <(git -C /home/paperspace/code/lerf-scene-graph show db08d57:$f) $DST/$f >> $DST/OVBENCH_PATCH.diff || true
done
echo "[setup] $DST ready; patch recorded in $DST/OVBENCH_PATCH.diff"
