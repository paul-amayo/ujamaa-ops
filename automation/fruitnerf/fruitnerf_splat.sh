#!/bin/bash
# fruitnerf_splat.sh <tree_0X> — splat per FruitNeRF tree for the splat-based fruit registry: v3 recipe (splatfacto + bilateral grid, cull alpha
# 0.005, 30k) on the undistorted 1620x1080 working set with FruitNeRF's poses, ALL frames in training (eval-mode all), init = the DA3 fused cloud
# (same frame as the poses) subsampled to 500k points.
set -uo pipefail; T=${1:?}; S=/home/paperspace/data/public/fruitnerf/high/$T; NS=/home/paperspace/code/nerf_new; L=/home/paperspace/logs/fruitnerf_${T}_splat.log
python3 - "$S" <<'PY'
import sys, json, numpy as np
from pathlib import Path
S = Path(sys.argv[1]); src = S / 'init_da3_fused.ply'; dst = S / 'init_da3_500k.ply'
if not dst.exists():
    data = src.read_bytes(); hdr_end = data.index(b'end_header\n') + len(b'end_header\n'); hdr = data[:hdr_end].decode()
    n = int([l for l in hdr.splitlines() if l.startswith('element vertex')][0].split()[-1]); props = [l.split() for l in hdr.splitlines() if l.startswith('property')]
    dt = np.dtype([(p[2], {'float': '<f4', 'double': '<f8', 'uchar': 'u1'}[p[1]]) for p in props]); v = np.frombuffer(data[hdr_end:hdr_end + n * dt.itemsize], dtype=dt)
    keep = np.random.default_rng(0).choice(n, size=min(500000, n), replace=False); v = v[np.sort(keep)]
    dst.write_bytes(hdr.replace(f'element vertex {n}', f'element vertex {len(v)}').encode() + v.tobytes())
tj = json.loads((S / 'transforms.json').read_text()); tj['ply_file_path'] = 'init_da3_500k.ply'; (S / 'transforms.json').write_text(json.dumps(tj, indent=1)); print('init ready')
PY
cd $NS && env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 5 pixi run ns-train splatfacto --data $S --output-dir $S/outputs --experiment-name sf_30000_bilateral --timestamp run --vis tensorboard \
  --max-num-iterations 30000 --steps-per-eval-all-images 1000000 --steps-per-eval-image 1000000 --steps-per-eval-batch 1000000 --pipeline.model.cull-alpha-thresh 0.005 --pipeline.model.use-bilateral-grid True \
  nerfstudio-data --eval-mode all > $L 2>&1
echo "[$(date '+%m-%d %H:%M:%S')] $T splat rc=$? -> $(ls $S/outputs/sf_30000_bilateral/splatfacto/run/nerfstudio_models/ 2>/dev/null | tail -1)" | tee -a /home/paperspace/logs/fruitnerf_$T.log
