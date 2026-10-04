#!/bin/bash
# Paired held-out eval of 05 chunk 1_0: the expo model vs the 01-recipe retrain on the SAME 71 views (the 1-in-10 list; both
# held them out: expo held out 365 of 659, a superset). The retrain chain's eval stage failed (h3dgs_01recipe had no
# camera_calibration/aligned; the aligned dir is symlinked from h3dgs_expo, whose aligned test.txt is the same 1-in-10 list).
set -u
E=/home/paperspace/data/citrus_all/05_13D_Jackal/experimental; L=/home/paperspace/logs/eval_05_1_0_paired.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
[ -e $E/h3dgs_01recipe/camera_calibration/aligned ] || ln -s $E/h3dgs_expo/camera_calibration/aligned $E/h3dgs_01recipe/camera_calibration/aligned
export PATH=/home/paperspace/miniconda3/envs/h3dgs/bin:/home/paperspace/code/_cuda12/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/paperspace/code/hierarchical-3d-gaussians
for P in h3dgs_expo h3dgs_01recipe; do
  timeout 1800 python /home/paperspace/logs/h3dgs_eval_chunk.py $E/$P --hier output/trained_chunks/1_0/hierarchy.hier_opt --taus 0 --only_chunk 1_0 --out output/eval_heldout71 --save 6 > /home/paperspace/logs/eval_05_1_0_$P.log 2>&1
  say "$P rc=$?: $(grep -aE '^\[eval\] tau' /home/paperspace/logs/eval_05_1_0_$P.log | cut -c1-240)"
done
python3 - $E <<'PY' | tee -a $L
import json, sys, numpy as np
E = sys.argv[1]; A = {r['name']: r for r in json.load(open(f'{E}/h3dgs_expo/output/eval_heldout71/scores.json'))}; B = {r['name']: r for r in json.load(open(f'{E}/h3dgs_01recipe/output/eval_heldout71/scores.json'))}
k = sorted(set(A) & set(B)); d = np.array([B[n]['psnr'] - A[n]['psnr'] for n in k])
print(f'[paired] {len(k)} views held out in both: expo mean {np.mean([A[n]["psnr"] for n in k]):.2f} median {np.median([A[n]["psnr"] for n in k]):.2f}; '
      f'01r mean {np.mean([B[n]["psnr"] for n in k]):.2f} median {np.median([B[n]["psnr"] for n in k]):.2f}; delta mean {d.mean():+.2f} median {np.median(d):+.2f}, 01r better on {(d > 0).sum()}/{len(k)}')
for key in ('psnr_nosky', 'psnr_fg'):
    kk = [n for n in k if A[n].get(key) is not None and B[n].get(key) is not None]
    if kk: print(f'[paired] {key}: {len(kk)} views, expo {np.mean([A[n][key] for n in kk]):.2f} -> 01r {np.mean([B[n][key] for n in kk]):.2f}')
PY
say "paired eval done"
