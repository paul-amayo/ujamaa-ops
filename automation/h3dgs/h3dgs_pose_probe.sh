#!/bin/bash
# h3dgs_pose_probe.sh <h3dgs proj dir> <chunk> [extra pose_probe.py args]
# Test-time pose refinement probe on one trained chunk (pose_probe.py in the H3DGS repo): frozen hierarchy, per-view
# 6-DoF correction, sky-masked PSNR before/after. Eval-only; touches nothing in the survey. Log: caller's redirect.
P=$1; C=$2; shift 2
export CUDA_HOME=/home/paperspace/code/_cuda12 PATH=/home/paperspace/code/_cuda12/bin:$PATH
cd /home/paperspace/code/hierarchical-3d-gaussians || exit 1
exec /home/paperspace/miniconda3/envs/h3dgs/bin/python -u pose_probe.py -s "$P/camera_calibration/chunks/$C" \
  --model_path "$P/output/trained_chunks/$C" --hierarchy "$P/output/trained_chunks/$C/hierarchy.hier_opt" \
  -i ../../rectified/images --eval --scaffold_file "$P/output/scaffold/point_cloud/iteration_30000" --fill "$@"
