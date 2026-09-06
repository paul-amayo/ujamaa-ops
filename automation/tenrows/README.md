# dec_2025_ten_rows (A300 / ZED / Ouster) — image+LiDAR stage-1 recipe (2026-09-06)

Working copies of the scripts live in `~/logs/` (paths hardcoded to
`/home/paperspace/data/klapmuts/dec_2025_ten_rows`); these are the committed snapshots.
Envs: `PY=nerf_new/.pixi/envs/default/bin/python3.10` with
`PYTHONPATH=aru_sil_core/src/interfaces/build/temp.linux-x86_64-3.10/lib` (aru_nerf_interface);
the writer binding (aru_py_logger, cp310 lib dir) in a SEPARATE process; InstantSplat env for
`plyfile`; GPU-stack COLMAP = `glomap/build_gpu/_deps/colmap-build/src/colmap/exe/colmap` with
the CUDA-12 `LD_LIBRARY_PATH` (see `tenrows_colmap_axes.sh`). Always `from PIL import Image`
before importing the binding.

## Order
1. `tenrows_ingest.sh` (mcap → monolithics), `tenrows_post_ingest.sh`.
2. **Pose frame fix** `tenrows_rollfix.py`: the recorded ZED odometry body frame is rolled 180°
   about the optical axis vs the images (measured vs COLMAP). `inc' = C·inc·C`, C=diag(-1,-1,1).
3. **INS yaw-drift correction** `ZED_IN=zed_transform_rollfix.monolithic OUT_SUFFIX=rollfix_inscorr
   tenrows_zed_inscorr.py` → `transform_lio_laserframe_rollfix_inscorr.monolithic`
   (laser-frame conjugation with the rig L2C; mirror sign must come out +1).
   Promote: `cp … monolithics/transform_lio.monolithic` (real file, no symlinks).
4. Survey cloud + row map: `tenrows_laser_cloud.py --monos … --transform … --out …`,
   `tenrows_topdown_v3.py --npz … --out …`.
5. Keyframes (20 cm on the camera path): `tenrows_kfcut_A.py` (reader process: selection +
   `prod/scratch_sam3/kf_%06d.png` + `kf_index.json`) then `tenrows_kfcut_B.py` (writer process:
   `image_left_kf20cm.monolithic`). Validate PNGs with `Image.load()` after any interrupted dump.
6. Row blocks: `build_row_blocks.py --data-dir <survey> --outcfg-name lio_row100 --max-kf 100`
   (writes OpenGL c2w since 2026-09-06; rig.json intrinsics fallback). Strip any stale
   `fg_masks` wiring.
7. LiDAR init per block: `tenrows_init_all.sh` (`lidar_init_per_block.py --root <survey root>`;
   passes the rig L2C; expect 250k–500k voxel points per block).
8. A/B arm (keyframes vs full stream, same held-out kf): `tenrows_ab_build.py block_013`
   (+ absolute split paths). Stage-1 fleet: `tenrows_stage1_driver.sh` (gated on the QLoRA
   pause flag; A/B pair first). Held-out metrics: `tenrows_ns_eval.sh <block…>` (all views;
   the tensorboard eval tag is a single image every 500 steps). Reports: `tenrows_ab_report.py`,
   `tenrows_blocks_report.py`, `tenrows_pose_strip.py`.
9. **Refined poses (+3 dB):** `tenrows_colmap_axes.sh <block>` (GPU SIFT, exhaustive, mapper with
   calibrated intrinsics fixed; prints camera-axis dots vs our poses) → `tenrows_refine_block.py
   <block>` (COLMAP poses Sim(3)-aligned into the LIO world, LiDAR init kept) →
   `tenrows_stage1_driver_ref.sh`.
Diagnostics: `tenrows_c2w_diag.py` (projector c2w vs pose formula), `tenrows_axes_test.py`
(init projected under both axis conventions), `tenrows_block_check.py` (init vs cameras top-down).

## Numbers (block_013, 87/13 kf split, 15k iters, ns-eval on the 13 held-out kf)
poses as recorded 12.11 dB → GL flip only 13.11 → roll-corrected 15.13 (COLMAP init) / 14.43
(LiDAR init) → COLMAP-refined in our world 17.52 → COLMAP-native 17.72. Full stream (233 views)
vs keyframes (87) on corrected poses: +0.81 dB PSNR, +0.044 SSIM, LPIPS −0.057. In-loop
camera optimiser: 11.41 (worse). Wall ≈ 6 min per 15k-iter block alone on the A100-40GB.
