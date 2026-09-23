# 2 · Poses — odometry to per-block camera poses worth training on

Pose quality is the single biggest lever on reconstruction quality we have
measured: on the same images, the same block went **12.1 dB (poses as
recorded) → 15.1 dB (frame conventions fixed) → 18.6 dB fleet median
(image-refined)**. Budget time here, not in training tricks.

## 2.1 Frame sanity (once per rig configuration)

Run the convention check — it solves a small photogrammetric reference on
one block and compares camera axes:

```bash
pixi run python pipeline/poses/check_conventions.py data/<survey_id> --block 0
```

Camera-axis agreement should be ≈ +1.0 on all three axes. A clean
**negative** pattern (e.g. x −0.99, y −0.99, z +0.99) means a constant
frame error between odometry and images — the capture-guide inversion
case. The fix is one stream-level conjugation, applied once and promoted;
the tool prints the transform it measured.
<!-- CONSOLIDATION-CONTRACT: today tenrows_colmap_axes.sh +
tenrows_rollfix.py; released as one check with an explicit apply step. -->

If you have INS: apply the yaw-drift correction next; the internal mirror
sign the fit reports must come out **+1** — a −1 fit is the fingerprint of
an unfixed frame error, not a property of your INS.

## 2.2 Keyframes and row blocks

```bash
pixi run python pipeline/poses/cut_keyframes.py data/<survey_id> --spacing 0.20
pixi run python pipeline/poses/build_row_blocks.py data/<survey_id> --max-kf 100
```

- **20 cm spacing is the shipped default** — measured: with good poses,
  the full frame stream gains nothing over 20 cm keyframes (+0.13 dB, a
  wash) and costs 2.7× per epoch. With *bad* poses the full stream looks
  +0.8 dB better — by averaging pose noise. Don't chase that; fix poses.
- Blocks are ~100-keyframe row segments; each trains independently
  (~6 min each on an A100-40, one at a time on smaller cards).
- Block poses are written **OpenGL c2w** — the contract every consumer
  assumes. (We once wrote OpenCV c2w into that contract; every run
  trained, none complained, all were ~1 dB worse. Convention bugs are
  silent — hence the §2.1 check.)

## 2.3 LiDAR initialisation

```bash
pipeline/poses/lidar_init_all.sh data/<survey_id>    # per block
```

Expect **250k–500k voxel points per block**. Min-range default 0.45 m
(clears rig self-returns; raising it to 0.6 m measurably deleted real
canopy — set it from your own §0 pre-flight band, not higher).

## 2.4 Image-based refinement (the +3 dB step — do not skip)

```bash
pipeline/poses/refine_blocks.sh data/<survey_id>
```

Per block: a photogrammetric solve on the block's keyframes → similarity-
aligned back into the survey's world frame → LiDAR init kept. Measured on
a 23-block survey: **every block improved, +2.0 to +4.9 dB, median +2.9**.

Two hard-won rules baked into the tool:
- **Global mapper (GLOMAP), not incremental SfM**: on repetitive row
  crops, the incremental mapper failed or mis-merged 10 of 23 blocks;
  the global mapper recovered all 23 at 1–10 cm alignment.
- **Refuse bad solves**: <90 % of frames registered or alignment residual
  > 0.5 m → the block keeps its odometry poses (a bad refinement is worse
  than none).

Do NOT use the trainer's in-loop camera optimiser instead: measured
−3.7 dB on our data.

## Verify

The driver prints per-block: frames registered, alignment residual (p50,
expect ≤ 0.1 m), camera-axis agreement (expect +0.99…). The real gate is
the next stage's held-out PSNR.

Next: [03_train_splats.md](03_train_splats.md).
