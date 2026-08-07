# Testing & reproducibility

Doctrine + ledgers. The daily dashboard parses the `AUTONOMY:` lines in
notebook entries and the tables in this file — keep formats intact.
Motivation (2026-08-04): three camera-sign errors in one session, a depth
npz misaligned for five weeks, and pipelines that work on one scene family
silently breaking on the next. "The skills and pipelines are there but being
broken by new datasets" — Paul.

## 1. The convention landmines (found by scan, 2026-08-04)

**Camera basis flips by BLOCK-GENERATION ERA, not by survey.** Every survey's
LIO json stores OpenCV c2w (camY = +y world, y down-positive). But
`nerf_per_block.py` applies `opencv_to_opengl` when writing arc blocks, so
citrus-01 arc blocks and gendia-osuga carry camY = −y **in the same
transforms.json format with nothing declaring it**. `camera_convention.py`
(tested, aru) is correct for 03/04/05/klapmuts row+pass blocks and silently
v-flipped for 01-arc and osuga. Any script mixing block poses with metric
LiDAR must check the basis first.

**Everything metric dies at gendia phone scenes.** COLMAP poses are ~2.5–3.4×
off metric (and different per scene), gravity is not axis-aligned, distortion
is strong (k1≈0.13, k2≈−0.2), images are portrait. Every hardcoded metre and
pixel (zmin/rmax/voxel/pads/px-radius/SAM3 upscale) is wrong there, silently.

## 2. Scene difference matrix

| | citrus 01–03 (13B) | citrus 04/05 (13D) | klapmuts | gendia osuga | gendia mango/pawpaw/sukuma |
|---|---|---|---|---|---|
| scene | separated trees, 5.9 m pitch | close-range trees ~4.2 m irregular | bushes | low crop, handheld arc | low-canopy crops, phone video |
| poses | LIO metric | LIO metric | LIO metric | LIO-ish, 188 kf/12 m | COLMAP, arbitrary units |
| lidar / GPS | yes / ENU | yes / 04 true WGS84 | yes / NO | no / no | no / no |
| camera | ZED 1280×720 k=0 | same | same | phone 1080×1920 fx1609 k1.04 | phone 2160×3840 fx~3150 k1.13 |
| block basis | arc=OpenGL, pass=OpenCV | OpenCV | OpenCV | OpenGL | COLMAP, gravity unaligned |
| sky | yes | yes | yes | little | none → sky sup meaningless |
| fruit | oranges ~7 cm | oranges | berries | — | mango big / pawpaw clustered / sukuma NONE |

## 3. Scene manifest (`scene.yaml` at each survey root) — SPEC, to build

Pins the axes above so scripts read instead of assume: `basis`
(opencv/opengl + world-up vector), `scale` (metric? factor), `pose_source`,
`sensors` (lidar/gps/stereo), `camera` (res, orientation, distortion),
`sky`, `canopy_class` (tree/bush/low-crop), `row_pitch` (or none),
`fruit` (type, expected px). Ingest computes the derivable ones and emits a
**pytest fixture per scene family** (1 frame + ≥4 known 3D points, like
`test_camera_convention.py`) so a converter changing convention fails the
suite, not a training run five weeks later.

## 4. Geometry gates (`geometry_gates.py` — to build, run beside FG-PSNR)

PSNR is appearance; geometry can be badly wrong under decent FG-PSNR.
1. **Free-space violation** (lidar scenes): opacity mass in front of
   z-buffered lidar depth — space lidar certifies empty. Floater metric.
2. **Occupancy IoU**: opacity-weighted splat means voxelized 10–20 cm vs
   survey cache. Missing-volume metric. (NN-to-dense-cloud stays BANNED.)
3. **Eval-view depth error**: rendered depth at eval poses vs z-buffer maps,
   median |ΔZ| on supervised px.
4. **Silhouette IoU** (works without lidar → the gendia gate): rendered
   alpha vs SAM3 fg masks at eval views.
5. **Cross-view fruit consistency**: frame-A fruit px → splat depth →
   frame B; distance to B's fruit masks (tracks the ~100 px issue).

Headline appearance numbers wherever sky supervision is on: **train-split
TREE + FRUIT PSNR** (the product is a flythrough near the trajectory; train
views are what the viewer shows). Full-image PSNR is retired. Eval PSNR at
this dataset's ~25 cm keyframe spacing is an INTERPOLATION test — gross-
breakage check only, never a tuning target (2026-08-05). The off-axis risk
of trajectory overfitting is floaters — gate 1, not PSNR.

## 4b. FRUIT SIGN-OFF (2026-08-06) — the full battery, or it did not happen

No fruit-bearing checkpoint is quoted, served, or promoted without ALL of
these, in one report (`automation/fruit_signoff.sh <run_dir>`):

| # | test | tool | pass bar |
|---|------|------|----------|
| 1 | appearance vs scene baseline | score_splat (train FG/TREE/FRUIT) + kf_542 render | >= baseline − noise; stage2 runs: geometry tensors BIT-IDENTICAL |
| 2 | per-level rendered NORM | fruit_signoff norms check | fruit-px norms at ~7.14, trees at their per-word targets ~4.5-5.3 (depth diagnostic; census-init era) |
| 3 | NO-WALK cross-level pointing | fruit_pointing_map --no-walk | THE headline: recall@fruit px at FP ~0 |
| 4 | walked pointing + FP anatomy | fruit_pointing_map | reference/query semantics; own-fruit fraction reported |
| 5 | ceiling control | fruit_pointing_map --gt-features | must stay 100/100/0 (scoring-path canary) |
| 6 | full relevancy eval | eval_r6_relevancy --no-negatives | object/row pointing, within-level fruit, IoU/AUC-PR, multi-frame |
| 7 | aligned gates | aru_sil_core/src/scripts/aligned_gates.py (HIGH_EMBEDDER_CKPT must match the run) | mass/coverage vs the run's OWN supervision |
| 8 | CONTAINMENT ladder | aru_sil_core/src/scripts/containment_eval.py | per-view row/tree/fruit best-IoU masks from single bare words (the paper-highlight test); words via live table, never literals |

Test 8 (cross-view 542->543 consistency) was DROPPED 2026-08-07 on Paul's
call: cross-view geometry is out of scope for the field verdict; per-view
performance gauges it. If cross-view identity ever returns as a requirement,
it re-enters as a NEW numbered test with a tool that resolves words through
the live vocab (the old s9b tool selected 0 gaussians after the v4_1k swap).

Sign-off = the table filled with numbers + Paul's explicit OK recorded in
the notebook entry. Partial suites are labelled PARTIAL and cannot sign off.
History that mandates this: within-level pointing quoted as cross-level
(2026-08-06), walk degeneracy faking 88.6%, level collapse scoring 96%.

## 5. Pipeline autonomy score (daily)

Every day's notebook entry set ends with one machine-parsed line:

    AUTONOMY: runs=<started> clean=<no-intervention> interventions=<human actions needed> debugged=<experiments that needed debugging before yielding a result>

Definitions: an *intervention* is any human/manual action a pipeline should
have handled (kill+relaunch, hand-editing a config, un-sticking a queue, a
permission the tooling forced). *Debugged* counts experiments whose first
result was invalid (wrong convention, misaligned data, broken metric) and
had to be re-run or re-measured. Honest zeros are rare; that is the point.
The dashboard plots clean/runs and interventions per day.

## 6. New-dataset-to-tassili ledger

Clock starts when the raw data lands on disk, stops when a queryable splat
is served on tassili (8001/8002). Dashboard surfaces the open rows.

| dataset | landed | first splat | on tassili | days | interventions | notes |
|---|---|---|---|---|---|---|
| citrus 04_13D | 2026-07 | 2026-07 | 2026-07 | ~5 | many | fruit ladder scene; survey-cache init 08-04 |
| citrus 02_13B redownload | 2026-08-01 | — | — | open | 1 | ingest pending |
| klapmuts dec_2025 (A300) | 2025-12-03 | — | — | open | — | ROS2 mcap, LIO route, net-new ingest |
| gendia beans_multi_row | 2026-07 | — | — | open | — | phone video, COLMAP-era |
| TEMPO-VINE | — | — | — | pending access | — | mcap + non-ZED rig |
