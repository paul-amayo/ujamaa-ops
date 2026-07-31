# Skill candidates — validated procedures awaiting promotion

Curated from sessions 2026-07-22/23. Each entry: what it does, the validated
assets, and what a skill write-up must carry. Promote with skill-creator when
Paul green-lights; several also imply EDITS to existing skills (noted).

## 1. citrus-absolute-association  (REPLACES the core of citrus-4d-association)
- **What:** cross-survey tree association in absolute WGS84 — per-survey GPS anchor,
  timestamp joins, GPS↔LIO latency correction, trajectory-corroborated common-mode
  shift, mutual-NN @ 1.5 m gate, ungameable residual-translation diagnostic.
- **Assets:** `aru_sil_core/src/scripts/tree_association_abs.py`,
  `calibrate_gps_lio_latency.py`, `sankofa_substrate/gps_lio_latency.json`.
- **Must carry:** why fitted offsets are FORBIDDEN (one-pitch neighbour-matching
  post-mortem, notebook 2026-07-23); the pitch-vs-residual sanity test; the
  component-not-tree caveat from clustering. Deprecation banner already added to
  the old skill (2026-07-23).

## 2. gps-monolithic-regen  (extends citrus-rosbag-pipeline)
- **What:** replicable regeneration of true-WGS84 `gps.monolithic` for any pre-fix
  survey: redownload only base_* bags (md5), sandboxed per-bag extraction, time-ordered
  stream concat, degrees+monotonic validation, install touching ONLY gps.monolithic
  (.enu_backup kept).
- **Assets:** `aru_sil_core/src/scripts/regen_gps_monolithics.sh` (idempotent,
  disk-guarded). Validated on 01/03 (2026-07-22→23).

## 3. lidar-cloud-topdown + detection placement
- **What:** survey-wide LiDAR cloud accumulation via LaserProjector (every-2nd-kf,
  0.1 m ground cells, count + canopy-top height), optional per-scan range clip
  (10 m) — plus the median-in-bbox placement that puts any per-frame detection at
  its LiDAR position (the tool that exposed the registry undercount).
- **Assets:** /tmp scripts `build_cloud_04*.py`, `place_dets.py` — need promoting
  into aru scripts/ with survey parameterization. Outputs in sankofa_substrate/.
- **Must carry:** y-is-down convention; ground = high-percentile top_y; full-range
  cloud reaches ±150 m (neighbour blocks) vs 10 m clip = "what clustering saw".

## 4. fruit-in-tree-masks  (recursive hierarchy detection)
- **What:** SAM3 fruit prompting inside existing tree-mask bboxes; parent-mask
  rejection; per-view counts with median/max aggregates.
- **Assets:** `aru_sil_core/src/scripts/fruit_in_tree_masks.py`,
  `validate_fruit_sam3.py`; outputs `sam3_v2/fruit_v1*.json` (04 done, 05 running).
- **Must carry (hard-won):** bbox-crop ≠ masking (rejection exists because crops
  include ground/neighbours); TILING MAKES IT WORSE (11→5 kept, tested — SAM3 needs
  scene context); illumination dominates (18-vs-0 same tree same frame, sun-relative
  angle); counts are per-VIEW visible fruit of a COMPONENT, not yield; prompt
  "fruit" ≥ "orange" ≈ "citrus fruit" on this imagery.

## 5. graph-embedder-template  (updates train-aru-block + full-pipeline guidance)
- **What:** per-survey HyperEmbedder training via the graph-only trainer with the
  validated c20cos20 recipe (contrastive 2.0, cos-recon 2.0, recon 1.0, temp 0.2,
  --no-level-norms --learn-curv, 1500ep/b16, ~3 min/survey) + retrieval validation
  (child→nearest-parent ≥95%; interpolation-path = same) + pairing rule
  (HIGH_EMBEDDER_CKPT one knob end-to-end).
- **Must carry:** the walk-decode Row metric is layout-dependent and reads ~random
  on healthy models — judge by 'Row (retrieval)'; radial-norm forcing DESTROYS
  retrieval; nesting emerges unforced (norms 5.16→2.48→1.03).

## 6. registry-qa  (clustering audit)
- **What:** decompose any registry tree into its placed member detections; 2-means
  separation vs tree pitch flags multi-tree merges (04 "tree 25" = 30 m segment);
  X/O maps of mapped vs unclustered detections; raw-vs-registry census comparison
  against the LiDAR cloud.
- **Assets:** /tmp `tree_decomp.py`, `dets_xo*.py`, `sem_stageA.py`+`sem_stageB.py`
  (semantic cloud: SAM3-masked LiDAR, two-env split, 10 m depth clip) — promote +
  parameterize.
- **Why it matters:** first-line QA before trusting any per-tree measurement;
  found the most upstream defect in the stack. The semantic cloud LOCALIZED it:
  110/112 census peaks are present in the mask-gated input, so the defect is in the
  voxel-CC step, not SAM3 — and re-clustering the saved semantic cloud needs no
  SAM3 re-run.

## 7. sankofa-ledger  (build + compare)
- **What:** observations ledger (tree×epoch; "row" reserved for planting rows) +
  cohort comparison with robust-z anomaly flags; NDVI columns via per-survey
  absolute sampling (`tree_ndvi_survey.py`).
- **Assets:** `ujamaa/sankofa/build_ledger.py`, `compare.py`,
  `LEDGER_OPEN_QUESTIONS.md`; substrate under `citrus_all/sankofa_substrate/`.
- **Must carry:** identities are clustering COMPONENTS (see #6) — ledger quality is
  bounded by registry quality; epoch dates from bag timestamps (13B pair is 2 days
  apart → comparison v1 is a repeatability baseline, not phenology).

## PARKED (get later): fruit-detector uplift
- Zero-shot ceiling reached (SAM3 ≈ YOLO-World, notebook 2026-07-23). Paths when
  picked up: fine-tune small-object detector on ~100 crops from our fired views;
  l-model@1280 sweep for the ground-fruit layer; exemplar-query re-scoring.
- Ready assets: ~/venvs/yolow2 env, /tmp/yolow_{validate,sweep_04,place_04}.py,
  figs/yolow_validation.png.

## OPEN TO-DOs carried out of the 2026-07-25/27 relevancy investigation
1. **Fruit-level collapse (blocker for the fruit demo).** Fruit px render at radius
   4.174 vs target 7.14; 0.0% of fruit px pick a fruit anchor when trees compete.
   Tree/row levels are healthy (95.8% / 99.1% pointing, IoU 0.702).
   **PLAN REVISED 2026-07-28 — FruitNeRF-style bootstrap (arXiv 2408.06190, IROS'24)
   replaces the bare 10x up-weight as first move.** Their result (F1 0.95 with noisy
   Grounded-SAM masks at 0.15 thresholds) is direct evidence that a SEPARATE binary
   fruit channel + multi-view fusion denoises exactly the scarcity that killed our
   level (~1% of gradient, each orange painted in ~1 frame). Three gated stages on
   block_001:
   (a) **Fruit-channel probe** — 1-channel binary fruit head on the HiGH splat
       (freeze clean 20k geometry, train channel only). Supervise with UNION of
       low-conf SAM3 fruit masks; no curation. NOTE fruit_v1*.json store counts
       only, NO masks — regenerate via build_tree_instances --prompt fruit
       --conf 0.05 --min-mask-area 40 --keep-frames <block frames> (scores now
       stored, so threshold is post-hoc). Gate: rendered density lights up on
       held-out oranges.
   (b) **Fruit instances** — sample channel -> fruit cloud -> cascaded clustering
       (DBSCAN + citrus size template ~7-9 cm, merge) -> assign to tree instances.
       Yields per-tree counts for Sankofa INDEPENDENT of the embedding, and the
       true fruit-node inventory (20 nodes is a floor).
   (c) **Bootstrap the hierarchy** — render channel into all frames -> dense
       pseudo-masks -> repaint semantic_v2_B fruit ids (cluster = node, parent =
       tree) -> retrain 5-level embedder + splat WITH the 10x up-weight.
       Gate: cross-level pointing (fruit px choosing fruit anchors, trees competing).
   Launch prep done 2026-07-28: /tmp/b001_keep_frames.json (124 frames),
   /tmp/fruit_prep_launch.sh (fires after the klapmuts sack chain frees the GPU).
2. **Checkpoint hygiene — apply to everything already trained.** Opacity resets
   (every 3000 steps until stop_split_at) corrupt any checkpoint saved on the
   boundary (measured 5.1x and 7.5x on two of ten). Verify each E-03 block's feature
   loss against its logged value before serving/exporting; offset steps_per_save.
3. **Re-export block_001** from the clean ladder ckpt (/tmp/ladder/FULL, step 20000)
   — that is the good artifact; the F4 export is corrupted.
4. **Serving formula** now negatives-free + steps=4 in clip_encoder (UNCOMMITTED in
   high/ — the commit was interrupted). Re-validate the viewer after committing.
5. **Vocabulary**: tree/fruit words were never chosen for separation (tree-word
   pairwise cosine p90 0.90, max 0.9993 — one pair is effectively identical).
   Consider greedy max-min word selection before the next embedder retrain.
6. **E-03 batch parked at 57/80** — resumable with automation/day_gpu_e03.sh.
7. `nvidia-smi` reports a driver/library mismatch (cosmetic — CUDA compute, training
   and allocator all work). Reboot only if a real CUDA failure appears.
