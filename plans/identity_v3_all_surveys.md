# Identity on the v3 models for Klapmuts, Lindendhof, the image farm and GenDIA (plan, 2026-10-12)

Paul (16:2x): "identity for klapmuts, lindendhof and all the image farm runs and gendia". The citrus surveys (82 cells) already
carry v3 identity. Identity on a v3 model needs five inputs; the order below follows what each survey already has.

| input | what it is | how it is made |
|---|---|---|
| A. v3 RGB model | splatfacto + bilateral grid, 60k | `train_chunk.sh` on a workspace (`build_chunk_ws.py`) |
| B. hierarchy | plants / rows / fruit with global ids | markers (SAM3 lifted with depth) → `build_marker_hierarchy` / census |
| C. embedder | HiGH hyperbolic embedder of B | `high/train_hyperembedder_graph.py` (citrus graph recipe: c20cos20, 1500 ep, `--no-level-norms`) |
| D. supervision | per-frame id maps | block process (`save_filtered_semantic_pngs` + `compile_supervision`) on keyframes |
| E. identity | census on A with D → seed B → text bank → verdicts | `v3_identity_cell.sh` / `v3_identity_from_maps.sh` |

## Inventory (measured 2026-10-12 16:3x)
| survey | A v3 | B hierarchy | C embedder | D maps | missing |
|---|---|---|---|---|---|
| Klapmuts April (4 cells + lane 2) | yes | yes | apr_2026_zed_v1g | 2,006 keyframe maps (25 blocks) | nothing → **running now** |
| Klapmuts Dec (10 lanes) | yes | registry v5 (sankofa_demo, 911 objects) | none | none (markers_v2_B + filtered_semantic_v2_B + sam3_v2 exist) | C, D |
| Lindendhof (5 lanes) | yes | none | none | none | keyframes, SAM3 (site prompt), lifting, B, C, D |
| image farm cabbage / pepper | yes (2 lanes) | yes | yes | yes | E only |
| image farm, 45 other segments | 1 (IMG_7975_s0 lane) | 10 of 45 (depth-DBSCAN) | none | none (SAM3 sam3_v2 on 45) | A (44), B (35), C, D |
| GenDIA osuga / osuga_baseline / sukuma | none (blocks deleted 10-08) | yes (3–6 objects) | hyper/*_v1c | markers + filtered_semantic | A, D |
| GenDIA mango / pawpaw | none | none | none | SAM3 only | A, B, C, D |
| GenDIA osuga_full | none | none | none | nothing (1,412 frames, no keyframes) | everything |

## Order (GPU-light first; each family closes with containment verdicts like the citrus cells)
1. April cells + lane 2 (maps exist; lane keyframes → stream frames by `kf_map.json`); cabbage and pepper on their v3 lanes.
2. Klapmuts Dec: embedder from the v5 registry, keyframe supervision by the block process, per-lane `kf_map` → census.
   Check: the v5 registry ids must be the ids the supervision paints (renumbering trap, see phone-segment-sidecar memory).
3. GenDIA osuga / osuga_baseline / sukuma: v3 workspaces from UNDISTORTED images (phone OPENCV k1/k2; maps undistorted with
   the same camera — `undistort_id_maps.py`), train, supervision, census.
4. Image farm 45 segments + GenDIA mango / pawpaw: v3 RGB (≈ 40 min each), lifting → hierarchy → embedder → supervision → census.
5. Lindendhof + osuga_full: full onboarding. **Needs Paul: the Lindendhof crop and SAM3 prompt (site.json).**

## Decisions and progress (2026-10-12 evening)
- **April: DONE** (16:45) — cells TREE eval 0.46–0.57, lane 2 0.68. **Cabbage / pepper on v3: DONE** — cabbage TREE 0.67 eval / 0.49
  train, pepper TREE 0.96, FRUIT (red) 0.32 eval.
- **Lindendhof (Paul 17:3x: "yeah works"):** trees = SAM3 "tree trunk" (the only reliable prompt; "plant" = weeds, canopy prompts
  merge whole hedges, no fruit visible) placed with the LiDAR, trunk BASE band only (the trees branch into leaders ~0.5 m apart);
  census = 1-D density peaks → 126 trees / 7 rows, pitch 1.3–1.6 m = the LiDAR trunk-band period 1.46 m; canopy split per tree in
  3D (nearest trunk along the row). `automation/lindendhof/`: trunk_masks.py → trunk_obs_lidar.py → tree_census_v2.py →
  lindendhof_identity_build.py → lindendhof_identity_chain.sh (embedder + per-lane v3 census + verdicts). Running.
- **Klapmuts Dec: same design**, anchor = "grow bag" (probe: 10/10 frames, best 0.88), canopy = "plant" pixels above the bag row
  split per bag in 3D. The scripts take PROMPT / WSMAP / ZLO / ZHI / ROW_SEP / ROW_HALF / TREE_HALF / CZLO / CZHI / MASKS / SITE.
  Bag masks running. Open: map the census ids onto the Sankofa v5 registry (911 bags) for the stage.
- **Phone surveys (image farm 39 base segments + GenDIA 6):** v3 RGB first via `image_farm_h3dgs_prep.py` (undistorted single
  chunk) → `v3_fleet_queue_phone_20261012.txt`; identity per crop afterwards (SAM3 sam3_v2 exists on most; maps undistorted with
  `undistort_id_maps.py`).

## Unmeasured
Embedder training time per survey (pepper's took one run of 1,500 epochs, time not logged); hierarchy quality on the
image-farm segments (10 existing ones have 3–6 objects); the image-farm GPU total ≈ 45 × 40 min ≈ 30 h for A alone.
