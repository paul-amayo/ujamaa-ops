# Prod readiness — surveys × UJAMAA agents

Generated 2026-08-14 20:50 UTC by `automation/build_prod_manifests.py` — do not edit by hand.
Each survey has `<root>/prod/` (symlinks to blessed artifacts + `prod.json`
checklist). **Agent servers mount `prod/` only; everything else in a survey
dir is experimental.** Nothing is physically moved — splat run configs
self-reference absolute paths and break on relocation.

Agents = adinkra panel agents (`ujamaa/adinkra/server.py`). Spoor and Hapi
run on demo-synthetic data and have no per-survey assets yet.

| survey | tassili | bateleur | sankofa | azalai | note |
|---|---|---|---|---|---|
| 01_13B_Jackal | 5/7 ✗ stage2,verdicts | **READY** | **READY** | **READY** |  |
| 02_13B_Jackal | 2/4 ✗ blocks,embedder | 3/4 ✗ topdown | **READY** | **READY** | ledger control epoch — registry+ledger by design, no splat planned |
| 03_13B_Jackal | 5/7 ✗ stage2,verdicts | 3/4 ✗ topdown | **READY** | **READY** |  |
| 04_13D_Jackal | 4/7 ✗ stage2,verdicts,registered | 3/4 ✗ topdown | 2/3 ✗ multi_epoch | **READY** |  |
| 05_13D_Jackal | 4/7 ✗ stage2,verdicts,registered | 3/4 ✗ topdown | 1/3 ✗ in_ledger,multi_epoch | **READY** |  |
| apr_2026_zed | 3/4 ✗ blocks | 3/4 ✗ topdown | 1/3 ✗ in_ledger,multi_epoch | 2/3 ✗ georef |  |
| dec_2025_a300 | 1/4 ✗ blocks,embedder,hierarchy | 1/4 ✗ hierarchy,registry,topdown | 1/3 ✗ in_ledger,multi_epoch | 1/3 ✗ rows,georef | pilot mcap — georef + SAM3 ledger seed |
| dec_2025_ten_rows | 0/4 ✗ kdomain,blocks,embedder,hierarchy | 1/4 ✗ hierarchy,registry,topdown | 1/3 ✗ in_ledger,multi_epoch | 2/3 ✗ rows | Dec ten-rows — ingest chain in progress (task #10) |

## Per-survey checklists

### 01_13B_Jackal
prod block config: `/home/paperspace/data/citrus_all/01_13B_Jackal/blocks_ns/lio_row`
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y (01_13B_Jackal)
  - [x] blocks: lio_row: 35 blocks
  - [ ] stage2: 0/35 blocks have stage2_censusinit_fw2 ckpt (missing ['block_000', 'block_001', 'block_002', 'block_003'])
  - [ ] verdicts: no verdicts_*.json in lio_row — run containment verdicts
  - [x] registered: splats.json present
  - [x] embedder: 01_13B_c20cos20 (canon)
  - [x] hierarchy: 289 obj / 38 rows (scene_graph)
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [x] hierarchy: 289 obj / 38 rows
  - [x] registry: global_ids.json
  - [x] topdown: topdown export current for this survey
- **sankofa**
  - [x] in_ledger: 289 observations in ledger_v2.json
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [x] multi_epoch: site 13B epochs in ledger: ['01_13B_Jackal', '02_13B_Jackal', '03_13B_Jackal']
- **azalai**
  - [x] site_geometry: /home/paperspace/data/citrus_all/site.json
  - [x] rows: 38 planting rows
  - [x] georef: gps.monolithic present

### 02_13B_Jackal — ledger control epoch — registry+ledger by design, no splat planned
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y (02_13B_Jackal)
  - [ ] blocks: no blocks_ns config with stage2 or splats.json
  - [ ] embedder: no embedder ckpt found
  - [x] hierarchy: 177 obj / 9 rows (scene_graph)
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [x] hierarchy: 177 obj / 9 rows
  - [x] registry: global_ids.json
  - [ ] topdown: topdown export slot holds 01_13B_Jackal (single-slot; re-export with export_bateleur_topdown.py /home/paperspace/data/citrus_all/02_13B_Jackal)
- **sankofa**
  - [x] in_ledger: 177 observations in ledger_v2.json
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [x] multi_epoch: site 13B epochs in ledger: ['01_13B_Jackal', '02_13B_Jackal', '03_13B_Jackal']
- **azalai**
  - [x] site_geometry: /home/paperspace/data/citrus_all/site.json
  - [x] rows: 9 planting rows
  - [x] georef: gps.monolithic present

### 03_13B_Jackal
prod block config: `/home/paperspace/data/citrus_all/03_13B_Jackal/blocks_ns/lidar_pass1_100cm`
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y (03_13B_Jackal)
  - [x] blocks: lidar_pass1_100cm: 2 blocks
  - [ ] stage2: 0/2 blocks have stage2_censusinit_fw2 ckpt (missing ['block_000', 'block_080'])
  - [ ] verdicts: no verdicts_*.json in lidar_pass1_100cm — run containment verdicts
  - [x] registered: splats.json present
  - [x] embedder: 03_13B_v2G (newest, no canon tag)
  - [x] hierarchy: 272 obj / 35 rows (scene_graph)
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [x] hierarchy: 272 obj / 35 rows
  - [x] registry: global_ids.json + Q4 reproduction verified
  - [ ] topdown: topdown export slot holds 01_13B_Jackal (single-slot; re-export with export_bateleur_topdown.py /home/paperspace/data/citrus_all/03_13B_Jackal)
- **sankofa**
  - [x] in_ledger: 303 observations in ledger_v2.json
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [x] multi_epoch: site 13B epochs in ledger: ['01_13B_Jackal', '02_13B_Jackal', '03_13B_Jackal']
- **azalai**
  - [x] site_geometry: /home/paperspace/data/citrus_all/site.json
  - [x] rows: 35 planting rows
  - [x] georef: gps.monolithic present

### 04_13D_Jackal
prod block config: `/home/paperspace/data/citrus_all/04_13D_Jackal/blocks_ns/lio_row6F`
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y (04_13D_Jackal)
  - [x] blocks: lio_row6F: 6 blocks
  - [ ] stage2: 1/6 blocks have stage2_censusinit_fw2 ckpt (missing ['block_000', 'block_001', 'block_002', 'block_004'])
  - [ ] verdicts: no verdicts_*.json in lio_row6F — run containment verdicts
  - [ ] registered: splats.json MISSING — export + register for the viewer
  - [x] embedder: 04_13D_v3vocab1k (canon)
  - [x] hierarchy: 103 obj / 11 rows (scene_graph_v4)
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [x] hierarchy: 103 obj / 11 rows
  - [x] registry: global_ids.json + Q4 reproduction verified
  - [ ] topdown: topdown export slot holds 01_13B_Jackal (single-slot; re-export with export_bateleur_topdown.py /home/paperspace/data/citrus_all/04_13D_Jackal)
- **sankofa**
  - [x] in_ledger: 85 observations in ledger_v2.json
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [ ] multi_epoch: site 13D epochs in ledger: ['04_13D_Jackal']
- **azalai**
  - [x] site_geometry: /home/paperspace/data/citrus_all/site.json
  - [x] rows: 11 planting rows
  - [x] georef: gps.monolithic present

### 05_13D_Jackal
prod block config: `/home/paperspace/data/citrus_all/05_13D_Jackal/blocks_ns/lio_row100`
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y (05_13D_Jackal)
  - [x] blocks: lio_row100: 43 blocks
  - [ ] stage2: 9/43 blocks have stage2_censusinit_fw2 ckpt (missing ['block_009', 'block_010', 'block_011', 'block_012'])
  - [ ] verdicts: 8 recorded, 7 pass floor 0.8; unrecorded blocks ['004', '009', '010', '011', '012']; failing ['000']
  - [ ] registered: splats.json MISSING — export + register for the viewer
  - [x] embedder: 05_13D_v1g (canon)
  - [x] hierarchy: 81 obj / 10 rows (scene_graph)
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [x] hierarchy: 81 obj / 10 rows
  - [x] registry: global_ids.json + Q4 reproduction verified
  - [ ] topdown: topdown export slot holds 01_13B_Jackal (single-slot; re-export with export_bateleur_topdown.py /home/paperspace/data/citrus_all/05_13D_Jackal)
- **sankofa**
  - [ ] in_ledger: not in ledger_v2.json — association ready (assoc_04_05_selfcheck.npz), ledger rebuild pending
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [ ] multi_epoch: site 13D epochs in ledger: ['04_13D_Jackal']
- **azalai**
  - [x] site_geometry: /home/paperspace/data/citrus_all/site.json
  - [x] rows: 10 planting rows
  - [x] georef: gps.monolithic present

### apr_2026_zed
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y (apr_2026_zed)
  - [ ] blocks: no blocks_ns config with stage2 or splats.json
  - [x] embedder: klapmuts_v2vocab1k (newest, no canon tag)
  - [x] hierarchy: 721 obj / 14 rows (scene_graph)
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [x] hierarchy: 721 obj / 14 rows
  - [x] registry: global_ids.json
  - [ ] topdown: topdown export slot holds 01_13B_Jackal (single-slot; re-export with export_bateleur_topdown.py /home/paperspace/data/klapmuts/apr_2026_zed)
- **sankofa**
  - [ ] in_ledger: not in ledger_v2.json
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [ ] multi_epoch: site klapmuts epochs in ledger: []
- **azalai**
  - [x] site_geometry: /home/paperspace/data/klapmuts/apr_2026_zed/site.json
  - [x] rows: 14 planting rows
  - [ ] georef: no GNSS on ZED rig — inherit georef via census association (planned)

### dec_2025_a300 — pilot mcap — georef + SAM3 ledger seed
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y (monolithics)
  - [ ] blocks: no blocks_ns config with stage2 or splats.json
  - [ ] embedder: no embedder ckpt found
  - [ ] hierarchy: no marker_hierarchy
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [ ] hierarchy: no marker_hierarchy
  - [ ] registry: no sam3_v2/global_ids.json
  - [ ] topdown: topdown export slot holds 01_13B_Jackal (single-slot; re-export with export_bateleur_topdown.py /home/paperspace/data/klapmuts/dec_2025_a300)
- **sankofa**
  - [ ] in_ledger: not in ledger_v2.json
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [ ] multi_epoch: site klapmuts epochs in ledger: []
- **azalai**
  - [x] site_geometry: /home/paperspace/data/klapmuts/site.json
  - [ ] rows: no rows in hierarchy
  - [ ] georef: no absolute georef source

### dec_2025_ten_rows — Dec ten-rows — ingest chain in progress (task #10)
- **tassili**
  - [ ] kdomain: kf20cm=MISSING lio_mono=y (monolithics)
  - [ ] blocks: no blocks_ns config with stage2 or splats.json
  - [ ] embedder: no embedder ckpt found
  - [ ] hierarchy: no marker_hierarchy
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [ ] hierarchy: no marker_hierarchy
  - [ ] registry: no sam3_v2/global_ids.json
  - [ ] topdown: topdown export slot holds 01_13B_Jackal (single-slot; re-export with export_bateleur_topdown.py /home/paperspace/data/klapmuts/dec_2025_ten_rows)
- **sankofa**
  - [ ] in_ledger: not in ledger_v2.json
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [ ] multi_epoch: site klapmuts epochs in ledger: []
- **azalai**
  - [x] site_geometry: /home/paperspace/data/klapmuts/site.json
  - [ ] rows: no rows in hierarchy
  - [x] georef: gnsscorr_raw.npz present

