# Prod readiness — surveys × UJAMAA agents

Generated 2026-08-14 21:41 UTC by `automation/build_prod_manifests.py` — do not edit by hand.

Layout per survey: `prod/{monos,tassili,bateleur,sankofa,azalai}` =
PHYSICAL folders, **unsafe to delete** (source data + current-best
outputs). `experimental/` = everything else, **safe to delete
wholesale**. Symlink shims at the old root paths keep existing
absolute references resolving; deleting experimental/ can never
dangle prod. prod-dir ≠ READY — prod holds the current best even
when the checklist is not yet green.

Agents = adinkra panel agents (`ujamaa/adinkra/server.py`). Spoor and
Hapi run on demo-synthetic data — no per-survey assets yet.

| survey | prod | experimental (deletable) | tassili | bateleur | sankofa | azalai | note |
|---|---|---|---|---|---|---|---|
| 01_13B_Jackal | 69G | 1G | 3/4 ✗ blocks | **READY** | **READY** | **READY** |  |
| 02_13B_Jackal | 100G | 1G | 2/4 ✗ blocks,embedder | **READY** | **READY** | **READY** | ledger control epoch — joined the splat rotation 2026-08-14 (stage2 contingent on painted semantics) |
| 03_13B_Jackal | 72G | 1G | 3/4 ✗ blocks | **READY** | **READY** | **READY** |  |
| 04_13D_Jackal | 123G | 36G | 4/7 ✗ stage2,verdicts,registered | **READY** | 2/3 ✗ multi_epoch | **READY** |  |
| 05_13D_Jackal | 109G | 21G | 4/7 ✗ stage2,verdicts,registered | **READY** | 1/3 ✗ in_ledger,multi_epoch | **READY** |  |
| apr_2026_zed | 27G | 23G | 3/4 ✗ blocks | **READY** | 1/3 ✗ in_ledger,multi_epoch | 2/3 ✗ georef |  |
| dec_2025_a300 | 12G | 1G | 1/4 ✗ blocks,embedder,hierarchy | 1/4 ✗ hierarchy,registry,topdown | 1/3 ✗ in_ledger,multi_epoch | 1/3 ✗ rows,georef | pilot mcap — georef + SAM3 ledger seed |
| dec_2025_ten_rows | 89G | 1G | 0/4 ✗ kdomain,blocks,embedder,hierarchy | 1/4 ✗ hierarchy,registry,topdown | 1/3 ✗ in_ledger,multi_epoch | 2/3 ✗ rows | Dec ten-rows — in week rotation, gated on pose-domain verification (INS=ENU0 frame) |

## Per-survey checklists

### 01_13B_Jackal
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y
  - [ ] blocks: no blocks_ns config with stage2 or splats.json
  - [x] embedder: 01_13B_c20cos20 (canon)
  - [x] hierarchy: 289 obj / 38 rows (scene_graph)
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [x] hierarchy: 289 obj / 38 rows
  - [x] registry: global_ids.json
  - [x] topdown: topdown export bateleur_orchard_topdown_01_13B_Jackal.json
- **sankofa**
  - [x] in_ledger: 289 observations in ledger_v2.json
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [x] multi_epoch: site 13B epochs in ledger: ['01_13B_Jackal', '02_13B_Jackal', '03_13B_Jackal']
- **azalai**
  - [x] site_geometry: /home/paperspace/data/citrus_all/site.json
  - [x] rows: 38 planting rows
  - [x] georef: gps.monolithic present

### 02_13B_Jackal — ledger control epoch — joined the splat rotation 2026-08-14 (stage2 contingent on painted semantics)
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y
  - [ ] blocks: no blocks_ns config with stage2 or splats.json
  - [ ] embedder: no embedder ckpt found
  - [x] hierarchy: 177 obj / 9 rows (scene_graph)
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [x] hierarchy: 177 obj / 9 rows
  - [x] registry: global_ids.json
  - [x] topdown: topdown export bateleur_orchard_topdown_02_13B_Jackal.json
- **sankofa**
  - [x] in_ledger: 177 observations in ledger_v2.json
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [x] multi_epoch: site 13B epochs in ledger: ['01_13B_Jackal', '02_13B_Jackal', '03_13B_Jackal']
- **azalai**
  - [x] site_geometry: /home/paperspace/data/citrus_all/site.json
  - [x] rows: 9 planting rows
  - [x] georef: gps.monolithic present

### 03_13B_Jackal
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y
  - [ ] blocks: no blocks_ns config with stage2 or splats.json
  - [x] embedder: 03_13B_v2G (newest, no canon tag)
  - [x] hierarchy: 272 obj / 35 rows (scene_graph)
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [x] hierarchy: 272 obj / 35 rows
  - [x] registry: global_ids.json
  - [x] topdown: topdown export bateleur_orchard_topdown_03_13B_Jackal.json
- **sankofa**
  - [x] in_ledger: 303 observations in ledger_v2.json
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [x] multi_epoch: site 13B epochs in ledger: ['01_13B_Jackal', '02_13B_Jackal', '03_13B_Jackal']
- **azalai**
  - [x] site_geometry: /home/paperspace/data/citrus_all/site.json
  - [x] rows: 35 planting rows
  - [x] georef: gps.monolithic present

### 04_13D_Jackal
prod block config: `/home/paperspace/data/citrus_all/04_13D_Jackal/prod/tassili/blocks_ns/lio_row6F`
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y
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
  - [x] topdown: topdown export bateleur_orchard_topdown_04_13D_Jackal.json
- **sankofa**
  - [x] in_ledger: 85 observations in ledger_v2.json
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [ ] multi_epoch: site 13D epochs in ledger: ['04_13D_Jackal']
- **azalai**
  - [x] site_geometry: /home/paperspace/data/citrus_all/site.json
  - [x] rows: 11 planting rows
  - [x] georef: gps.monolithic present

### 05_13D_Jackal
prod block config: `/home/paperspace/data/citrus_all/05_13D_Jackal/prod/tassili/blocks_ns/lio_row100`
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y
  - [x] blocks: lio_row100: 43 blocks
  - [ ] stage2: 8/43 blocks have stage2_censusinit_fw2 ckpt (missing ['block_000', 'block_009', 'block_010', 'block_011'])
  - [ ] verdicts: 8 recorded, 7 pass floor 0.8; unrecorded blocks ['004', '009', '010', '011', '012']; failing ['000']
  - [ ] registered: splats.json MISSING — export + register for the viewer
  - [x] embedder: 05_13D_v1g (canon)
  - [x] hierarchy: 81 obj / 10 rows (scene_graph)
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [x] hierarchy: 81 obj / 10 rows
  - [x] registry: global_ids.json + Q4 reproduction verified
  - [x] topdown: topdown export bateleur_orchard_topdown_05_13D_Jackal.json
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
  - [x] kdomain: kf20cm=y lio_mono=y
  - [ ] blocks: no blocks_ns config with stage2 or splats.json
  - [x] embedder: klapmuts_v2vocab1k (newest, no canon tag)
  - [x] hierarchy: 721 obj / 14 rows (scene_graph)
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [x] hierarchy: 721 obj / 14 rows
  - [x] registry: global_ids.json
  - [x] topdown: topdown export bateleur_orchard_topdown_apr_2026_zed.json
- **sankofa**
  - [ ] in_ledger: not in ledger_v2.json
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [ ] multi_epoch: site klapmuts epochs in ledger: []
- **azalai**
  - [x] site_geometry: /home/paperspace/data/klapmuts/apr_2026_zed/prod/azalai/site.json
  - [x] rows: 14 planting rows
  - [ ] georef: no GNSS on ZED rig — inherit georef via census association (planned)

### dec_2025_a300 — pilot mcap — georef + SAM3 ledger seed
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y
  - [ ] blocks: no blocks_ns config with stage2 or splats.json
  - [ ] embedder: no embedder ckpt found
  - [ ] hierarchy: no marker_hierarchy
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [ ] hierarchy: no marker_hierarchy
  - [ ] registry: no sam3_v2/global_ids.json
  - [ ] topdown: topdown slot holds 01_13B_Jackal — run export_bateleur_topdown.py /home/paperspace/data/klapmuts/dec_2025_a300
- **sankofa**
  - [ ] in_ledger: not in ledger_v2.json
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [ ] multi_epoch: site klapmuts epochs in ledger: []
- **azalai**
  - [x] site_geometry: /home/paperspace/data/klapmuts/site.json
  - [ ] rows: no rows in hierarchy
  - [ ] georef: no absolute georef source

### dec_2025_ten_rows — Dec ten-rows — in week rotation, gated on pose-domain verification (INS=ENU0 frame)
- **tassili**
  - [ ] kdomain: kf20cm=MISSING lio_mono=y
  - [ ] blocks: no blocks_ns config with stage2 or splats.json
  - [ ] embedder: no embedder ckpt found
  - [ ] hierarchy: no marker_hierarchy
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [ ] hierarchy: no marker_hierarchy
  - [ ] registry: no sam3_v2/global_ids.json
  - [ ] topdown: topdown slot holds 01_13B_Jackal — run export_bateleur_topdown.py /home/paperspace/data/klapmuts/dec_2025_ten_rows
- **sankofa**
  - [ ] in_ledger: not in ledger_v2.json
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [ ] multi_epoch: site klapmuts epochs in ledger: []
- **azalai**
  - [x] site_geometry: /home/paperspace/data/klapmuts/site.json
  - [ ] rows: no rows in hierarchy
  - [x] georef: gnsscorr_raw.npz present

