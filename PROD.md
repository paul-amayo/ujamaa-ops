# Prod readiness — surveys × UJAMAA agents

Generated 2026-08-20 00:31 UTC by `automation/build_prod_manifests.py` — do not edit by hand.

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
| 01_13B_Jackal | 121G | 41G | 4/7 ✗ stage2,verdicts,hierarchy | 3/4 ✗ hierarchy | **READY** | 2/3 ✗ rows |  |
| 02_13B_Jackal | 195G | 58G | 4/7 ✗ stage2,verdicts,hierarchy | 3/4 ✗ hierarchy | **READY** | 2/3 ✗ rows | ledger control epoch — joined the splat rotation 2026-08-14 (stage2 contingent on painted semantics) |
| 03_13B_Jackal | 114G | 40G | 4/7 ✗ stage2,verdicts,hierarchy | 3/4 ✗ hierarchy | **READY** | 2/3 ✗ rows |  |
| 04_13D_Jackal | 156G | 113G | 4/7 ✗ stage2,verdicts,hierarchy | 3/4 ✗ hierarchy | 2/3 ✗ multi_epoch | 2/3 ✗ rows |  |
| 05_13D_Jackal | 110G | 139G | 4/7 ✗ stage2,verdicts,hierarchy | 3/4 ✗ hierarchy | 1/3 ✗ in_ledger,multi_epoch | 2/3 ✗ rows |  |
| apr_2026_zed | 65G | 57G | 4/7 ✗ stage2,verdicts,hierarchy | 3/4 ✗ hierarchy | 1/3 ✗ in_ledger,multi_epoch | 1/3 ✗ rows,georef |  |
| dec_2025_a300 | 12G | 1G | 1/4 ✗ blocks,embedder,hierarchy | 1/4 ✗ hierarchy,registry,topdown | 1/3 ✗ in_ledger,multi_epoch | 0/3 ✗ site_geometry,rows,georef | pilot mcap — georef + SAM3 ledger seed |
| dec_2025_ten_rows | 92G | 3G | 1/7 ✗ blocks,stage2,verdicts,registered,embedder,hierarchy | 1/4 ✗ hierarchy,registry,topdown | 1/3 ✗ in_ledger,multi_epoch | 1/3 ✗ site_geometry,rows | Dec ten-rows — in week rotation, gated on pose-domain verification (INS=ENU0 frame) |

## Per-survey checklists

### 01_13B_Jackal
prod block config: `/home/paperspace/data/citrus_all/01_13B_Jackal/prod/tassili/blocks_ns/lio_row100`
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y
  - [x] blocks: lio_row100: 71 blocks
  - [ ] stage2: 3/71 blocks have stage2_censusinit_* ckpt (missing ['block_003', 'block_004', 'block_005', 'block_006'])
  - [ ] verdicts: 2 recorded, 1 pass floor 0.8; unrecorded blocks ['002', '003', '004', '005', '006']; failing ['000']
  - [x] registered: splats.json present
  - [x] embedder: 01_13B_v1g (newest in prod/bateleur)
  - [ ] hierarchy: no marker_hierarchy
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [ ] hierarchy: no marker_hierarchy
  - [x] registry: global_ids.json
  - [x] topdown: topdown export bateleur_orchard_topdown_01_13B_Jackal.json
- **sankofa**
  - [x] in_ledger: 289 observations in ledger_v2.json
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [x] multi_epoch: site 13B epochs in ledger: ['01_13B_Jackal', '02_13B_Jackal', '03_13B_Jackal']
- **azalai**
  - [x] site_geometry: /home/paperspace/data/citrus_all/site.json
  - [ ] rows: no rows in hierarchy
  - [x] georef: gps.monolithic present

### 02_13B_Jackal — ledger control epoch — joined the splat rotation 2026-08-14 (stage2 contingent on painted semantics)
prod block config: `/home/paperspace/data/citrus_all/02_13B_Jackal/prod/tassili/blocks_ns/lio_row100`
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y
  - [x] blocks: lio_row100: 27 blocks
  - [ ] stage2: 2/27 blocks have stage2_censusinit_* ckpt (missing ['block_002', 'block_003', 'block_004', 'block_005'])
  - [ ] verdicts: 2 recorded, 0 pass floor 0.8; unrecorded blocks ['002', '003', '004', '005', '006']; failing ['000', '001']
  - [x] registered: splats.json present
  - [x] embedder: 02_13B_v1g (canon)
  - [ ] hierarchy: no marker_hierarchy
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [ ] hierarchy: no marker_hierarchy
  - [x] registry: global_ids.json
  - [x] topdown: topdown export bateleur_orchard_topdown_02_13B_Jackal.json
- **sankofa**
  - [x] in_ledger: 177 observations in ledger_v2.json
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [x] multi_epoch: site 13B epochs in ledger: ['01_13B_Jackal', '02_13B_Jackal', '03_13B_Jackal']
- **azalai**
  - [x] site_geometry: /home/paperspace/data/citrus_all/site.json
  - [ ] rows: no rows in hierarchy
  - [x] georef: gps.monolithic present

### 03_13B_Jackal
prod block config: `/home/paperspace/data/citrus_all/03_13B_Jackal/prod/tassili/blocks_ns/lio_row100`
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y
  - [x] blocks: lio_row100: 105 blocks
  - [ ] stage2: 2/105 blocks have stage2_censusinit_* ckpt (missing ['block_002', 'block_003', 'block_004', 'block_005'])
  - [ ] verdicts: 2 recorded, 2 pass floor 0.8; unrecorded blocks ['002', '003', '004', '005', '006']
  - [x] registered: splats.json present
  - [x] embedder: 03_13B_v1g (newest in prod/bateleur)
  - [ ] hierarchy: no marker_hierarchy
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [ ] hierarchy: no marker_hierarchy
  - [x] registry: global_ids.json
  - [x] topdown: topdown export bateleur_orchard_topdown_03_13B_Jackal.json
- **sankofa**
  - [x] in_ledger: 303 observations in ledger_v2.json
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [x] multi_epoch: site 13B epochs in ledger: ['01_13B_Jackal', '02_13B_Jackal', '03_13B_Jackal']
- **azalai**
  - [x] site_geometry: /home/paperspace/data/citrus_all/site.json
  - [ ] rows: no rows in hierarchy
  - [x] georef: gps.monolithic present

### 04_13D_Jackal
prod block config: `/home/paperspace/data/citrus_all/04_13D_Jackal/prod/tassili/blocks_ns/lio_row100`
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y
  - [x] blocks: lio_row100: 48 blocks
  - [ ] stage2: 2/48 blocks have stage2_censusinit_* ckpt (missing ['block_002', 'block_003', 'block_004', 'block_005'])
  - [ ] verdicts: 1 recorded, 1 pass floor 0.8; unrecorded blocks ['000', '002', '003', '004', '005']
  - [x] registered: splats.json present
  - [x] embedder: 04_13D_v1g (newest in prod/bateleur)
  - [ ] hierarchy: no marker_hierarchy
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [ ] hierarchy: no marker_hierarchy
  - [x] registry: global_ids.json + Q4 reproduction verified
  - [x] topdown: topdown export bateleur_orchard_topdown_04_13D_Jackal.json
- **sankofa**
  - [x] in_ledger: 85 observations in ledger_v2.json
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [ ] multi_epoch: site 13D epochs in ledger: ['04_13D_Jackal']
- **azalai**
  - [x] site_geometry: /home/paperspace/data/citrus_all/site.json
  - [ ] rows: no rows in hierarchy
  - [x] georef: gps.monolithic present

### 05_13D_Jackal
prod block config: `/home/paperspace/data/citrus_all/05_13D_Jackal/prod/tassili/blocks_ns/lio_row100`
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y
  - [x] blocks: lio_row100: 43 blocks
  - [ ] stage2: 2/43 blocks have stage2_censusinit_* ckpt (missing ['block_002', 'block_003', 'block_004', 'block_005'])
  - [ ] verdicts: 2 recorded, 1 pass floor 0.8; unrecorded blocks ['002', '003', '004', '005', '006']; failing ['001']
  - [x] registered: splats.json present
  - [x] embedder: 05_13D_v1g (canon)
  - [ ] hierarchy: no marker_hierarchy
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [ ] hierarchy: no marker_hierarchy
  - [x] registry: global_ids.json + Q4 reproduction verified
  - [x] topdown: topdown export bateleur_orchard_topdown_05_13D_Jackal.json
- **sankofa**
  - [ ] in_ledger: not in ledger_v2.json — association ready (assoc_04_05_selfcheck.npz), ledger rebuild pending
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [ ] multi_epoch: site 13D epochs in ledger: ['04_13D_Jackal']
- **azalai**
  - [x] site_geometry: /home/paperspace/data/citrus_all/site.json
  - [ ] rows: no rows in hierarchy
  - [x] georef: gps.monolithic present

### apr_2026_zed
prod block config: `/home/paperspace/data/klapmuts/apr_2026_zed/prod/tassili/blocks_ns/lio_row100`
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y
  - [x] blocks: lio_row100: 25 blocks
  - [ ] stage2: 2/25 blocks have stage2_censusinit_* ckpt (missing ['block_002', 'block_003', 'block_004', 'block_005'])
  - [ ] verdicts: 2 recorded, 0 pass floor 0.8; unrecorded blocks ['002', '003', '004', '005', '006']; failing ['000', '001']
  - [x] registered: splats.json present
  - [x] embedder: apr_2026_zed_v1g (newest in prod/bateleur)
  - [ ] hierarchy: no marker_hierarchy
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [ ] hierarchy: no marker_hierarchy
  - [x] registry: global_ids.json
  - [x] topdown: topdown export bateleur_orchard_topdown_apr_2026_zed.json
- **sankofa**
  - [ ] in_ledger: not in ledger_v2.json
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [ ] multi_epoch: site klapmuts epochs in ledger: []
- **azalai**
  - [x] site_geometry: /home/paperspace/data/klapmuts/apr_2026_zed/prod/azalai/site.json
  - [ ] rows: no rows in hierarchy
  - [ ] georef: no GNSS on ZED rig — inherit georef via census association (planned)

### dec_2025_a300 — pilot mcap — georef + SAM3 ledger seed
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y
  - [ ] blocks: no blocks_ns config with stage2 or splats.json
  - [ ] embedder: no embedder in prod/bateleur/embedder (trains in-slot)
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
  - [ ] site_geometry: no site.json (planting geometry)
  - [ ] rows: no rows in hierarchy
  - [ ] georef: no absolute georef source

### dec_2025_ten_rows — Dec ten-rows — in week rotation, gated on pose-domain verification (INS=ENU0 frame)
prod block config: `/home/paperspace/data/klapmuts/dec_2025_ten_rows/prod/tassili/blocks_ns/lio_row100`
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y
  - [ ] blocks: lio_row100: 0 blocks
  - [ ] stage2: 0/0 blocks have stage2_censusinit_* ckpt
  - [ ] verdicts: no verdicts_*.json in lio_row100 — run containment verdicts
  - [ ] registered: splats.json MISSING — export + register for the viewer
  - [ ] embedder: no embedder in prod/bateleur/embedder (trains in-slot)
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
  - [ ] site_geometry: no site.json (planting geometry)
  - [ ] rows: no rows in hierarchy
  - [x] georef: gnsscorr_raw.npz present

