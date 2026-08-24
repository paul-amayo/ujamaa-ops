# Prod readiness — surveys × UJAMAA agents

Generated 2026-08-24 17:00 UTC by `automation/build_prod_manifests.py` — do not edit by hand.

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
| 01_13B_Jackal | 97G | 1G | 4/7 ✗ stage2,verdicts,registered | **READY** | **READY** | **READY** |  |
| 02_13B_Jackal | 130G | 1G | 5/7 ✗ verdicts,registered | **READY** | **READY** | **READY** | ledger control epoch — joined the splat rotation 2026-08-14 (stage2 contingent on painted semantics) |
| 03_13B_Jackal | 95G | 1G | 4/7 ✗ stage2,verdicts,registered | **READY** | **READY** | **READY** |  |
| 04_13D_Jackal | 95G | 1G | 5/7 ✗ verdicts,registered | **READY** | 2/3 ✗ multi_epoch | **READY** |  |
| 05_13D_Jackal | 97G | 9G | 5/7 ✗ verdicts,registered | **READY** | 1/3 ✗ in_ledger,multi_epoch | **READY** |  |
| apr_2026_zed | 44G | 1G | 4/7 ✗ stage2,verdicts,registered | **READY** | 1/3 ✗ in_ledger,multi_epoch | 2/3 ✗ georef |  |

## Per-survey checklists

### 01_13B_Jackal
prod block config: `/home/paperspace/data/citrus_all/01_13B_Jackal/prod/tassili/blocks_ns/lio_row100`
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y
  - [x] blocks: lio_row100: 71 blocks
  - [ ] stage2: 22/71 blocks have stage2_censusinit_* ckpt (missing ['block_022', 'block_023', 'block_024', 'block_025'])
  - [ ] verdicts: 22 recorded, 6 pass floor 0.8; unrecorded blocks ['022', '023', '024', '025', '026']; failing ['001', '003', '006', '007', '008', '009', '010', '012', '013', '014', '015', '016', '017', '018', '019', '021']
  - [ ] registered: splats.json MISSING — export + register for the viewer
  - [x] embedder: 01_13B_v1g (newest in prod/bateleur)
  - [x] hierarchy: 290 obj / 40 rows (scene_graph)
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [x] hierarchy: 290 obj / 40 rows
  - [x] registry: global_ids.json
  - [x] topdown: topdown export bateleur_orchard_topdown_01_13B_Jackal.json
- **sankofa**
  - [x] in_ledger: 289 observations in ledger_v2.json
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [x] multi_epoch: site 13B epochs in ledger: ['01_13B_Jackal', '02_13B_Jackal', '03_13B_Jackal']
- **azalai**
  - [x] site_geometry: /home/paperspace/data/citrus_all/site.json
  - [x] rows: 40 planting rows
  - [x] georef: gps.monolithic present

### 02_13B_Jackal — ledger control epoch — joined the splat rotation 2026-08-14 (stage2 contingent on painted semantics)
prod block config: `/home/paperspace/data/citrus_all/02_13B_Jackal/prod/tassili/blocks_ns/lio_row100`
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y
  - [x] blocks: lio_row100: 27 blocks
  - [x] stage2: 27/27 blocks have stage2_censusinit_* ckpt
  - [ ] verdicts: 27 recorded, 4 pass floor 0.8; failing ['000', '001', '002', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '018', '020', '021', '022', '023', '024', '026']
  - [ ] registered: splats.json MISSING — export + register for the viewer
  - [x] embedder: 02_13B_v1g (canon)
  - [x] hierarchy: 182 obj / 28 rows (scene_graph)
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [x] hierarchy: 182 obj / 28 rows
  - [x] registry: global_ids.json
  - [x] topdown: topdown export bateleur_orchard_topdown_02_13B_Jackal.json
- **sankofa**
  - [x] in_ledger: 177 observations in ledger_v2.json
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [x] multi_epoch: site 13B epochs in ledger: ['01_13B_Jackal', '02_13B_Jackal', '03_13B_Jackal']
- **azalai**
  - [x] site_geometry: /home/paperspace/data/citrus_all/site.json
  - [x] rows: 28 planting rows
  - [x] georef: gps.monolithic present

### 03_13B_Jackal
prod block config: `/home/paperspace/data/citrus_all/03_13B_Jackal/prod/tassili/blocks_ns/lio_row100`
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y
  - [x] blocks: lio_row100: 105 blocks
  - [ ] stage2: 23/105 blocks have stage2_censusinit_* ckpt (missing ['block_023', 'block_024', 'block_025', 'block_026'])
  - [ ] verdicts: 23 recorded, 4 pass floor 0.8; unrecorded blocks ['023', '024', '025', '026', '027']; failing ['000', '002', '003', '004', '006', '007', '008', '009', '010', '012', '013', '014', '015', '016', '017', '018', '019', '021', '022']
  - [ ] registered: splats.json MISSING — export + register for the viewer
  - [x] embedder: 03_13B_v1g (newest in prod/bateleur)
  - [x] hierarchy: 309 obj / 40 rows (scene_graph)
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [x] hierarchy: 309 obj / 40 rows
  - [x] registry: global_ids.json
  - [x] topdown: topdown export bateleur_orchard_topdown_03_13B_Jackal.json
- **sankofa**
  - [x] in_ledger: 303 observations in ledger_v2.json
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [x] multi_epoch: site 13B epochs in ledger: ['01_13B_Jackal', '02_13B_Jackal', '03_13B_Jackal']
- **azalai**
  - [x] site_geometry: /home/paperspace/data/citrus_all/site.json
  - [x] rows: 40 planting rows
  - [x] georef: gps.monolithic present

### 04_13D_Jackal
prod block config: `/home/paperspace/data/citrus_all/04_13D_Jackal/prod/tassili/blocks_ns/lio_row100`
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y
  - [x] blocks: lio_row100: 48 blocks
  - [x] stage2: 48/48 blocks have stage2_censusinit_* ckpt
  - [ ] verdicts: 48 recorded, 25 pass floor 0.8; failing ['000', '003', '004', '005', '012', '013', '014', '015', '017', '020', '023', '024', '025', '029', '030', '032', '033', '037', '038', '040', '041', '043', '046']
  - [ ] registered: splats.json MISSING — export + register for the viewer
  - [x] embedder: 04_13D_v1g (newest in prod/bateleur)
  - [x] hierarchy: 107 obj / 8 rows (scene_graph)
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [x] hierarchy: 107 obj / 8 rows
  - [x] registry: global_ids.json
  - [x] topdown: topdown export bateleur_orchard_topdown_04_13D_Jackal.json
- **sankofa**
  - [x] in_ledger: 85 observations in ledger_v2.json
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [ ] multi_epoch: site 13D epochs in ledger: ['04_13D_Jackal']
- **azalai**
  - [x] site_geometry: /home/paperspace/data/citrus_all/site.json
  - [x] rows: 8 planting rows
  - [x] georef: gps.monolithic present

### 05_13D_Jackal
prod block config: `/home/paperspace/data/citrus_all/05_13D_Jackal/prod/tassili/blocks_ns/lio_row100`
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y
  - [x] blocks: lio_row100: 43 blocks
  - [x] stage2: 43/43 blocks have stage2_censusinit_* ckpt
  - [ ] verdicts: 43 recorded, 24 pass floor 0.8; failing ['002', '003', '005', '009', '011', '016', '019', '023', '024', '025', '026', '028', '029', '030', '031', '035', '036', '038', '040']
  - [ ] registered: splats.json MISSING — export + register for the viewer
  - [x] embedder: 05_13D_v1g (canon)
  - [x] hierarchy: 100 obj / 8 rows (scene_graph)
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [x] hierarchy: 100 obj / 8 rows
  - [x] registry: global_ids.json
  - [x] topdown: topdown export bateleur_orchard_topdown_05_13D_Jackal.json
- **sankofa**
  - [ ] in_ledger: not in ledger_v2.json — association ready (assoc_04_05_selfcheck.npz), ledger rebuild pending
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [ ] multi_epoch: site 13D epochs in ledger: ['04_13D_Jackal']
- **azalai**
  - [x] site_geometry: /home/paperspace/data/citrus_all/site.json
  - [x] rows: 8 planting rows
  - [x] georef: gps.monolithic present

### apr_2026_zed
prod block config: `/home/paperspace/data/klapmuts/apr_2026_zed/prod/tassili/blocks_ns/lio_row100`
- **tassili**
  - [x] kdomain: kf20cm=y lio_mono=y
  - [x] blocks: lio_row100: 25 blocks
  - [ ] stage2: 24/25 blocks have stage2_censusinit_* ckpt (missing ['block_017'])
  - [ ] verdicts: 23 recorded, 0 pass floor 0.8; unrecorded blocks ['017', '024']; failing ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '018', '019', '020', '021', '022', '023']
  - [ ] registered: splats.json MISSING — export + register for the viewer
  - [x] embedder: apr_2026_zed_v1g (newest in prod/bateleur)
  - [x] hierarchy: 481 obj / 16 rows (scene_graph)
- **bateleur**
  - [x] trajectory: lio/odom mono present
  - [x] hierarchy: 481 obj / 16 rows
  - [x] registry: global_ids.json
  - [x] topdown: topdown export bateleur_orchard_topdown_apr_2026_zed.json
- **sankofa**
  - [ ] in_ledger: not in ledger_v2.json
  - [x] one_datum: ledger_v2.json (single 03-datum)
  - [ ] multi_epoch: site klapmuts epochs in ledger: []
- **azalai**
  - [x] site_geometry: /home/paperspace/data/klapmuts/apr_2026_zed/prod/azalai/site.json
  - [x] rows: 16 planting rows
  - [ ] georef: no GNSS on ZED rig — inherit georef via census association (planned)

