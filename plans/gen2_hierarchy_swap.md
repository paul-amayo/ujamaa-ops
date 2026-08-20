# Generation-2 hierarchy swap (gen2)

**Written 2026-08-20.** One coordinated hierarchy generation per survey,
batching every pending hierarchy-touching change into a SINGLE embedder
retrain + stage2 re-seed per survey — because per the artifact-determinism
doctrine, any hierarchy change zeroes every seeded block downstream, so the
retrain is the unit of cost and features must ride together, never alone.

## What rides the train

| component | surveys | state |
|---|---|---|
| Row fix: July-style centroid RANSAC rows | apr | SOLVED in experimental (2026-08-20): `ransac_init` on census centroids, dir 0.985, thr 0.8, outlier bucket EXCLUDED — ~22 clean bed strands, bucket 3-4 pts. Fig `topdown_apr_july_centroid_solve.png` |
| Census upgrade: mv10 (783 sacks) | apr | BUILT + audited (beats 573 prod on every axis; mv5=1357 over-fragments). `bateleur/sam3_mv10` + `markers_mv10.monolithic` |
| Fruit level (ids ≥ 200) | 04, 05 | Recipe validated Jul 31 on 01 (strict filter); NOT yet run on 04/05 |
| Supervision hygiene extension | 01, 02, 03 | OPTIONAL / undecided — 13B trio still pre-hygiene; fold in here or defer to gen3 |

## Gates (all must hold before any survey's swap)

1. **Tree-level rotation complete for that survey** (don't zero a survey the
   week queue is still filling; small surveys complete ~Aug 24-25).
2. **M0 landed**: the row stage productionized in `build_marker_hierarchy`
   (see below) — apr's swap depends on it; 04/05's doesn't but should use it.
3. **Paul's explicit go per survey** (prod doctrine: swaps are blessed, not
   automatic — R10's removal made requeue/reseed a human decision).

## M0 — productionize the centroid row solve (no prod impact)

Replace the row stage in `build_marker_hierarchy.py`:
- Input: **census centroids** (all of them, from `global_ids.json` stats) —
  NOT the lifted 2+-frame markers (July's regime; proven 2026-08-20).
- Solver: `HighInterface.ransac_init(pts_xz, num_models, thr, dx, dy, dir)`
  via the Aug-18 binding (`src/interfaces/build/temp.../aru_nerf_interface`).
- **Outlier bucket (label == num_models) is NOT a row** — the July bug that
  became every catch-all row. Bucket members stay row-less (row_id -1).
- Params from site.json: `dominant_direction_threshold` (0.985),
  `coral_outlier_threshold_m` (0.8), NEW `row_num_models` key (60 apr;
  default 40 preserves citrus).
- Map row membership back onto hierarchy objects; tree node ids unchanged.
- Pilot: rebuild apr + 05 hierarchies to experimental/, diff row stats vs
  current (apr: expect ~22 rows, no catch-all; 05: expect its 9 rows stable
  — regression check that citrus is unharmed).
- Reference implementation: `automation/july_centroid_solve.py` (committed
  2026-08-20; run it for the standalone plot anytime).

## M1 — apr swap (first: smallest, all pieces proven)

1. Promote census: site.json `cluster_min_voxels: 20 -> 10`; re-run the
   census in-place (or bless `sam3_mv10` -> `sam3_v2`); markers + hierarchy
   rebuild via M0 builder. apr tree ids REMAP (573->783) — safe: apr is not
   in the sankofa ledger yet (PROD.md sankofa in_ledger ✗).
2. Supervision recompile (sack ids, per-block).
3. Embedder retrain (apr_v1h) against the gen2 hierarchy.
4. Re-seed sweep: stage2 census-init re-run per block on existing stage1s
   (~10 min/block cache-warm × 25 blocks ≈ 4-5 h GPU).
5. Verdicts re-run; expect the b000-profile (0.75-0.85) to extend fleet-wide
   now that row GT is per-bed; far-small sacks stay the known floor-binding
   entities (verdict battery unchanged by decision 2026-08-20).

## M2 — 04 + 05 fruit level

1. SAM3 fruit pass, image mode, strict filter as site keys
   (`sam3_fruit_prompt: fruit`, conf + strict_area 300-3000 + fill 0.45 +
   aspect 2.2 + relscore 0.5) — ~1 GPU-evening per survey. Fruit masks
   filter = roundness only in-tree (doctrine).
2. Fruit inventory: detection-driven (never preseeded — Jul 31: hand
   inventory dropped 46 true oranges); assign fruit->parent tree by
   mask-in-tree + depth; fruit nodes ids ≥ 200, parented to trees.
   Tree/row ids UNCHANGED (census untouched) — 04's true-WGS84 ledger
   entries stay valid.
3. Supervision v3 compile (trees + fruit), embedder retrain per survey
   (fruit = innermost hyperbolic radius), re-seed sweep (04: 48 blocks,
   05: 43 — ~8 h GPU each cache-warm).
4. Fruit-level verdicts activate automatically (the ids ≥ 200 criterion in
   the battery stops returning EMPTY).

## M3 (optional) — 13B trio hygiene

01/02/03 supervision is pre-hygiene (partition doctrine: only 04/05
regenerated). If their gen2 happens, fold hygiene + (if fruit visible in
13B imagery — it is the orange orchard) the fruit level, same train.
Decision deferred; 03 (105 blocks) re-seed ≈ 18 h GPU — schedule with care.

## Order + rough calendar

apr (M1) -> 05 (M2) -> 04 (M2) -> 13B trio (M3, if at all).
Earliest start: when the tree rotation clears each survey (~Aug 24-25 for
apr/02/05/04). Total GPU for M1+M2 ≈ 20-24 h — roughly two nights, can run
as the week queue's successor with the same slot/verdict machinery.

## Risks

- **Embedder retrain is the hazard** (artifact-determinism doctrine): this
  plan exists to pay it once per survey. Nothing else in the chain is
  stochastic (hierarchy rebuilds byte-identical given inputs).
- apr id remap: re-anchor anything that referenced old apr ids (none in
  ledger today; check azalai site rows consumers).
- Fruit recall at 13D fg-PSNR: b004-class blocks (24 dB) fine; if a survey
  has 17-19 dB backlit lanes (apr-style), fruit supervision there will be
  sparse — record, don't fight.
