# Sample bundle — build spec (internal, not shipped)

What `scripts/get_sample_data.sh` downloads. This is the T1 load-bearing
artifact: the quickstart is only as good as this bundle. Build it once,
hash it, host it, and the docs' promises become testable.

## Source

Citrus survey `04_13D_Jackal` — the same site as the published Adinkra
evaluation, so the quickstart's example questions provably work against it.

## Contents (target ≤ 2 GB; likely far under)

| item | source on the A100 | est. size |
|---|---|---|
| trained splat(s) for the walkthrough | `prod/tassili/blocks_ns/lio_row100/block_*/splats/*.splat` (+ `.br`) | ~30–90 MB |
| render-service inputs | **MEASURED 2026-09-22**: `render_service` streams from nerfstudio checkpoint runs (`RENDER_RUN_GLOB=…/stage2_censusinit_*/high/*/config.yml`), ~600 MB/block, 43 blocks = 25.3 GB total → ship **3 contiguous blocks (~1.8 GB)** as a walkable sub-orchard. Lean alternative: client-side `.splat` rendering (30 MB/block, whole survey ~100 MB) — requires the front-end comeback already listed as web_deployment.md P2 option A; if that lands first, the bundle drops under 200 MB and T1 needs no streamed renderer at all. | 1.8 GB (streamed) or <0.2 GB (client-side) |
| plant registry | `prod/bateleur/scene_graph/marker_hierarchy.json` (frozen `.pre_fruit` variant = the evaluated one) | 30 KB |
| change ledger | `sankofa_substrate/ledger_v2.json` | ~1 MB |
| quality record | `prod/tassili/.../verdicts_censusinit_fw2.json` | ~100 KB |
| survey hierarchy for the viewer | `survey_hierarchy.bin` + `.json` (built by `build_survey_hierarchy.py`) | TBD |
| MANIFEST.json | sha256 per file, capture date, licence/terms note | — |

Explicitly NOT in the bundle: raw imagery, rosbags, GPS beyond the local
ENU frame (anonymisation), anything from the berry farm (single-site keeps
the permission conversation simple).

## Blockers / decisions

1. **Farm permission in writing** covering redistribution of the derived
   artifacts (splats are photographic in character — treat them as
   imagery for consent purposes, even anonymised). Owner: Paul. Fallback:
   klapmuts if citrus declines (loses ledger + eval-question alignment).
2. **Anonymisation pass**: strip GPS/WGS84 fields (`tree_wgs84.json` stays
   out; check marker_hierarchy for embedded lat/lon), scrub provenance
   paths/hostnames from every JSON `_provenance` block.
3. **Survey coherence**: the eval-aligned registry/ledger are survey 04
   (107 plants — the quickstart's example answers), but current prod
   serving streams survey 05 (the roll-fixed rebuild, glref runs,
   `RENDER_CKPT_POSES=opengl`). The bundle must pair blocks and registry
   from ONE survey — decide 04 (eval alignment) vs 05 (best
   reconstruction) at build time and re-verify the quickstart's answers
   against whichever ships.
4. **Streamed vs client-side rendering** (table above) — decides bundle
   size 10× and whether `demo_up.sh` needs the render service at all.
4. **Hosting**: GitHub release asset (2 GB/file limit — fine) vs HF
   dataset (better bandwidth, versioning). Lean: GitHub release on the
   umbrella repo, sha256 in MANIFEST checked by the download script.

## Definition of done

- Bundle downloads on a machine that has never seen our infrastructure,
  `demo_up.sh` serves it, and every quickstart step (§3–§5, including the
  four example questions) passes as written.
- MANIFEST hashes verified by `get_sample_data.sh`.
- Terms note inside the bundle states the farm-permission scope.
