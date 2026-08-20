# CPU week ledger — 2026-08-20 → 08-27

Standing queue of CPU-only work to advance alongside the GPU rotation.
Worked by scheduled Claude sessions (daily cron) and live sessions alike:
take the TOP unchecked task, work it to a committable state, tick it with a
date + result line, stop. Conventions: nice -n 15 for heavy CPU (GLOMAP
bursts in the GPU slots own the cores), commit-after-green, notebook entry
per work session, ground truth from artifacts not logs. Each morning tick
ALSO: republish lab_notebook/dashboard.html to the canonical artifact
(https://claude.ai/code/artifact/3a2156c8-d3b4-4a2b-9968-923c007dc45a) so
Paul's view stays current without asking.

- [x] P0 (pilot): hierarchy checklist cells lie — checker probes a
      pre-migration path ("no marker_hierarchy" fleet-wide while the files
      exist under prod/bateleur/scene_graph/). Fix in
      automation/build_prod_manifests.py, regen PROD.md, verify cells flip.
      DONE 2026-08-20: checker probed <root>/scene_graph; now falls back to
      prod/bateleur/scene_graph. 6 surveys flipped to hierarchy=[x]. d1e5ef6
- [ ] P1: gen2 M0 — centroid row stage into build_marker_hierarchy.py per
      plans/gen2_hierarchy_swap.md M0: census centroids in (all, from
      global_ids stats), HighInterface.ransac_init (Aug-18 binding path),
      outlier bucket => row_id -1 (never a row), site keys (row_num_models
      NEW, default 40), pilot apr+05 to experimental/ + citrus regression
      diff. Reference: automation/july_centroid_solve.py.
- [ ] P2: Sankofa structural metrics per tree per epoch — canopy extent /
      height / volume proxies from banked LiDAR cluster members in each
      registry (no splat needed); output per-tree per-epoch table keyed by
      ledger_v2 canonical ids; wire into compare.py replacing the proxy.
- [ ] P3: per-tree NDVI series via the citrus-tree-ndvi skill (Sentinel-2,
      all 4 ledger epochs + dec if georef allows); join onto P2's table.
- [ ] P4: Bateleur timeline scrubber real — drive it from ledger_v2 epochs;
      refresh ALL stale topdown exports (apr shows 81 trees vs 783 census;
      regenerate export_bateleur_topdown for every survey post-gen2 too).
- [ ] P5: dec_2025_a300 CPU half — kf20cm cut from odom (~220 kf) → blocks
      → census snap → berry-in-plant supervision compile (~15 px plant-mask
      dilation). GPU stage1 waits for a post-rotation window.
- [ ] P6: hygiene sweep — fix the 1 RED test in aru_sil_core suite; retire
      /tmp path refs in the 11 flagged active scripts (volatile-dependency
      class); teach_repeat legacy quarantine decision note for Paul.
- [ ] P7 (slack only): semantic pose-graph prototype — tree correspondences
      as loop-closure constraints (ceres/g2o over banked ledger pairs);
      success = the 1.4 m common-mode shift absorbed by the graph, 02
      inherits geometry. Research; timebox one day.

Out of scope for this queue (GPU): SAM3 fruit passes, DINOv2 pre-flight,
any training — they ride gen2 / post-rotation windows per the plan.
