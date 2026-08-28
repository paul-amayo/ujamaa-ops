# CPU week ledger — 2026-08-20 → 08-31 (weekend catch-up armed 08-28)

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
2026-08-21: the 08:23 tick was MISSED — the session died with the VM reboot
(09:23); re-armed 09:40 in the recovery session, P1 still top.

- [x] P1: gen2 M0 — centroid row stage into build_marker_hierarchy.py per
      plans/gen2_hierarchy_swap.md M0: census centroids in (all, from
      global_ids stats), HighInterface.ransac_init (Aug-18 binding path),
      outlier bucket => row_id -1 (never a row), site keys (row_num_models
      NEW, default 40), pilot apr+05 to experimental/ + citrus regression
      diff. Reference: automation/july_centroid_solve.py.
      DONE 2026-08-21: census RANSAC row stage + row_min_members (default 3)
      in build_marker_hierarchy (aru_sil_core f2773f6a); pilot on ALL six
      surveys to experimental/gen2_m0 (code 4962286): citrus keeps every real
      row (05 8, 04 8, 01 37, 03 39), apr catch-all 229 -> 72 (mv10: 53).
      M1/M3 inputs: apr band 0.6 is clean on mv10 (0 wide) but not on 573;
      13B dir gate 0.985 = 0 thieves (row-end fragments bucketed); 03 needs
      row_num_models >= 60. Notebook 2026-08-21 "gen2 M0 LANDED".
- [ ] P2: Sankofa structural metrics per tree per epoch — canopy extent /
      height / volume proxies from banked LiDAR cluster members in each
      registry (no splat needed); output per-tree per-epoch table keyed by
      ledger_v2 canonical ids; wire into compare.py replacing the proxy.
      2026-08-28 A100 attempt (automation/tree_struct_metrics.py, census
      argmax over stage1 gaussians): NOT measurement-grade — ownership
      smears along ray corridors (t72/b013 owned cloud spans 10x23 m, 0
      gaussians within 3 m of the hierarchy anchor; anchor-trim keeps 7%
      and reads 0.6 m "heights" = ground band). Same blend-attribution
      smear the fruit round-trip exposed. Next: per-block visibility
      weighting or W-mass weighting, cross-check vs init_lidar.ply — OR
      the banked-registry path once registries are rsynced from the VM
      (not on this box; no ssh config A100->VM yet). Diagnosis session
      work, not unattended-queue work.
- [ ] P3: per-tree NDVI series via the citrus-tree-ndvi skill (Sentinel-2,
      all 4 ledger epochs + dec if georef allows); join onto P2's table.
      08-28 note: blocked on this box — needs per-tree WGS84 from the
      registries (VM) or a gps_anchor_lio-derived anchor for census xyz.
- [ ] P4: Bateleur timeline scrubber real — drive it from ledger_v2 epochs;
      refresh ALL stale topdown exports (apr shows 81 trees vs 783 census;
      regenerate export_bateleur_topdown for every survey post-gen2 too).
      08-28: export refresh HALF queued (weekend queue topdown_01..05, all
      5 on-box surveys); scrubber half still open.

## Weekend catch-up queue (2026-08-28, Paul: "queue non GPU roadmap items")
Durable (setsid) sequential queue: automation/weekend_cpu_queue.sh, logs in
~/logs/weekend_cpu_20260828/ (STATUS file). Items: DATASETS.md regen [DONE,
6 surveys], regression Tier A [PASS], topdown exports x5, 10 fps 3D fruit
dedup fleet on every fruit-bearing tree of 04+05 (fruit3d_{prep,detect,
cluster}.py — validated same day on 05 t72), cross-epoch v3 join on
corroborated 3D counts (fruit_cross_epoch_04_05_v3.json). CronCreate ticks
were DENIED by the permission classifier this session — re-arm judgment
ticks in any new session with one line ("weekend tick per cpu_week_ledger").
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

## Session-death recovery (added 2026-08-21 after the VM reboot)
Durable — survive session death AND reboot: the GPU queue itself (setsid),
automation/queue_watchdog.sh (USER crontab `:13/:43` — `crontab -l`, not
/etc/crontab), daily_dashboard.sh (06:15 HTML rebuild), pointers/markers in
~/logs/week_prod_20260814_state/, this ledger.
Session-bound — die with the Claude session; re-arm in ANY new session with one
line ("revive the autonomous week"):
  1. persistent Monitor: `tail -n0 -F ~/logs/week_prod_20260814.log
     ~/logs/queue_watchdog.log | grep -E --line-buffered
     "CIRCUIT-BREAKER|ABORT|FAILED|SUP-SPARSE|ROUND-[0-9]+-DONE|WEEK-DONE|WEEK QUEUE START|GPU ready|SLOT .* OK|relaunching|NOT restarting"`
  2. CronCreate `23 8 * * *` (7-day expiry): queue-health glance (tail log,
     watchdog log, audit_pointers.sh, df floor 15G, queue alive) → republish
     lab_notebook/dashboard.html with Artifact url=
     https://claude.ai/code/artifact/3a2156c8-d3b4-4a2b-9968-923c007dc45a →
     top unchecked task here (~3 h, niced, commit-after-green, notebook) →
     status note to Paul.
Lessons: cron PATH has no ~/.pixi/bin — every cron-launched script exports
PATH itself (watchdog + queue fixed 08-21). Never run ollama (cpu-only Gemma,
~8 G RAM) or the adinkra :8003 server beside the rotation: a gsplat_cuda JIT
rebuild (8× nvcc ≈ 2–2.5 G each) OOM-killed on the 29 G box and cost 04 b008
(08-21 06:47–07:16). A transiently-failed block's requeue is Paul's call:
`audit_pointers.sh`, rewind the .next, archive the FAILED marker.
