# UJAMAA pillars — the science narrative

The dashboard renders this file. It lives in `lab_notebook/` on purpose: it is
a notebook artifact, not code, so it can be updated in the SAME commit as the
day's entry. The dashboard flags how many notebook entries have landed since
this file last changed — keep that counter at zero.

Format (parsed by automation/build_dashboard.py):
`## Name — tagline` / `**Q:** question` / `- [status] claim — evidence` /
`**Next:** what happens next`.  status ∈ good | warning | serious | critical

## Tassili — survey → queryable hierarchical splat
**Q:** Can a farm survey become a 3D field you can QUERY BY LEVEL — section, row, tree, fruit — from one hyperbolic embedding (depth == radius on the Lorentz manifold)?
- [good] Tree and row levels separate in the rendered field — 99.6-100% cross-level tree pointing on every block trained since the compiler landed
- [good] Supervision is now COMPILED, not painted — detection ledger + hierarchy + named filter spec → uint16 id maps + manifest (pinned id→word and id→level tables, source hashes, per-gate drop counts); colours are a one-way viz export
- [good] Fruit level revived and now beats the old contaminated baseline — aligned gates s7 32.3% → s8 44.5% → s9b 53.3% vs S4's 51.7%, using ~1/3 the supervision pixels
- [good] Two levers found for the fruit level — fruit-aware densification (force-split protected coarse gaussians) and extending the split window to 19k; tau 3.0 REGRESSED (noisy single-observation anchors)
- [good] Measurement ruler fixed — gates now score each run against ITS OWN supervision; the old template scored residue-paint positions and under-measured every clean-label run
- [serious] Fruit identity vs fruit GEOMETRY — carriers sat on the right viewing ray at the wrong depth (~79 px, colour R-B 8.7 vs 65.1 at true fruit). EVIDENCE PREDATES CENSUS-INIT (July zero-init era); the view-based battery since passed (recall 0.94, pointing 98.9) but carrier DEPTH has not been re-measured on a census-init checkpoint — re-measure before quoting either way
- [serious] The reconstruction substrate is view-overfit — train FG ~21 dB vs eval FG ~10 dB on the same block; every semantic number sits on top of this
- [good] Vocabulary redesigned for separation (v4_1k: 1024 tree + 64 fruit, farthest-point CLIP) — swap + embedder retrain landed 2026-08-06: tree->row retrieval 100% (beats 99.03), fruit->tree 100%; max collision 0.65 vs old 0.92/1.00
- [good] **PAPER HIGHLIGHT (Paul 2026-08-07): census-init makes the whole hierarchy queryable per-view** — single bare words segment row (IoU 0.80/0.87) ⊃ tree (0.80-0.91) ⊃ fruit (recall 0.94, absent fruit correctly empty) in kf_542 via containment; no-walk cross-level pointing 98.9/98.9 at FP 10.4%
- [good] Paper Figure 1 exists (five iterations under Paul's art direction, 2026-08-09..12) — (a) NAIP aerial band: real 0.6 m imagery, epoch-1 dots + epoch-2 rings on individual visible canopies, 1,250+ plants / 3 sites / 5 epochs; (b) SAM3+CLIP baseline clash: flat scores, the ordinal query ranks the third tree LOWEST; (c) HiGH multi-place multi-level relevancy + no-response panel; hero drafts A/B banked (perpendicular novel views substrate-limited); source promoted to high/paper_figs/fig_teaser.py
**Next:** serve a census-init splat on tassili; klapmuts census-init stage2 once Paul picks block length; LERF-3D trained baseline for the experiments section.

## Bateleur — top-down farm state: instances and rows
**Q:** Can a trustworthy PLANT REGISTRY (instances, rows, counts) be built from noisy per-frame detections + LiDAR, and transfer across sites?
- [good] Citrus registry + QA method — census audit exposed a 24% undercount (112 vs 85); v4 NMS recluster canonical
- [good] klapmuts census, second site and new crop — 866 instances, median size 0.99 m = measured pitch
- [good] klapmuts training substrate BUILT (2026-08-08 overnight): sack supervision compiled for all 10 row blocks (v4_1k word table), embedder klapmuts_v2vocab1k retrained, bg00 stage-1 fleet done — census-init stage2 is one command away (block-length question open)
- [good] Row structure solved at the RANSAC init — 14 clean rows in 0.12 s with dir gate ±10° and thr = spacing/2; the CORAL optimiser is a no-op at λ=β=0 and destructive at 0.5
- [good] The top-down hierarchy is the identity authority — per-frame tree supervision is now its PROJECTION (mask shape from SAM3, identity from LiDAR-snapped census clusters), not a re-derivation
- [good] Transfer methodology — 9+ silent site/hardware constants externalised to rig.json + site.json per dataset
- [warning] 02's row fit collapsed (n_rows=1) on its sparse 9-leg coverage — 02 row_id unusable until re-fit; association/ledger unaffected (centroid-based)
- [warning] LEGACY blocks predate the block-length rule — citrus 04 31.9 m, klapmuts 44-50 m, citrus 05 57-64 m. The rule now EXISTS and is enforced for new builds (Paul 2026-08-13: ~100 kf == ~20 m, site-independent; pipeline default). Re-cut legacy configs when their stage2 work re-runs; the canonical 04 fruit substrate still sits on 31.9 m blocks
- [warning] Two-sided duplicates and far-side rows remain in the census — ~128 two-plant merges
- [serious] Citrus registries have never been re-verified under K-indexed poses
**Next:** finish the 05 length baseline (third site), then set block length PER SITE and enforce it in build_row_blocks.

## Sankofa — the tree ledger across time
**Q:** Is it the SAME plant across epochs — without a shared datum — and what changed biologically?
- [good] **Dec-2025 mcap ingest UNBLOCKED (2026-08-09)** — rosbags AnyReader opens the mcap natively and husky_bag_to_monolithic was ALREADY WRITTEN for the A300 topics (Paul's call: no new scripts). Round-trip proven; full ZED stack incl. registered depth; LiDAR is PointCloud2 (July's PointCloud fear was wrong); INS+ZED odom agree to 1.7% over a 44 m row pass at 0.78 m/s; INS z drifts 3.8 m (LiDAR owns the vertical)
- [good] Per-epoch prompts are a SCENE FACT (scene.json prompts.dec_2025) — Dec bags answer to 'plant pot'/'grow bag' (12 dets) and NOT 'sack' (0); Jan is the mirror (sacks yes, berries none). Berry association measured: berry-in-PLANT containment 92.9% px vs pot-column 28.8% — recipe = berry ⊂ plant ⊂ (pot adjacency | 3D census snap)
- [good] Dec SAM3 ledgers banked: 290/290 frames (plant pot / plant / berry) in sam3_ledger_v0 — the ingest-independent half of the Dec fruit level
- [good] DEC TOP-DOWN REGISTRY (2026-08-09) — SAM3 ledgers × native ZED depth × odom fused to world coordinates: 199 pot clusters (166 solid vs ~176 expected) + 299 berry observations; berry mass sits in the 5–10 m stretch (real non-uniform fruiting), fallen ground berries auto-rejected by the plant gate
- [good] Absolute WGS84 association on citrus — 01/03/04 associated at 99% match, latency calibrated (+300 ms)
- [good] **02 EPOCH LANDED (2026-08-12) — the ledger now holds all four surveys**: 854 obs / 452 canonical trees; 02↔03 136 pairs @0.77 m (81% of 02, tighter 1.5 m gate after the day-datum shift); 02 is a SAME-DAY repeat of 01 (3 min gap) → a zero-growth CONTROL epoch that calibrates the Comparison noise floor; anchor-delta triangle closes to 4 cm (per-DAY RTK datums — fresh pose-graph-lead evidence)
- [good] Systematic 1.4 m common-mode shift found and corroborated two ways — correcting it took 219 → 272 pairs at a TIGHTER gate, which is the semantic pose-graph lead
- [good] Ledger v0 — 677 observations / 402 canonical trees / 275 multi-epoch
- [good] Dec-2025 A300 routing settled — the bag carries 3D LiDAR + INS odom + ZED stereo, so it goes the LIO/odometry route; COLMAP would discard the LiDAR and return scale-free poses
- [good] Blueberries are detectable — SAM3 image mode on dec_0144 gives 58 masks for 'berry' at fill 0.76, geometry close to citrus oranges, so the in-tree recipe should transfer; qualified multi-word prompts return nothing
- [warning] A300 lidar→camera extrinsics still unmeasured — needed for the LiDAR-init path on Dec blocks (the rest of the "net-new ingest" fear retired 2026-08-09: mcap reads natively, cloud IS PointCloud2, existing husky tool wrote the monolithics)
- [resolved] ledger georeference datum — Paul caught per-survey WGS84 anchors 6.6 m apart in Fig 1 (2026-08-12); build_ledger_v2.py now re-anchors every survey to the 03 datum after each rebuild (verified 0.000 m pair medians; 02 auto-re-anchored on landing). Consumers read v2; v1 stays as per-survey provenance. Residual kernel: fig_teaser band (a) still carries its own inline correction (drop when it reads v2); compare.py + adinkra read v1 but are unaffected
- [warning] The ledger compares a structure PROXY, not phenology
**Next:** kf20cm keyframe cut from odom (~220 kf) → blocks → LiDAR init → census snap → berry-in-plant supervision (ledgers banked, ~15 px plant-mask dilation in the compile); DINOv2 Dec→April retrieval pre-flight still unrun.

## Adinkra — natural-language query over the twin
**Q:** Can plain language select geometry — "the fruiting trees in row 7" — through the hierarchical embedding?
- [good] Per-panel agents live in the ujamaa repo (ollama Gemma 4); sankofa panel wired to the ledger digest
- [good] The scoring mechanics are now fully dissected — per-step walk decomposition tools; walked max-over-steps is LEVEL-BLIND by construction (extends every pixel to all depths; ceiling glows the whole parent mask at 1.0)
- [good] CONTAINMENT scoring discovered (the walk function's own extrapolation mask, previously unused): match ancestors + own point only. Open-vocab masks-inside-masks: "cassette" -> tree-84 mask IoU 0.847; "nectar" -> fruit blobs inside it (recall 0.77); absent fruit ("persimmon" in 542) -> correctly EMPTY where walked scoring would fabricate the whole canopy
- [resolved] Containment quality tracked per-tree radial health — and "radial health" turned out to be UNFINISHED GROWTH from zero-init, not decay: census-init lifts tree 73 containment 0.27 -> 0.80, tree 84 -> 0.911, rows 0.80/0.87 (2026-08-07)
- [good] Vocabulary opened: v4_1k proposal (1024 tree + 64 fruit, wordfreq-filtered farthest-point CLIP sampling) — max collision 0.65 vs current 0.92/1.00; found 'copper'+'coral' in BOTH current lists; 416 words < klapmuts 866 objects
**Next:** containment as a named viewer mode (tassili panel); row/section words get the farthest-point treatment; level_target_norms stays a deliberately-unused variable.

## Cross-cutting — reconstruction quality
**Q:** Is the underlying splat good enough for any of the above to mean anything?
- [resolved] The train/eval gap is REFRAMED, not a bug — eval frames sit 25 cm from train neighbours (interpolation test) and the product is a flythrough near the trajectory. Doctrine: headline = train TREE/FRUIT PSNR; eval demoted to breakage check; floaters (geometry gate) are the real off-axis risk
- [good] Block length matters, but NOT uniformly — klapmuts eval 11.3 → 16.2 dB going 47.9 → 19 m, while citrus eval FALLS 11.6 → 7.2 dB over the same shortening; the cap is a klapmuts-shaped result, not a law
- [good] Sky supervision earns its place — eval FG 9.49 → 10.51 dB and 33% fewer gaussians (the sky floaters are gone)
- [resolved] Depth supervision RETIRED as a direction (record, not a live risk) — two invalid measurements (misaligned rows, then metres-vs-internal-units), honest units-fixed result +0.34 eval FG for −1.7 train FG. Assets kept: z-buffered survey-cache depth maps + tested camera_convention.py
- [good] LOD-on-trees is the working lever — tree-weighted photometric L1 (bg=0.0 winner): train TREE 17.92 → 19.14, FRUIT 12.31 → 13.14, 15% fewer gaussians; ground keeps only 7–8% of budget in every run so eradicating it buys nothing
- [good] scene.json manifests exist (TESTING.md §3, implemented) — 04's pins the treelod_bg00_v1 recipe as THE scene baseline; scene_baseline.sh runs whatever the manifest declares
- [good] REGRESSION HARNESS exists (2026-08-12, roadmap Phase 1 closed) — Tier A: K-domain consistency, ledger_v2 datum invariants, embedder round-trip; Tier B: block_001 scene-baseline probe @2k iters vs committed expected values; unified pipeline hardened the same day (--registry-only, K consistency gate) — proven by 02's fresh run
- [good] **POSE JSONS KILLED (2026-08-13, Paul's call — monos-only)** — kf_domain.py derives K/ts/poses straight from the kf + lio monolithics (exact TransformMap semantics, float64; validated ≤7.4 mm vs the legacy C++ jsons, whose float32 drift it BEATS); kf_poses API unchanged so every consumer inherited; association regression bit-identical (136 pairs @0.77 m); harness A1 now shows ALL FIVE surveys K-consistent — the "legacy skew" lived in the dead jsons. Still owed: first kf-native block build end-to-end; Q4 registry re-verification (clusters built against old poses)
- [good] Two-stage recipe VALIDATED end-to-end — stage2_fruitchild: appearance BIT-IDENTICAL to bg00 (19.14/13.14) + fruit level real at its own radius: NO-WALK pointing 94.6% recall / 3.5% FP (was 0/0), fruit-px norms 6.7 -> target 7.15. Three resume-era bugs found+fixed on the way (optimizer orphaning, freeze-dies-at-load, canary info clobber)
- [resolved] "Radial deflation" was ZERO-INIT UNDERTRAINING, proven by the census-init experiment — per-gaussian interaction census (autograd through the rasterizer = exact blend weights) -> majority-label feature init -> one 44-min refine puts trees 73/60/84 at cos 1.00/1.00/0.99 with radius on target to 2 decimals (were 0.57/0.50/0.67 at half radius). Zero-init stage2 retired; census-init is the standing recipe. Collision/coverage/distance/conflict were all measured and eliminated first (2026-08-07)
- [good] **FIRST FRUIT SIGN-OFF GRANTED (Paul, 2026-08-07)** — stage2_censusinit_fw2 is block_001's canonical fruit checkpoint; battery = 8 real tests (cross-view dropped, containment ladder added as test 8 after it shipped missing from the block_003 verdict)
- [good] Census-init REPLICATED on block_003 — no-walk 100/100 at FP 14.7%; containment tree 0.919 / row 0.859; the weak-tree pattern (tree 11, 0.46) persists cross-block
- [resolved] Fruit halo is REFINEMENT-born, not init-born — margin-init (fruit share >= 0.6) produced a field identical to 3 decimals (17 demoted gaussians re-converge). Fruit verdicts: blob-level metric + competition masks; px-IoU retired for ~8 px objects
**Next:** loss-side fruit gradient gating if the halo ever matters; second-site census-init (klapmuts, all prerequisites banked).

## Spoor · Azalai · Hapi — design-stage pillars
**Q:** Deliberately "coming soon" in the demo — scope control is the plan, not a failure.
**Next:** design mockups only until the shipped slice is solid.
