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
- [serious] Fruit identity is not attached to fruit GEOMETRY — carriers sit on the right viewing ray at the wrong depth (on-orange in kf_542, on foliage in kf_543, ~79 px away, colour R-B 8.7 vs 65.1 at true fruit)
- [serious] The reconstruction substrate is view-overfit — train FG ~21 dB vs eval FG ~10 dB on the same block; every semantic number sits on top of this
- [warning] Vocabulary was never chosen for separation — tree-word cosine p90 0.90, max 0.9993
**Next:** fix the substrate before more semantics — depth supervision (now correctly aligned), pose refinement, early stop at the eval peak; then re-test protection with min_frames as a multi-view consistency filter.

## Bateleur — top-down farm state: instances and rows
**Q:** Can a trustworthy PLANT REGISTRY (instances, rows, counts) be built from noisy per-frame detections + LiDAR, and transfer across sites?
- [good] Citrus registry + QA method — census audit exposed a 24% undercount (112 vs 85); v4 NMS recluster canonical
- [good] klapmuts census, second site and new crop — 866 instances, median size 0.99 m = measured pitch
- [good] Row structure solved at the RANSAC init — 14 clean rows in 0.12 s with dir gate ±10° and thr = spacing/2; the CORAL optimiser is a no-op at λ=β=0 and destructive at 0.5
- [good] The top-down hierarchy is the identity authority — per-frame tree supervision is now its PROJECTION (mask shape from SAM3, identity from LiDAR-snapped census clusters), not a re-derivation
- [good] Transfer methodology — 9+ silent site/hardware constants externalised to rig.json + site.json per dataset
- [serious] Every site's blocks were cut per row-pass and ignore the length rule — citrus 04 31.9 m, klapmuts 44-50 m, citrus 05 57-64 m against a documented ~24 m cap
- [warning] Two-sided duplicates and far-side rows remain in the census — ~128 two-plant merges
- [serious] Citrus registries have never been re-verified under K-indexed poses
**Next:** finish the 05 length baseline (third site), then set block length PER SITE and enforce it in build_row_blocks.

## Sankofa — the tree ledger across time
**Q:** Is it the SAME plant across epochs — without a shared datum — and what changed biologically?
- [good] Absolute WGS84 association on citrus — 01/03/04 associated at 99% match, latency calibrated (+300 ms)
- [good] Systematic 1.4 m common-mode shift found and corroborated two ways — correcting it took 219 → 272 pairs at a TIGHTER gate, which is the semantic pose-graph lead
- [good] Ledger v0 — 677 observations / 402 canonical trees / 275 multi-epoch
- [good] Dec-2025 A300 routing settled — the bag carries 3D LiDAR + INS odom + ZED stereo, so it goes the LIO/odometry route; COLMAP would discard the LiDAR and return scale-free poses
- [good] Blueberries are detectable — SAM3 image mode on dec_0144 gives 58 masks for 'berry' at fill 0.76, geometry close to citrus oranges, so the in-tree recipe should transfer; qualified multi-word prompts return nothing
- [serious] Dec-2025 ingest is net-new — ROS2 mcap, sensor_msgs/PointCloud (not PointCloud2), no raw IMU topic, A300 lidar→camera extrinsics unknown
- [warning] The ledger compares a structure PROXY, not phenology
**Next:** build the mcap reader (images + INS-odom poses + cloud) → kf20cm dump → blocks → LiDAR init; DINOv2 Dec→April retrieval pre-flight still unrun.

## Adinkra — natural-language query over the twin
**Q:** Can plain language select geometry — "the fruiting trees in row 7" — through the hierarchical embedding?
- [good] Per-panel agents live in the ujamaa repo (ollama Gemma 4); sankofa panel wired to the ledger digest
- [good] The scoring mechanics are now fully dissected — per-step walk decomposition tools; walked max-over-steps is LEVEL-BLIND by construction (extends every pixel to all depths; ceiling glows the whole parent mask at 1.0)
- [good] CONTAINMENT scoring discovered (the walk function's own extrapolation mask, previously unused): match ancestors + own point only. Open-vocab masks-inside-masks: "cassette" -> tree-84 mask IoU 0.847; "nectar" -> fruit blobs inside it (recall 0.77); absent fruit ("persimmon" in 542) -> correctly EMPTY where walked scoring would fabricate the whole canopy
- [serious] Containment quality tracks per-tree radial health — deflated tree 73 segments at IoU 0.22 (spills into 84); depth-corrected 84 at 0.85. Per-level norm maintenance is the gate
- [good] Vocabulary opened: v4_1k proposal (1024 tree + 64 fruit, wordfreq-filtered farthest-point CLIP sampling) — max collision 0.65 vs current 0.92/1.00; found 'copper'+'coral' in BOTH current lists; 416 words < klapmuts 866 objects
**Next:** Paul strike-list on v4_1k -> embedder retrain (with level_target_norms ON) -> stage2 -> full sign-off; containment as a named viewer mode.

## Cross-cutting — reconstruction quality
**Q:** Is the underlying splat good enough for any of the above to mean anything?
- [resolved] The train/eval gap is REFRAMED, not a bug — eval frames sit 25 cm from train neighbours (interpolation test) and the product is a flythrough near the trajectory. Doctrine: headline = train TREE/FRUIT PSNR; eval demoted to breakage check; floaters (geometry gate) are the real off-axis risk
- [good] Block length matters, but NOT uniformly — klapmuts eval 11.3 → 16.2 dB going 47.9 → 19 m, while citrus eval FALLS 11.6 → 7.2 dB over the same shortening; the cap is a klapmuts-shaped result, not a law
- [good] Sky supervision earns its place — eval FG 9.49 → 10.51 dB and 33% fewer gaussians (the sky floaters are gone)
- [serious] Depth supervision is a BUST after two invalid measurements — misaligned rows (S6..s9b + klapmuts sweep), then a metres-vs-internal-units loss in both "aligned" runs; the honest units-fixed result was +0.34 eval FG for −1.7 train FG. Direction retired; z-buffered survey-cache depth maps + tested camera_convention.py remain as assets
- [good] LOD-on-trees is the working lever — tree-weighted photometric L1 (bg=0.0 winner): train TREE 17.92 → 19.14, FRUIT 12.31 → 13.14, 15% fewer gaussians; ground keeps only 7–8% of budget in every run so eradicating it buys nothing
- [good] scene.json manifests exist (TESTING.md §3, implemented) — 04's pins the treelod_bg00_v1 recipe as THE scene baseline; scene_baseline.sh runs whatever the manifest declares
- [good] Two-stage recipe VALIDATED end-to-end — stage2_fruitchild: appearance BIT-IDENTICAL to bg00 (19.14/13.14) + fruit level real at its own radius: NO-WALK pointing 94.6% recall / 3.5% FP (was 0/0), fruit-px norms 6.7 -> target 7.15. Three resume-era bugs found+fixed on the way (optimizer orphaning, freeze-dies-at-load, canary info clobber)
- [serious] Tree level radially DEFLATED (px median 1.8-2.4 vs 4.67 target; blending shrinkage) — the next norm-maintenance target after fruit
- [good] FRUIT SIGN-OFF battery formalized (TESTING.md 4b, fruit_signoff.sh): 8 tests incl. no-walk cross-level pointing (the headline), ceiling control, FP anatomy; partial suites cannot sign off
**Next:** cross-view identity (test 8, still failing) and tree-level norm maintenance; then second block replication.

## Spoor · Azalai · Hapi — design-stage pillars
**Q:** Deliberately "coming soon" in the demo — scope control is the plan, not a failure.
**Next:** design mockups only until the shipped slice is solid.
