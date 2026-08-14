# Tassili serving POC — 35-block flythrough without sacrificing detail

**Started:** 2026-07-13 · **Scope:** one week · **Branch:** `aru_sil_core/src` @ `splat/walkthrough`
**Goal:** serve survey 01 (35 `lio_row` splats, 2.43 GiB PLY, 44.2 M gaussians) as a smooth
browser flythrough on the tassili viewer (port 8001), with full detail wherever the camera is.

## Why this shape

The viewer skeleton (distance-ranked streaming, trajectory prefetch, sphere placeholders,
Brotli, COOP/COEP for SharedArrayBuffer) is already right. What flows through it is wrong:
raw 70 MB PLYs parsed in-browser, ~1.3 M gaussians per `addSplatScene` (rebuild hitch),
only 3 resident blocks. Levers, by measured leverage:

1. **Format** — PLY → `.splat` (32 B/gauss) now, `.ksplat`/`.RAD` when node tooling is in.
   ~5× less network + near-zero parse. Lossless in practice.
2. **Prune** — measured on block_005: 20 % of gaussians below alpha 0.02, 37 % below 0.05;
   spatial outliers span 117 m on a ~50 m row (block radii up to 102 m). Gated by a
   render-diff fidelity check, so "no sacrificed detail" is a number, not a vibe.
3. **Crop overlap** — each block independently trained its own copy of the shared
   background → Voronoi-crop gaussians to the block whose camera trajectory owns them.
4. **Two-tier LOD** — per block, a far tier (~18 % importance-sampled) that is *always*
   resident (35 × ~6 MiB ≈ 210 MiB, ~7 M gauss) replaces sphere placeholders; full tier
   streams in by camera distance. Full detail near, real splats far, no spheres.
   (This is Hierarchical-3DGS with the cut quantized to block granularity — true
   continuous LOD needs a cut-selection renderer no web library has, except possibly
   Spark 2.0, hence the spike below.)
5. **Loader tuning** — residency budgeted by gaussian count (~5–6 M full-tier) not block
   count; drop hysteresis; prefetch N+2 along the walkthrough path.

## Day plan / status

- [x] **Day 1 — stage + baseline.** `stage_splat_manifests.py` (new) wrote
  `index.json`/`splats.json` for `lio_row`; spatial `survey_hierarchy.bin` built (48
  nodes); server restarted → `HIGH_DATA_ROOT=01_13B_Jackal HIGH_SPLAT_CONFIG=lio_row`.
  Verified: 35/35 ready, 289 trees, `/tassili/` 200. Baseline: 2.43 GiB, mean block
  71 MiB / 1.26 M gauss, localhost fetch 0.68 s. Browser FPS baseline **pending Paul's
  tunnel session**. Lab notebook entry ✓.
- [x] **Day 2 — compaction pipeline.** `compact_block_splats.py`: alpha prune + scale
  cap + Voronoi crop + Gumbel-top-k far tier (18 %). Final sweep at
  `--min-alpha 0.02 --crop-margin 8`: 44.2 M → 34.6 M gauss, 1.03 GiB full `.splat`
  + 190 MiB far tier. Findings: pruning is the whole win (crop at margin 8 = no-op
  guard-rail); Brotli on `.splat` is 95 % — skipped.
- [x] **Day 3 — fidelity gate.** `render_fidelity_gate.py` (gsplat, nerf_new pixi):
  **35/35 PASS**. Gate redesigned during iteration: (A) prune-only GT-delta ≤ 0.5 dB —
  measured ≤ +0.20 dB, median +0.02; (B) compact scene vs baseline scene *at equal
  residency* (own full + ≤20 m neighbour fulls + far tiers vs original PLYs same
  layout), GT-delta mean ≤ 0.5 / worst ≤ 1.5 dB — measured −0.04…−3.74 dB mean
  (compact BEATS originals: pruning removes multi-resident low-alpha haze), worst
  view +0.81 dB. Gotcha: `transforms.json` is OpenCV c2w, not GL (`--convention cv`).
  Server + loader landed same day: compact/far serving live on 8001 (31 MiB + 5.6 MiB
  per block verified), `splat_loader.js` rewritten two-tier. Browser exercise pending.
- [ ] **Day 4 — Spark 2.0 spike.** `build-lod` → `.RAD` on pruned blocks; minimal
  standalone `static/spark_test.html` (three.js + Spark SplatMesh streaming LOD, 16 M
  splat GPU pool). Needs node (install via pixi). Check `relevancy.js` coupling to
  `@mkkellogg` internals before committing to a swap.
- [ ] **Day 5 — integrate winner + demo.** Either Spark, or two-tier in
  `splat_loader.js` (far tier always resident, gaussian-budget residency, hysteresis,
  N+2 prefetch) + `server.py` manifest extensions (`far_url`, gauss counts). Scripted
  full-survey flythrough with HUD (FPS / resident gauss / MB fetched) vs Day-1 baseline.
  **Paul sign-off checkpoint.**

## Landing zones

- Viewer: `interfaces/splat_viewer/static/{splat_loader,hierarchy_lod,main}.js`, `server.py`
- Offline: `scripts/stage_splat_manifests.py` ✓, `scripts/compact_block_splats.py` ✓,
  `scripts/render_fidelity_gate.py` (Day 3)
- Data (not git): `blocks_ns/lio_row/{index,splats}.json` ✓, `survey_hierarchy.bin` ✓,
  per-block `exported/compact/*`
- Records: `lab_notebook/2026-07.md` (Day-1 entry ✓)

## Browser shakeout findings (2026-07-13 evening, via Claude-in-Chrome)

- **Brotli `.splat.br` actively breaks gaussian-splats-3d 0.4.7**: the lib stream-reads
  against Content-Length (compressed) while the browser hands it decompressed bytes —
  first load hung forever, subsequent adds rejected. `.br` siblings DELETED (they were
  only 95 % anyway). Lesson: never serve Content-Encoding'd splat payloads to this lib.
- **A hung/failed addSplatScene wedges the viewer instance** (sort worker never spawns;
  every later add rejects with the generic "Could not load file"). Needs a page reload
  to recover — loader should get a watchdog/re-init path eventually.
- **Tunnel bandwidth reality: ~0.09 MB/s** (25.5 MB in 278 s). At this rate: full tier
  ≈ 6 min/block, whole far-tier backdrop ≈ 35 min. The two-tier design is right but the
  payload needs the quantized-format step (.ksplat/.RAD ~4-8×, SOGS ~15-20×) to be
  usable over this link — OR the demo browser must sit nearer the server. Spark spike
  is now the main event, not a nice-to-have.
- Chrome pauses rAF when the window is occluded → loader only progresses while the tab
  is visible. Not a bug, but explains "nothing happens" reports.

## Constraints & notes

- No pushes this week (Paul, 2026-07-13); commit-on-green locally; nothing deleted —
  compacted assets live *alongside* originals, originals untouched.
- Expected end state: ~2.4 GiB → ~500–600 MiB assets; far field always real splats;
  full detail near camera; add-scene hitches hidden in prefetch dwell time.
- Known risk: 7 M always-resident far-tier gaussians may stress the per-frame sort on
  the target GPU → fallback is merging far tiers into one static backdrop scene
  (`HIGH_USE_MERGED` path half-exists already).
- Found in passing: `find` intermittently missed existing files under
  `blocks_ns/lio_row/block_00{0,1}` (NFS-ish staleness?) — re-verify with `ls` before
  trusting a negative.
