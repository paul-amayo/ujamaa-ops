# Autonomous experiment week — 2026-07-14 → 2026-07-20

**Mode:** Paul away / hands-off. Claude drives via a recurring supervisor session
(every ~3 h): advances the queue, reads metrics (psnr skill), applies the decision
gates below, writes lab-notebook entries, commits green results locally, and
notifies on gate decisions + failures. Weekly report generated Sunday.

**Hard rules for the week**
- Nothing is deleted. Reclaimable files move to `/home/paperspace/data/_quarantine/`
  (Paul empties it on return).
- **No pushes at all this week** (Paul's call, 2026-07-13): all commits stay local,
  including today's unpushed review-pending commits. Paul reviews on return.
  Commit-on-green is authorized (local commits to the exp branches).
- Disk guard: no new run starts under 120 GB free; supervisor quarantines stale
  training caches first, then pauses the queue and notifies.
- Every pipeline-code change re-runs the regression fixture (block_000 pass-0,
  FG-PSNR 23.6 ± 0.7 dB) before the queue continues.
- Every run — pass or fail — gets a notebook entry. Failures: quarantine the item,
  continue the queue (never wedge the week on one block).
- All training on experiment branch `exp/2026-07-14-week` in both repos; green
  results committed same day with the run's notebook entry ref.

---

## Track A — Semantics: R-retrain validation → rollout (GPU, the week's main line)

Today's radial-norm fix means all existing splats trained against collapsed (r≈0.5)
hierarchy targets. Question: does a correctly-normed embedder improve relevancy?

- **A1.** Retrain HyperEmbedders with config-loaded `level_target_norms`
  (`train_hyperembedder_graph.py`): `01_13B_v2R`, `03_13B_v2R`. Sanity: decoded
  leaf norms ≈ trained target norms, not 0.5.
- **A2.** Rebuild lookup targets + retrain the **6-block A/B set** (3 blocks × 2
  surveys), two arms: old embedder (baseline, exists — reuse) vs v2R. C-config,
  20k, depth-sup.
- **A3.** Relevancy harness (`full_relevancy_eval` family) on both arms.
- **GATE G1:** v2R object-relevancy ≥ baseline + 3 pts (01_13B baseline 76 %)
  AND row-relevancy not degraded →
  - **PASS → A4:** rollout — rebuild targets + retrain remaining 01_13B blocks
    (~29) and 03_13B half-row blocks; re-run harness per survey; refresh the
    served demo splats.
  - **FAIL → A5:** stop rollout. One diagnostic pass (decode path on v2R
    features, confusion matrix by hierarchy level), notebook write-up, GPU goes
    to Track C. Do not iterate embedder configs unattended.

## Track B — 4D backbone: 02 + 05 epochs (CPU/IO, parallel throughout)

- **B1.** Verify 02_13B (77 GB) and 05_13D downloads complete + checksums; finish
  downloads if partial. Combine bags. (Pre-check: system python3 numpy/scipy.)
- **B2.** 02_13B ingest → tree registry → 4D association vs 01/03 (ENU 2D-translation
  registration — pre-WGS84 survey).
  **GATE G3:** match rate ≥ 90 % (01↔03 achieved 99 %) → add epoch to registry;
  else quarantine, diagnostic notebook entry, do NOT force-merge into the ledger.
- **B3.** 05_13D ingest → registry → associate vs 04_13D (true-WGS84 pair).
- **B4.** Sankofa substrate: per-tree NDVI time series (citrus-tree-ndvi) for every
  registered epoch; canonical ledger table (tree_id × epoch × {centroid, NDVI,
  canopy proxy}) written to the Sankofa POC's expected format.
- **B5.** If A-rollout leaves GPU headroom: train 2–3 keystone 02/05 blocks so each
  epoch has at least one demo-grade splat.

## Track C — Splat quality: sky objective ablation (GPU, after G1 or on A-fail)

Dark-floater sky is an objective problem (black bg + sky-loss 0.05 + masked
metrics ⇒ sky unsupervised).

- **C1.** 4 variants × 2 blocks (block_000 + worst-sky block): (i) baseline,
  (ii) random-color bg, (iii) sky-alpha penalty λ=0.01, (iv) λ=0.05.
- **GATE G2:** pick variant with no dark floaters (sky-region mean luminance /
  alpha stats + render strip) at FG-PSNR within 0.3 dB of baseline.
  Winner → apply to the demo showcase blocks; no clear winner → notebook
  write-up, leave baseline.
- **C2 (fill).** Half-row repartition rollout (`build_row_blocks`) on remaining
  full-row 03 configs; criterion: front-frame PSNR flat (no −2.7 dB/100-frame slope).

## Day-0 setup (needs Paul present, ~today)

1. Review + authorize push of today's unpushed commits (`c50c995f`, `577d444c`,
   `1fbe48e`, `d091981`, ujamaa `0d098f8`, `9014e16`) — or defer to end-of-week.
2. Approve quarantine policy + the standing authorizations (commit-on-green,
   end-of-week push, notification channel).
3. Claude: create `exp/2026-07-14-week` branches; pin regression fixture as a
   script; write queue-runner + disk-guard; smoke-test one embedder retrain,
   one block train, one association step end-to-end; start supervisor loop.

## Rough load plan (single GPU)

| Day | GPU (booked ≈h / 24) | GPU filler (Track E) | CPU |
|---|---|---|---|
| Mon | A1+A2+A3 (~9h) | E-03 halves overnight (~15h) | B1 |
| Tue | **G1** → A4 rollout (~24h) | — | B2 → **G3** |
| Wed | A4 (~24h) | — | B3, B4 |
| Thu | A4 finishes (~10h) | E-04 keystones (~14h) | B4, D1 |
| Fri | C1 sky ablation (~7h) | E-05 + E-02 keystones (~17h) | D1/D2 |
| Sat | **G2** → demo blocks (~5h) | E leftovers + E-RETARGET (~19h) | D2/D3 |
| Sun | refresh served splats (~4h) | buffer | report + consolidation |

Estimate basis: ~50 min per 20k C-config block (smoke-test iteration rate); supervisor
re-projects daily from the first measured block. If G1 fails, A4's ~41h flows to E/C/D.
End state target: 01 complete (already), 03 complete, 02/04/05 keystone splats — every
registered epoch demo-visible.

## End-of-week deliverables
- Relevancy: v2R vs baseline verdict with numbers (G1 outcome).
- 4–5-epoch tree registry + Sankofa ledger substrate (G3 outcome per epoch).
- Sky-objective verdict (G2) and updated demo blocks.
- Lab notebook complete for every run; green work committed; weekly summary +
  proposed next-week plan in `lab_notebook/2026-07.md`.
