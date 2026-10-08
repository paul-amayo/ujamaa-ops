# Recipe v3 — splatfacto chunks with co-trained exposure (proposal, 2026-10-08)

Status: DRAFT for Paul ("can this be the new prod recipe, v3"). Measured basis: lab_notebook/2026-10.md, 10-08 A/B on 05 chunk 1_0.

## What v3 is
Per chunk: the H3DGS view set (in-cell + outside views that see in, the H3DGS export's split: train / 1-in-10 held-out) trained with
**stock nerfstudio splatfacto + bilateral grid** (`--pipeline.model.use-bilateral-grid True`, `nerfstudio-data --eval-mode filename`,
init = the chunk's points3D.ply), 30k iterations, no hierarchy, no post-opt. Exposure is co-trained (the grid), not fitted afterwards.

## Why (measured on 05 chunk 1_0, identical views / poses / init / held-out list)
| | training views (walkthrough) median / mean | held-out raw | time | gaussians |
|---|---|---|---|---|
| v2 = H3DGS 60k + 15k post-opt, exposure affine | 24.29 / 24.14 | 15.76 | 3.2 h | 12 M |
| v3 = splatfacto 30k + bilateral grid (grid applied) | 23.64 / 24.13 | 16.47 | 17 min | ~0.7 M |
| splatfacto 60k + bilateral grid | (pending) | | ~35 min | |
- Same walkthrough quality, better generalisation, 11x faster, 16x smaller; two chunks per 40 GB card.
- Post-geometry exposure is NOT enough (frozen-geometry affine + colour refit: 22.9); the grid must be co-trained.
- The held-out ceiling (~16-17 dB) is the drive's geometry, shared by every recipe tried.

## What has to be built before it can be prod (THE BAR in the prod skill = blessed config + verdicts + registration)
1. **Chunk builder**: H3DGS export chunks -> per-chunk nerfstudio workspace (transforms.json with train_filenames / test_filenames,
   ply init). Exists as a one-off for 05 1_0 (`splatfacto_1_0/`); needs a script over all chunks of a survey (~1 day incl. a survey run).
2. **Serving**: a render service for splatfacto checkpoints that applies the trained grid of the nearest trained camera for recorded
   walks (the HIER_EXPOSURE=nearest idea) and a mean/identity grid for free poses; `launch_api` already has a nerfstudio
   render-service branch (the deprecated side-car's `render_service:app`) to start from. Must render at >= the stage's frame rate.
3. **Identity (the long pole)**: the native census (`hier_census.py`), seed (`build_census_init.py`), text bank and
   `native_identity.feature_pass` are written against the hierarchy (cut + interpolation weights). A flat-gaussian version is simpler
   (no cut, no interpolation): census = per-gaussian alpha weight on labelled pixels via gsplat's rasterization gradient; serve =
   per-gaussian 32-d features rasterised the same way. The deprecated side-car did exactly this on nerfstudio checkpoints
   (`gaussian_interaction_census.py`), so the maths exists. Budget: 2-3 days to parity on the containment verdicts.
4. **Checks before blessing** (same scorers as today): training-view PSNR >= v2 on the demo chunks (05 1_0 / 0_0, 01 3_1); held-out
   >= v2; tree containment IoU >= the native cells (05 1_0: 0.70 held-out); fruit no worse (0.25); walkthrough replay with
   per-view grids visually clean on the stage; two chunks concurrently measured, not assumed.
5. **Not covered by v3**: survey-scale LoD (one merged model rendered at a budget) - H3DGS's hierarchy remains the tool for that
   view; v3 serves chunks, as the demo does today.

## Open measurements
- 60k + bilateral (running); 01 3_1 and 05 0_0 under v3 (are the gains general?); a Klapmuts lane under v3 (the 8 M lane recipe
  is H3DGS 60k: 29.7 training / 19.8 held-out); render speed of the grid at 1280 px.
