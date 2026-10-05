# Disk audit 2026-10-05 (Paul: "what's taking all the memory, rogue checkpoints? we had 300 GB free a few days ago")
Free: 48 GB of 2.0 TB (98 %). Written since 10-02: ~198 GB (174 GB in 174 files > 50 MB, 24 GB in 35,532 smaller files).
Measured with find -newermt 2026-10-02 and du --apparent-size. rm is Paul's: nothing below has been deleted or moved.
Do it AFTER the demo; until then the side-cars are the stage's revert path.

## Candidates (~190 GB)
| GB | path | why it can go |
|---|---|---|
| 75 | citrus_all/05_13D_Jackal/experimental/h3dgs_sidecar_chunks/ (chunk_1_0_expo alone 52) | side-car deprecated (Paul 10-04); the stage serves native identity since 10-05 |
| 15 | citrus_all/01_13B_Jackal/experimental/h3dgs_sidecar_chunks/ | side-car deprecated |
| 15 | image_farm/gwakungu/2026-05-16/IMG_7993_s0_cabbage/demo_root/experimental/h3dgs_sidecar_chunks/ | cabbage side-car (v2_e400 embedder, IoU 0.02); KEEP h3dgs_native/ and prod/ next to it |
| 40 | citrus_all/01_13B_Jackal/prod/tassili/blocks_ns/lio_row100/block_*/splat_runs_FEATFIX/era_census07_20260822/ (71 ckpts) | old-era block seeds, kept "for the record" at the 10-03 CORAL/median swap; they decode only through the retrained-away embedder |
| 21 | features_A_block.bin in 10 h3dgs_native cells | seed A never served; every stage and cut uses features_B_bg2share.bin |
| 11 | citrus_all/05_13D_Jackal/experimental/h3dgs_01recipe/ | the 01-recipe retrain of 05 1_0: not better on training views (10-04) |
| 9 | citrus_all/01_13B_Jackal/experimental/h3dgs_native/chunk_3_1_sam3_era_liftmin_20261003/ | superseded 3_1 cell (pre-median-lifting) |
| ~5 | demo_video_v2/*/frames*, logs/demo_chunks/*/frames | PNG frames of videos already encoded (mp4s kept) |

## KEEP (served on the stage tonight)
- 01 h3dgs (chunk 3_1 hier_opt + scaffold) + h3dgs_native/chunk_3_1_sam3/{features_B_bg2share.bin, text_bank.npz} + prod embedder 01_13B_v1g
- 05 h3dgs_expo (chunk 1_0) + h3dgs_native/chunk_1_0_sam3/{features_B_bg2share.bin, text_bank.npz} + prod embedder 05_13D_v1g
- IMG_7993_s0/h3dgs_8m + IMG_7993_s0_cabbage/demo_root/experimental/h3dgs_native/chunk_lane_ud + demo_root/prod/bateleur/embedder/IMG_7993_s0_cabbage_v3g
- klapmuts dec lane2/h3dgs_e2, kendu IMG_7975_s0/h3dgs, the h3dgs_demo walk roots, every *_nocap.mp4 / cut mp4
