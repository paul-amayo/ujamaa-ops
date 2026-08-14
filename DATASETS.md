# Dataset registry

Generated 2026-08-14 20:32 UTC by `automation/build_dataset_registry.py` — do not edit by hand;
re-run the generator after ingest/pipeline milestones and commit the diff.
Ledger source: `ledger_v2.json`.

| survey | size | stage | bag | monos | lio mono | kf mono | kf PNGs | markers | hierarchy | masks | blocks | splats | ledger obs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01_13B_Jackal | 189G | **splats** | — | img+lidar+gps | y | y | 6031 | y | 289 obj / 38 rows (scene_graph) | y | lio_arc_size15.0_ov0.10_kf20cm:102, lio_row:35 | 35 | 289 |
| 02_13B_Jackal | 100G | **registry** | y | img+lidar+gps | y | y | 2041 | y | 177 obj / 9 rows (scene_graph) | — | — | — | 177 |
| 03_13B_Jackal | 447G | **splats** | — | img+lidar+gps | y | y | 8102 | y | 272 obj / 35 rows (scene_graph) | y | lidar_pass1_100cm:2, lio_arc_size15.0_ov0.10_kf20cm_dedup:66, lio_row_halves:80 | 3 | 303 |
| 04_13D_Jackal | 158G | **blocks** | — | img+lidar+gps | y | y | 3543 | y | 103 obj / 11 rows (scene_graph_v4) | — | lio_row6F:11 | — | 85 |
| 05_13D_Jackal | 126G | **blocks** | — | img+lidar+gps | y | y | 3386 | y | 81 obj / 10 rows (scene_graph) | y | lio_arc_size15.0_ov0.10_kf20cm:48, lio_row:19, lio_row100:44 | — | — |
| klapmuts apr_2026 (ZED) | 49G | **blocks** | — | img+lidar | y | y | 2019 | y | 721 obj / 14 rows (scene_graph) | — | lio_row:16, lio_row100_preview:25 | — | — |
| klapmuts dec_2025 pilot (A300 mcap) | 12G | **registry** | y | img+lidar | y | y | 193 | y | SAM3 ledger 290 frames | — | — | — | — |
| klapmuts dec_2025 ten-rows (A300 mcap) | 89G | **ingested** | y | img+lidar+gnss | y | — | — | — | — | — | — | — | — |

Stage ladder: raw → combined → ingested (monolithics+LIO) → registry
(markers+hierarchy) → blocks → splats. `ledger obs` = observations this
survey contributes to the sankofa ledger.
