# Dataset registry

Generated 2026-08-28 20:06 UTC by `automation/build_dataset_registry.py` — do not edit by hand;
re-run the generator after ingest/pipeline milestones and commit the diff.
Ledger source: `ledger_v2.json`.

| survey | size | stage | bag | monos | lio mono | kf mono | kf PNGs | markers | hierarchy | masks | blocks | splats | ledger obs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01_13B_Jackal | 155G | **raw** | — | — | — | — | — | — | — | — | — | — | 289 |
| 02_13B_Jackal | 130G | **raw** | — | — | — | — | — | — | — | — | — | — | 177 |
| 03_13B_Jackal | 172G | **raw** | — | — | — | — | — | — | — | — | — | — | 303 |
| 04_13D_Jackal | 106G | **raw** | — | — | — | — | — | — | — | — | — | — | 85 |
| 05_13D_Jackal | 129G | **raw** | — | — | — | — | — | — | — | — | — | — | — |
| klapmuts apr_2026 (ZED) | 44G | **raw** | — | — | — | — | — | — | — | — | — | — | — |

Stage ladder: raw → combined → ingested (monolithics+LIO) → registry
(markers+hierarchy) → blocks → splats. `ledger obs` = observations this
survey contributes to the sankofa ledger.
