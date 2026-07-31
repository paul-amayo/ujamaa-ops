# UJAMAA research workspace

This directory holds the UJAMAA digital-twin research programme (gaussian-splat orchard
reconstruction + 4D tree monitoring). Paul Amayo, UCT. Goal and milestones live in
**UJAMAA_ROADMAP.md** — read it when planning multi-day work.

## Conventions (post-doc rigor)
- **Log every run** (training, pipeline, association — including failures) as an entry in
  `lab_notebook/<YYYY-MM>.md` using the template in `lab_notebook/README.md`.
- **Commit after green:** when a run/experiment works, prompt Paul to commit that day and
  tag known-good recipes. Uncommitted work caused the June 2026 regression post-mortem.
- **Conclusions → memory, records → notebook.** Durable findings get a memory file;
  raw run records stay in the notebook.
- Metrics via the psnr skill (tensorboard events), not visual inspection, where possible.

## Environment facts
- ns-train / ns-process-data: pixi shell in `nerf_new/` — NOT system python.
- Rosbag ingest (`zed_bag_to_monolithics`): needs SYSTEM python3 with numpy+scipy.
- Splat viewer serves on port 8001 (SSH tunnel); query server on 8002.
- Key repos: `high/` (HiGH method), `aru_sil_core/` (viewer + pipeline core),
  `nerf_new/` (nerfstudio env), `ujamaa/` (design handoff bundle — the demo's product spec).
- Surveys live under `/home/paperspace/data/citrus_all/<id>/`.
