# JS-2 continuation on the A100 (11 Nov 2026, from 20:03 UTC): plan, adds no rule

The design of record stays `DESIGN_20261112.md` (ops ae6119f, 19:40 UTC 11 Nov, frozen; copy committed here in a4f5085).
This note fixes how the A100 agent finishes the test before it runs anything further. It changes no rule, threshold,
column or reading of the design. Everything the test produces is **sensitivity, not adopted**.

## What I had seen before writing this

The handover files (BRIEF, STATE, design, code, both sensitivity tables and both recount files, all byte-identical to the
copies in the two `sensitivity_20261112/` directories on the box), the plan `refs/ops/mustc_jirs.md` in full, the three
MuST-C memory notes, the code paths of `mustc_ledger.py` that read marks (`load_marks`, one reader) and of
`mustc_splat_geom.py` that take `--hand-init`, the 230628 registration records, and twelve stake pictures (the three zoom
sheets, five per-ortho crops, two CHM crops, two full-polygon orthos). The Mac agent had computed and read every cell of
both tables (STATE.md); I have read them too. No job of the Mac agent is running (checked by process list, its logs and
the DONE files `DONE_plot198`, `DONE_plot178`, both 19:49 UTC).

## Remaining steps, in order

1. **Stake readings** by the design's rule (rod / plant / undetermined, 5 cm of the moved position on the ortho nearest in
   time), for laser 230621 id 41 (SB7_080723) and laser 230525 id 4 (SB7_240623), with the reference rod as the control.
   Where the rule cannot be applied as written (230525 has no laser-to-ortho link on plot 178), the reading is given under
   the Mac agent's stated deviation (raw position, 10 cm circle) and labelled so. Readings are an agent's, not a person's.
2. **Input md5 after**: the same command as `input_md5_before.txt` (`find census results -type f -not -path
   '*/sensitivity_20261112/*'` from each plot directory) into `input_md5_after.txt`, and the scripts' md5 into
   `scripts_md5_after.txt`; both diffed against `_before`. Any difference is reported, not reconciled.
3. **Transcription check (added verification, not in the design):** a new script `spotcheck_a100.py` that imports neither
   the MuST-C modules nor `mustc_sens.py` / `verify_sens.py` and reads only the records of record (`ledger.json`,
   `census_truth.json`, `growth.json`, `noise_floor.json`, `cross_sensor_floor.json`, `dsm_vs_laser.json`,
   `registration.json`, `lmi_assoc.json`) and the two table JSONs. It checks that the as-registered column reports the
   record's numbers, and recomputes the item-30 intervals (Wilson by its own formula), the item-31 counts (per-stop p95 from
   `dsm_vs_laser.json` against `cross_sensor_floor.json`), the item-33 verdicts (`noise_floor.json` p95 at 0.15 m against
   0.095 m) and the item-32 stop sets (`registration.json`). Differences are reported as found.
4. **Hand initialisation of 230628_pos00-03**: not attempted. From the code, state exactly what Paul must supply (file,
   format, values), what exists to help, what else the current code needs before such a stop can enter (the check of
   record), which table cells it could change, and the machine time from the records. Any human-time figure is labelled an
   estimate.
5. **Figure (one, three panels), drawn from the table JSONs only:** (a) item 31: per-stop per-plant |dh95| p95 against
   2 g and the M2d floors, both plots; (b) item 33: 230613 same-day position p95 against match radius, both plots, with the
   9.5 cm cap and the laser's pooled p95; (c) item 30: false-removal rate with its 95 % Wilson interval beside the same-day
   rate, splat and laser, both plots, with the 0.5 line. Every panel labelled "sensitivity, not adopted".
6. **Write** `results/RESULTS.md` (this is the brief's `SENSITIVITY_20261112.md`: verdict of the test first, the table,
   per item the change, what it moves and by how much, the classification by the design's criteria, the sceptical
   reviewer; the marks cells; stake checks; hand-initialisation needs; deviations; what was and was not verified),
   `results/SECTION_4_8_READING.md` (two or three honest framings, no manuscript edits) and `results/NOTEBOOK_ENTRY.md`;
   copy new code to `code/`; commit results and code by pathspec under the lock; keep STATE.md's A100 section current.

No GPU is used. Writes go only to the two `sensitivity_20261112/` directories and this test folder.
