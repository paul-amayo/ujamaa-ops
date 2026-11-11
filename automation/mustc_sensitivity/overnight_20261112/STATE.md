HANDED OVER 2026-11-11 19:57:14 UTC

# JS-2 state (MuST-C sensitivity to rule changes 30-33; everything here is SENSITIVITY, NOT ADOPTED)

Refreshed 19:57 UTC 11 Nov 2026 by the Mac-side agent at the stop for handover. The Mac agent does nothing further.
Binding rules: the common rules file sits only on the Mac scratchpad (design before scores, never tune a rule after a
score, MuST-C records read-only, pathspec commits of own files only, never git inside /home/paperspace/code, British
spelling, "check" never "gate", "Section 4" never the symbol, no em dashes, hand marks never substituted).

## Design of record
`automation/mustc_sensitivity/DESIGN_20261112.md`, ops repo (github paul-amayo/ujamaa-ops, main), commit **ae6119f**,
committed and pushed 19:40 UTC 11 Nov 2026, before any number. Copy: `design/DESIGN_20261112.md` here. Frozen.

## Have sensitivity numbers been computed or looked at? YES, all of them
The whole table for both plots was computed after the design commit (19:45 and 19:49 UTC) and I have read every cell.
The stake crops exist and I have read 6 of the 14 pictures (listed below). Nothing remains blind.

## Done (complete, not partial)
- Code, committed f1c0844 (Mac `automation/mustc_sensitivity/`; copies in `code/` here and in each `bin/` on the box):
  `mustc_sens.py` (driver; imports byte-identical copies of mustc_ledger / mustc_sameday_floor / mustc_lmi_census from its
  own bin/, calls `compute()` only, never `main()`, which writes into the census directory; clauses behind
  --item30/31/32/33; `--all-columns` writes the table), `verify_sens.py` (independent recount, no MuST-C import),
  `stake_zoom.py` (ortho crops).
- Tables and recount, committed 72ed071 (same Mac directory; copies in `results/` here):
  `sensitivity_table_plot198|178.json|md`, `verify_sens_plot198|178.json`.
- Box, everything under `/mnt/shared2/must-c/plot198/census/sensitivity_20261112/` and
  `/mnt/shared2/must-c/plot178/census/sensitivity_20261112/`: `bin/`, `logs/`, `input_md5_before.txt`,
  `scripts_md5_before.txt`, `runs/{as_registered,item32_b_thresholds_rederived,item32_a_registered_thresholds,marks_withheld}/`
  `{census_truth,ledger,growth}.json`, `sensitivity_table_plot<p>.json|md`, `verify_sens_plot<p>.json`, `DONE_plot<p>`;
  plot 178 also `stakes/SB7_080723/`, `stakes/SB7_240623/` (unmodified `mustc_stakes_check.py`), `stakes/zoom/`.
- Checks passed: as-registered rerun against `ledger.json` of record, 0 differences on both plots (gates, pairs, polygons,
  tracks); item-32 line (a) against the stored `sensitivity.exclusion.plant_pattern`, 0 differences; independent recount
  201 of 201 (plot 198) and 103 of 103 (plot 178) comparisons agree.

## What the table shows (read it from the JSON; these are the headlines)
Plot 198. No failed test becomes a pass under any item or under all four.
- Item 30: splat specificity clause "met" (9 of 28, bound 20.1) becomes "no verdict" (same-day rate 0.72 = 56 of 78;
  false-removal rate 0.32, Wilson 0.18-0.51). T5 still fails on the removal clause (4 against 8). Laser ceiling
  unchanged: not met (15 of 43 against 5.7). Under the other reading of "the rate" (false-removal rate) nothing moves.
- Item 31: T7 still fails; 4 of 12 stops at or below the M2d per-plant floor (5.7 / 7.5 / 14.4 cm for 230601 / 230613 /
  230621: 0 of 4, 1 of 4, 3 of 4); as registered 0 of 12 against 2 g = 6.4 cm. My prediction was about 6.
- Item 32 (T2 and plant-pattern check): 7 of 18 stops enter (230601 x4, 230613_pos03, 230621_pos00-01); 230613 same-day
  floor does not exist; ledger falls back to r_assoc 9.5 cm and g 2.8 cm (laser values); T3 undecidable; T4 230613 falls
  from 0.905 (pass, 60 marks) to 0.667 (fail, 20 marks; 0.556 with registered thresholds); T6 undecidable (n 2 and n 0);
  T7 0 of 7; T5: the code says "fail" (0 against 8) but the one 230613 stop left images 0 of the 10 laser plants in the
  polygon, so it is not a test of the change rule (report as not evaluable for lack of coverage, give the code's word
  beside). The plan had said "undecidable" for T5 here.
- Item 33: nothing moves on plot 198 (12.1 cm at 0.15 m, n 18; 16.2 cm at 0.20 m). Laser 9.7 cm: not resolvable too.
Plot 178. Item 33 moves T3 from "radius-dependent" to "resolvable" (9.08 cm at 0.15 m, n 18; 10.4 cm at 0.20 m), the only
verdict any item moves in the favourable direction; note the 0.15 m value equals the 0.10 m value (no match added).
Item 30 leaves the clause standing: same-day rate exactly 0.50 (18 of 36), not above 0.5, so "met" (6 of 14, bound 7.0);
fragile. Item 31: no M2d on this plot; against plot 198's floor 1 of 9 (as registered 8 of 9 against 2 g = 16 cm because
g is capped at 8 cm: there item 31 is the stricter clause). Item 32: 1 of 11 stops enters, everything undecidable.
Marks: with the marks withheld only the two T4 cells change; T1, T2, T3, T5, T6, T7 are identical on both plots. The
found-again decomposition, added-by-class splits and census recall/precision (not table rows) also read marks.

## Partial
- Stake pictures (plot 178 `stakes/zoom/`), agent reading, 6 of 14 read: reference rod (230621 id 1 on the 230623 ortho):
  a pale pink rod in the circle and an orange one nearby. **230621 id 41 (SB7_080723):** on 230623 and 230614 the
  position is the leaf edge of a beet plant over shadowed soil, no rod; the laser height map shows a 19 cm2 fragment
  joined by a thin strip to the plant to the north-east: reading so far "plant" (a leaf fragment), so the stake rule
  wrongly flags it. **230525 id 4 (SB7_240623):** on 230526 a white card with a QR code (about 6 x 11 cm, a plot label)
  lies inside the 10 cm circle; on 230517 bare soil: reading so far "not vegetation" (a label, rod class), so the flag is
  right. NOT yet read: `zoom_230621_id41_..._ortho230708.jpg`, `zoom_230525_id4_..._ortho230602.jpg`,
  `..._id4_..._chm230525.png`, the two other reference-rod crops, the full-polygon PNGs of `mustc_stakes_check.py`.
- `input_md5_after.txt` not written.

## Not done; exact next steps, in order
1. Finish reading the stake pictures; state both readings as an agent's, not a person's.
2. For both plots run the md5 list again into `input_md5_after.txt` and diff against `_before` (must be identical):
   `cd /mnt/shared2/must-c/plot<p> && find census results -type f -not -path '*/sensitivity_20261112/*' -print0 | sort -z | xargs -0 md5sum`
3. Write `SENSITIVITY_20261112.md` (the table from the JSON; per item: one-sentence change, verdicts moved, principled or
   not by the design's criteria, the sceptical reviewer; marks cells; stake checks; hand-initialisation needs; deviations)
   and `SECTION_4_8_READING.md` (two or three honest framings, what each claims and concedes; no manuscript edits) into
   `automation/mustc_sensitivity/` of the ops repo and commit with a pathspec commit (not from /home/paperspace/code).
4. Hand initialisation of 230628_pos00-03: do NOT attempt. Report what Paul must give: per stop three numbers for
   `mustc_splat_geom.py run --stop <stop> --hand-init X_UTM Y_UTM YAW_DEG` (levelled frame origin in UTM and yaw, from
   hand-clicked correspondences between the stop's nadir render and the reference); I had not yet checked whether a
   clicking aid exists or how long it takes, so that estimate is still to be made from the script.

## Deviations from the design so far (to be listed in the results)
- T5 cell text gained the polygon-coverage count, and the item-30 overall text names the judged specificity clause, after
  the first run (wording only; no rule, number or verdict word changed).
- Stake crops: laser 230525 has no measured laser-to-ortho link on plot 178, so its crops show the raw position with a
  10 cm circle instead of a moved position with 5 cm; two extra ortho dates (230708, 230602) and one known rod were added.
- Plot 178, item 31: 230616 stops are read against the laser date their own record names (230613).

## Learned
- `ledger.json` of record was created 16:46 UTC 11 Nov on both plots (the plan says 15:57); numbers are the plan's.
- Item 30's threshold falls exactly on plot 178's rate; item 31 is looser than 2 g on plot 198 and tighter on plot 178.
- No job of mine is running on the box; nothing was stopped; no output is partial except as listed.

## A100 session (Ujamaa A100 agent), 11-12 Nov 2026

- 20:03 UTC: took over. No Mac job running (process list, logs, DONE_plot198 / DONE_plot178 at 19:49 UTC). Folder copies of
  code, design and results are byte-identical to the box copies. Committed as received: a4f5085 (20:03:14 UTC).
- 20:08 UTC: continuation plan `design/A100_CONTINUATION_20261111.md` committed 357e5e1 (adds no rule; design of record
  stays DESIGN_20261112.md, ops ae6119f).
- Stake pictures: all three zoom sheets, five per-ortho crops, two CHM crops and two full-polygon orthos read (12 images).
- Running: nothing. Next: md5 after-check, spotcheck_a100.py, figure, RESULTS.md, SECTION_4_8_READING.md, NOTEBOOK_ENTRY.md.
- 20:08 UTC: input md5 after = before on both plots (222 and 134 files; 36 scripts identical); lists in
  `results/verification/` and beside the `_before` files on the box.
- 20:10 UTC: `spotcheck_a100.py` (records of record only): 44 of 44 (plot 198) and 37 of 37 (plot 178) checks agree.
  Post hoc (not a rule): plot 178 230616_pos00's T7 p95 18.6 cm includes 10 plants harvested on 15 June (laser 230613
  predates the harvest); 7.0 cm without them.
- 20:13-20:18 UTC: figure `results/fig_js2_sensitivity.png` (items 31, 33, 30; read by eye after three layout passes);
  compact tables `results/table_compact_plot198|178.md` generated from the JSON; new code copied to `code/`.
- Running: nothing. Next: RESULTS.md, SECTION_4_8_READING.md, NOTEBOOK_ENTRY.md, commit.
- 20:24:50 UTC: results and code committed, d47844b: `results/RESULTS.md` (= the brief's SENSITIVITY_20261112.md),
  `results/SECTION_4_8_READING.md`, `results/NOTEBOOK_ENTRY.md`, `results/fig_js2_sensitivity.png`, compact tables,
  `results/verification/`, `results/stakes/`, `code/spotcheck_a100.py`, `code/compact_table.py`, `code/fig_js2_sensitivity.py`.
- 20:25 UTC: records (222 + 134 files) and the 36 scripts re-checked identical to `_before` after every A100 script ran.
- **DONE.** Nothing running; nothing partial. Stake readings complete (id 41 = plant, rule misfires; id 4 = plot label,
  flag right). Open for Paul only: decisions on items 30-33, the check for hand-initialised stops (`registration_ok`
  requires hand_init None), plot 178 T7 across the 15 June harvest (post hoc), the stake rule's citation, the hand marks.
