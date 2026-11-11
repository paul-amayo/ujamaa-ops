## 2026-11-12 · JS-2: MuST-C verdicts under the proposed rule changes 30-33 (sensitivity, not adopted)
- **Goal:** for each of items 30-33 of `plans/mustc_jirs.md` ("Revisions 11 Nov 2026") and for all four together, which
  MuST-C verdicts T1-T7 move and by how much, so that Paul can decide them with numbers; which cells wait for his hand marks;
  the two outstanding stake checks; what the hand initialisation of 230628_pos00-03 needs; a reading of how Section 4.8 can
  be framed. Overnight test, started on the Mac, finished on the A100.
- **Command/config:** design of record `DESIGN_20261112.md` (ops ae6119f, 19:40 UTC, before any number). Box:
  `/mnt/shared2/must-c/venv/bin/python mustc_sens.py --plot 198|178 --out-dir .../census/sensitivity_20261112 --all-columns`
  (imports byte-identical copies of `mustc_ledger.py`, `mustc_sameday_floor.py`, `mustc_lmi_census.py`; calls `compute()`
  only; clauses behind `--item30..33`), `verify_sens.py` (independent recount), `mustc_stakes_check.py` (unmodified) and
  `stake_zoom.py`; A100 additions `spotcheck_a100.py`, `compact_table.py`, `fig_js2_sensitivity.py`. No GPU.
- **Data:** finished records of plots 198 and 178 under `/mnt/shared2/must-c/plot{198,178}/census/` (read-only; md5 of all
  222 and 134 record files and of the 36 scripts identical before and after). Outputs in the two `sensitivity_20261112/`
  directories and in `/mnt/shared/overnight_20261112/JS2_mustc_sensitivity/results/`.
- **Result:** plot 198: no failed test becomes a pass under any item or all four. Item 30: splat T5 specificity "met"
  (9 of 28, bound 20.1 at same-day rate 0.72 = 56/78) becomes "no verdict" (false-removal 0.32, Wilson 0.18-0.51); T5 stays
  fail (4 against 8). Item 31: T7 stays fail, 0/12 within 2 g = 6.4 cm becomes 4/12 within the M2d floor (two by <= 2 mm).
  Item 32: 7/18 stops enter; T3 and T6 undecidable, T4 230613 0.905 -> 0.667 (20 marks), T5 removal not a test (no 230613
  coverage of the polygon). Item 33: nothing moves (12.1 cm at 0.15 m, n 18). Plot 178: item 33 moves T3 "radius-dependent"
  -> "resolvable" (9.08 cm at 0.15 m, n 18; 10.4 at 0.20 m), the only favourable move; item 31 is stricter there (1/9 vs
  8/9); item 30 leaves the clause standing (rate exactly 0.50); item 32 leaves 1/11 stops. Marks: only the two T4 rows of
  plot 198 change without marks; every other cell is final with respect to them. Stakes (agent readings): laser 230621 id 41
  (SB7_080723) is a beet leaf, the stake rule misfires; laser 230525 id 4 (SB7_240623) is a plot label on a stake, the flag is
  right. Hand initialisation: three numbers per stop (`--hand-init X_UTM Y_UTM YAW_DEG`), but `registration_ok` excludes any
  hand-initialised stop by construction, so a rule for checking such stops is needed first; it would move only T2. Post hoc,
  not a rule: plot 178's one T7 failure (230616_pos00, 18.6 cm) compares 10 plants harvested on 15 June; 7.0 cm without
  them. Verification: as-registered rerun 0 differences; recount 201/201 and 103/103; A100 transcription check 44/44 and
  37/37. Files: `results/RESULTS.md`, `results/fig_js2_sensitivity.png`, `results/SECTION_4_8_READING.md`.
- **Verdict:** keep as evidence for Paul's decision on items 30-33; nothing adopted. The items change the wording of the
  MuST-C story, not its outcome on the primary plot. Follow-up for Paul: decide items 30-33; the check for hand-initialised
  stops; whether plot 178's T7 should exclude the harvested polygon on 230616; cite the stake rule for SB7_150623 only; the
  hand marks.
