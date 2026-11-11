# JS-2 results: the MuST-C verdicts under the four proposed rule changes (sensitivity, not adopted)

Overnight test JS-2 for the JIRS paper (Section 4.8, the MuST-C public field trial), 11 to 12 Nov 2026. This file is the
brief's `SENSITIVITY_20261112.md` under the A100 name. **Everything below is SENSITIVITY, NOT ADOPTED.** The verdicts of
record are the as-registered column; items 30 to 33 of `plans/mustc_jirs.md` ("Revisions 11 Nov 2026") are Paul's decision
and none is applied to any record, script default or design.

- Design of record: `design/DESIGN_20261112.md`, ops repo commit ae6119f, 19:40 UTC 11 Nov, before any number (Mac agent);
  copy committed here as received in a4f5085 (20:03:14 UTC). A100 continuation plan, adding no rule:
  `design/A100_CONTINUATION_20261111.md`, 357e5e1 (20:08:02 UTC).
- Computation: Mac agent, 19:45 to 19:49 UTC (code ops f1c0844, tables ops 72ed071; copies in `code/` and here). The A100
  agent finished the stake readings, the md5 check, an added transcription check, the figure and this text.
- Box working directories: `/mnt/shared2/must-c/plot198/census/sensitivity_20261112/` and
  `/mnt/shared2/must-c/plot178/census/sensitivity_20261112/` (the second is `/mnt/shared/must-c_bulk/plot178/...`).

## 1. Verdict of this test (the design's rule: an item "moves" a verdict when the cell's verdict word differs from the as-registered cell, otherwise it "moves a number only")

**Plot 198 (the primary site): no failed test becomes a pass under any item, nor under all four together, under either
resolution of item 32.**
- Item 30 moves one verdict: the splat's T5 specificity clause, "met" (9 of 28 unchanged on-row plants called removed,
  bound 20.1 at the same-day rate 0.72 = 56 of 78) becomes "no verdict". It was the only clause the splat met. T5 stays
  "fail" on its removal clause (the splat calls 4 removed, the bar is the laser's 8 of 10).
- Item 31 moves a number only: T7's plant clause stays "fail", 0 of 12 stops within 2 g = 6.4 cm becomes 4 of 12 stops
  within the M2d floor (230601 0 of 4, 230613 1 of 4, 230621 3 of 4).
- Item 32 (T2 check and plant-pattern check) lets 7 of 18 stops enter instead of 14. The 230613 same-day floor cannot be
  measured (one stop that day), so T3 association and T3 g become "undecidable", T6 becomes "undecidable" on both intervals
  (n 2 and n 0), T4 on 230613 falls from "pass" (0.905, 60 marks) to "fail" (0.667, 20 marks), the splat's specificity
  clause becomes "not evaluable". T5's removal clause reads "not met" in the code (0 against 8), but the one 230613 stop that
  enters images 0 of the 10 laser plants inside SB8_020623, so it is not a test of the change rule.
- Item 33 moves nothing on plot 198 (12.1 cm at the 0.15 m radius, n 18, against the 9.5 cm cap).

**Plot 178 (the second site):**
- Item 33 moves T3 association from "radius-dependent" to "resolvable" (9.08 cm at 0.15 m, n 18; 10.4 cm at 0.20 m, n 19):
  the only verdict any item moves in the favourable direction, by 4 mm under the cap.
- Item 31 moves T7's word from "fail" (8 of 9 stops within 2 g = 16.0 cm, g capped at 8 cm) to "not evaluable on its own
  floor; against plot 198's M2d: fail" (1 of 9). Here item 31 is the stricter clause.
- Item 30 moves nothing: the splat's same-day rate is exactly 0.50 (18 of 36), not above 0.5, so the clause stands as "met"
  (6 of 14 against a bound of 7.0).
- Item 32 lets 1 of 11 stops enter; every downstream verdict becomes "undecidable" except T7 (0 of 1 stop within 2 g =
  6.7 cm, fail).

The design's prediction (the Mac agent's, written before any number) was right on items 30, 32 and 33 and on "no failed
test becomes a pass"; for item 31 it predicted about 6 of 12 stops within the floor and the count is 4 of 12. For item 32
it predicted T5 "undecidable"; the code says "not met" with zero coverage, which this report treats as not a test.

## 2. The table

Each cell gives the verdict and the number it rests on with its denominator, as the table JSON holds it. "=" means the
verdict and the number are identical to the as-registered cell. The full-text tables (every cell written out) are
`sensitivity_table_plot198.md` and `sensitivity_table_plot178.md`; the JSON holds every number behind them. T4 rests on the
PROVISIONAL premarks (no `marks_<date>.csv` exists). Plot 178 has no marks, hence no T4 row, and no M2d of its own.

### Plot 198 (primary site)

Plot 198. SENSITIVITY, NOT ADOPTED; the verdicts of record are the as-registered column. "=" = verdict and number identical to the as-registered cell.

| test / clause | as registered | item 30 | item 31 | item 32 | item 33 | all four (a) | all four (b) |
|---|---|---|---|---|---|---|---|
| T1 reprojection <= 0.70 px | **pass**. 18 of 18 stops, 0.43-0.58 px | = | = | = | = | = | = |
| T2 check (soil residual, yaw) | **14 of 18**. 14 of 18 stops pass the T2 check; 7 of 18 also pass the plant-pattern check | = | = | = | = | = | = |
| T2 stops entering the ledger | **14 of 18**. 14 of 18 under 'registration_ok'; per day 230601 4, 230613 4, 230621 4, 230628 2 | = | = | **7 of 18**. 7 of 18 under 'plant_pattern'; per day 230601 4, 230613 1, 230621 2, 230628 0 | = | = | **7 of 18**. 7 of 18 under 'plant_pattern'; per day 230601 4, 230613 1, 230621 2, 230628 0 |
| T2 scale flag /s-1/ > 1 % | **11 of 18 flagged**. days with every stop flagged: 230621, 230628 | = | = | = | = | = | = |
| T3 association resolvable at the 9.5 cm cap | **not resolvable**. 230613 same-day position p95 8.7 cm at 0.10 m (n 12), 12.1 cm at 0.15 m (n 18), 16.2 cm at 0.20 m (n 23); cap 9.5 cm; read at 0.15 and 0.20 m together | = | = | **undecidable**. 230613: 0 matched same-day plants (fewer than two stops enter that day); read at 0.15 and 0.20 m together | **not resolvable**. 230613 same-day position p95 8.7 cm at 0.10 m (n 12), 12.1 cm at 0.15 m (n 18), 16.2 cm at 0.20 m (n 23); cap 9.5 cm; read at 0.15 m alone | **not resolvable**. 230613 same-day position p95 8.7 cm at 0.10 m (n 12), 12.1 cm at 0.15 m (n 18), 16.2 cm at 0.20 m (n 23); cap 9.5 cm; read at 0.15 m alone | **undecidable**. 230613: 0 matched same-day plants (fewer than two stops enter that day); read at 0.15 m alone |
| T3 laser ceiling, same clause (beside) | **not resolvable**. laser two-scanner same-day position p95 9.7 cm at the 0.15 m cap (n 145, pooled 230601/13/21) | = | = | = | = | = | = |
| T3 growth threshold g <= 8 cm | **pass**. 230613 same-day /dh95/ p95 3.2 cm (n 12); g in force 3.2 cm | = | = | **undecidable**. 230613 same-day floor not measurable under this stop set; g in force 2.8 cm (laser same-day /dh95/ p95 (pooled cap 0.15), capped at 0.08) | = | = | **undecidable**. 230613 same-day floor not measurable under this stop set; g in force 2.8 cm (laser same-day /dh95/ p95 (pooled cap 0.15), capped at 0.08) |
| T3 flag on growth from 230601 | **flagged**. 230601 own same-day /dh95/ p95 10.3 cm (n 5) against 8 cm | = | = | = | = | = | = |
| T4 230601 splat recall >= 0.9 x laser recall | **fail**. ratio 0.678: splat 40 of 63 marks, laser 59 of 63; provisional marks | = | = | **fail**. ratio 0.667: splat 40 of 63 marks, laser 60 of 63; provisional marks | = | = | **fail**. ratio 0.667: splat 40 of 63 marks, laser 60 of 63; provisional marks |
| T4 230613 splat recall >= 0.9 x laser recall | **pass**. ratio 0.905: splat 38 of 60 marks, laser 42 of 60; provisional marks | = | = | **fail**. ratio 0.667: splat 12 of 20 marks, laser 18 of 20; provisional marks | = | = | **fail**. ratio 0.667: splat 12 of 20 marks, laser 18 of 20; provisional marks |
| T5 SB8_020623 removal clause (restated 4 Nov) | **not met**. splat calls 4 removed of the 5 plants both sensors record inside; bar = the laser's 8 of 10; truth 10 removed; the later day's stops entering image 9 of the 10 laser plants inside | = | = | **not met**. splat calls 0 removed of the 5 plants both sensors record inside; bar = the laser's 8 of 10; truth 10 removed; the later day's stops entering image 0 of the 10 laser plants inside (NO COVERAGE of the polygon: not a test of the change rule) | = | = | **not met**. splat calls 0 removed of the 5 plants both sensors record inside; bar = the laser's 8 of 10; truth 10 removed; the later day's stops entering image 0 of the 10 laser plants inside (NO COVERAGE of the polygon: not a test of the change rule) |
| T5 SB8_020623 removal clause, pre-registered >= 9 (beside) | **not met**. 4 against 9 | = | = | **not met**. 0 against 9 | = | = | **not met**. 0 against 9 |
| T5 SB8_020623 specificity clause, splat | **met**. 9 of 28 unchanged on-row plants called removed; bound 20.1 at the same-day rate 0.72 (56 of 78) | **no verdict**. 9 of 28 unchanged on-row plants called removed; bound 20.1 at the same-day rate 0.72 (56 of 78); false-removal rate 0.32 (95 % interval 0.18-0.51), same-day rate interval 0.61-0.81 | = | **not evaluable**. 7 of 14 unchanged on-row plants called removed; same-day rate not measurable | = | **no verdict**. 9 of 28 unchanged on-row plants called removed; bound 20.1 at the same-day rate 0.72 (56 of 78); false-removal rate 0.32 (95 % interval 0.18-0.51), same-day rate interval 0.61-0.81 | **not evaluable**. 7 of 14 unchanged on-row plants called removed; same-day rate not measurable; false-removal rate 0.50 (95 % interval 0.27-0.73) |
| T5 SB8_020623 specificity, laser ceiling (beside) | **not met**. 15 of 43 unchanged on-row plants called removed; bound 5.7 at the same-day rate 0.13 (10 of 76) | **not met**. 15 of 43 unchanged on-row plants called removed; bound 5.7 at the same-day rate 0.13 (10 of 76); false-removal rate 0.35 (95 % interval 0.22-0.50), same-day rate interval 0.07-0.23 | = | **not met**. 13 of 43 unchanged on-row plants called removed; bound 5.7 at the same-day rate 0.13 (10 of 76) | = | **not met**. 15 of 43 unchanged on-row plants called removed; bound 5.7 at the same-day rate 0.13 (10 of 76); false-removal rate 0.35 (95 % interval 0.22-0.50), same-day rate interval 0.07-0.23 | **not met**. 13 of 43 unchanged on-row plants called removed; bound 5.7 at the same-day rate 0.13 (10 of 76); false-removal rate 0.30 (95 % interval 0.19-0.45), same-day rate interval 0.07-0.23 |
| T5 SB8_020623 overall | **fail**. removal clause not met (4 against 8), specificity clause met; detection-limited (fewer plants recorded by both than the bar) | **fail**. removal clause not met (4 against 8); specificity clause not judged (no verdict); detection-limited (fewer plants recorded by both than the bar) | = | **fail**. removal clause not met (0 against 8), specificity clause not evaluable; detection-limited (fewer plants recorded by both than the bar) | = | **fail**. removal clause not met (4 against 8); specificity clause not judged (no verdict); detection-limited (fewer plants recorded by both than the bar) | **fail**. removal clause not met (0 against 8); specificity clause not judged (not evaluable); detection-limited (fewer plants recorded by both than the bar) |
| T6 230601->230613 | **pass**. n 12 plants paired in both sensors; median /dh splat - dh laser/ 1.3 cm against g 3.2 cm; calls agree 12 of 12 (need 0.9); flagged by the T3 clause on 230601 | = | = | **undecidable**. n 2 plants paired in both sensors; median /dh splat - dh laser/ 2.4 cm against g 2.8 cm; calls agree 2 of 2 (need 0.9); flagged by the T3 clause on 230601 | = | = | **undecidable**. n 2 plants paired in both sensors; median /dh splat - dh laser/ 2.4 cm against g 2.8 cm; calls agree 2 of 2 (need 0.9); flagged by the T3 clause on 230601 |
| T6 230613->230621 | **fail**. n 9 plants paired in both sensors; median /dh splat - dh laser/ 3.0 cm (0.4 cm with the M2e bound subtracted) against g 3.2 cm; calls agree 7 of 9 (need 0.9) | = | = | **undecidable**. undecidable (fewer than 5 plants paired in both sensors) | = | = | **undecidable**. undecidable (fewer than 5 plants paired in both sensors) |
| T6 230621->230628 | **not evaluable**. no laser within 13 days of the 230628 stops (design: against UAV2 only) | = | = | = | = | = | = |
| T6 long span 230601->230621 (beside) | **pass**. n 10; median 4.7 cm (2.2 cm with the bound); calls agree 9 of 10 | = | = | **undecidable**. n 2; median 4.4 cm (1.9 cm with the bound); calls agree 1 of 2 | = | = | **undecidable**. n 2; median 4.4 cm (1.9 cm with the bound); calls agree 1 of 2 |
| T7 soil clause (after the M2d offset is removed; by construction) | **12 of 12**. soil bias within the M2d soil floor on 12 of 12 stops | = | = | **7 of 7**. soil bias within the M2d soil floor on 7 of 7 stops | = | = | **7 of 7**. soil bias within the M2d soil floor on 7 of 7 stops |
| T7 plant clause | **fail**. 0 of 12 stops with p95 <= 2 g = 6.4 cm; per-plant /dh95/ p95 7.4-14.9 cm; median difference -8.6 to -2.8 cm; 2 entering stop(s) without a laser date not evaluable | = | **fail**. 4 of 12 stops at or below the M2d per-plant floor (230601: 5.7 cm, 230613: 7.5 cm, 230621: 14.4 cm); per-plant /dh95/ p95 7.4-14.9 cm; median difference -8.6 to -2.8 cm; 2 entering stop(s) not evaluable | **fail**. 0 of 7 stops with p95 <= 2 g = 5.6 cm; per-plant /dh95/ p95 8.7-10.9 cm; median difference -7.7 to -2.8 cm | = | **fail**. 4 of 12 stops at or below the M2d per-plant floor (230601: 5.7 cm, 230613: 7.5 cm, 230621: 14.4 cm); per-plant /dh95/ p95 7.4-14.9 cm; median difference -8.6 to -2.8 cm; 2 entering stop(s) not evaluable | **fail**. 2 of 7 stops at or below the M2d per-plant floor (230601: 5.7 cm, 230613: 7.5 cm, 230621: 14.4 cm); per-plant /dh95/ p95 8.7-10.9 cm; median difference -7.7 to -2.8 cm |

Settings: as registered: 14 stops, r_assoc 8.75 cm, g 3.22 cm; item 30: 14 stops, r_assoc 8.75 cm, g 3.22 cm; item 31: 14 stops, r_assoc 8.75 cm, g 3.22 cm; item 32: 7 stops, r_assoc 9.50 cm, g 2.81 cm; item 33: 14 stops, r_assoc 8.75 cm, g 3.22 cm; all four (a): 14 stops, r_assoc 8.75 cm, g 3.22 cm; all four (b): 7 stops, r_assoc 9.50 cm, g 2.81 cm.

### Plot 178 (second site)

Plot 178. SENSITIVITY, NOT ADOPTED; the verdicts of record are the as-registered column. "=" = verdict and number identical to the as-registered cell.

| test / clause | as registered | item 30 | item 31 | item 32 | item 33 | all four (a) | all four (b) |
|---|---|---|---|---|---|---|---|
| T1 reprojection <= 0.70 px | **pass**. 11 of 11 stops, 0.46-0.55 px | = | = | = | = | = | = |
| T2 check (soil residual, yaw) | **9 of 11**. 9 of 11 stops pass the T2 check; 1 of 11 also pass the plant-pattern check | = | = | = | = | = | = |
| T2 stops entering the ledger | **9 of 11**. 9 of 11 under 'registration_ok'; per day 230601 3, 230613 3, 230616 3 | = | = | **1 of 11**. 1 of 11 under 'plant_pattern'; per day 230601 1, 230613 0, 230616 0 | = | = | **1 of 11**. 1 of 11 under 'plant_pattern'; per day 230601 1, 230613 0, 230616 0 |
| T2 scale flag /s-1/ > 1 % | **1 of 11 flagged**. days with every stop flagged: none | = | = | = | = | = | = |
| T3 association resolvable at the 9.5 cm cap | **radius-dependent**. 230613 same-day position p95 9.1 cm at 0.10 m (n 18), 9.1 cm at 0.15 m (n 18), 10.4 cm at 0.20 m (n 19); cap 9.5 cm; read at 0.15 and 0.20 m together | = | = | **undecidable**. 230613: 0 matched same-day plants (fewer than two stops enter that day); read at 0.15 and 0.20 m together | **resolvable**. 230613 same-day position p95 9.1 cm at 0.10 m (n 18), 9.1 cm at 0.15 m (n 18), 10.4 cm at 0.20 m (n 19); cap 9.5 cm; read at 0.15 m alone | **resolvable**. 230613 same-day position p95 9.1 cm at 0.10 m (n 18), 9.1 cm at 0.15 m (n 18), 10.4 cm at 0.20 m (n 19); cap 9.5 cm; read at 0.15 m alone | **undecidable**. 230613: 0 matched same-day plants (fewer than two stops enter that day); read at 0.15 m alone |
| T3 laser ceiling, same clause (beside) | **resolvable**. laser two-scanner same-day position p95 7.6 cm at the 0.15 m cap (n 151, pooled 230601/13/21) | = | = | = | = | = | = |
| T3 growth threshold g <= 8 cm | **fail**. 230613 same-day /dh95/ p95 11.9 cm (n 18); g in force 8.0 cm | = | = | **undecidable**. 230613 same-day floor not measurable under this stop set; g in force 3.4 cm (laser same-day /dh95/ p95 (pooled cap 0.15), capped at 0.08) | = | = | **undecidable**. 230613 same-day floor not measurable under this stop set; g in force 3.4 cm (laser same-day /dh95/ p95 (pooled cap 0.15), capped at 0.08) |
| T3 flag on growth from 230601 | **undecidable**. 230601 own same-day /dh95/ p95 3.7 cm (n 4) against 8 cm | = | = | **undecidable**. 230601 own same-day /dh95/ p95 n/a (n 0) against 8 cm | = | = | **undecidable**. 230601 own same-day /dh95/ p95 n/a (n 0) against 8 cm |
| T5 SB7_150623 removal clause (restated 4 Nov) | **not met**. splat calls 9 removed of the 9 plants both sensors record inside; bar = the laser's 14 of 16; truth 15 removed; the later day's stops entering image 9 of the 16 laser plants inside | = | = | **undecidable**. undecidable (230613 or 230616 splat census missing under the 'plant_pattern' exclusion) | = | = | **undecidable**. undecidable (230613 or 230616 splat census missing under the 'plant_pattern' exclusion) |
| T5 SB7_150623 removal clause, pre-registered >= 9 (beside) | **met**. 9 against 9 | = | = | **undecidable**. undecidable (230613 or 230616 splat census missing under the 'plant_pattern' exclusion) | = | = | **undecidable**. undecidable (230613 or 230616 splat census missing under the 'plant_pattern' exclusion) |
| T5 SB7_150623 specificity clause, splat | **met**. 6 of 14 unchanged on-row plants called removed; bound 7.0 at the same-day rate 0.50 (18 of 36) | **met**. 6 of 14 unchanged on-row plants called removed; bound 7.0 at the same-day rate 0.50 (18 of 36); false-removal rate 0.43 (95 % interval 0.21-0.67), same-day rate interval 0.34-0.66 | = | **undecidable**. undecidable (230613 or 230616 splat census missing under the 'plant_pattern' exclusion) | = | **met**. 6 of 14 unchanged on-row plants called removed; bound 7.0 at the same-day rate 0.50 (18 of 36); false-removal rate 0.43 (95 % interval 0.21-0.67), same-day rate interval 0.34-0.66 | **undecidable**. undecidable (230613 or 230616 splat census missing under the 'plant_pattern' exclusion) |
| T5 SB7_150623 specificity, laser ceiling (beside) | **not met**. 11 of 26 unchanged on-row plants called removed; bound 5.5 at the same-day rate 0.21 (17 of 81) | **not met**. 11 of 26 unchanged on-row plants called removed; bound 5.5 at the same-day rate 0.21 (17 of 81); false-removal rate 0.42 (95 % interval 0.26-0.61), same-day rate interval 0.14-0.31 | = | **undecidable**. undecidable (230613 or 230616 splat census missing under the 'plant_pattern' exclusion) | = | **not met**. 11 of 26 unchanged on-row plants called removed; bound 5.5 at the same-day rate 0.21 (17 of 81); false-removal rate 0.42 (95 % interval 0.26-0.61), same-day rate interval 0.14-0.31 | **undecidable**. undecidable (230613 or 230616 splat census missing under the 'plant_pattern' exclusion) |
| T5 SB7_150623 overall | **fail**. removal clause not met (9 against 14), specificity clause met; detection-limited (fewer plants recorded by both than the bar) | = | = | **undecidable**. undecidable (230613 or 230616 splat census missing under the 'plant_pattern' exclusion) | = | = | **undecidable**. undecidable (230613 or 230616 splat census missing under the 'plant_pattern' exclusion) |
| T6 230601->230613 | **fail**. n 12 plants paired in both sensors; median /dh splat - dh laser/ 1.1 cm against g 8.0 cm; calls agree 7 of 12 (need 0.9); not a T6 interval of the design (second site, g capped) | = | = | **undecidable**. undecidable (splat or laser census missing at one date under the 'plant_pattern' exclusion) | = | = | **undecidable**. undecidable (splat or laser census missing at one date under the 'plant_pattern' exclusion) |
| T7 soil clause (after the M2d offset is removed; by construction) | **not evaluable**. no M2d measured for this plot | = | = | = | = | = | = |
| T7 plant clause | **fail**. 8 of 9 stops with p95 <= 2 g = 16.0 cm; per-plant /dh95/ p95 7.4-18.6 cm; median difference -7.1 to -1.6 cm | = | **not evaluable on its own floor; against plot 198's M2d: fail**. 1 of 9 stops at or below the M2d per-plant floor (230601: 5.7 cm, 230613: 7.5 cm); per-plant /dh95/ p95 7.4-18.6 cm; median difference -7.1 to -1.6 cm | **fail**. 0 of 1 stops with p95 <= 2 g = 6.7 cm; per-plant /dh95/ p95 12.4-12.4 cm; median difference -7.1 to -7.1 cm | = | **not evaluable on its own floor; against plot 198's M2d: fail**. 1 of 9 stops at or below the M2d per-plant floor (230601: 5.7 cm, 230613: 7.5 cm); per-plant /dh95/ p95 7.4-18.6 cm; median difference -7.1 to -1.6 cm | **not evaluable on its own floor; against plot 198's M2d: fail**. 0 of 1 stops at or below the M2d per-plant floor (230601: 5.7 cm); per-plant /dh95/ p95 12.4-12.4 cm; median difference -7.1 to -7.1 cm |

Settings: as registered: 9 stops, r_assoc 9.08 cm, g 8.00 cm; item 30: 9 stops, r_assoc 9.08 cm, g 8.00 cm; item 31: 9 stops, r_assoc 9.08 cm, g 8.00 cm; item 32: 1 stops, r_assoc 7.57 cm, g 3.37 cm; item 33: 9 stops, r_assoc 9.08 cm, g 8.00 cm; all four (a): 9 stops, r_assoc 9.08 cm, g 8.00 cm; all four (b): 1 stops, r_assoc 7.57 cm, g 3.37 cm.

Figure: `fig_js2_sensitivity.png`, three panels: (a) item 31, each stop's per-plant |dh95| p95 against 2 g and the M2d
floors; (b) item 33, the 230613 same-day position p95 against the match radius, with the cap, the censoring limit and the
laser; (c) item 30, the false-removal rates with 95 % Wilson intervals beside the same-day rates.

## 3. Per item: the change, what it moves, whether it is principled, and the sceptical reviewer

The classification uses the criteria written in the design before any number: *mis-specified* (the clause compares a
quantity with a threshold that measures a different quantity, judged from the text), *trivially satisfiable* (the bound
admits more than half of the candidates being called removed), *internally inconsistent* (two texts of the record prescribe
different things), *merely makes a failure pass* (none of these holds and a fail turns into a pass). Each classification is
my judgement and is labelled so.

### Item 30, T5 specificity clause (sensitivity, not adopted)
**Change.** Report each sensor's false-removal rate with its interval beside the same-day rate, and make no pass or fail of
the specificity clause while the same-day rate exceeds 0.5.

**What it moves.** Plot 198: the splat's clause, "met" (9 of 28 against a bound of 20.1), becomes "no verdict" because the
same-day rate is 0.72 (56 of 78; Wilson 0.61-0.81). The splat's false-removal rate is 0.32 (Wilson 0.18-0.51;
Clopper-Pearson 0.16-0.52). T5 overall stays "fail". The laser ceiling is outside the T5 verdict and does not move: 15 of
43 = 0.35 (Wilson 0.22-0.50) against a bound of 5.7 at its rate of record 0.132 (the mean of the two scanners' fractions,
6 of 37 and 4 of 39; pooled 10 of 76, Wilson 0.07-0.23), "not met" under either rule. Plot 178: nothing moves; the splat's
same-day rate is 0.50 (18 of 36, Wilson 0.34-0.66), which is not above 0.5, so the clause stands as "met" (6 of 14 = 0.43,
Wilson 0.21-0.67, against 7.0); the laser's 11 of 26 = 0.42 against 5.5 (rate 0.21) stays "not met". Under the other
reading of "the rate" (the false-removal rate itself) nothing moves on either plot (0.32 and 0.43 are below 0.5).

**Principled or not (judgement).** On plot 198 the registered clause is *trivially satisfiable* by the design's criterion:
its bound admits 20.1 of 28 candidates (72 %) being called removed. On plot 178 the bound admits exactly half (7.0 of 14),
which is not more than half, so the criterion is not met there, by the smallest margin the counts allow. The clause is not
mis-specified: it compares false removals with the number expected under no change at the sensor's own same-day rate, which
is the right kind of null. Item 30 removes a pass that carries no information; it does not make a failure pass.

**Sceptical reviewer.** The 0.5 was written after the 0.72 was seen, and any cut between 0.50 and 0.72 removes the plot-198
pass while keeping plot 178's. With the item the splat's specificity is never judged on this crop, so the clause can no
longer fail for the sensor it was written for; reporting 0.32 (0.18-0.51) is honest but claims nothing. The binomial
intervals assume independent plants, while the plants share one registration per stop and a row structure, so the true
uncertainty is wider. The laser fails its own bound under every rule (15 against 5.7), which suggests the same-day rate,
taken from two scanners in one pass, is not the null for a 12-day interval in which plants grow, merge and change shape;
item 30 does not touch that.

### Item 31, T7 plant clause (sensitivity, not adopted)
**Change.** Test each stop's per-plant |h95 splat minus h95 laser| p95 against the M2d floor (UAV1 against the laser,
per-plant |diff| p95 of the same dates) instead of 2 g, and report the bias as the finding.

**What it moves.** Plot 198: the verdict stays "fail"; 0 of 12 stops within 2 g = 6.4 cm becomes 4 of 12 within the floor.
Per stop (p95 against floor, cm): 230601 9.05 / 10.65 / 8.85 / 8.70 against 5.74 (0 of 4); 230613 7.69 / **7.43** / 11.06 /
10.62 against 7.49 (1 of 4, by 0.6 mm); 230621 **9.32** / **10.87** / **14.22** / 14.87 against 14.43 (3 of 4, the third by
2 mm). The bias (median difference per stop) is -2.8 to -8.6 cm. The two 230628 stops that enter have no laser and are not
evaluable under either clause. Plot 178 (no M2d of its own; read against plot 198's floors of the same dates, flagged):
1 of 9 (230616_pos02, 7.39 against 7.49), where 2 g as registered admits 8 of 9 because g is capped at 8 cm (2 g = 16 cm).

**Principled or not (judgement).** The registered clause is *mis-specified* by the design's criterion: it compares a
cross-sensor difference (splat against laser) with 2 g, a same-sensor repeatability of the splat, so the bound tightens as
the splat becomes more repeatable while the tested quantity is a bias between sensors (the plan's argument, which the text
supports). Item 31 compares like with like in kind, a cross-sensor per-plant |diff| p95, though of another pair (UAV1 and
laser). It turns no fail into a pass on either plot.

**Sceptical reviewer.** M2d is UAV photogrammetry against the laser; passing it says the splat is as good as the UAV1
product, not that it agrees with the laser. The splat is registered to that same UAV1 cloud, so the floor is not independent
of the tested quantity. Three of the four passes come from the 230621 floor (14.4 cm), the loosest, which spans the two days
and the 14.65 mm of rain between the laser scan and the UAV1 flight; two of the four are within 2 mm of their floor, so the
count of 4 is not stable. "Report the bias as the finding" turns a failure into a result; a reviewer would accept it only
stated as a systematic error with its cause, that both image-based sensors read beet tops 3 to 9 cm below the laser.

**Post hoc, not a rule (found while checking the table, not part of the design).** Plot 178's one stop outside 2 g as
registered, 230616_pos00 (p95 18.6 cm), is compared with the 230613 laser, three days earlier, across the SB7_150623 harvest
of 15 June (its registration record says so). Ten of its 26 compared plants lie inside that polygon (dilated 0.10 m); nine
of them read 9 to 21 cm lower in the splat, as harvested plants would. Without them its p95 is 7.0 cm (n 16), which would
put plot 178 at 9 of 9 within 2 g as registered and 2 of 9 within the borrowed M2d floor. Whether the second site's T7
should exclude the harvested polygon on 230616 is a rule question for Paul; nothing is changed here.

### Item 32, stop exclusion (sensitivity, not adopted)
**Change.** Fix which statement governs the stop exclusion; the column computes the second statement, the T2 check and the
plant-pattern check, with the thresholds re-derived as the ledger's `main()` would.

**What it moves.** Plot 198: 7 of 18 stops enter (230601_pos00-03, 230613_pos03, 230621_pos00-01). With one stop on 230613
the same-day floor does not exist; r_assoc falls back to the laser's pooled p95 capped at 9.5 cm and g to the laser's
2.8 cm. T3 association and g "undecidable"; T4 230613 0.905 (pass, 38 against 42 of 60 marks) becomes 0.667 (fail, 12
against 18 of 20 marks; 0.556 with the registered thresholds, line (a)); T4 230601 0.678 becomes 0.667; T5 removal clause
"not met" by the code (0 against 8) with no coverage of the polygon on 230613, so not a test; splat specificity "not
evaluable" (7 of 14 called removed; no same-day rate); T6 "undecidable" (n 2, n 0); T7 0 of 7 within 2 g = 5.6 cm (all four
(b): 2 of 7 within the M2d floor). Plot 178: 1 of 11 stops enters (230601_pos01); every downstream verdict is
"undecidable" except T7 (0 of 1 within 2 g = 6.7 cm).

**Principled or not (judgement).** The record is *internally inconsistent* by the design's criterion: the header of
`mustc_splat_geom.py` says the floor and ledger skip stops whose plant-pattern check fails, while the code comment of
08:05 UTC 4 Nov says they read the composite T2 check. Fixing which governs is principled. The second statement moves
verdicts towards "undecidable" and one towards "fail"; it makes no failure pass. The plan records that the T2 check was
chosen on the texts before any ledger run; I cannot verify the order of that decision from the files.

**Sceptical reviewer.** The plant-pattern check's only basis on record is the two detector revisions of 4 Nov (the plan's
own words in item 32), and the stops that fail it sit at its edge (matched fraction 0.40-0.50 against 0.5, or residual
3.1-3.8 cm against 3.0 cm). If the
check measures the two detectors' disagreement rather than the registration, as the 08:05 comment argues, excluding stops by
it discards data for the wrong reason; if it measures the registration, the 230613 floor that sets r_assoc and g rests on
three stops whose registration is not trusted. Either way a reviewer will ask for the 230613 registration accuracy from an
independent measure (the split-half laser shift test, +1.2 to +6.0 cm east, is the one on record).

### Item 33, T3 wording (sensitivity, not adopted)
**Change.** Read the association verdict from the 0.15 m match radius alone and state the 0.20 m value beside it.

**What it moves.** Plot 198: nothing; 12.1 cm at 0.15 m (n 18) and 16.2 cm at 0.20 m (n 23) both exceed the 9.5 cm cap.
Plot 178: "radius-dependent" becomes "resolvable": 9.08 cm at 0.15 m (n 18) against 9.5 cm, with 10.4 cm at 0.20 m (n 19)
beside it. The laser's own line is unchanged by the item: 9.7 cm (n 145, not resolvable) on plot 198, 7.6 cm (n 151,
resolvable) on plot 178. Under all four (b) both plots are "undecidable". Beside it, outside the item: the ledger record's
own run at r_assoc 0.095 m (plot 198) gives T4 230613 0.952, T6 230601->230613 pass on n 14 and 230613->230621 fail on n 10;
no item proposes that radius, and it is quoted only because someone reading item 33 might draw it.

**Principled or not (judgement).** The clause of 4 Nov is *internally inconsistent* by the design's criterion: it says the
p95 must be measured at a radius well above the cap, else it is censored, and it says a p95 within 10 % of its radius is
censored. At 0.10 m with 8.75 cm the first statement says censored (0.10 m is not well above 0.095 m) and the second says
not censored. Fixing one uncensored radius is principled; choosing 0.15 m over 0.20 m is a judgement (0.15 m is the laser's
adopted cap and below the 0.19 m in-row spacing, while at 0.20 m a match can reach the neighbouring plant). The verdict it
moves was "radius-dependent", not a fail, so by the criterion it does not make a failure pass, but its only effect is
favourable.

**Sceptical reviewer.** On plot 178 the p95 at 0.15 m equals the p95 at 0.10 m because no pair is matched between the two
radii (18 and 18), and one more pair at 0.20 m lifts it to 10.4 cm; a 95th percentile of 18 distances sits among the two
largest, so the 4 mm margin rests on one or two plants. The radius was named after the 0.15 and 0.20 m values of both plots
were known. Association that resolves on plot 178 buys nothing downstream there: its |dh95| p95 is 11.9 cm, g is capped at
8 cm and growth over its 12-day span is not callable. On plot 198, the primary site, association stays not resolvable for
both sensors.

### All four together (sensitivity, not adopted)
Plot 198, (a) item 32 read as the T2 check: T5 specificity "no verdict", T7 4 of 12 within the floor (fail), T3 "not
resolvable", everything else as registered. (b) item 32 read as the second statement: the item-32 column with T7 at 2 of 7
within the floor and the splat specificity "not evaluable" (7 of 14 = 0.50, Wilson 0.27-0.73). Plot 178: (a) T3
"resolvable", T7 "not evaluable on its own floor; against plot 198's M2d: fail" (1 of 9); (b) everything undecidable except
T1, T2 and T7 (0 of 1 against plot 198's floor). No failed test becomes a pass on either plot.

## 4. What waits for Paul's hand marks and what is final

- The ledger reads marks through one function, `load_marks()` (`marks_<date>.csv`, else `premarks_<date>.csv`), and
  `frames.json` holds no entry derived from marks (soil phase correlation and laser-to-ortho links; laser 230621 is
  plant-based, from the clouds). A run with both withheld (in memory) changes the T4 records only, on plot 198: both T4
  verdicts become "undecidable (no splat census or marks)". T1, T2, T3, T5, T6 and T7 are identical with and without marks
  on both plots.
- **Cells that change when the marks arrive:** the two T4 rows of plot 198 in all seven columns (as registered and items 30,
  31, 33, all four (a): 0.678 and 0.905; item 32 and all four (b): 0.667 and 0.667). The 230613 cell passes by 0.005 on the
  premarks, so it can turn either way.
- **Final with respect to the marks:** every other cell of both tables. "Final" means only that; a decision on items 30 to
  33, a hand initialisation, or a `frames.json` re-run would still change cells.
- Outside the table, the found-again decomposition, the weed or beet split of added calls, census recall and precision,
  Fig 8 (a) classes and (c), and the marks columns of Table 7 read the marks. Separately, the T5 precheck (seedlings present
  in the 230601_pos00 crops before the 2 June harvest) rests on an agent's reading; the human count is still pending.

## 5. Remaining machine-only checks

### Stake checks (plan item 25; `mustc_stakes_check.py` run unmodified, plus 0.5 m crops)
Reading rule of the design: "rod" if a compact object that is not vegetation lies within 5 cm of the moved position on the
ortho nearest in time; "plant" if only green vegetation is there; "undetermined" otherwise. These are an agent's readings
of the pictures, not a person's. Pictures in `stakes/` (three sheets; the per-ortho crops and polygon views are on the box).
- **Control, laser 230621 id 1 in SB7_150623** (one of the twelve known rods; h95 36 cm, 23 cm2): on 230623 a pale pink rod
  lies inside the 5 cm circle at the moved position, with an orange rod further north in the crop; the laser height map
  shows an isolated elongated object. Reading: rod. The rule recognises a known rod.
- **Laser 230621 id 41 in SB7_080723** (h95 21 cm, 19 cm2): on 230623 (moved position; link 0.6 cm) the 5 cm circle holds
  shadowed soil, a beet petiole and the edge of a large beet leaf, and nothing orange, pink or white; on 230614 the moved
  circle covers a beet leaf pointing west from a plant about 10 cm to the east; on 230708 closed beet canopy; the laser
  height map shows a 19 cm2 fragment joined by a one-cell strip about 10 cm long to the plant canopy to the north.
  Reading: **plant** (most likely a leaf of a neighbouring beet that the instancing split off). The stake rule (h95 >
  0.20 m and area < 40 cm2) flags it wrongly, by 1 cm of height. It lies in SB7_080723, which no table row uses.
- **Laser 230525 id 4 in SB7_240623** (h95 31 cm, 6 cm2): 230525 has no laser-to-ortho link on plot 178, so the moved
  position of the rule cannot be formed (deviation: raw position, 10 cm circle). On 230526, one day after the scan, a white
  card with a QR code and a red mark, about 6 by 11 cm (a plot label), lies inside the circle, its centre about 5 cm and its
  nearest corner within about 2 cm of the raw position (read by eye against the 10 cm scale bar); on 230602 the same card
  casts a long thin shadow to the north-west, so it stands above the soil on a stake; on 230517 bare soil. Apart from a
  seedling about 9 cm to the west, the card is the only object within 10 cm, and a two-leaf seedling cannot return 31 cm.
  Reading: **not vegetation, a plot label on a stake (rod class)**; the flag is right. By the rule as written: not
  applicable (no moved position).
- Consequence: the two stake-like flags inside polygons that the plan lists as unchecked (plan item 25) split one right
  (a label) and one wrong (a leaf). The stake rule should not be cited beyond SB7_150623 without that line. The two
  marginal tall instances of plot 198 (230711 and 230726, outside the scored dates) were not checked here.

### Hand initialisation of 230628_pos00-03 (not attempted, nothing substituted)
- **What Paul gives, per stop:** three numbers for `mustc_splat_geom.py run --stop 230628_pos0N --hand-init X_UTM Y_UTM
  YAW_DEG` (box: `/mnt/shared2/must-c/scripts/`, plot 198 default): the EPSG 25832 easting and northing at which the origin
  of the stop's levelled rig frame lands, and the yaw in degrees by which that frame's x axis is turned anticlockwise from
  east (the code builds the coarse transform as a rotation by YAW_DEG plus that translation; height comes from the
  reference soil). They come from hand-clicked correspondences (the plan says four) between the stop's nadir render
  `/mnt/shared/must-c_bulk/work/splat_geom/230628_pos0N/ortho_render_5mm.png` (5 mm pixels, levelled frame: x = xmin +
  (u + 0.5) 0.005, y = ymax - (v + 0.5) 0.005, with xmin and ymax in the `meta` of `ortho_render_5mm.npz`; renders
  523-544 by 548-565 pixels, about 2.7 by 2.8 m) and the UAV1 230623 ortho, the coarse reference of the 230628 route; the
  fine fit to UAV2 230629 then runs by itself.
- **What exists:** the four renders (4 Nov). There is no tool for clicking correspondences or fitting the rigid transform in
  `automation/mustc/` (the marks page clicks plant centres on orthos only), so a small fitting step is needed besides the
  clicks.
- **What else is needed, a rule decision:** the T2 check of record (`registration_ok`) requires `hand_init is None`
  (`mustc_splat_geom.py` lines 1845 and 2060), so a hand-initialised stop fails it by construction and would still be
  excluded, whereas the design's T2 text has it enter, marked. Paul must say what check a hand-initialised stop must pass
  (for example the soil residual alone, or with the plant-pattern check against UAV2), and the code must follow.
- **What it would change in this table:** the T2 entry row only (from 14 of 18 stops entering to at most 18 of 18) and the
  descriptive 230628 rows.
  230628 has no laser within 13 days, so T6 230621->230628 and T7 stay "not evaluable" by design; T3, T4 and T5 use other
  dates.
- **How long:** the machine part is measured: `run_stop` took 209-251 s per 230628 stop on 4 Nov (`dsm_vs_laser.json`
  wall_s), then the floor and the ledger take seconds. The human part is not measured. It is the step the automatic search
  failed on (closed canopy on a 0.5 m row lattice; reverse candidate at 0.77-0.97 of the peak), so the person needs features
  that break the lattice (gaps, weeds, labels, wheel marks) visible in both images.

## 6. Deviations from the design
1. (Mac agent) The T5 cell text gained the polygon-coverage count, and the item-30 overall text names the clause judged,
   after the first run (wording only; no rule, number or verdict word changed).
2. (Mac agent) Stake crops for laser 230525: no laser-to-ortho link exists on plot 178, so the raw position with a 10 cm
   circle replaces the moved position with 5 cm; two extra ortho dates (230708, 230602) and one known rod were added.
3. (Mac agent) Plot 178, item 31: the 230616 stops are read against the laser date their own record names (230613).
4. (A100) Added verification not in the design (`spotcheck_a100.py`), and one post hoc diagnostic (plot 178,
   230616_pos00 across the harvest), labelled post hoc in Section 3.
5. (A100) Results are written in this folder (README mapping), not in the Mac ops repo's `automation/mustc_sensitivity/`;
   this file stands for `SENSITIVITY_20261112.md`.

## 7. What was verified and what was not
Verified:
- The as-registered rerun of `mustc_ledger.compute()` reproduces `ledger.json` of record (the T1 to T7 verdict records, pairs, polygons, tracks):
  0 differences on both plots (Mac agent). Item 32 line (a) reproduces the stored `sensitivity.exclusion.plant_pattern`:
  0 differences.
- `verify_sens.py` (imports no MuST-C module; its own stop selection, same-day floor, censuses, association, T4 to T7 and
  intervals): 201 of 201 comparisons agree on plot 198, 103 of 103 on plot 178 (Mac agent).
- `spotcheck_a100.py` (A100; reads only the records of record; its own Wilson formula, cross-checked against scipy): 44 of
  44 checks agree on plot 198, 37 of 37 on plot 178 (T1 from `rig_ws_summary.csv`, T2 stop sets from `registration.json`,
  T3 and item 33 from `noise_floor.json`, T4 from `census_truth.json`, T5 and item 30 from `ledger.json`, T7 and item 31
  recomputed from the per-plant lists of `dsm_vs_laser.json` against `cross_sensor_floor.json`).
- Inputs untouched: the md5 of every record file is the same before and after (222 files on plot 198, 134 on plot 178), and
  the 36 MuST-C scripts are unchanged (`verification/`).
- By reading code: the driver's four clauses against the design text; the single marks reader; the hand-initialisation
  path and its exclusion by `registration_ok`.
- By eye: twelve pictures (three zoom sheets, five per-ortho crops, two height-map crops, two polygon views) for the stake
  readings; the figure after rendering.
Not verified:
- Anything that needs Paul's hand marks (T4 rests on premarks); the seedling count of the T5 precheck.
- What the stake objects are, beyond what the pictures show (agent readings).
- The registrations themselves (taken from the records of record), and the independence the binomial intervals assume.
- The post hoc 230616 value was not put through the ledger; it is a diagnostic only.
- The human time of the hand initialisation; the palette validator of the figure could not run (no `node` on the box), so
  the figure uses the reference palette's first two slots, which that reference documents as passing.

## 8. Needs Paul
1. Items 30 to 33: adopt or not; nothing is adopted here. No item turns a failed test into a pass on plot 198.
2. Plot 178 T7: whether the 230616 stops' comparison with the 230613 laser should exclude SB7_150623 (post hoc finding).
3. The check a hand-initialised stop must pass, before the four 230628 hand initialisations can change anything.
4. The stake rule: cite it for SB7_150623 only, or with the id 41 misfire stated.
5. The hand marks (T4 and the marks-dependent figures outside the table).

## Files
`results/RESULTS.md` (this), `results/SECTION_4_8_READING.md`, `results/NOTEBOOK_ENTRY.md`, `results/fig_js2_sensitivity.png`,
`results/table_compact_plot198|178.md`, `results/sensitivity_table_plot198|178.json|md`, `results/verify_sens_plot198|178.json`,
`results/verification/` (spot-check JSONs and log, md5 lists), `results/stakes/` (sheets and stake records); code in `code/`
(`mustc_sens.py`, `verify_sens.py`, `stake_zoom.py` from the Mac agent; `spotcheck_a100.py`, `compact_table.py`,
`fig_js2_sensitivity.py` from the A100). Commits: a4f5085 (as received), 357e5e1 (continuation plan); the commit that adds
this file is recorded in STATE.md.
