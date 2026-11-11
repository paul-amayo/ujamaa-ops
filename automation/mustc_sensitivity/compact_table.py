#!/usr/bin/env python3
"""JS-2: compact form of the sensitivity tables (SENSITIVITY, NOT ADOPTED).  Reads sensitivity_table_plot<p>.json only.
A cell identical to the as-registered cell in verdict and basis prints as "="; any other cell prints its verdict (bold) and
its basis as the table JSON holds it (a few fixed phrases shortened, listed in ABBREV).  Writes table_compact_plot<p>.md.

Box: /mnt/shared2/must-c/venv/bin/python compact_table.py --out-dir <dir>
"""
import argparse
import json

ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", required=True)
A = ap.parse_args()
SRC = {"198": "/mnt/shared2/must-c/plot198/census/sensitivity_20261112/sensitivity_table_plot198.json",
       "178": "/mnt/shared2/must-c/plot178/census/sensitivity_20261112/sensitivity_table_plot178.json"}
ABBREV = [("230613: no matched same-day plants (no overlapping pair, no instances in the overlap, or fewer than two stops pass the exclusion rule) (0 matched plants)",
           "230613: 0 matched same-day plants (fewer than two stops enter that day)"),
          ("unchanged on-row plants called removed", "unchanged on-row plants called removed"),
          (" (beside; not in the verdict)", " (beside)"), (" (beside; not a T6 interval)", " (beside)"),
          ("PROVISIONAL marks", "provisional marks")]
COLS = ["as registered", "item 30", "item 31", "item 32", "item 33", "all four (a: item 32 = T2 check)", "all four (b: item 32 = T2 and plant-pattern)"]
HEAD = ["as registered", "item 30", "item 31", "item 32", "item 33", "all four (a)", "all four (b)"]


def short(t):
    for a, b in ABBREV:
        t = t.replace(a, b)
    return t.replace("|", "/")


for p, path in SRC.items():
    T = json.load(open(path)); C = T["columns"]; reg = C["as registered"]
    L = [f"Plot {p}. SENSITIVITY, NOT ADOPTED; the verdicts of record are the as-registered column. \"=\" = verdict and number identical to the as-registered cell.", "",
         "| test / clause | " + " | ".join(HEAD) + " |", "|---|" + "---|" * len(HEAD)]
    for r in T["rows"]:
        cells = []
        for cn in COLS:
            c = C[cn].get(r)
            if c is None:
                cells.append("n/a")
            elif cn != "as registered" and c.get("verdict") == reg.get(r, {}).get("verdict") and c.get("basis") == reg.get(r, {}).get("basis"):
                cells.append("=")
            else:
                cells.append(f"**{c['verdict']}**. {short(c['basis'])}")
        L.append(f"| {short(r)} | " + " | ".join(cells) + " |")
    st = [f"{h}: {len(C[cn]['_settings']['stops_entering'])} stops, r_assoc {C[cn]['_settings']['r_assoc_m'] * 100:.2f} cm, g {C[cn]['_settings']['g_m'] * 100:.2f} cm" for h, cn in zip(HEAD, COLS)]
    L += ["", "Settings: " + "; ".join(st) + "."]
    with open(f"{A.out_dir}/table_compact_plot{p}.md", "w") as f:
        f.write("\n".join(L) + "\n")
    n_eq = sum(1 for r in T["rows"] for cn in COLS[1:] if C[cn].get(r) and C[cn][r].get("verdict") == reg.get(r, {}).get("verdict") and C[cn][r].get("basis") == reg.get(r, {}).get("basis"))
    print(f"plot {p}: {len(T['rows'])} rows; {n_eq} of {len(T['rows']) * 6} non-registered cells identical to the as-registered cell; wrote {A.out_dir}/table_compact_plot{p}.md")
