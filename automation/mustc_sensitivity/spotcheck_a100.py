#!/usr/bin/env python3
"""JS-2 A100 continuation (11 Nov 2026): transcription check of the sensitivity tables. SENSITIVITY, NOT ADOPTED.

Independent of the MuST-C modules and of mustc_sens.py / verify_sens.py: it reads only the records of record
(ledger.json, census_truth.json, noise_floor.json, cross_sensor_floor.json, dsm_vs_laser.json, registration.json,
lmi_assoc.json, results/rig_ws_summary.csv, truth.json) and the two table JSONs, and checks that each table cell reports
the record's number, recomputing the item-30 intervals (own closed form), the item-31 and 2 g counts (per-plant lists of
dsm_vs_laser.json against cross_sensor_floor.json), the item-33 verdicts and the item-32 stop sets.  One post hoc
diagnostic (labelled so, not a rule) on plot 178's 230616 stops.  Writes spotcheck_a100_plot<p>.json into each plot's
sensitivity_20261112 directory; prints one line per comparison.

Box: OMP_NUM_THREADS=3 /mnt/shared2/must-c/venv/bin/python spotcheck_a100.py
"""
import csv
import json
import math
import os

import numpy as np
from matplotlib.path import Path

Z = 1.959963984540054
CAP_R = 0.095
OUT = {}


def J(p):
    with open(p) as f:
        return json.load(f)


def wilson(k, n):
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return [c - h, c + h]


def check(plot, what, mine, table, tol=1e-6):
    if isinstance(mine, float) or isinstance(table, float):
        ok = mine is not None and table is not None and abs(float(mine) - float(table)) <= tol
    else:
        ok = mine == table
    OUT[plot]["checks"].append({"what": what, "record_or_recount": mine, "table": table, "agree": bool(ok)})
    print(("OK   " if ok else "DIFF ") + f"plot {plot} | {what}: record/recount {mine} | table {table}")


def p95_abs(plants):
    d = [q["splat_h95"] - q["laser_h95"] for q in plants if q.get("splat_h95") is not None and q.get("n_splat_pts", 0) >= 20]
    return (float(np.percentile(np.abs(d), 95)), len(d)) if d else (None, 0)


def in_dilated(poly, xy, dil):
    poly = np.asarray(poly, float); xy = np.asarray(xy, float).reshape(-1, 2)
    inside = Path(poly).contains_points(xy)
    best = np.full(len(xy), np.inf)
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]; v = q - p
        t = np.clip(((xy - p) @ v) / float(v @ v), 0, 1)
        best = np.minimum(best, np.hypot(*(xy - (p + t[:, None] * v)).T))
    return inside | (best <= dil)


for plot in ("198", "178"):
    OUT[plot] = {"label": "SENSITIVITY, NOT ADOPTED: A100 transcription check against the records of record", "checks": [], "post_hoc": {}}
    C = f"/mnt/shared2/must-c/plot{plot}/census"; RES = f"/mnt/shared2/must-c/plot{plot}/results"
    TAB = J(f"{C}/sensitivity_20261112/sensitivity_table_plot{plot}.json")
    col = TAB["columns"]; reg_col = col["as registered"]
    L = J(f"{C}/ledger.json"); G = L["gates_T1_T7"]; CT = J(f"{C}/census_truth.json"); NF = J(f"{C}/noise_floor.json")
    REG = J(f"{C}/registration.json"); DSM = J(f"{C}/dsm_vs_laser.json"); AS = J(f"{C}/lmi_assoc.json")
    XS = J("/mnt/shared2/must-c/plot198/census/cross_sensor_floor.json"); TR = J(f"{C}/truth.json")

    # ---- T1 from the rig summary CSV
    rows = list(csv.DictReader(open(f"{RES}/rig_ws_summary.csv")))
    px = [float(r["ba_swap_reproj_px"]) if r["ba_adopted"] == "swap" else float(r["ba_noswap_reproj_px"]) for r in rows]
    pxs = [float(r["ba_swap_reproj_px"]) for r in rows]
    basis = reg_col["T1 reprojection <= 0.70 px"]["basis"]
    check(plot, "T1 basis (n pass of n, swap px range from rig_ws_summary.csv)", f"{sum(p <= 0.70 for p in pxs)} of {len(pxs)} stops, {min(pxs):.2f}-{max(pxs):.2f} px", basis)

    # ---- T2 from registration.json (raw fields)
    stops = sorted(k for k in REG if not k.startswith("_") and "__" not in k)
    ok = [s for s in stops if (REG[s].get("registration_ok") or {}).get("ok")]
    pp = [s for s in ok if (REG[s].get("gate") or {}).get("pass") is True]
    nflag = sum(1 for s in stops if any(f.startswith("|s-1| > 1 %") for f in (REG[s].get("flags") or [])))
    check(plot, "T2 check count", f"{len(ok)} of {len(stops)}", reg_col["T2 check (soil residual, yaw)"]["verdict"])
    check(plot, "T2 + plant-pattern count (item 32 stop set size)", f"{len(pp)} of {len(stops)}", col["item 32"]["T2 stops entering the ledger"]["verdict"])
    check(plot, "item 32 stop set (names)", pp, col["item 32"]["_settings"]["stops_entering"])
    check(plot, "as-registered stop set (names)", ok, reg_col["_settings"]["stops_entering"])
    check(plot, "T2 scale flag count", f"{nflag} of {len(stops)} flagged", reg_col["T2 scale flag |s-1| > 1 %"]["verdict"])

    # ---- T3 / item 33 from noise_floor.json
    t3 = NF["T3_adopted"]["230613"]; pr = t3["position_p95_by_radius_m"]; nr = t3["n_matched_by_radius"]
    asreg = "not resolvable" if (pr["0.150"] > CAP_R and pr["0.200"] > CAP_R) else ("resolvable" if (pr["0.150"] <= CAP_R and pr["0.200"] <= CAP_R) else "radius-dependent")
    check(plot, "T3 as registered (0.15 and 0.20 m together)", asreg, reg_col["T3 association resolvable at the 9.5 cm cap"]["verdict"])
    i33 = "resolvable" if pr["0.150"] <= CAP_R else "not resolvable"
    check(plot, "T3 item 33 (0.15 m alone)", i33, col["item 33"]["T3 association resolvable at the 9.5 cm cap"]["verdict"])
    check(plot, "T3 p95 at 0.15 m", pr["0.150"], col["item 33"]["T3 association resolvable at the 9.5 cm cap"]["item33"]["p95_at_0.15_m"], 1e-12)
    check(plot, "T3 n at 0.15 / 0.20 m", (nr["0.150"], nr["0.200"]), (col["item 33"]["T3 association resolvable at the 9.5 cm cap"]["item33"]["n_at_0.15"], col["item 33"]["T3 association resolvable at the 9.5 cm cap"]["item33"]["n_at_0.20"]))
    lp = AS["same_day_floor"]["pooled_growth_dates_cap_0.15"]
    check(plot, "T3 laser ceiling verdict", "resolvable" if lp["dist_p95_m"] <= CAP_R else "not resolvable", reg_col["T3 laser ceiling, same clause (beside)"]["verdict"])
    OUT[plot]["t3_numbers"] = {"p95_by_radius_m": pr, "n_by_radius": nr, "laser_pooled_p95_m": lp["dist_p95_m"], "laser_n": lp["n_matched"], "g_p95_m": t3.get("g_p95_measured_m")}

    # ---- T4 from census_truth.json
    for d in ("230601", "230613"):
        v = (CT.get("T4") or {}).get(d)
        rn = f"T4 {d} splat recall >= 0.9 x laser recall"
        if isinstance(v, dict) and "ratio" in v and rn in reg_col:
            check(plot, f"T4 {d} ratio", v["ratio"], reg_col[rn]["ratio"], 1e-12)
            check(plot, f"T4 {d} verdict", "pass" if v["ratio"] >= 0.9 else "fail", reg_col[rn]["verdict"])

    # ---- T5 and item 30 from ledger.json
    tk = "T5_second_site" if "T5_second_site" in G else "T5"; T5 = G[tk]; PN = T5["polygon"]
    cr = T5["clause_removals_restated_4nov"]
    check(plot, "T5 removal clause value/bar", (cr["value"], cr["bar"]), (reg_col[f"T5 {PN} removal clause (restated 4 Nov)"]["value"], reg_col[f"T5 {PN} removal clause (restated 4 Nov)"]["bar"]))
    OUT[plot]["item30"] = {}
    for sens, key, rowname in (("splat", "specificity_splat", f"T5 {PN} specificity clause, splat"), ("laser", "specificity_laser", f"T5 {PN} specificity, laser ceiling (beside; not in the verdict)")):
        sp = T5[key]; k, n, rate = sp["n_removed_called"], sp["n_on_row_unchanged_candidates"], sp["same_day_false_change_rate"]
        cell30 = col["item 30"][rowname]; w = wilson(k, n)
        check(plot, f"item 30 {sens}: counts in basis", f"{k} of {n} unchanged on-row plants called removed" in cell30["basis"], True)
        check(plot, f"item 30 {sens}: same-day rate", rate, cell30["item30"]["same_day_rate"], 1e-12)
        check(plot, f"item 30 {sens}: Wilson low", w[0], cell30["item30"]["false_removal_wilson95"][0], 1e-9)
        check(plot, f"item 30 {sens}: Wilson high", w[1], cell30["item30"]["false_removal_wilson95"][1], 1e-9)
        v30 = "no verdict" if rate > 0.5 else ("met" if k <= rate * n else "not met")
        check(plot, f"item 30 {sens}: verdict", v30, cell30["verdict"])
        check(plot, f"as registered {sens}: verdict", "met" if k <= rate * n else "not met", reg_col[rowname]["verdict"])
        OUT[plot]["item30"][sens] = {"k": k, "n": n, "false_removal_rate": k / n, "wilson95": w, "same_day_rate": rate, "bound": rate * n, "verdict_item30": v30}
        try:
            from scipy.stats import binomtest
            ci = binomtest(k, n).proportion_ci(0.95, method="wilson")
            check(plot, f"item 30 {sens}: Wilson low (scipy cross-check of my formula)", float(ci.low), w[0], 1e-9)
        except ImportError:
            pass

    # ---- T7 and item 31 from dsm_vs_laser.json and cross_sensor_floor.json
    floors = {p["pair"]: p["per_plant"]["registered"]["abs_diff_p95_m"] for p in XS["pairs"]}
    two_g = G["T7"]["two_g_m"]; per = {}
    for s in ok:
        vl = (DSM.get(s) or {}).get("vs_laser") or {}
        b = ((vl.get("by_footprint") or {}).get("nview6") or {}).get("b") or {}
        plants = (b.get("per_plant") or {}).get("plants") or []
        if not vl.get("laser_date") or not plants:
            continue
        p95, n = p95_abs(plants)
        ref = REG[s]["reference"]; pair = f"laser_{vl['laser_date']}_vs_{ref['kind']}_{ref['date']}"
        per[s] = {"p95_m": p95, "n": n, "pair": pair, "floor_m": floors.get(pair), "le_2g": p95 <= two_g, "le_floor": (p95 <= floors[pair]) if pair in floors else None,
                  "record_p95_m": G["T7"]["per_stop"].get(s, {}).get("per_plant_absdiff_p95_m")}
        check(plot, f"T7 {s} per-plant p95 recomputed vs ledger record", p95, per[s]["record_p95_m"], 1e-9)
    n2g = sum(v["le_2g"] for v in per.values())
    rec31 = col["item 31"]["T7 plant clause"]["item31"]
    check(plot, "T7 as registered: stops with p95 <= 2 g", (n2g, len(per)), (reg_col["T7 plant clause"]["n_pass"], reg_col["T7 plant clause"]["n"]))
    ev = {s: v for s, v in per.items() if v["le_floor"] is not None}
    check(plot, "item 31: stops at or below the M2d floor", (sum(v["le_floor"] for v in ev.values()), len(ev)), (rec31["n_within_floor"], rec31["n_evaluated"]))
    OUT[plot]["t7_per_stop"] = per; OUT[plot]["two_g_m"] = two_g

    # ---- post hoc (not a rule): plot 178's 230616 stops are read against the 230613 laser, across the 15 June harvest
    if plot == "178":
        poly = TR["destructive"]["SB7_150623"]["polygon_utm"]; ph = {}
        for s in [s for s in per if s.startswith("230616")]:
            plants = DSM[s]["vs_laser"]["by_footprint"]["nview6"]["b"]["per_plant"]["plants"]
            plants = [q for q in plants if q.get("splat_h95") is not None and q.get("n_splat_pts", 0) >= 20]
            xy = np.array([[q["x"], q["y"]] for q in plants]); ins = in_dilated(poly, xy, 0.10)
            rest = [q for q, i in zip(plants, ins) if not i]
            d_in = [q["splat_h95"] - q["laser_h95"] for q, i in zip(plants, ins) if i]
            p95_rest = p95_abs(rest)
            ph[s] = {"n_plants": len(plants), "n_inside_SB7_150623_dilated_0.10": int(ins.sum()), "diff_inside_m": [round(x, 3) for x in d_in],
                     "p95_all_m": per[s]["p95_m"], "p95_without_inside_m": p95_rest[0], "n_without_inside": p95_rest[1]}
            print(f"POST HOC plot 178 {s}: {int(ins.sum())} of {len(plants)} compared laser-230613 plants lie inside SB7_150623 (dilated 0.10 m); "
                  f"p95 {per[s]['p95_m']:.4f} m with them, {p95_rest[0]:.4f} m without; differences inside {[round(x, 3) for x in d_in]}")
        OUT[plot]["post_hoc"]["t7_230616_vs_230613_laser_across_harvest"] = {"label": "post hoc, not a rule", "per_stop": ph}

    OUT[plot]["n_checks"] = len(OUT[plot]["checks"]); OUT[plot]["n_diff"] = sum(1 for c in OUT[plot]["checks"] if not c["agree"])
    with open(f"{C}/sensitivity_20261112/spotcheck_a100_plot{plot}.json", "w") as f:
        json.dump(OUT[plot], f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else (bool(o) if isinstance(o, np.bool_) else str(o)))
    print(f"plot {plot}: {OUT[plot]['n_checks']} checks, {OUT[plot]['n_diff']} DIFF; wrote {C}/sensitivity_20261112/spotcheck_a100_plot{plot}.json")
