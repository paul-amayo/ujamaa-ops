#!/usr/bin/env python3
"""JS-2 (12 Nov 2026): SENSITIVITY, NOT ADOPTED, of the MuST-C verdicts T1-T7 (JIRS Section 4.8) to the four proposed rule
changes, items 30-33 of plans/mustc_jirs.md "Revisions 11 Nov 2026".  Design of record: DESIGN_20261112.md (same directory).

This driver imports byte-identical copies of mustc_ledger.py / mustc_sameday_floor.py / mustc_lmi_census.py from its own
directory (bin/ on the box) and calls mustc_ledger.compute() only; main() of the ledger (the function that writes into the
census directory) is never called.  It reads the finished records and writes only under --out-dir.  Each changed clause sits
behind an explicit flag:
  --item30  T5 specificity: false-removal rate with its interval beside the same-day rate; no pass or fail while the same-day
            rate exceeds 0.5
  --item31  T7 plant clause: per-plant |dh95| p95 against the cross-sensor floor M2d (UAV1 against the laser) instead of 2 g
  --item32  stop exclusion: T2 check AND plant-pattern check (the second of the two statements); thresholds re-derived as
            mustc_ledger.main() would derive them under that statement
  --item33  T3 association verdict read from the 0.15 m radius alone (0.20 m beside)
With --all-columns the seven columns of the design are computed in one run and the table is written.

Box: /mnt/shared2/must-c/venv/bin/python mustc_sens.py --plot 198|178 --out-dir <dir> --all-columns
"""
import argparse
import copy
import glob
import json
import math
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mustc_ledger as ML  # noqa: E402
import mustc_sameday_floor as sdf  # noqa: E402

CAP_R, CAP_G = ML.CAP_R, ML.CAP_G
ITEM30_RATE = 0.5      # item 30, as written
ITEM33_RADIUS = "0.150"  # item 33, as written
Z95 = 1.959963984540054
LABEL = "SENSITIVITY, NOT ADOPTED (JS-2, 12 Nov 2026): the verdicts of record are the as-registered column"


def wilson(k, n):
    if not n:
        return None
    p = k / n; den = 1 + Z95 ** 2 / n
    c = (p + Z95 ** 2 / (2 * n)) / den; h = Z95 * math.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def clopper(k, n):
    if not n:
        return None
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)); hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def jdiff(a, b, path="", tol=1e-9, out=None, limit=60):
    """Paths at which two JSON-like objects differ (floats within tol are equal)."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'record' if k in b else 'rerun'}")
            else:
                jdiff(a[k], b[k], f"{path}/{k}", tol, out, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: list length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                jdiff(x, y, f"{path}[{i}]", tol, out, limit)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        if abs(a - b) > tol:
            out.append(f"{path}: {a} vs {b}")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


class Site:
    """Context of one plot, loaded as mustc_ledger.main() loads it (read-only)."""

    def __init__(self, plot):
        self.plot = plot; self.P = ML.PLOT_CFG[plot]; self.C = C = self.P["census"]
        sdf.set_plot(plot)
        J = ML.jload
        self.truth = J(f"{C}/truth.json"); self.assoc = J(f"{C}/lmi_assoc.json"); self.nf = J(f"{C}/noise_floor.json")
        self.xs = J(f"{C}/cross_sensor_floor.json") if os.path.exists(f"{C}/cross_sensor_floor.json") else None
        self.xs198 = J(f"{ML.PLOT_CFG['198']['census']}/cross_sensor_floor.json")
        self.nf198 = J(f"{ML.PLOT_CFG['198']['census']}/noise_floor.json")
        self.dsm = J(f"{C}/dsm_vs_laser.json"); self.REG = J(f"{C}/registration.json")
        self.F = J(f"{C}/frames.json") if plot == "198" else ML.frames_178(C)
        self.stops_all = sorted(x for x in {re.sub(r"__b$", "", os.path.basename(p)[:-5]) for p in glob.glob(f"{C}/splat_census/*.json")} if "__" not in x)
        self.status = sdf.registration_status(self.stops_all)
        inputs = ["truth.json", "lmi_assoc.json", "noise_floor.json", "cross_sensor_floor.json", "dsm_vs_laser.json", "registration.json", "frames.json", "t5_gate.json", "georef_prep/frames_prep_178.json"]
        self.inputs_mtime = {f: time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(os.path.getmtime(f"{C}/{f}"))) + " UTC" for f in inputs if os.path.exists(f"{C}/{f}")}
        self.DSMd = {k: v for k, v in self.dsm.items() if not k.startswith("_") and "__" not in k}
        self.days_all = sorted({s.split("_")[0] for s in self.stops_all})
        self.record = {k: J(f"{C}/{n}") for k, n in (("ct", "census_truth.json"), ("led", "ledger.json"), ("gr", "growth.json"))}
        self._sd = {}

    def nf_under(self, rule, nf=None, primary=None):
        """noise_floor record as it would read if `rule` governed the stop set: T3_adopted and T7 from exclusion_variants[rule]
        (the primary set's T7 where the variant's is not decided, as that record states)."""
        nf = self.nf if nf is None else nf
        if rule == "registration_ok":
            return nf
        v = nf["exclusion_variants"][rule]; out = copy.deepcopy(nf)
        out["T3_adopted"] = copy.deepcopy(v["T3_adopted"])
        out["T7"] = copy.deepcopy(v["T7"]) if v["T7"].get("decided") else copy.deepcopy(nf["T7"])
        out["_js2_note"] = f"T3_adopted / T7 taken from exclusion_variants['{rule}'] (in memory; noise_floor.json is not rewritten)"
        return out

    def thresholds(self, rule):
        """r_assoc, g, g_laser, method, r_source, g_source exactly as mustc_ledger.main() derives them, with the noise-floor
        record read under `rule` (lines 1148-1171 of mustc_ledger.py, 11 Nov)."""
        nf = self.nf_under(rule); nf198 = self.nf_under(rule, nf=self.nf198) if self.plot != "198" else nf
        nf_thr, thr_plot = nf, self.plot
        if not nf.get("T3_adopted", {}).get("230613", {}).get("decidable") and self.plot != "198":
            nf_thr = nf198; thr_plot = "198"
        method = nf.get("T7", {}).get("method") or "a"
        t3 = nf_thr.get("T3_adopted", {}).get("230613", {})
        laser_pool = self.assoc["same_day_floor"]["pooled_growth_dates_cap_0.15"]
        if t3.get("decidable"):
            r_assoc = float(t3["r_assoc_m"]); r_source = {"source": f"plot {thr_plot} noise_floor.json T3_adopted['230613'].r_assoc_m", "method": t3["method"], "p95_measured_m": t3["r_assoc_p95_measured_m"], "n_matched_plants": t3["n_matched_plants"],
                                                          "measured_at_match_radius_m": t3.get("match_radius_m"), "censored_by_10pct_clause": t3.get("censored_by_10pct_clause"), "uncensored_p95_by_radius_m": t3.get("position_p95_by_radius_m"), "association_verdict": t3.get("association_verdict")}
        else:
            r_assoc = min(float(laser_pool["dist_p95_m"]), CAP_R); r_source = {"source": "laser same-day p95 (lmi_assoc same_day_floor pooled cap 0.15), capped at 0.095", "p95_measured_m": laser_pool["dist_p95_m"]}
        r_source["laser_uncensored_same_day_p95_m_cap_0.15"] = laser_pool["dist_p95_m"]
        if t3.get("decidable"):
            g = float(t3["g_m"]); g_source = {"source": f"plot {thr_plot} noise_floor.json T3_adopted['230613'].g_m", "method": t3["method"], "p95_measured_m": t3["g_p95_measured_m"], "capped_at_0.08": t3["g_capped_at_0.08"]}
        else:
            g = min(float(laser_pool["absdh95_p95_m"]), CAP_G); g_source = {"source": "laser same-day |dh95| p95 (pooled cap 0.15), capped at 0.08"}
        g_laser = float(laser_pool["absdh95_p95_m"]); g_source["laser_g_m"] = g_laser; g_source["laser_g_source"] = "lmi_assoc same_day_floor pooled_growth_dates_cap_0.15 absdh95_p95_m"
        t3_601 = nf.get("T3_adopted", {}).get("230601", {})
        g_source["230601_own_p95_absdh95_m"] = t3_601.get("g_p95_measured_m"); g_source["230601_flags"] = t3_601.get("T3_flags")
        return {"r_assoc": r_assoc, "g": g, "g_laser": g_laser, "method": method, "r_source": r_source, "g_source": g_source, "nf": nf}

    def sameday(self, rule, r, method):
        """Same-day splat floor on the stops entering under `rule` at radius r: rate and counts per day."""
        key = (rule, round(r, 6), method)
        if key not in self._sd:
            st = sdf.select_stops(self.stops_all, self.status, rule)
            dd, _ = sdf.compute_days(st, self.days_all, r, 0.08, self.DSMd, verbose=False)
            self._sd[key] = {d: {"rate": dd[d]["pooled"][method]["false_change_rate"], "n_unmatched": dd[d]["pooled"][method]["n_unmatched_total"], "n_in_overlap": dd[d]["pooled"][method]["n_in_overlap_total"],
                                 "n_matched": dd[d]["pooled"][method]["position_diff"].get("n", 0), "stops": dd[d]["stops"]} for d in dd}
        return self._sd[key]

    def run(self, rule, thr=None, r=None, g=None, marks=True):
        """One ledger run (mustc_ledger.compute) under the stop-exclusion `rule`.  thr: thresholds dict (default: derived under
        `rule`); r / g override it (item 32 line (a): the registered values with the stricter stop set)."""
        thr = dict(thr or self.thresholds(rule))
        if r is not None:
            thr["r_assoc"] = r
        if g is not None:
            thr["g"] = g
        ctx = {"truth": self.truth, "assoc": self.assoc, "nf": thr["nf"], "xs": self.xs, "dsm": self.dsm, "REG": self.REG, "status": self.status, "F": self.F, "inputs_mtime": self.inputs_mtime}
        sd = self.sameday(rule, thr["r_assoc"], thr["method"])
        orig = ML.load_marks
        if not marks:   # marks withheld (in memory): which cells keep their value without any marks file
            ML.load_marks = lambda *a, **k: (None, None)
        try:
            res = ML.compute(self.P, self.plot, self.C, thr["r_assoc"], thr["g"], thr["g_laser"], thr["method"], rule, "reference_stop", ctx, {d: v["rate"] for d, v in sd.items()},
                             stake_excl=True, r_source=thr["r_source"], g_source=thr["g_source"], want_fig=False)
        finally:
            ML.load_marks = orig
        res["thr"] = {k: thr[k] for k in ("r_assoc", "g", "g_laser", "method")}; res["sameday"] = sd; res["rule"] = rule
        return res


# ---------------------------------------------------------------------------------------------------------------- changed clauses
def item30_clause(spec, sd_counts, sensor):
    """Item 30 as written.  spec: the ledger's specificity record; sd_counts: counts behind the same-day rate (or None)."""
    if not spec or not spec.get("n_on_row_unchanged_candidates"):
        return {"verdict": "not evaluable (no unchanged on-row candidates)", "sensor": sensor}
    n, k, sd = spec["n_on_row_unchanged_candidates"], spec["n_removed_called"], spec.get("same_day_false_change_rate")
    rec = {"sensor": sensor, "n_candidates": n, "n_removed_called": k, "false_removal_rate": k / n, "false_removal_wilson95": wilson(k, n), "false_removal_clopper_pearson95": clopper(k, n),
           "same_day_rate": sd, "same_day_counts": sd_counts, "bound_as_written": (sd * n if sd is not None else None), "met_as_written": spec.get("within_expected"),
           "interval_note": "binomial intervals; the plants share a registration and a row structure, so the true uncertainty is wider"}
    if sd_counts and sd_counts.get("n_in"):
        rec["same_day_wilson95"] = wilson(sd_counts["n_unmatched"], sd_counts["n_in"])
    if sd is None:
        rec["verdict"] = "not evaluable (same-day rate not measurable: no overlapping same-day pair under this stop set)"
    elif sd > ITEM30_RATE:
        rec["verdict"] = "no verdict (same-day rate > 0.5)"
    else:
        rec["verdict"] = "met" if k <= sd * n else "not met"
    # the other reading of "the rate" (the false-removal rate itself); beside, not the column
    rec["other_reading_rate_is_false_removal_rate"] = ("no verdict (false-removal rate > 0.5)" if k / n > ITEM30_RATE else ("not evaluable" if sd is None else ("met" if k <= sd * n else "not met")))
    return rec


def laser_sd_counts(assoc, d):
    v = (assoc["same_day_floor"].get(d) or {}).get("by_cap", {}).get("0.15")
    if not v:
        p = assoc["same_day_floor"]["pooled_growth_dates_cap_0.15"]
        return {"n_unmatched": None, "n_in": p.get("n_interior_pooled"), "note": "pooled record; unmatched count not stored"}
    return {"n_unmatched": int(sum(v["n_unmatched_interior"].values())), "n_in": int(sum(v["n_interior"].values())), "note": "two scanners pooled (the rate of record is the mean of the two scanners' fractions)"}


def item31_t7(site, res):
    """Item 31 as written: per stop, per-plant |dh95| p95 against the M2d per-plant |diff| p95 of the pair laser_<laser date>_vs_<reference>."""
    t7 = res["gates"]["T7"]; xs = site.xs if site.xs is not None else site.xs198
    floors = {p["pair"]: p["per_plant"]["registered"] for p in xs["pairs"]}
    per = {}
    for s, v in t7.get("per_stop", {}).items():
        ld = (site.dsm.get(s, {}).get("vs_laser") or {}).get("laser_date")
        ref = (site.REG.get(s, {}).get("reference") or {})
        pair = f"laser_{ld}_vs_{ref.get('kind')}_{ref.get('date')}"
        fl = floors.get(pair)
        p95 = v.get("per_plant_absdiff_p95_m")
        per[s] = {"per_plant_n": v.get("per_plant_n"), "per_plant_absdiff_p95_m": p95, "per_plant_diff_median_m": v.get("per_plant_diff_median_m"), "m2d_pair": pair,
                  "m2d_per_plant_abs_diff_p95_m": (fl or {}).get("abs_diff_p95_m"), "m2d_per_plant_diff_median_m": (fl or {}).get("diff_median_m"), "m2d_n_plants": (fl or {}).get("n_plants"),
                  "within_floor": (bool(p95 <= fl["abs_diff_p95_m"]) if (fl and p95 is not None) else None)}
    ev = {s: v for s, v in per.items() if v["within_floor"] is not None}
    n_ok = sum(1 for v in ev.values() if v["within_floor"])
    not_ev = sorted(set(res["led"]["stops_entering"]) - set(ev))
    out = {"per_stop": per, "n_evaluated": len(ev), "n_within_floor": n_ok, "stops_not_evaluable": not_ev,
           "not_evaluable_reason": "no laser date for the stop's rig day (230628), or no M2d pair for the stop's laser date and reference",
           "floor_source": "own cross_sensor_floor.json" if site.xs is not None else "plot 198 cross_sensor_floor.json (this plot has no M2d of its own): flagged",
           "bias_median_range_m": ([min(v["per_plant_diff_median_m"] for v in ev.values()), max(v["per_plant_diff_median_m"] for v in ev.values())] if ev else None)}
    if not ev:
        out["verdict"] = "undecidable (no stop evaluable)"
    else:
        out["verdict"] = "pass" if n_ok == len(ev) else "fail"
        if site.xs is None:
            out["verdict"] = "not evaluable on its own floor; against plot 198's M2d: " + out["verdict"]
    return out


def item33_t3(t3, laser_pool):
    """Item 33 as written: verdict from the 0.15 m radius alone; 0.20 m beside."""
    pr = t3.get("position_p95_by_radius_m") or {}; nr = t3.get("n_matched_by_radius") or {}
    p15, p20 = pr.get(ITEM33_RADIUS), pr.get("0.200")
    rec = {"p95_at_0.15_m": p15, "n_at_0.15": nr.get(ITEM33_RADIUS), "p95_at_0.20_m": p20, "n_at_0.20": nr.get("0.200"), "p95_at_0.10_m": pr.get("0.100"), "n_at_0.10": nr.get("0.100"), "cap_m": CAP_R,
           "laser_pooled_p95_cap_0.15_m": laser_pool["dist_p95_m"], "laser_n_matched": laser_pool["n_matched"], "laser_verdict_same_rule": "resolvable" if laser_pool["dist_p95_m"] <= CAP_R else "not resolvable"}
    if not t3.get("decidable"):
        rec["verdict"] = "undecidable (fewer than 5 matched same-day plants at 0.10 m)"
    elif p15 is None:
        rec["verdict"] = "undecidable (no measurement at 0.15 m)"
    else:
        rec["verdict"] = "resolvable" if p15 <= CAP_R else "not resolvable"
    return rec


def word(v):
    v = str(v or "")
    for w in ("not resolvable", "radius-dependent", "resolvable", "undecidable", "not evaluable", "no verdict", "not met", "met", "pass", "fail", "not defined", "not computed"):
        if v.startswith(w):
            return w
    return v[:40]


def cm(x, nd=1):
    return "n/a" if x is None else f"{x * 100:.{nd}f} cm"


def frac(k, n):
    return f"{k} of {n}"


# ---------------------------------------------------------------------------------------------------------------- cells
def cells(site, res, items):
    """One column of the table: {row: {verdict, basis, ...numbers}} from one compute() result and the set of items applied."""
    G = res["gates"]; led, ct, gr = res["led"], res["ct"], res["gr"]; P = site.P; out = {}
    thr = res["thr"]; laser_pool = site.assoc["same_day_floor"]["pooled_growth_dates_cap_0.15"]
    out["_settings"] = {"exclusion": res["rule"], "stops_entering": led["stops_entering"], "r_assoc_m": thr["r_assoc"], "g_m": thr["g"], "g_laser_m": thr["g_laser"], "method": thr["method"], "items_applied": sorted(items),
                        "r_assoc_source": led["rules"]["r_assoc_source"].get("source"), "g_source": led["rules"]["g_source"].get("source")}
    t1 = G["T1"]
    out["T1 reprojection <= 0.70 px"] = {"verdict": word(t1.get("verdict")), "basis": f"{frac(t1.get('n_pass'), t1.get('n_stops'))} stops, {t1.get('min_px'):.2f}-{t1.get('max_px'):.2f} px", "n_pass": t1.get("n_pass"), "n": t1.get("n_stops")}
    t2 = G["T2"]; n_ok = sum(b["n_registration_ok"] for b in t2["by_day"].values()); n_all = sum(b["n"] for b in t2["by_day"].values()); n_pp = sum(1 for v in t2["per_stop"].values() if v["registration_ok"] and v["plant_pattern_pass"])
    out["T2 check (soil residual, yaw)"] = {"verdict": frac(n_ok, n_all), "basis": f"{frac(n_ok, n_all)} stops pass the T2 check; {frac(n_pp, n_all)} also pass the plant-pattern check", "n_pass": n_ok, "n": n_all, "n_also_plant_pattern": n_pp}
    out["T2 stops entering the ledger"] = {"verdict": frac(len(led["stops_entering"]), n_all), "basis": f"{frac(len(led['stops_entering']), n_all)} under '{res['rule']}'; per day " + ", ".join(f"{d} {sum(1 for s in led['stops_entering'] if s.startswith(d))}" for d in P["rig_days"]), "n": len(led["stops_entering"])}
    n_flag = sum(b["n_scale_flag"] for b in t2["by_day"].values())
    out["T2 scale flag |s-1| > 1 %"] = {"verdict": f"{frac(n_flag, n_all)} flagged", "basis": "days with every stop flagged: " + (", ".join(sorted(d for d, b in t2["by_day"].items() if b["n_scale_flag"] == b["n"])) or "none"), "n_flag": n_flag, "n": n_all}
    # T3
    t3g = G["T3"]; t3 = t3g.get("noise_floor_T3_adopted_230613") or {}; pr = t3.get("position_p95_by_radius_m") or {}; nr = t3.get("n_matched_by_radius") or {}
    basis3 = (f"230613 same-day position p95 {cm(pr.get('0.100'))} at 0.10 m (n {nr.get('0.100')}), {cm(pr.get('0.150'))} at 0.15 m (n {nr.get('0.150')}), {cm(pr.get('0.200'))} at 0.20 m (n {nr.get('0.200')}); cap 9.5 cm"
              if t3.get("decidable") else f"230613: {t3.get('reason') or 'not decidable'} ({t3.get('n_matched_plants', 0)} matched plants)")
    if "33" in items:
        r33 = item33_t3(t3, laser_pool)
        out["T3 association resolvable at the 9.5 cm cap"] = {"verdict": word(r33["verdict"]), "basis": basis3 + "; read at 0.15 m alone", "item33": r33}
    else:
        out["T3 association resolvable at the 9.5 cm cap"] = {"verdict": word(t3g.get("verdict")), "basis": basis3 + "; read at 0.15 and 0.20 m together", "verdict_text": t3g.get("verdict")}
    out["T3 laser ceiling, same clause (beside)"] = {"verdict": "resolvable" if laser_pool["dist_p95_m"] <= CAP_R else "not resolvable", "basis": f"laser two-scanner same-day position p95 {cm(laser_pool['dist_p95_m'])} at the 0.15 m cap (n {laser_pool['n_matched']}, pooled 230601/13/21)"}
    gv = t3g.get("g_verdict") or ""
    out["T3 growth threshold g <= 8 cm"] = {"verdict": "pass" if gv.startswith("pass") else ("fail" if t3.get("g_p95_measured_m") is not None else "undecidable"),
                                             "basis": (f"230613 same-day |dh95| p95 {cm(t3.get('g_p95_measured_m'))} (n {t3.get('n_matched_plants')}); g in force {cm(thr['g'])}" if t3.get("g_p95_measured_m") is not None
                                                       else f"230613 same-day floor not measurable under this stop set; g in force {cm(thr['g'])} ({led['rules']['g_source'].get('source')})")}
    t3b = t3g.get("noise_floor_T3_adopted_230601") or {}
    if t3b:
        out["T3 flag on growth from 230601"] = {"verdict": "flagged" if t3b.get("g_capped_at_0.08") else ("not flagged" if t3b.get("decidable") else "undecidable"),
                                                 "basis": f"230601 own same-day |dh95| p95 {cm(t3b.get('g_p95_measured_m'))} (n {t3b.get('n_matched_plants')}) against 8 cm"}
    # T4
    for d in P["t4_dates"]:
        v = ct["T4"].get(d)
        if isinstance(v, dict) and "ratio" in v:
            out[f"T4 {d} splat recall >= 0.9 x laser recall"] = {"verdict": ct["T4"]["verdict"].get(d), "provisional": bool(v.get("provisional_marks")), "ratio": v["ratio"],
                                                                 "basis": f"ratio {v['ratio']:.3f}: splat {frac(v['splat_n_matched'], v['n_truth_marks_in_footprint'])} marks, laser {frac(v['laser_n_matched'], v['n_truth_marks_in_footprint'])}" + ("; PROVISIONAL marks" if v.get("provisional_marks") else "")}
        else:
            out[f"T4 {d} splat recall >= 0.9 x laser recall"] = {"verdict": word(ct["T4"]["verdict"].get(d)), "basis": str(ct["T4"]["verdict"].get(d))}
    # T5
    tk = "T5_second_site" if "T5_second_site" in G else "T5"; T5 = G[tk]; name = T5.get("polygon", "")
    lab = f"T5 {name}" if name else "T5"
    if "clause_removals_restated_4nov" in T5:
        cr, c9 = T5["clause_removals_restated_4nov"], T5["clause_removals_preregistered_ge_9"]
        bsr = (led.get("polygons", {}).get(name, {}) or {}).get("both_sensors_recorded") or {}
        cov = bsr.get("n_laser_inside_in_splat_imaged_region_k1")
        cov_txt = "" if cov is None else f"; the later day's stops entering image {cov} of the {T5['n_laser_inside_k']} laser plants inside" + (" (NO COVERAGE of the polygon: not a test of the change rule)" if cov == 0 else "")
        out[f"{lab} removal clause (restated 4 Nov)"] = {"verdict": "met" if cr["met"] else "not met", "value": cr["value"], "bar": cr["bar"], "n_laser_inside_in_splat_region_k1": cov,
                                                         "basis": f"splat calls {cr['value']} removed of the {T5['n_both_sensors_recorded']} plants both sensors record inside; bar = the laser's {cr['bar']} of {T5['n_laser_inside_k']}; truth {T5['truth_n_removed']} removed" + cov_txt}
        out[f"{lab} removal clause, pre-registered >= 9 (beside)"] = {"verdict": "met" if c9["met"] else "not met", "basis": f"{c9['value']} against 9", "value": c9["value"]}
        k1s, k1l = T5["dates_splat"][1], T5["dates_laser"][1]
        sp_s, sp_l = T5.get("specificity_splat"), T5.get("specificity_laser")
        sd_s = res["sameday"].get(k1s); sdc_s = {"n_unmatched": sd_s["n_unmatched"], "n_in": sd_s["n_in_overlap"], "n_matched": sd_s["n_matched"]} if sd_s else None
        r30s = item30_clause(sp_s, sdc_s, "splat"); r30l = item30_clause(sp_l, laser_sd_counts(site.assoc, k1l), "laser")

        def spec_basis(sp, r30, with_ci):
            if not sp or not sp.get("n_on_row_unchanged_candidates"):
                return "no candidates"
            b = f"{frac(sp['n_removed_called'], sp['n_on_row_unchanged_candidates'])} unchanged on-row plants called removed"
            if sp.get("same_day_false_change_rate") is None:
                b += "; same-day rate not measurable"
            else:
                b += f"; bound {sp['expected_false_removals']:.1f} at the same-day rate {sp['same_day_false_change_rate']:.2f}"
                c = r30.get("same_day_counts") or {}
                if c.get("n_unmatched") is not None:
                    b += f" ({frac(c['n_unmatched'], c['n_in'])})"
            if with_ci:
                w = r30["false_removal_wilson95"]; b += f"; false-removal rate {r30['false_removal_rate']:.2f} (95 % interval {w[0]:.2f}-{w[1]:.2f})"
                if r30.get("same_day_wilson95"):
                    w2 = r30["same_day_wilson95"]; b += f", same-day rate interval {w2[0]:.2f}-{w2[1]:.2f}"
            return b
        if "30" in items:
            out[f"{lab} specificity clause, splat"] = {"verdict": word(r30s["verdict"]), "basis": spec_basis(sp_s, r30s, True), "item30": r30s}
            out[f"{lab} specificity, laser ceiling (beside; not in the verdict)"] = {"verdict": word(r30l["verdict"]), "basis": spec_basis(sp_l, r30l, True), "item30": r30l}
            judged = [cr["met"]] + ([r30s["verdict"] == "met"] if r30s["verdict"] in ("met", "not met") else [])
            ov = "fail" if not all(judged) else "pass"
            note = f", specificity clause {r30s['verdict']}" if r30s["verdict"] in ("met", "not met") else f"; specificity clause not judged ({word(r30s['verdict'])})"
            out[f"{lab} overall"] = {"verdict": ov, "basis": f"removal clause {'met' if cr['met'] else 'not met'} ({cr['value']} against {cr['bar']}){note}" + ("; detection-limited (fewer plants recorded by both than the bar)" if T5["n_both_sensors_recorded"] < cr["bar"] else "")}
        else:
            vs = ("met" if sp_s.get("within_expected") else "not met") if (sp_s and sp_s.get("within_expected") is not None) else "not evaluable"
            vl = ("met" if sp_l.get("within_expected") else "not met") if (sp_l and sp_l.get("within_expected") is not None) else "not evaluable"
            out[f"{lab} specificity clause, splat"] = {"verdict": vs, "basis": spec_basis(sp_s, r30s, False)}
            out[f"{lab} specificity, laser ceiling (beside; not in the verdict)"] = {"verdict": vl, "basis": spec_basis(sp_l, r30l, False)}
            out[f"{lab} overall"] = {"verdict": word(T5.get("verdict")), "basis": f"removal clause {'met' if cr['met'] else 'not met'} ({cr['value']} against {cr['bar']}), specificity clause {vs}" + ("; detection-limited (fewer plants recorded by both than the bar)" if T5["n_both_sensors_recorded"] < cr["bar"] else ""), "verdict_text": T5.get("verdict")}
    else:
        for r_ in ("removal clause (restated 4 Nov)", "removal clause, pre-registered >= 9 (beside)", "specificity clause, splat", "specificity, laser ceiling (beside; not in the verdict)", "overall"):
            out[f"{lab or 'T5'} {r_}"] = {"verdict": word(T5.get("verdict")), "basis": str(T5.get("verdict"))}
    # T6
    for key, t in G["T6"].items():
        k, k1 = key.split("->")
        if k1 not in P["laser_of_rig_day"]:
            out[f"T6 {key}"] = {"verdict": "not evaluable", "basis": "no laser within 13 days of the 230628 stops (design: against UAV2 only)" if site.plot == "198" else str(t.get("verdict"))}; continue
        if not t.get("n"):
            out[f"T6 {key}"] = {"verdict": word(t.get("verdict")), "basis": str(t.get("verdict"))}; continue
        n = t["n"]; ag = t.get("agreement_frac_signed_corrected", t["agreement_frac"]); med = t.get("median_abs_diff_minus_bound_m", t["median_abs_diff_m"])
        b = f"n {n} plants paired in both sensors; median |dh splat - dh laser| {cm(t['median_abs_diff_m'])}"
        if "median_abs_diff_minus_bound_m" in t:
            b += f" ({cm(med)} with the M2e bound subtracted)"
        b += f" against g {cm(t['g_m'])}; calls agree {frac(int(round(ag * n)), n)} (need 0.9)"
        if t.get("flag"):
            b += "; flagged by the T3 clause on 230601"
        if site.plot != "198":
            b += "; not a T6 interval of the design (second site, g capped)"
        out[f"T6 {key}"] = {"verdict": word(t.get("verdict")), "basis": b, "n": n, "median_m": med, "agreement": ag}
    for key, rec in gr.get("long_span", {}).items():
        t = rec.get("T6_like")
        if t and t.get("n"):
            n = t["n"]; ag = t.get("agreement_frac_signed_corrected", t["agreement_frac"])
            out[f"T6 long span {key} (beside; not a T6 interval)"] = {"verdict": word(t.get("verdict")), "basis": f"n {n}; median {cm(t['median_abs_diff_m'])}" + (f" ({cm(t['median_abs_diff_minus_bound_m'])} with the bound)" if "median_abs_diff_minus_bound_m" in t else "") + f"; calls agree {frac(int(round(ag * n)), n)}"}
        else:
            out[f"T6 long span {key} (beside; not a T6 interval)"] = {"verdict": "undecidable", "basis": str(rec.get("error") or (t or {}).get("verdict"))}
    # T7
    t7 = G["T7"]
    if t7.get("n_stops"):
        ps = t7["per_stop"]; p95s = [v["per_plant_absdiff_p95_m"] for v in ps.values() if v["per_plant_absdiff_p95_m"] is not None]; meds = [v["per_plant_diff_median_m"] for v in ps.values() if v["per_plant_diff_median_m"] is not None]
        out["T7 soil clause (after the M2d offset is removed; by construction)"] = {"verdict": (frac(t7["n_soil_within_m2d"], t7["n_stops_with_m2d_floor"]) if t7["n_stops_with_m2d_floor"] else "not evaluable"),
                                                                                   "basis": (f"soil bias within the M2d soil floor on {frac(t7['n_soil_within_m2d'], t7['n_stops_with_m2d_floor'])} stops" if t7["n_stops_with_m2d_floor"] else "no M2d measured for this plot")}
        rng = f"per-plant |dh95| p95 {min(p95s) * 100:.1f}-{max(p95s) * 100:.1f} cm; median difference {min(meds) * 100:.1f} to {max(meds) * 100:.1f} cm"
        n_noev = len(led["stops_entering"]) - t7["n_stops"]
        if "31" in items:
            r31 = item31_t7(site, res)
            fl = sorted({(v["m2d_pair"], v["m2d_per_plant_abs_diff_p95_m"]) for v in r31["per_stop"].values() if v["m2d_per_plant_abs_diff_p95_m"] is not None})
            out["T7 plant clause"] = {"verdict": r31["verdict"], "basis": f"{frac(r31['n_within_floor'], r31['n_evaluated'])} stops at or below the M2d per-plant floor (" + ", ".join(f"{p.split('_vs_')[0][6:]}: {f * 100:.1f} cm" for p, f in fl) + f"); {rng}" + (f"; {len(r31['stops_not_evaluable'])} entering stop(s) not evaluable" if r31["stops_not_evaluable"] else ""), "item31": r31}
        else:
            out["T7 plant clause"] = {"verdict": word(t7.get("verdict")), "basis": f"{frac(t7['n_per_plant_p95_le_2g'], t7['n_stops'])} stops with p95 <= 2 g = {t7['two_g_m'] * 100:.1f} cm; {rng}" + (f"; {n_noev} entering stop(s) without a laser date not evaluable" if n_noev else ""),
                                      "n_pass": t7["n_per_plant_p95_le_2g"], "n": t7["n_stops"], "two_g_m": t7["two_g_m"]}
    else:
        out["T7 soil clause (after the M2d offset is removed; by construction)"] = {"verdict": "undecidable", "basis": str(t7.get("verdict"))}
        out["T7 plant clause"] = {"verdict": "undecidable", "basis": str(t7.get("verdict"))}
    return out


def write_json(p, o):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(ML.to_py(o), f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", choices=["198", "178"], default="198"); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--item30", action="store_true"); ap.add_argument("--item31", action="store_true"); ap.add_argument("--item32", action="store_true"); ap.add_argument("--item33", action="store_true")
    ap.add_argument("--all-columns", action="store_true", help="the seven columns of the design, the faithfulness check, item 32 line (a) and the marks-withheld run")
    a = ap.parse_args()
    for p_ in (ML.PLOT_CFG["198"]["census"], ML.PLOT_CFG["178"]["census"]):   # never write into a census directory itself
        assert os.path.abspath(a.out_dir) != os.path.abspath(p_) and "sensitivity_20261112" in os.path.abspath(a.out_dir), "out-dir must be the sensitivity directory"
    os.makedirs(a.out_dir, exist_ok=True)
    site = Site(a.plot); t0 = time.time()
    if not a.all_columns:
        items = {k for k, f in (("30", a.item30), ("31", a.item31), ("32", a.item32), ("33", a.item33)) if f}
        res = site.run("plant_pattern" if "32" in items else "registration_ok")
        col = cells(site, res, items)
        write_json(f"{a.out_dir}/column_{'_'.join(sorted(items)) or 'as_registered'}_plot{a.plot}.json", {"label": LABEL, "column": col})
        print(json.dumps(ML.to_py({k: (v.get("verdict"), v.get("basis")) for k, v in col.items() if not k.startswith("_")}), indent=1)); return
    # ---- runs
    reg = site.run("registration_ok")
    pp_b = site.run("plant_pattern")                                                  # item 32, line (b): thresholds re-derived
    pp_a = site.run("plant_pattern", thr=site.thresholds("registration_ok"))          # item 32, line (a): registered thresholds, stricter stop set
    nomarks = site.run("registration_ok", marks=False)
    for name, res in (("as_registered", reg), ("item32_b_thresholds_rederived", pp_b), ("item32_a_registered_thresholds", pp_a), ("marks_withheld", nomarks)):
        for key, fn in (("ct", "census_truth.json"), ("led", "ledger.json"), ("gr", "growth.json")):
            rec = dict(res[key]); rec["_js2"] = {"label": LABEL, "run": name, "thresholds": res["thr"], "sameday_counts": res["sameday"]}
            write_json(f"{a.out_dir}/runs/{name}/{fn}", rec)
    # ---- faithfulness: the as-registered rerun against the record
    faith = {}
    for key in ("ct", "led", "gr"):
        mine = json.loads(json.dumps(ML.to_py(reg[key]["gates_T1_T7"]))); rec = site.record[key]["gates_T1_T7"]
        faith[key] = jdiff(mine, rec)
    led_extra = jdiff(json.loads(json.dumps(ML.to_py({k: reg["led"][k] for k in ("pairs", "polygons", "tracks", "stops_entering")}))), {k: site.record["led"][k] for k in ("pairs", "polygons", "tracks", "stops_entering")})
    stored_pp = (site.record["led"].get("sensitivity") or {}).get("exclusion", {}).get("plant_pattern")
    line_a = jdiff(json.loads(json.dumps(ML.to_py(ML.summarise(pp_a)))), stored_pp) if stored_pp else ["no stored sensitivity.exclusion.plant_pattern"]
    # ---- marks: which gates keep their value when the marks are withheld
    g1 = json.loads(json.dumps(ML.to_py(reg["gates"]))); g2 = json.loads(json.dumps(ML.to_py(nomarks["gates"])))
    marks_dep = {k: jdiff(g1[k], g2[k], limit=8) for k in g1}
    # ---- columns
    cols = {"as registered": cells(site, reg, set()), "item 30": cells(site, reg, {"30"}), "item 31": cells(site, reg, {"31"}), "item 32": cells(site, pp_b, {"32"}), "item 33": cells(site, reg, {"33"}),
            "all four (a: item 32 = T2 check)": cells(site, reg, {"30", "31", "33"}), "all four (b: item 32 = T2 and plant-pattern)": cells(site, pp_b, {"30", "31", "32", "33"})}
    beside = {"item 32 line (a): stricter stop set, registered thresholds": cells(site, pp_a, {"32"}), "marks withheld (as registered otherwise)": cells(site, nomarks, set())}
    rows = []
    for c in list(cols.values()) + list(beside.values()):
        for r_ in c:
            if not r_.startswith("_") and r_ not in rows:
                rows.append(r_)
    order = lambda r_: (int(re.match(r"T(\d)", r_).group(1)), rows.index(r_))
    rows = sorted(rows, key=order)
    moved = {cn: {r_: {"moves_verdict": (c.get(r_, {}).get("verdict") != cols["as registered"].get(r_, {}).get("verdict")), "moves_basis": (c.get(r_, {}).get("basis") != cols["as registered"].get(r_, {}).get("basis"))} for r_ in rows} for cn, c in cols.items() if cn != "as registered"}
    marks_rows = {r_: {"verdict_changes_without_marks": cols["as registered"].get(r_, {}).get("verdict") != beside["marks withheld (as registered otherwise)"].get(r_, {}).get("verdict"),
                       "basis_changes_without_marks": cols["as registered"].get(r_, {}).get("basis") != beside["marks withheld (as registered otherwise)"].get(r_, {}).get("basis")} for r_ in rows}
    rec_sens_r = {k: {"T4": v.get("T4"), "T4_verdict": v.get("T4_verdict"), "T5": v.get("T5"), "T6": {p: {q: w for q, w in x.items() if q in ("n", "median_abs_diff_m", "agreement_frac", "verdict", "median_abs_diff_minus_bound_m", "agreement_frac_signed_corrected")} for p, x in (v.get("T6") or {}).items()}}
                  for k, v in ((site.record["led"].get("sensitivity") or {}).get("r_assoc") or {}).items()}
    table = {"label": LABEL, "plot": a.plot, "created_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "script": os.path.abspath(__file__), "design": "automation/mustc_sensitivity/DESIGN_20261112.md",
             "record_created": {k: site.record[k].get("created") for k in site.record}, "rows": rows, "columns": cols, "beside": beside, "moved": moved,
             "faithfulness_as_registered_vs_record": {"gates_diffs": faith, "ledger_pairs_polygons_tracks_diffs": led_extra, "n_diffs": sum(len(v) for v in faith.values()) + len(led_extra)},
             "item32_line_a_vs_stored_sensitivity": {"diffs": line_a, "n_diffs": len(line_a)}, "marks_withheld_gate_diffs": marks_dep, "marks_rows": marks_rows,
             "record_r_assoc_sensitivity_quoted_beside_item33": rec_sens_r, "runtime_s": round(time.time() - t0, 1)}
    write_json(f"{a.out_dir}/sensitivity_table_plot{a.plot}.json", table)
    # ---- markdown
    cn = list(cols)
    L = [f"### Plot {a.plot}: {LABEL}", "", "| test / clause | " + " | ".join(cn) + " |", "|---|" + "---|" * len(cn)]
    for r_ in rows:
        L.append(f"| {r_} | " + " | ".join((f"**{c[r_]['verdict']}**. {c[r_]['basis']}" if r_ in c else "n/a").replace("|", "/") for c in cols.values()) + " |")
    L += ["", "Settings per column: " + "; ".join(f"{k}: {len(c['_settings']['stops_entering'])} stops, r_assoc {c['_settings']['r_assoc_m'] * 100:.2f} cm, g {c['_settings']['g_m'] * 100:.2f} cm" for k, c in cols.items())]
    with open(f"{a.out_dir}/sensitivity_table_plot{a.plot}.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"plot {a.plot}: faithfulness diffs {table['faithfulness_as_registered_vs_record']['n_diffs']}; item 32 line (a) vs stored diffs {len(line_a)}; wrote {a.out_dir}/sensitivity_table_plot{a.plot}.json|md in {time.time() - t0:.1f} s")
    for k, v in faith.items():
        for d in v[:10]:
            print("  FAITH", k, d)
    for d in led_extra[:10]:
        print("  FAITH led", d)
    for d in line_a[:10]:
        print("  LINE-A", d)
    with open(f"{a.out_dir}/DONE_plot{a.plot}", "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S UTC\n", time.gmtime()))


if __name__ == "__main__":
    main()
