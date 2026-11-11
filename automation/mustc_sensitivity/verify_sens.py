#!/usr/bin/env python3
"""JS-2 (12 Nov 2026): independent recount of the sensitivity table (SENSITIVITY, NOT ADOPTED) from the per-item records.
Imports none of the MuST-C modules: own stop selection (raw fields of registration.json), own same-day floor (raster overlap,
minimum-filter erosion, brute-force mutual nearest neighbours), own per-day census, association, cross-sensor match, T4, T5, T6,
T7 and intervals.  Reads the finished records (read-only) and sensitivity_table_plot<plot>.json; writes verify_sens_plot<plot>.json
beside it and prints one line per comparison.

Box: /mnt/shared2/must-c/venv/bin/python verify_sens.py --plot 198|178 --sens-dir <sensitivity_20261112 dir>
"""
import argparse
import csv
import json
import os

import numpy as np
from matplotlib.path import Path
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument("--plot", default="198"); ap.add_argument("--sens-dir", required=True)
A = ap.parse_args()
PLOT = A.plot
C = f"/mnt/shared2/must-c/plot{PLOT}/census"; RES = f"/mnt/shared2/must-c/plot{PLOT}/results"; C198 = "/mnt/shared2/must-c/plot198/census"
J = lambda p: json.load(open(p))
CFG = {"198": dict(band=(357415.6, 357421.7, 5610200.3, 5610201.95), rig=("230601", "230613", "230621"), laser_of={"230601": "230601", "230613": "230613", "230621": "230621"}, marks={"230601": "230602", "230613": "230614"},
                   change=("SB8_020623", "230601", "230613", "230613"), growth=(("230601", "230613"), ("230613", "230621")), m2e_k1="230621", stake=None, tkey="T5"),
       "178": dict(band=(357415.7, 357421.65, 5610203.85, 5610205.75), rig=("230601", "230613", "230616"), laser_of={"230601": "230601", "230613": "230613"}, marks={},
                   change=("SB7_150623", "230613", "230616", "230621"), growth=(("230601", "230613"),), m2e_k1=None, stake=("230616", "230621"), tkey="T5_second_site")}[PLOT]
TAB = J(f"{A.sens_dir}/sensitivity_table_plot{PLOT}.json")
REG, TRUTH, ASSOC, DSM = J(f"{C}/registration.json"), J(f"{C}/truth.json"), J(f"{C}/lmi_assoc.json"), J(f"{C}/dsm_vs_laser.json")
XS = J(f"{C198}/cross_sensor_floor.json")
rows = []; out = {"label": "SENSITIVITY, NOT ADOPTED: independent recount", "plot": PLOT, "comparisons": []}


def cmp(name, mine, theirs, tol=1e-9):
    if isinstance(mine, float) or isinstance(theirs, float):
        ok = mine is not None and theirs is not None and abs(mine - theirs) <= tol
    else:
        ok = mine == theirs
    out["comparisons"].append({"what": name, "recount": mine, "table": theirs, "agree": bool(ok)})
    print(("OK   " if ok else "DIFF ") + name + f": recount {mine} | table {theirs}")


# ------------------------------------------------------------------------------------------------ frames
def frames():
    if PLOT == "198":
        return {k: (v["dE_m"], v["dN_m"]) for k, v in J(f"{C}/frames.json")["frames"].items() if v.get("dE_m") is not None}
    Pp = J(f"{C}/georef_prep/frames_prep_178.json"); fr = {"uav1_230602": (0.0, 0.0)}
    for k, v in Pp["ortho_pairs"].items():
        fr["uav1_" + k.split("->")[0]] = (v["grey_soil"]["dE_mm"] / 1e3, v["grey_soil"]["dN_mm"] / 1e3)
    for v in Pp["laser_to_ortho"].values():
        q = v["soil_texture"] if v["soil_texture"].get("accepted") else v["nn_two_pass"]; o = fr["uav1_" + v["ortho_date"]]
        fr["laser_" + v["laser_date"]] = (q["dE_mm"] / 1e3 + o[0], q["dN_mm"] / 1e3 + o[1])
    return fr


FR = frames()


# ------------------------------------------------------------------------------------------------ geometry helpers
def mnn(a, b, r):
    """mutual nearest neighbours within r: list of (i, j, d), brute force"""
    a = np.asarray(a, float).reshape(-1, 2); b = np.asarray(b, float).reshape(-1, 2)
    if not len(a) or not len(b):
        return []
    D = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    best_b = D.argmin(axis=1); best_a = D.argmin(axis=0)
    return [(i, int(best_b[i]), float(D[i, best_b[i]])) for i in range(len(a)) if best_a[best_b[i]] == i and D[i, best_b[i]] <= r]


def two_pass(a, b, r, minp=5):
    m = mnn(a, b, r)
    if len(m) >= minp:
        sh = np.median(np.array([b[j] - a[i] for i, j, _ in m]), axis=0); m = mnn(a + sh, b, r)
    return m


def seg_dist(P, X):
    P = np.asarray(P, float); X = np.asarray(X, float).reshape(-1, 2); best = np.full(len(X), np.inf)
    for i in range(len(P)):
        p, q = P[i], P[(i + 1) % len(P)]; v = q - p
        t = np.clip(((X - p) @ v) / max(float(v @ v), 1e-12), 0.0, 1.0)
        best = np.minimum(best, np.hypot(*(X - (p + t[:, None] * v)).T))
    return best


def in_dil(P, X, dil):
    X = np.asarray(X, float).reshape(-1, 2)
    if not len(X):
        return np.zeros(0, bool)
    return Path(np.asarray(P, float)).contains_points(X) | (seg_dist(P, X) <= dil)


def wilson(k, n, z=1.959963984540054):
    """Wilson score interval as the roots of (p_hat - p)^2 = z^2 p (1 - p) / n"""
    ph = k / n; a = 1 + z * z / n; b = -(2 * ph + z * z / n); c = ph * ph
    disc = np.sqrt(b * b - 4 * a * c)
    return [float((-b - disc) / (2 * a)), float((-b + disc) / (2 * a))]


# ------------------------------------------------------------------------------------------------ stops
STOPS = sorted(k for k in REG if not k.startswith("_") and "__" not in k and os.path.exists(f"{C}/splat_census/{k}__b.json"))


def t2_ok(s):
    v = REG[s]; cr = ((v.get("coarse") or {}).get("yaw_prior") or {}).get("candidate_score_ratio_second_over_first"); soil = (((v.get("fine") or {}).get("fixed_scale") or {}).get("soil_points_vs_ref") or {}).get("median_m")
    return bool(cr is not None and cr <= 0.7 and soil is not None and soil <= 0.02 and "hand-initialised" not in (v.get("flags") or []))


def stop_set(rule):
    return [s for s in STOPS if t2_ok(s) and (rule == "t2" or (REG[s].get("gate") or {}).get("pass") is True)]


_cache = {}


def census(s):
    if s not in _cache:
        S = J(f"{C}/splat_census/{s}__b.json"); fp = S["footprint_nview6_raster"]; z = np.load(fp["npz"])
        sh = np.array(FR.get(f"{S['reference'].get('sensor') or 'uav1'}_{S['reference']['date']}", (np.nan, np.nan)))
        I = S["instances"]
        _cache[s] = dict(xy_raw=np.array([[i["x_utm"], i["y_utm"]] for i in I]).reshape(-1, 2), h=np.array([i["h95_m"] for i in I]), in6=np.array([i["in_footprint_nview6"] for i in I], bool), row=np.array([i["on_row"] for i in I], bool),
                         nview6=z[fp["key"]], inpoly=z["in_polygon"], meta=[float(v) for v in fp["meta"]], shift=sh, soil_poly=np.array(S["footprint_soil_polygon_utm"]), date=S["date"])
    return _cache[s]


def lookup(R, meta, X):
    X = np.asarray(X, float).reshape(-1, 2); x0, y0, cell, nx, ny = meta
    ix = np.floor((X[:, 0] - x0) / cell).astype(int); iy = np.floor((X[:, 1] - y0) / cell).astype(int)
    ok = (ix >= 0) & (ix < int(nx)) & (iy >= 0) & (iy < int(ny)); o = np.zeros(len(X), bool); o[ok] = R[ix[ok], iy[ok]]
    return o


# ------------------------------------------------------------------------------------------------ same-day floor (own raster route)
def sameday(stops, day, radius, erode=0.08):
    ds = [s for s in stops if s.startswith(day)]; pos = []; dh = []; n_in = n_un = 0
    for a in range(len(ds)):
        for b in range(a + 1, len(ds)):
            Sa, Sb = census(ds[a]), census(ds[b]); ma_, mb_ = Sa["meta"], Sb["meta"]; cell = ma_[2]
            gx0 = min(ma_[0], mb_[0]); gy0 = min(ma_[1], mb_[1])
            gx1 = max(ma_[0] + ma_[3] * cell, mb_[0] + mb_[3] * cell); gy1 = max(ma_[1] + ma_[4] * cell, mb_[1] + mb_[4] * cell)
            nx, ny = int(round((gx1 - gx0) / cell)) + 2, int(round((gy1 - gy0) / cell)) + 2
            G = []
            for S_, m_ in ((Sa, ma_), (Sb, mb_)):
                g = np.zeros((nx, ny), bool); ox, oy = int(round((m_[0] - gx0) / cell)), int(round((m_[1] - gy0) / cell))
                g[ox:ox + S_["inpoly"].shape[0], oy:oy + S_["inpoly"].shape[1]] = S_["inpoly"]; G.append(g)
            ov = G[0] & G[1]
            if ov.sum() < 100:
                continue
            rr = int(round(erode / cell)); yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]; disk = (xx ** 2 + yy ** 2) <= rr * rr
            ove = ndimage.minimum_filter(ov.astype(np.uint8), footprint=disk, mode="constant", cval=0).astype(bool)
            if ove.sum() < 100:
                continue
            gm = (gx0, gy0, cell, nx, ny)
            ia = lookup(ove, gm, Sa["xy_raw"]); ib = lookup(ove, gm, Sb["xy_raw"])
            m = mnn(Sa["xy_raw"][ia], Sb["xy_raw"][ib], radius)
            pos += [d for _, _, d in m]; dh += [abs(float(Sa["h"][ia][i] - Sb["h"][ib][j])) for i, j, _ in m]
            n_in += int(ia.sum() + ib.sum()); n_un += int(ia.sum() + ib.sum() - 2 * len(m))
    return dict(n=len(pos), pos_p95=float(np.percentile(pos, 95)) if pos else None, dh_p95=float(np.percentile(dh, 95)) if dh else None, n_in=n_in, n_un=n_un, rate=(n_un / n_in if n_in else None))


# ------------------------------------------------------------------------------------------------ censuses in the experiment frame
STAKE = {d: v["ids"] for d, v in ASSOC.get("stake_rule", {}).get("per_date", {}).items()}


def laser(d):
    P = J(f"{C}/lmi_{d}.json")["plants"]; sh = np.array(FR[f"laser_{d}"]); b = CFG["band"]
    xy = np.array([[p["x"], p["y"]] for p in P]); ids = np.array([p["id"] for p in P]); stake = np.isin(ids, np.array(STAKE.get(d, []), int))
    inb = (xy[:, 0] >= b[0]) & (xy[:, 0] <= b[1]) & (xy[:, 1] >= b[2]) & (xy[:, 1] <= b[3]); keep = inb & ~stake
    box = (b[0] + sh[0], b[1] + sh[0], b[2] + sh[1], b[3] + sh[1])
    return dict(xy=(xy + sh)[keep], h=np.array([p["height_p95"] for p in P])[keep], row=np.array([p["row"] >= 0 for p in P])[keep], id=ids[keep], stake_xy=(xy + sh)[inb & stake],
                reg=lambda X, box=box: (lambda X_: (X_[:, 0] >= box[0]) & (X_[:, 0] <= box[1]) & (X_[:, 1] >= box[2]) & (X_[:, 1] <= box[3]))(np.asarray(X, float).reshape(-1, 2)))


def day_census(stops, d, r):
    ds = [s for s in stops if s.startswith(d)]
    if not ds:
        return None
    xy = []; h = []; row = []; own = []; dc = []
    for k, s in enumerate(ds):
        S = census(s); m = S["in6"]; p = S["xy_raw"][m] + S["shift"]
        xy.append(p); h.append(S["h"][m]); row.append(S["row"][m]); own += [k] * int(m.sum()); dc.append(np.hypot(*(p - (S["soil_poly"] + S["shift"]).mean(0)).T))
    xy = np.vstack(xy); h = np.concatenate(h); row = np.concatenate(row); own = np.array(own); dc = np.concatenate(dc)
    parent = list(range(len(xy)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for a in range(len(ds)):
        for b in range(a + 1, len(ds)):
            ia, ib = np.nonzero(own == a)[0], np.nonzero(own == b)[0]
            for i, j, _ in mnn(xy[ia], xy[ib], r):
                ra, rb = root(int(ia[i])), root(int(ib[j]))
                if ra != rb:
                    parent[rb] = ra
    groups = {}
    for i in range(len(xy)):
        groups.setdefault(root(i), []).append(i)
    G = list(groups.values())

    def reg(X, ds=ds):
        X = np.asarray(X, float).reshape(-1, 2); o = np.zeros(len(X), bool)
        for s in ds:
            S = census(s); o |= lookup(S["nview6"], S["meta"], X - S["shift"])
        return o
    return dict(xy=np.array([xy[g].mean(0) for g in G]).reshape(-1, 2), h=np.array([h[g[int(np.argmin(dc[g]))]] for g in G]), row=np.array([row[g].any() for g in G], bool), reg=reg, stops=ds)


def assoc(A, B, r):
    m = two_pass(A["xy"], B["xy"], r); paired = {i: j for i, j, _ in m}
    un = np.array([i for i in range(len(A["xy"])) if i not in paired], int)
    rem = set(un[B["reg"](A["xy"][un])].tolist()) if len(un) else set()
    return paired, rem


def run_config(name, stops, r, g, gl, col):
    """Recount of one column's ledger-dependent cells."""
    print(f"--- {name}: {len(stops)} stops, r {r:.4f}, g {g:.4f}, g_laser {gl:.4f}")
    cells = TAB["columns"].get(col) or TAB["beside"][col]
    SP = {d: day_census(stops, d, r) for d in CFG["rig"]}; SP = {d: v for d, v in SP.items() if v is not None}
    LA = {d: laser(d) for d in sorted(set(CFG["laser_of"].values()) | {CFG["change"][3]})}
    if CFG["stake"] and CFG["stake"][0] in SP:
        ds_, dl_ = CFG["stake"]; S_ = SP[ds_]; sx = LA[dl_]["stake_xy"]
        if len(sx) and len(S_["xy"]):
            keep = np.array([np.min(np.hypot(*(sx - q).T)) > r for q in S_["xy"]])
            for k_ in ("xy", "h", "row"):
                S_[k_] = S_[k_][keep]
    # T4
    for d, md in CFG["marks"].items():
        row_name = f"T4 {d} splat recall >= 0.9 x laser recall"
        if d not in SP:
            cmp(f"{name} | {row_name}", "no splat census", cells[row_name]["verdict"]); continue
        p = f"{C}/hand_marks/marks_{md}.csv"
        if not os.path.exists(p):
            p = f"{C}/hand_marks/premarks_{md}.csv"
        R = list(csv.DictReader(l for l in open(p) if not l.startswith("#"))); sh = np.array(FR[f"uav1_{md}"])
        M = np.array([[float(x["x_utm"]), float(x["y_utm"])] for x in R]) + sh; cls = np.array([x["class"].strip().lower() for x in R])
        tr = np.isin(cls, ["beet", "weed"]) & SP[d]["reg"](M); n = int(tr.sum())
        ns = len(mnn(SP[d]["xy"], M[tr], r)); L_ = LA[CFG["laser_of"][d]]; lsel = SP[d]["reg"](L_["xy"]); nl = len(mnn(L_["xy"][lsel], M[tr], r))
        ratio = (ns / n) / (nl / n) if nl else None
        cmp(f"{name} | {row_name}: ratio", float(ratio), cells[row_name].get("ratio"))
        cmp(f"{name} | {row_name}: counts in basis", f"splat {ns} of {n} marks, laser {nl} of {n}" in cells[row_name]["basis"], True)
        cmp(f"{name} | {row_name}: verdict", "pass" if ns / n >= 0.9 * nl / n else "fail", cells[row_name]["verdict"])
    # T5
    PN, K0, K1S, K1L = CFG["change"]; poly = TRUTH["destructive"][PN]["polygon_utm"]; allp = [v["polygon_utm"] for v in TRUTH["destructive"].values()]
    rn = f"T5 {PN} removal clause (restated 4 Nov)"
    if K0 in SP and K1S in SP:
        lp, lrem = assoc(LA[K0], LA[K1L], r); sp, srem = assoc(SP[K0], SP[K1S], r)
        lin = np.nonzero(in_dil(poly, LA[K0]["xy"], 0.10))[0]
        x = {i: j for i, j, _ in two_pass(SP[K0]["xy"], LA[K0]["xy"], r)}; inv = {j: i for i, j in x.items()}
        both = [(inv[int(l)], int(l)) for l in lin if int(l) in inv]
        val = sum(1 for s_, _ in both if s_ in srem); bar = sum(1 for l in lin if int(l) in lrem)
        cov = int(SP[K1S]["reg"](LA[K0]["xy"][lin]).sum())
        cmp(f"{name} | {rn}: splat removed among both", val, cells[rn].get("value")); cmp(f"{name} | {rn}: bar (laser removed inside)", bar, cells[rn].get("bar"))
        cmp(f"{name} | {rn}: n both in basis", f"of the {len(both)} plants both sensors record inside" in cells[rn]["basis"], True)
        cmp(f"{name} | {rn}: laser plants inside imaged by the later day's stops", cov, cells[rn].get("n_laser_inside_in_splat_region_k1"))

        def spec(A_, rem, B_):
            X = A_["xy"]; outside = ~np.any([in_dil(q, X, 0.10) for q in allp], axis=0); cand = np.nonzero(A_["row"] & B_["reg"](X) & outside)[0]
            return int(len(cand)), int(sum(1 for c in cand if int(c) in rem))
        (ns, ks), (nl, kl) = spec(SP[K0], srem, SP[K1S]), spec(LA[K0], lrem, LA[K1L])
        sd = sameday(stops, K1S, r)
        v = ASSOC["same_day_floor"][K1L]["by_cap"]["0.15"]; lrate = float(np.mean([v["n_unmatched_interior"][k] / v["n_interior"][k] for k in v["n_interior"]])); l_un, l_in = sum(v["n_unmatched_interior"].values()), sum(v["n_interior"].values())
        for sens, n_, k_, rate, cnt in (("splat", ns, ks, sd["rate"], (sd["n_un"], sd["n_in"])), ("laser", nl, kl, lrate, (l_un, l_in))):
            row_name = f"T5 {PN} specificity clause, splat" if sens == "splat" else f"T5 {PN} specificity, laser ceiling (beside; not in the verdict)"
            cell = cells[row_name]
            cmp(f"{name} | {row_name}: counts", f"{k_} of {n_} unchanged on-row plants called removed" in cell["basis"], True)
            if "item30" in cell:
                i30 = cell["item30"]
                cmp(f"{name} | {row_name}: same-day rate", None if rate is None else float(rate), i30["same_day_rate"])
                cmp(f"{name} | {row_name}: same-day counts", list(cnt), [i30["same_day_counts"]["n_unmatched"], i30["same_day_counts"]["n_in"]])
                w = wilson(k_, n_); cmp(f"{name} | {row_name}: Wilson low", w[0], i30["false_removal_wilson95"][0]); cmp(f"{name} | {row_name}: Wilson high", w[1], i30["false_removal_wilson95"][1])
                try:
                    from scipy.stats import binomtest
                    ci = binomtest(k_, n_).proportion_ci(0.95, method="wilson"); cmp(f"{name} | {row_name}: Wilson (scipy) low", float(ci.low), i30["false_removal_wilson95"][0], 1e-6)
                    ce = binomtest(k_, n_).proportion_ci(0.95, method="exact"); cmp(f"{name} | {row_name}: Clopper-Pearson (scipy) high", float(ce.high), i30["false_removal_clopper_pearson95"][1], 1e-6)
                except ImportError:
                    print("     (scipy.stats.binomtest not available)")
                verdict = "not evaluable" if rate is None else ("no verdict" if rate > 0.5 else ("met" if k_ <= rate * n_ else "not met"))
                cmp(f"{name} | {row_name}: item-30 verdict", verdict, cell["verdict"])
            else:
                verdict = "not evaluable" if rate is None else ("met" if k_ <= rate * n_ else "not met")
                cmp(f"{name} | {row_name}: verdict as written", verdict, cell["verdict"])
    else:
        cmp(f"{name} | {rn}: undecidable (a day's census absent)", "undecidable", cells[rn]["verdict"])
    # T6
    m2e = next((p for p in XS["pairs"] if p["pair"] == "uav2_230621_vs_uav2_230623"), None)["per_plant"]["registered"]
    call = lambda v, t: np.where(v > t, 1, np.where(v < -t, -1, 0))
    for k, k1 in CFG["growth"]:
        row_name = f"T6 {k}->{k1}"; cell = cells[row_name]
        if not (k in SP and k1 in SP):
            cmp(f"{name} | {row_name}: undecidable (a day's census absent)", "undecidable", cell["verdict"]); continue
        sp, _ = assoc(SP[k], SP[k1], r); lk, lk1 = CFG["laser_of"][k], CFG["laser_of"][k1]; lp, _ = assoc(LA[lk], LA[lk1], r)
        xk = {i: j for i, j, _ in two_pass(SP[k]["xy"], LA[lk]["xy"], r)}; xk1 = {i: j for i, j, _ in two_pass(SP[k1]["xy"], LA[lk1]["xy"], r)}
        R = [(SP[k1]["h"][sp[i]] - SP[k]["h"][i], LA[lk1]["h"][lp[p]] - LA[lk]["h"][p]) for i, p in xk.items() if i in sp and p in lp and xk1.get(sp[i]) == lp[p]]
        n = len(R)
        if n < 5:
            cmp(f"{name} | {row_name}: undecidable with n", ("undecidable", n), (cell["verdict"], cell.get("n", n if f"n {n} " in cell["basis"] or "fewer than 5" in cell["basis"] else None))); continue
        a = np.array(R); dhs, dhl = a[:, 0], a[:, 1]
        if k1 == CFG["m2e_k1"]:
            med = float(np.median(np.clip(np.abs(dhs - dhl) - m2e["abs_diff_median_m"], 0, None))); ag = float((call(dhs - m2e["diff_median_m"], g) == call(dhl, gl)).mean())
        else:
            med = float(np.median(np.abs(dhs - dhl))); ag = float((call(dhs, g) == call(dhl, gl)).mean())
        cmp(f"{name} | {row_name}: n", n, cell.get("n")); cmp(f"{name} | {row_name}: median", med, cell.get("median_m")); cmp(f"{name} | {row_name}: agreement", ag, cell.get("agreement"))
        cmp(f"{name} | {row_name}: verdict", "pass" if (med <= g and ag >= 0.9) else "fail", cell["verdict"])
    # T7 (both clauses of record and item 31)
    floors = {}
    for p in XS["pairs"]:
        pl = [q for q in p["per_plant"]["per_plant"] if q.get("other_h95_reg") is not None and q.get("n_other_reg", 0) >= p["per_plant"]["registered"]["min_points_other"]]
        floors[p["pair"]] = (float(np.percentile(np.abs([q["other_h95_reg"] - q["ref_h95"] for q in pl]), 95)), len(pl)) if pl else (None, 0)
    n2g = nfl = nev = 0
    for s in stops:
        v = (DSM.get(s, {}).get("vs_laser") or {}).get("by_footprint", {}).get("nview6", {}).get("b")
        if not v or not v["per_plant"].get("plants"):
            continue
        d_ = np.array([q["splat_h95"] - q["laser_h95"] for q in v["per_plant"]["plants"] if q["splat_h95"] is not None and q.get("n_splat_pts", 20) >= 20])
        if not len(d_):
            continue
        p95 = float(np.percentile(np.abs(d_), 95)); n2g += p95 <= 2 * g
        ref = REG[s]["reference"]; fl = floors.get(f"laser_{DSM[s]['vs_laser']['laser_date']}_vs_{ref['kind']}_{ref['date']}", (None, 0))[0]
        if fl is not None:
            nev += 1; nfl += p95 <= fl
    cell = cells["T7 plant clause"]
    if "item31" in cell:
        cmp(f"{name} | T7 plant clause (item 31): stops within the M2d floor", (int(nfl), int(nev)), (cell["item31"]["n_within_floor"], cell["item31"]["n_evaluated"]))
        for pr_, fv in sorted({(x["m2d_pair"], x["m2d_per_plant_abs_diff_p95_m"]) for x in cell["item31"]["per_stop"].values() if x["m2d_per_plant_abs_diff_p95_m"] is not None}):
            cmp(f"{name} | M2d per-plant |diff| p95 recomputed from the plant list, {pr_}", floors[pr_][0], fv, 1e-6)
    elif cell["verdict"] != "undecidable":
        cmp(f"{name} | T7 plant clause (2 g): stops with p95 <= 2 g", int(n2g), cell.get("n_pass"))


# ================================================================================================ configurations
laser_pool = ASSOC["same_day_floor"]["pooled_growth_dates_cap_0.15"]
S_t2, S_pp = stop_set("t2"), stop_set("pp")
set_reg = TAB["columns"]["as registered"]["_settings"]; set_32 = TAB["columns"]["item 32"]["_settings"]
cmp("stops entering, T2 check (from the raw registration fields)", S_t2, set_reg["stops_entering"]); cmp("stops entering, T2 and plant-pattern check", S_pp, set_32["stops_entering"])
# thresholds as registered from the own same-day floor (230613, 0.10 m), and under the stricter set by the fallback rule
f10 = sameday(S_t2, "230613", 0.10)
if PLOT == "198":
    cmp("as registered r_assoc = own 230613 same-day position p95 at 0.10 m (cap 0.095)", min(f10["pos_p95"], 0.095), set_reg["r_assoc_m"]); cmp("as registered g = own 230613 same-day |dh95| p95 (cap 0.08)", min(f10["dh_p95"], 0.08), set_reg["g_m"])
else:
    cmp("as registered r_assoc = own 230613 same-day position p95 at 0.10 m (cap 0.095)", min(f10["pos_p95"], 0.095), set_reg["r_assoc_m"]); cmp("as registered g = own 230613 same-day |dh95| p95, capped at 0.08", min(f10["dh_p95"], 0.08), set_reg["g_m"])
fpp = sameday(S_pp, "230613", 0.10)
cmp("stricter set: 230613 same-day matched plants", fpp["n"], 0)
cmp("stricter set r_assoc = laser pooled p95 capped at 0.095", min(laser_pool["dist_p95_m"], 0.095), set_32["r_assoc_m"]); cmp("stricter set g = laser pooled |dh95| p95 capped at 0.08", min(laser_pool["absdh95_p95_m"], 0.08), set_32["g_m"])
# item 33 from the own floor at 0.15 and 0.20 m
i33 = TAB["columns"]["item 33"]["T3 association resolvable at the 9.5 cm cap"]["item33"]
f15, f20 = sameday(S_t2, "230613", 0.15), sameday(S_t2, "230613", 0.20)
cmp("item 33: 230613 p95 at 0.15 m", f15["pos_p95"], i33["p95_at_0.15_m"]); cmp("item 33: n at 0.15 m", f15["n"], i33["n_at_0.15"]); cmp("item 33: p95 at 0.20 m (beside)", f20["pos_p95"], i33["p95_at_0.20_m"]); cmp("item 33: n at 0.20 m", f20["n"], i33["n_at_0.20"])
cmp("item 33: verdict", "resolvable" if f15["pos_p95"] <= 0.095 else "not resolvable", i33["verdict"])
cmp("as registered T3 reading (0.15 and 0.20 m together)", ("not resolvable" if (f15["pos_p95"] > 0.095 and f20["pos_p95"] > 0.095) else "resolvable" if (f15["pos_p95"] <= 0.095 and f20["pos_p95"] <= 0.095) else "radius-dependent"),
    TAB["columns"]["as registered"]["T3 association resolvable at the 9.5 cm cap"]["verdict"])
gl = float(laser_pool["absdh95_p95_m"])
run_config("as registered", S_t2, set_reg["r_assoc_m"], set_reg["g_m"], gl, "as registered")
run_config("item 30", S_t2, set_reg["r_assoc_m"], set_reg["g_m"], gl, "item 30")
run_config("item 31", S_t2, set_reg["r_assoc_m"], set_reg["g_m"], gl, "item 31")
run_config("item 32 (b)", S_pp, set_32["r_assoc_m"], set_32["g_m"], gl, "item 32")
run_config("item 32 (a)", S_pp, set_reg["r_assoc_m"], set_reg["g_m"], gl, "item 32 line (a): stricter stop set, registered thresholds")
run_config("all four (a)", S_t2, set_reg["r_assoc_m"], set_reg["g_m"], gl, "all four (a: item 32 = T2 check)")
run_config("all four (b)", S_pp, set_32["r_assoc_m"], set_32["g_m"], gl, "all four (b: item 32 = T2 and plant-pattern)")
out["n_comparisons"] = len(out["comparisons"]); out["n_diff"] = sum(1 for c in out["comparisons"] if not c["agree"])
json.dump(out, open(f"{A.sens_dir}/verify_sens_plot{PLOT}.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print(f"plot {PLOT}: {out['n_comparisons']} comparisons, {out['n_diff']} DIFF; wrote {A.sens_dir}/verify_sens_plot{PLOT}.json")
