#!/usr/bin/env python3
"""gen2 M0 pilot diff — prod hierarchy vs the census-solver rebuild.

For each survey: reads prod/bateleur/scene_graph/marker_hierarchy.json and
experimental/gen2_m0/<name>/marker_hierarchy.json, prints one table row,
writes lab_notebook/figs/gen2_m0_<name>.png (prod | gen2 top-downs, rows as
lines through their members) and experimental/gen2_m0/report.json per root.
Ground truth = the JSON artifacts, never the build logs.

  wide rows  = rows with >3 members whose x-span exceeds 1.5 x row spacing —
               a line crossing neighbouring rows (the "thief" signature)
  ARI        = adjusted Rand index of per-object row labels prod vs gen2 over
               objects present in both (row_id -1 counts as its own label);
               n/a when the object id sets differ (census remap, apr_mv10)
Usage: gen2_m0_diff.py [name ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402

CITRUS = Path("/home/paperspace/data/citrus_all")
KLAP = Path("/home/paperspace/data/klapmuts")
FIGS = Path("/home/paperspace/code/lab_notebook/figs")
# name -> (survey root, row spacing m)
SPECS = {
    "05": (CITRUS / "05_13D_Jackal", 4.0),
    "04": (CITRUS / "04_13D_Jackal", 4.0),
    "01": (CITRUS / "01_13B_Jackal", 4.0),
    "03": (CITRUS / "03_13B_Jackal", 4.0),
    "02": (CITRUS / "02_13B_Jackal", 4.0),
    "apr": (KLAP / "apr_2026_zed", 1.6),
    "apr_mv10": (KLAP / "apr_2026_zed", 1.6),
    # experiment variants (same prod baseline)
    "02_dir985": (CITRUS / "02_13B_Jackal", 4.0),
    "02_dir95": (CITRUS / "02_13B_Jackal", 4.0),
    "01_dir985": (CITRUS / "01_13B_Jackal", 4.0),
    "03_dir985": (CITRUS / "03_13B_Jackal", 4.0),
    "apr_band06": (KLAP / "apr_2026_zed", 1.6),
    "apr_mv10_band06": (KLAP / "apr_2026_zed", 1.6),
}


def row_stats(objs, spacing):
    rows = {}
    for o in objs:
        if o["row_id"] >= 0:
            rows.setdefault(o["row_id"], []).append(o)
    sizes = sorted((len(v) for v in rows.values()), reverse=True)
    wide = 0
    for v in rows.values():
        if len(v) > 3:
            xs = [m["xyz"][0] for m in v]
            if max(xs) - min(xs) > 1.5 * spacing:
                wide += 1
    rowless = sum(1 for o in objs if o["row_id"] < 0)
    return {"n_rows": len(rows), "sizes": sizes, "largest": sizes[0] if sizes else 0,
            "wide": wide, "rowless": rowless}


def ari(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    M = np.zeros((ia.max() + 1, ib.max() + 1), dtype=np.int64)
    np.add.at(M, (ia, ib), 1)

    def comb(x):
        x = np.asarray(x, dtype=np.float64)
        return x * (x - 1) / 2.0
    sum_ij = comb(M).sum()
    sum_a = comb(M.sum(1)).sum()
    sum_b = comb(M.sum(0)).sum()
    n = comb(len(a))
    exp = sum_a * sum_b / n if n else 0.0
    mx = 0.5 * (sum_a + sum_b)
    return 1.0 if mx == exp else float((sum_ij - exp) / (mx - exp))


def panel(ax, objs, title):
    rows = {}
    for o in objs:
        rows.setdefault(o["row_id"], []).append(o)
    order = sorted((r for r in rows if r >= 0),
                   key=lambda r: np.mean([m["xyz"][0] for m in rows[r]]))
    for i, r in enumerate(order):
        m = np.array([[o["xyz"][0], o["xyz"][2]] for o in rows[r]])
        m = m[np.argsort(m[:, 1])]
        c = cm.tab20(i % 20)
        ax.plot(m[:, 0], m[:, 1], "-", color=c, lw=1.1, alpha=0.85, zorder=2)
        ax.scatter(m[:, 0], m[:, 1], c=[c], s=14, edgecolors="k", linewidths=0.25,
                   zorder=3)
    if -1 in rows:
        m = np.array([[o["xyz"][0], o["xyz"][2]] for o in rows[-1]])
        ax.scatter(m[:, 0], m[:, 1], marker="x", s=22, c="0.5", zorder=1,
                   label=f"row-less ({len(m)})")
        ax.legend(loc="lower left", fontsize=8)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x cross-row (m)")
    ax.set_ylabel("z along-row (m)")
    ax.grid(alpha=0.25)


def main():
    names = sys.argv[1:] or [n for n in SPECS
                             if (SPECS[n][0] / "experimental/gen2_m0" / n /
                                 "marker_hierarchy.json").is_file()]
    reports = {}
    print(f"{'survey':9s} {'objs':>5s} | {'prod rows':>9s} {'largest':>7s} {'wide':>4s} "
          f"| {'gen2 rows':>9s} {'largest':>7s} {'wide':>4s} {'rowless':>7s} "
          f"{'bucket':>6s} {'census':>6s} | {'ARI':>5s}")
    for name in names:
        root, spacing = SPECS[name]
        prod_p = root / "prod/bateleur/scene_graph/marker_hierarchy.json"
        gen2_p = root / "experimental/gen2_m0" / name / "marker_hierarchy.json"
        if not gen2_p.is_file():
            print(f"{name:9s} (no gen2 build at {gen2_p})")
            continue
        prod = json.loads(prod_p.read_text())
        gen2 = json.loads(gen2_p.read_text())
        ps = row_stats(prod["objects"], spacing)
        gs = row_stats(gen2["objects"], spacing)
        cen = (gen2.get("_provenance") or {}).get("census") or {}
        pid = {o["id"]: o["row_id"] for o in prod["objects"]}
        gid = {o["id"]: o["row_id"] for o in gen2["objects"]}
        common = sorted(set(pid) & set(gid))
        same_ids = len(common) >= 0.8 * min(len(pid), len(gid)) and len(pid) == len(gid)
        a = ari([pid[i] for i in common], [gid[i] for i in common]) if same_ids else None
        rep = {"n_objects_prod": len(pid), "n_objects_gen2": len(gid), "prod": ps,
               "gen2": gs, "census_rows": cen.get("n_rows"), "census_bucket": cen.get("n_bucket"),
               "n_census": cen.get("n_census"), "row_num_models": cen.get("row_num_models"),
               "row_band_m": cen.get("row_band_m"), "dir_threshold": cen.get("dir_threshold"),
               "ari_prod_vs_gen2": a, "gen2_path": str(gen2_p)}
        reports.setdefault(str(root), {})[name] = rep
        print(f"{name:9s} {len(gid):5d} | {ps['n_rows']:9d} {ps['largest']:7d} {ps['wide']:4d} "
              f"| {gs['n_rows']:9d} {gs['largest']:7d} {gs['wide']:4d} {gs['rowless']:7d} "
              f"{str(cen.get('n_bucket')):>6s} {str(cen.get('n_census')):>6s} | "
              f"{'n/a' if a is None else f'{a:5.3f}':>5s}")
        fig, axes = plt.subplots(1, 2, figsize=(15, 9))
        panel(axes[0], prod["objects"],
              f"{name} PROD (CORAL on markers): {ps['n_rows']} rows, largest {ps['largest']}, "
              f"wide {ps['wide']}")
        panel(axes[1], gen2["objects"],
              f"{name} gen2 M0 (census RANSAC): {gs['n_rows']} rows, largest {gs['largest']}, "
              f"wide {gs['wide']}; bucket {cen.get('n_bucket')}/{cen.get('n_census')}")
        fig.suptitle(f"gen2 M0 pilot — {name}: band {cen.get('row_band_m')} m, dir thr "
                     f"{cen.get('dir_threshold')}, num_models {cen.get('row_num_models')}",
                     fontsize=11)
        out = FIGS / f"gen2_m0_{name}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=110)
        plt.close(fig)
    for root, rep in reports.items():
        p = Path(root) / "experimental/gen2_m0/report.json"
        old = json.loads(p.read_text()) if p.is_file() else {}
        old.update(rep)
        p.write_text(json.dumps(old, indent=1))
        print(f"report -> {p}")


if __name__ == "__main__":
    main()
