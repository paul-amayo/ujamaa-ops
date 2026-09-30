"""Demo registries + the combined Sankofa ledger (2026-09-30, AIC FS demo).

Klapmuts: Paul — "the old april registry is now out of date as it was joining bags together". The plant registry for
both Klapmuts surveys is therefore DERIVED FROM klapmuts_ledger_v5.json (peer session, 09-30: depth-masked SAM3 grow-bag
instances at the 2.5 m mask window; Dec 911 / Apr 912 bags; 825 mutual-NN pairs <= 0.5 m, median 0.23 m):
  * objects = the survey's bags. Apr in the ledger's (April) frame; Dec in its NATIVE frame (the stored Dec->Apr
    registration inverted), which is the frame of the Dec lane-2 H3DGS walk (checked 09-30: 120 bags within 2 m of the
    lane-2 path in its horizontal plane). xyz = [x, y, 0] (z up).
  * rows = bag LINES: 1-D gaps > GAP m across the rows (x of the April frame); lines with < MIN_LINE bags are stragglers,
    attached to the nearest main line. Row numbers are shared by both surveys (main lines matched in the April frame).
Sankofa: one combined ledger in the ledger_v2 observation schema (canonical_id, farm, epoch, source_survey,
source_tree_id, row_id, ndvi_sentinel2) so /api/sankofa/* and Adinkra read every farm from one file, keyed by "farm":
  * Citrus A (13B): ledger_v2 observations verbatim (surveys 01/02/03);
  * Citrus B (13D): REBUILT from assoc_04_05_v4 + the gen2 hierarchies (ledger_v2's 13D side predates them: survey 04
    only, old ids) — the same construction as ujamaa/project/export_bateleur_timeline.py: a pair shares canonical
    "13D:<05 id>", unpaired 04 trees are "13D:u04:<id>", stale pairs (ids no longer in the hierarchies) dropped;
    each observation carries the tree's 3D-confirmed fruit ("fruit_confirmed", n_fruit3d). Dates from the ledger:
    05 = 2023-07-16, 04 = 2023-07-18. Per-tree fruit CHANGE is not a fair statistic here (only ~2 pairs had a fair
    look in both surveys, 08-29 v4 verdict); farm totals are (154 vs 108);
  * Klapmuts ("KL"): one observation per bag per survey; a matched pair shares a canonical id.
Recording dates from the data (not the folder names): Dec 2025-12-03, Apr 2026-04-15.
  usage: build_demo_ledger.py [--out-dir /home/paperspace/data/sankofa_demo]"""
import argparse, json
from pathlib import Path
import numpy as np

KL = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/sankofa/klapmuts_ledger_v5.json")
CITRUS = Path("/home/paperspace/data/citrus_all/sankofa_substrate/ledger_v2.json")
C13D = {"05_13D_Jackal": "2023-07-16", "04_13D_Jackal": "2023-07-18"}
ASSOC = Path("/home/paperspace/data/citrus_all/sankofa_substrate/assoc_04_05_v4.npz")
GAP, MIN_LINE = 0.45, 10
DEC_SV, APR_SV = "klapmuts_dec_2025", "klapmuts_apr_2026"
DEC_DATE, APR_DATE = "2025-12-03", "2026-04-15"


def main_lines(xs):
    order = np.argsort(xs); x = xs[order]; cuts = np.where(np.diff(x) > GAP)[0] + 1
    groups = np.split(order, cuts)
    return [float(xs[g].mean()) for g in groups if len(g) >= MIN_LINE]


def assign_rows(xs, centres):
    c = np.asarray(centres); return (np.abs(xs[:, None] - c[None, :]).argmin(1) + 1).astype(int)


def registry(P, rows, survey, note):
    objs = [{"id": i, "xyz": [round(float(p[0]), 3), round(float(p[1]), 3), 0.0]} for i, p in enumerate(P)]
    out_rows = [{"id": int(r), "object_ids": [int(i) for i in np.where(rows == r)[0]]} for r in sorted(set(rows.tolist()))]
    return {"source": f"klapmuts_ledger_v5.json ({survey}): {note}", "n_objects": len(objs), "objects": objs,
            "rows": out_rows, "fruits": [], "trees": []}


def citrus_b_obs():
    H = {sv: json.loads(Path(f"/home/paperspace/data/citrus_all/{sv}/prod/bateleur/scene_graph/marker_hierarchy.json").read_text()) for sv in C13D}
    ids = {sv: {o["id"] for o in H[sv]["objects"]} for sv in H}
    row = {sv: {o["id"]: o.get("row_id") for o in H[sv]["objects"]} for sv in H}
    fruit = {sv: {f["tree_id"]: f.get("n_fruit3d") or 0 for f in (H[sv].get("fruits") or [])} for sv in H}
    z = np.load(ASSOC, allow_pickle=True)
    pairs = [(int(p[0]), int(p[1])) for p in z["pairs"]]
    pairs = [(a, b) for a, b in pairs if a in ids["04_13D_Jackal"] and b in ids["05_13D_Jackal"]]
    p04 = {a: b for a, b in pairs}; p05 = {b for _, b in pairs}
    out = []
    def ob(sv, tid, cid):
        return {"canonical_id": cid, "farm": "13D", "epoch": C13D[sv], "epoch_type": "survey", "source_survey": sv,
                "source_tree_id": tid, "row_id": row[sv].get(tid), "fruit_confirmed": fruit[sv].get(tid, 0), "ndvi_sentinel2": None}
    for b in sorted(ids["05_13D_Jackal"]):
        out.append(ob("05_13D_Jackal", b, f"13D:{b}"))
    for a in sorted(ids["04_13D_Jackal"]):
        out.append(ob("04_13D_Jackal", a, f"13D:{p04[a]}" if a in p04 else f"13D:u04:{a}"))
    print(f"[demo-ledger] Citrus B: 05 {len(ids['05_13D_Jackal'])} trees, 04 {len(ids['04_13D_Jackal'])} trees, {len(pairs)} pairs "
          f"(dropped {len(z['pairs']) - len(pairs)} stale); confirmed fruit 05 {sum(fruit['05_13D_Jackal'].values())} / 04 {sum(fruit['04_13D_Jackal'].values())}")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default="/home/paperspace/data/sankofa_demo"); a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    d = json.loads(KL.read_text()); assert d["version"] == "v5", d["version"]
    B = np.array(d["B_positions"], float); A_apr = np.array(d["A_positions_apr_frame"], float)
    r = d["registration_A_to_B"]; th = np.radians(r["yaw_deg"])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); t = np.array(r["t"], float)
    A_nat = (A_apr - t) @ R                                   # inverse of p_apr = R p_dec + t
    centres = sorted(set(round(c, 1) for c in main_lines(B[:, 0])) | set())
    # one centre list for both surveys: April's main lines, plus any Dec main line with no April line within 0.6 m
    for c in main_lines(A_apr[:, 0]):
        if min(abs(c - x) for x in centres) > 0.6: centres.append(round(c, 1))
    centres = sorted(centres)
    rows_B, rows_A = assign_rows(B[:, 0], centres), assign_rows(A_apr[:, 0], centres)
    note = f"{len(centres)} bag lines, gap {GAP} m, lines < {MIN_LINE} bags attached to the nearest line"
    (out / "klapmuts_registry_apr_2026_v5.json").write_text(json.dumps(registry(B, rows_B, "April 2026, April frame", note)))
    (out / "klapmuts_registry_dec_2025_v5.json").write_text(json.dumps(registry(A_nat, rows_A, "December 2025, native (lane-2 walk) frame", note)))

    obs = [o for o in json.loads(CITRUS.read_text())["observations"] if o.get("farm") != "13D"]
    obs += citrus_b_obs()
    pairB = {p["B_id"]: p for p in d["pairs"]}; pairA = {p["A_id"]: p for p in d["pairs"]}
    for b in range(len(B)):
        cid = f"KL:{b}"
        obs.append({"canonical_id": cid, "farm": "KL", "epoch": APR_DATE, "epoch_type": "survey", "source_survey": APR_SV,
                    "source_tree_id": b, "row_id": int(rows_B[b]), "xy_apr_frame": [round(float(v), 3) for v in B[b]],
                    "match_dist_m": pairB[b]["dist_m"] if b in pairB else None, "ndvi_sentinel2": None})
    for a_ in range(len(A_apr)):
        cid = f"KL:{pairA[a_]['B_id']}" if a_ in pairA else f"KL:dec{a_}"
        obs.append({"canonical_id": cid, "farm": "KL", "epoch": DEC_DATE, "epoch_type": "survey", "source_survey": DEC_SV,
                    "source_tree_id": a_, "row_id": int(rows_A[a_]), "xy_apr_frame": [round(float(v), 3) for v in A_apr[a_]],
                    "match_dist_m": pairA[a_]["dist_m"] if a_ in pairA else None, "ndvi_sentinel2": None})
    meta = {"built": "2026-09-30", "by": "automation/sankofa/build_demo_ledger.py",
            "sources": {"citrus": str(CITRUS), "klapmuts": str(KL)},
            "klapmuts": {"dec_bags": len(A_apr), "apr_bags": len(B), "pairs": len(d["pairs"]), "rows": len(centres),
                         "row_centres_apr_frame": centres}}
    (out / "ledger_demo_v1.json").write_text(json.dumps({"meta": meta, "observations": obs}))
    print(f"[demo-ledger] Klapmuts rows {len(centres)} (centres {centres}); Apr {len(B)} bags, Dec {len(A_apr)} bags, "
          f"{len(d['pairs'])} pairs; per-row Apr/Dec: " + ", ".join(f"{i+1}:{(rows_B == i+1).sum()}/{(rows_A == i+1).sum()}" for i in range(len(centres))))
    print(f"[demo-ledger] combined observations {len(obs)} -> {out / 'ledger_demo_v1.json'}")


if __name__ == "__main__":
    main()
