"""FEWSHOT_DIR=<folder> selects a detector run (default fewshot = stock). Aggregate the few-shot recipe over a survey's fruit3d trees (fleet_05.sh outputs): per tree and pooled - confirmed oranges, stock
recall on confirmed orange-views, exemplar gains at K = 1/3/5 (pooled over test views), extra unmatched detections.
python3 fleet_summary.py [survey root, default 05] -> prints a table + writes <root>/prod/scratch_sam3/fewshot_summary.json"""
import glob, json, os, sys
import numpy as np
R = sys.argv[1] if len(sys.argv) > 1 else "/home/paperspace/data/citrus_all/05_13D_Jackal"; FD = os.environ.get("FEWSHOT_DIR", "fewshot"); rows = []; pool = {"views": 0, "hit": 0, "oranges": 0, "never": 0}; ex = {1: [0, 0, 0, 0, 0, 0], 3: [0, 0, 0, 0, 0, 0], 5: [0, 0, 0, 0, 0, 0]}
for W in sorted(glob.glob(f"{R}/prod/scratch_sam3/fruit3d_t*/"), key=lambda p: int(p.rstrip("/").split("_t")[-1])):
    F = W + FD + "/"; t = int(W.rstrip("/").split("_t")[-1]); r = {"tree": t}
    if not os.path.exists(F + "score_refined_kfonly_thr6.json"): rows.append({**r, "status": "no score"}); continue
    sc = json.load(open(F + "score_refined_kfonly_thr6.json")); tr = json.load(open(F + "tracks.json"))
    r.update(tracks=len(tr), confirmed=sc["confirmed_oranges"], views=sc["orange_views"], stock_recall=round(sc["stock_recall_views"], 3), never_found=sc["oranges_never_found_by_stock"])
    pool["views"] += sc["orange_views"]; pool["hit"] += round(sc["stock_recall_views"] * sc["orange_views"]); pool["oranges"] += sc["confirmed_oranges"]; pool["never"] += sc["oranges_never_found_by_stock"]
    if os.path.exists(F + "exemplar_refined_kfonly_thr6.json"):
        for e in json.load(open(F + "exemplar_refined_kfonly_thr6.json")):
            a = ex[e["K"]]; a[0] += e["n_test"]; a[1] += e["base_hit"]; a[2] += e["ex_hit"]; a[3] += 1; a[4] += e["base_unmatched"]; a[5] += e["ex_unmatched"]
        e1 = [e for e in json.load(open(F + "exemplar_refined_kfonly_thr6.json")) if e["K"] == 1]
        if e1: r["K1"] = f"{sum(e['base_hit'] for e in e1)}/{sum(e['n_test'] for e in e1)} -> {sum(e['ex_hit'] for e in e1)}"
    rows.append(r)
print("| tree | tracks | confirmed | views | stock recall | never found | K=1 base -> exemplar |"); print("|---|---|---|---|---|---|---|")
for r in rows: print(f"| {r['tree']} | {r.get('tracks', '')} | {r.get('confirmed', '')} | {r.get('views', '')} | {r.get('stock_recall', r.get('status', ''))} | {r.get('never_found', '')} | {r.get('K1', '')} |")
ok = [r for r in rows if "confirmed" in r]; print(f"\n{len(ok)} trees scored, {sum(1 for r in rows if 'status' in r)} without a score; confirmed oranges {pool['oranges']} ({pool['never']} never found by stock), views {pool['views']}, pooled stock recall {pool['hit'] / max(pool['views'], 1):.3f}")
for K, a in ex.items():
    if a[0]: print(f"K={K}: {a[3]} frames, {a[0]} test views: recall text-only {a[1] / a[0]:.3f} -> +{K} exemplars {a[2] / a[0]:.3f}; unmatched detections per frame {a[4] / a[3]:.1f} -> {a[5] / a[3]:.1f}")
json.dump({"rows": rows, "pooled": pool, "exemplar": {str(k): v for k, v in ex.items()}}, open(f"{R}/prod/scratch_sam3/{FD}_summary.json", "w"), indent=1)
