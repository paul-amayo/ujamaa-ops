"""Run the demo question bank through the live launch API (:8011 -> Adinkra :8013 -> Gemma): every item x language,
record the answer, action, target, language detected and seconds; CSV + JSONL for review."""
import json, time, csv, sys, urllib.request
BANK = "/home/paperspace/code/automation/adinkra_www/demo_bank_20260930.json"; OUT = "/home/paperspace/logs/demo_bank_run_20261001"
bank = json.load(open(BANK)); rows = []
# --errors-only: rerun just the (id, lang) pairs that errored in an earlier run (same OUT files, appended)
ONLY = None
if len(sys.argv) > 1 and sys.argv[1] == "--errors-only":
    ONLY = {(r["id"], r["lang"]) for r in map(json.loads, open(OUT + ".jsonl")) if r.get("error")}
    print(f"rerunning {len(ONLY)} errored items")
for it in bank["items"]:
    for lang, q in it["text"].items():
        if ONLY is not None and (it["id"], lang) not in ONLY: continue
        body = json.dumps({"query": q, "panel": it["panel"], "lang": "auto", "survey": it["survey"]}).encode(); t0 = time.time()
        try:
            req = urllib.request.Request("http://127.0.0.1:8011/api/adinkra/query", data=body, headers={"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=240).read()); err = ""
        except Exception as e:
            r = {}; err = f"{type(e).__name__}: {e}"[:120]
        row = {"survey": it["survey"], "panel": it["panel"], "id": it["id"], "lang": lang, "lang_detected": r.get("lang"), "question": q,
               "expect": it["expect"], "action": r.get("action"), "target": r.get("target"), "answer": (r.get("reasoning") or r.get("detail") or ""),
               "guarded": r.get("guarded", False), "secs": round(time.time() - t0, 1), "error": err}
        rows.append(row); print(f"{it['survey']:18s} {it['panel']:8s} {it['id']:14s} {lang} {row['secs']:5.1f}s {row['action']!s:10s} | {row['answer'][:90]}", flush=True)
        with open(OUT + ".jsonl", "a") as f: f.write(json.dumps(row, ensure_ascii=False) + "\n")
allrows = {}
for r in map(json.loads, open(OUT + ".jsonl")): allrows[(r["id"], r["lang"], r["survey"])] = r   # latest answer per item wins
with open(OUT + ".csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(allrows.values())
print(f"done: {len(rows)} answers -> {OUT}.csv; errors {sum(1 for r in rows if r['error'])}, guarded {sum(1 for r in rows if r['guarded'])}, median secs {sorted(r['secs'] for r in rows)[len(rows)//2]}")
