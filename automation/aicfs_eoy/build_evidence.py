#!/usr/bin/env python3
"""Build the UJAMAA evidence bundle behind the AIC:FS end-of-year data request, served by the pre-launch demo app on the
box at /app/evidence/ (the launch app's StaticFiles mount of ujamaa/app, html=True, so each folder needs its own index.html).

  python3 automation/aicfs_eoy/build_evidence.py [--out /home/paperspace/code/ujamaa/app/evidence]

Runs on the A100 (stdlib only). Copies the raw evaluation records the form cites and computes the summary tables from
them (latency and outcomes per language from the Adinkra r3 run; the demo-bank grounding score), writes an index.html
that maps each form row to its files, then scans the whole bundle for the citrus orchards' location (never published)
and refuses to finish if it finds it. Nothing with per-tree WGS84 coordinates of the citrus surveys is copied."""
import csv, html, json, os, re, shutil, statistics, subprocess, sys
from pathlib import Path

OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path("/home/paperspace/code/ujamaa/app/evidence")
H = Path("/home/paperspace"); OPS = H / "code"
FORM = Path(os.environ.get("AICFS_FORM_DIR", OPS / "automation/aicfs_eoy"))   # the filled form; staged outside the checkout when the box's ops clone lags origin
R3 = H / "data/experimental_adinkra_www_20260827_r3"
SRC = {
    "adinkra_r3": [R3 / f for f in ["passA_citrus.jsonl", "passA_klap.jsonl", "passB_citrus.jsonl", "passB_klap.jsonl",
                                     "passC_citrus.jsonl", "passC_klap.jsonl", "adinkra_review_package_r3.csv", "freeze.json"]],
    "demo_bank_20261001": [OPS / "automation/adinkra_www/demo_bank_20260930.json", H / "logs/demo_bank_run_20261001.csv",
                           H / "logs/demo_bank_run_20261001.jsonl"],
    "multilingual": [H / "data/experimental_multiling_m1/summary.txt", OPS / "plans/g0_negative_result.md",
                     OPS / "automation/adinkra_www/questions.json", OPS / "automation/adinkra_www/www_fig_numbers.json"],
    "identity": [H / "logs/e1_protocol_table_v3.md", H / "code/aru_sil_core/src/docs/paper/numbers/e1_v3.json"],
    "reconstruction": [H / "logs/v3_fleet_results.jsonl"],   # the run-of-show plan names the citrus location: never copied; its PSNR table is re-typed below
    "sankofa": [H / "data/klapmuts/dec_2025_ten_rows/experimental/sankofa/klapmuts_ledger_v5.json"],
    "data": [OPS / "DATASETS.md"],
    "form": [FORM / "AICFS_EOY_2026_UJAMAA_interim.md", FORM / "AICFS_EOY_2026_UJAMAA.md", FORM / "AICFS_EOY_2026_UJAMAA.csv"],
}
LOCATION = re.compile(r"riverside|california|\b-117\.\d", re.I)   # the citrus orchards' location: never published (latitude alone collides with Klapmuts local-frame metres)

if OUT.exists(): shutil.rmtree(OUT)
OUT.mkdir(parents=True)
copied = {}
for group, files in SRC.items():
    (OUT / group).mkdir(exist_ok=True)
    for f in files:
        if not f.exists(): print("MISSING", f); continue
        shutil.copy2(f, OUT / group / f.name); copied.setdefault(group, []).append(f.name)

# ---- Adinkra r3: latency and outcomes per language, from the jsonl records (field `secs` per query)
rows = []
for f in sorted((OUT / "adinkra_r3").glob("pass*.jsonl")):
    for line in open(f):
        line = line.strip()
        if not line: continue
        d = json.loads(line); d["_file"] = f.name; rows.append(d)
def q(xs, p): xs = sorted(xs); return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]
langs = sorted({r["lang"] for r in rows}); passes = sorted({r.get("tag", "") for r in rows})
with open(OUT / "adinkra_r3/latency_by_language.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["lang", "pass", "n", "median_s", "p90_s", "max_s", "answered", "no_answer", "empty"])
    for lang in langs:
        for tag in ["ALL"] + passes:
            sel = [r for r in rows if r["lang"] == lang and (tag == "ALL" or r.get("tag") == tag)]
            if not sel: continue
            secs = [float(r["secs"]) for r in sel if r.get("secs") is not None]
            w.writerow([lang, tag, len(sel), round(statistics.median(secs), 1), round(q(secs, 0.9), 1), round(max(secs), 1),
                        sum(1 for r in sel if r.get("action") not in ("no_answer", None) and not r.get("empty")),
                        sum(1 for r in sel if r.get("action") == "no_answer"), sum(1 for r in sel if r.get("empty"))])
total = len(rows); empty = sum(1 for r in rows if r.get("empty")); noans = sum(1 for r in rows if r.get("action") == "no_answer")
r3_summary = {"run": "Adinkra x Way With Words r3, 2026-08-25 (freeze.json)", "queries": total, "languages": langs, "empty": empty,
              "honest_no_answer": noans, "honest_no_answer_pct": round(100 * noans / total, 1) if total else None,
              "latency_median_s_by_language": {l: round(statistics.median([float(r["secs"]) for r in rows if r["lang"] == l]), 1) for l in langs}}
json.dump(r3_summary, open(OUT / "adinkra_r3/summary.json", "w"), indent=1, ensure_ascii=False)

# ---- demo bank 1 Oct 2026: grounding score from the run CSV
bank = list(csv.DictReader(open(OUT / "demo_bank_20261001/demo_bank_run_20261001.csv")))
n_err = sum(1 for b in bank if (b.get("error") or "").strip()); n_ok = len(bank) - n_err
bank_summary = {"run": "demo bank, 2026-10-01 (demo_bank_run_20261001.csv)", "question_language_pairs": len(bank), "completed": n_ok,
                "failed": n_err, "failures": [{"survey": b["survey"], "id": b["id"], "lang": b["lang"], "error": b["error"][:120]} for b in bank if (b.get("error") or "").strip()],
                "latency_median_s_by_language": {l: round(statistics.median([float(b["secs"]) for b in bank if b["lang"] == l and b.get("secs")]), 1)
                                                 for l in sorted({b["lang"] for b in bank})},
                "note": "Expected answers are in demo_bank_20260930.json; correctness of the 87 completed answers was checked by the team against the farms' records on 1 Oct 2026 (lab notebook)."}
json.dump(bank_summary, open(OUT / "demo_bank_20261001/summary.json", "w"), indent=1, ensure_ascii=False)

# ---- reconstruction: the summit quality table (H3DGS, measured 29 Sep to 1 Oct 2026), re-typed from the notebook of record
with open(OUT / "reconstruction/psnr_table_summit_2026-10.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["farm", "survey_or_model", "psnr_median_training_views_in_cell_dB", "psnr_held_out_dB", "shown_at_summit", "source"])
    w.writerows([
        ["Citrus farm A", "01, chunk 3_1, improved H3DGS recipe", 28.23, "", "yes", "lab notebook 2026-09-30/10-01 (sky-masked 27.56)"],
        ["Citrus farm B", "05, chunk 1_0, improved H3DGS recipe", 26.28, "", "yes", "lab notebook 2026-10-01 (203 in-cell training views; sky-masked 26.55; chunk 0_0 27.54)"],
        ["Klapmuts", "Dec 2025 lane 2, h3dgs_e2", 29.72, 19.82, "yes", "lab notebook 2026-09-27"],
        ["Kendu Bay (phone)", "IMG_7975_s0 ground crop, lane-style H3DGS 8 M", 26.06, 18.74, "yes", "lab notebook 2026-10-01"],
        ["Gwakungu cabbages (phone)", "IMG_7993_s0, H3DGS 8 M", 23.43, 18.16, "yes, below the 25 dB bar by decision", "lab notebook 2026-10-01"],
        ["Gwakungu chillies (phone)", "IMG_7990_s1, H3DGS 8 M", 21.92, 15.19, "no", "lab notebook 2026-10-01"],
    ])
    w.writerow(["note", "'in cell' = training views whose cameras lie inside the 30 m chunk; v3_fleet_results.jsonl holds the later production-recipe runs with their held-out scores per chunk", "", "", "", ""])

# ---- Sankofa: Klapmuts ledger v5 counts (the file itself is copied; positions are in the April survey frame)
L = json.load(open(OUT / "sankofa/klapmuts_ledger_v5.json"))
def count(k):
    v = L.get(k); return len(v) if hasattr(v, "__len__") and not isinstance(v, str) else v
sank = {"file": "klapmuts_ledger_v5.json (built 2026-09-30)", "keys": {k: count(k) for k in L.keys()}}
json.dump(sank, open(OUT / "sankofa/summary.json", "w"), indent=1)

# ---- farm catalogue, public fields only
S = json.load(open(H / "code/ujamaa/app/sites.json"))
cat = [{"farm": f.get("name"), "id": f.get("id"), "surveys": [{"id": s.get("id"), "date": s.get("date"), "capture": s.get("capture"), "crop": s.get("crop")}
        for s in f.get("surveys", [])]} for f in S.get("farms", [])]
json.dump(cat, open(OUT / "data/farm_catalogue.json", "w"), indent=1, ensure_ascii=False)

# ---- index
ROWS = [
 ("1.1 Model inputs / data quality", "data/DATASETS.md, data/farm_catalogue.json", "surveys held, by farm and platform"),
 ("1.2 Model functionality", "reconstruction/psnr_table_summit_2026-10.csv, sankofa/, multilingual/g0_negative_result.md", "what was built, with dates"),
 ("1.3 Model coverage, 2.2 Product coverage", "data/farm_catalogue.json, multilingual/questions.json", "farms, surveys, languages"),
 ("1.4 Model accuracy and precision", "reconstruction/ (PSNR), identity/ (held-out IoU, E1 protocol), sankofa/ (825 of 912), adinkra_r3/summary.json, multilingual/summary.txt", "measured runs"),
 ("2.1 Product performance", "adinkra_r3/latency_by_language.csv, demo_bank_20261001/summary.json", "latency per language; 87 of 89 grounded"),
 ("2.4 Product accessibility, 2.7 Product trust", "adinkra_r3/pass*.jsonl (every query, answer and no_answer, with timing)", "refusal behaviour by language"),
 ("The form itself", "form/", "filled rows (md, csv) and the source-traced record"),
]
idx = ["<!doctype html><meta charset=utf-8><title>UJAMAA evidence</title>",
       "<style>body{font:15px/1.5 system-ui;margin:2rem auto;max-width:60rem;padding:0 1rem;color:#222}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.4rem .6rem;vertical-align:top}code{background:#f3f3f3;padding:0 .2rem}</style>",
       "<h1>UJAMAA: evidence behind the AIC:FS end-of-year data request</h1>",
       "<p>UCT African Robotics Unit. Raw records and the tables computed from them; each file is as written by the run that produced it. "
       "Contact: paul.amayo@uct.ac.za.</p>",
       f"<p><b>Adinkra r3 (25 Aug 2026):</b> {total} queries in {len(langs)} languages; {empty} empty; {noans} honest refusals ({r3_summary['honest_no_answer_pct']} %). "
       f"Median latency (s): " + ", ".join(f"{l} {v}" for l, v in r3_summary["latency_median_s_by_language"].items()) + ".</p>",
       f"<p><b>Demo bank (1 Oct 2026):</b> {len(bank)} question-language pairs, {n_ok} completed, {n_err} failed.</p>",
       "<table><tr><th>Form row</th><th>Files</th><th>What they show</th></tr>"]
for a, b, c in ROWS: idx.append(f"<tr><td>{html.escape(a)}</td><td><code>{html.escape(b)}</code></td><td>{html.escape(c)}</td></tr>")
idx.append("</table><h2>Folders</h2><ul>")
for group in SRC: idx.append(f"<li><a href='{group}/'>{group}/</a>: " + ", ".join(sorted(p.name for p in (OUT / group).iterdir())) + "</li>")
idx.append("</ul><p>Browse any folder for the files. Built by <code>automation/aicfs_eoy/build_evidence.py</code>.</p>")
(OUT / "index.html").write_text("\n".join(idx))
for group in SRC:   # the static mount has no directory listing: one index per folder
    files = sorted(p for p in (OUT / group).iterdir() if p.name != "index.html")
    (OUT / group / "index.html").write_text("<!doctype html><meta charset=utf-8><title>" + group + "</title><style>body{font:15px/1.6 system-ui;margin:2rem auto;max-width:50rem;padding:0 1rem}</style>"
        f"<p><a href='../'>evidence</a> / <b>{group}</b></p><ul>" + "".join(f"<li><a href='{p.name}'>{p.name}</a> ({p.stat().st_size // 1024} KB)</li>" for p in files) + "</ul>")

# ---- location scan: the citrus orchards' location is never published
hits = []
for p in OUT.rglob("*"):
    if p.is_file():
        try: t = p.read_text(errors="ignore")
        except Exception: continue
        for m in LOCATION.finditer(t): hits.append((str(p.relative_to(OUT)), t[max(0, m.start() - 50):m.end() + 30].replace("\n", " ")))
if hits:
    for h in hits[:20]: print("LOCATION HIT", h)
    shutil.rmtree(OUT); raise SystemExit(f"refused: {len(hits)} location hits; bundle removed")
size = subprocess.check_output(["du", "-sh", str(OUT)]).decode().split()[0]
print(f"evidence bundle at {OUT}: {sum(len(v) for v in copied.values())} files copied, {size}; r3 {total} queries, bank {n_ok}/{len(bank)}")
