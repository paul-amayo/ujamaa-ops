#!/usr/bin/env python3
"""Dataset registry generator (roadmap Phase 0).

Scans every survey on disk and writes DATASETS.md — one file answering
"what data exists and how far through the pipeline is it". Re-run after
any ingest/pipeline milestone; the file is committed so the registry has
history.

  python3 automation/build_dataset_registry.py            # writes DATASETS.md
  python3 automation/build_dataset_registry.py --stdout   # preview

States are DERIVED from artifacts on disk (never hand-maintained):
  ingest    combined.bag / monolithics / LIO poses / kf20cm / kf_images
  registry  scene_graph markers + marker_hierarchy.json (objects/rows)
  masks     sky_masks/.done + fg_masks/.done
  blocks    blocks_ns/<cfg>/block_* dirs, splats.json entries
  ledger    observations contributed to sankofa ledger_v2 (fallback v1)
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

CITRUS = Path("/home/paperspace/data/citrus_all")
KLAP = Path("/home/paperspace/data/klapmuts")
LEDGER_V2 = CITRUS / "sankofa_substrate/ledger_v2.json"
LEDGER_V1 = CITRUS / "sankofa_substrate/ledger_v1.json"
OUT = Path("/home/paperspace/code/DATASETS.md")


def du_gb(path: Path) -> str:
    try:
        out = subprocess.run(["du", "-sBG", str(path)], capture_output=True,
                             text=True, timeout=600).stdout
        return out.split()[0]
    except Exception:
        return "?"


def count(glob_dir: Path, pattern: str) -> int:
    return len(list(glob_dir.glob(pattern))) if glob_dir.is_dir() else 0


def ledger_obs_by_survey() -> dict:
    src = LEDGER_V2 if LEDGER_V2.exists() else LEDGER_V1
    if not src.exists():
        return {}
    d = json.load(open(src))
    flat = []
    for v in (d.values() if isinstance(d, dict) else d):
        flat.extend(v if isinstance(v, list) else [v])
    obs = {}
    for r in flat:
        s = r.get("source_survey")
        if s:
            obs[s] = obs.get(s, 0) + 1
    obs["_file"] = src.name
    return obs


def hierarchy_stats(root: Path) -> str:
    cands = sorted(root.glob("scene_graph*/marker_hierarchy*.json"),
                   key=lambda p: p.stat().st_mtime)
    if not cands:
        return "—"
    hier = cands[-1]
    try:
        d = json.load(open(hier))
        p = d.get("_provenance", {})
        n_obj = p.get("n_objects") or len(d.get("objects", []))
        n_rows = p.get("n_rows") or len(d.get("rows", []))
        return f"{n_obj} obj / {n_rows} rows ({hier.parent.name})"
    except Exception:
        return "unreadable"


def survey_row(root: Path, ledger_obs: dict) -> dict:
    r = {"id": root.name, "path": str(root), "size": du_gb(root)}
    r["combined_bag"] = (root / "combined.bag").exists()
    flags = [(any(root.glob("image_left*.monolithic")), "img"),
             ((root / "laser.monolithic").exists(), "lidar"),
             ((root / "gps.monolithic").exists(), "gps")]
    r["monolithics"] = "+".join(n for ok, n in flags if ok) or False
    r["lio"] = (root / "transform_lio.monolithic").exists()
    r["kf20cm"] = (root / "image_left_kf20cm.monolithic").exists()
    r["kf_pngs"] = count(root / "kf_images", "kf_*.png")
    r["markers"] = bool(list(root.glob("scene_graph*/markers_v2*.monolithic")))
    r["hierarchy"] = hierarchy_stats(root)
    r["masks"] = (root / "sky_masks/.done").exists() and (root / "fg_masks/.done").exists()
    blocks = sorted((root / "blocks_ns").glob("*/")) if (root / "blocks_ns").is_dir() else []
    r["blocks"] = {b.name: len(list(b.glob("block_*"))) for b in blocks} or {}
    r["splats"] = 0
    for b in blocks:
        sj = b / "splats.json"
        if sj.exists():
            try:
                d = json.load(open(sj))
                entries = d["blocks"] if isinstance(d, dict) and "blocks" in d else d
                r["splats"] += len(entries)
            except Exception:
                pass
    r["ledger_obs"] = ledger_obs.get(root.name, 0)
    return r


def stage(r: dict) -> str:
    if r["splats"]:
        return "splats"
    if r["blocks"]:
        return "blocks"
    if r["hierarchy"] != "—":
        return "registry"
    if r["lio"]:
        return "ingested"
    if r["combined_bag"]:
        return "combined"
    return "raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()
    ledger_obs = ledger_obs_by_survey()
    rows = []
    for root in sorted(CITRUS.glob("*_*_Jackal")):
        if root.is_dir():
            rows.append(survey_row(root, ledger_obs))
    if KLAP.is_dir():
        k = survey_row(KLAP, ledger_obs)
        k["id"] = "klapmuts (Jan ZED)"
        rows.append(k)
        dec = KLAP / "dec_2025_a300"
        if dec.is_dir():
            mono_dir = dec / "monolithics"
            d = {"id": "klapmuts dec_2025 (A300 mcap)", "path": str(dec),
                 "size": du_gb(dec),
                 "combined_bag": any(dec.glob("*.mcap")) or (dec / "combined.bag").exists(),
                 "monolithics": "+".join(n for ok, n in
                                         [(any(mono_dir.glob("image_left*.monolithic")), "img"),
                                          ((mono_dir / "laser.monolithic").exists(), "lidar")]
                                         if ok) or False,
                 "lio": (mono_dir / "transform_lio.monolithic").exists()
                        or (mono_dir / "zed_transform.monolithic").exists(),
                 "kf20cm": (mono_dir / "image_left_kf20cm.monolithic").exists(),
                 "kf_pngs": count(dec / "kf_images", "kf_*.png"),
                 "markers": bool(count(dec / "sam3_ledger_v0", "*.npz")),
                 "hierarchy": (f"SAM3 ledger {count(dec / 'sam3_ledger_v0', '*.npz')} frames"
                               if count(dec / "sam3_ledger_v0", "*.npz") else "—"),
                 "masks": False, "blocks": {},
                 "splats": 0, "ledger_obs": 0}
            rows.append(d)

    ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    L = ["# Dataset registry",
         "",
         f"Generated {ts} by `automation/build_dataset_registry.py` — do not edit by hand;",
         "re-run the generator after ingest/pipeline milestones and commit the diff.",
         f"Ledger source: `{ledger_obs.get('_file', 'none')}`.",
         "",
         "| survey | size | stage | bag | monos | lio mono | kf mono | kf PNGs | markers | hierarchy | masks | blocks | splats | ledger obs |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    yn = lambda b: (b if isinstance(b, str) else "y") if b else "—"
    for r in rows:
        blocks = ", ".join(f"{k}:{v}" for k, v in r["blocks"].items()) if r["blocks"] else "—"
        L.append(f"| {r['id']} | {r['size']} | **{stage(r)}** | {yn(r['combined_bag'])} "
                 f"| {yn(r['monolithics'])} | {yn(r['lio'])} | {yn(r['kf20cm'])} "
                 f"| {r['kf_pngs'] or '—'} | {yn(r['markers'])} | {r['hierarchy']} "
                 f"| {yn(r['masks'])} | {blocks} | {r['splats'] or '—'} | {r['ledger_obs'] or '—'} |")
    L += ["",
          "Stage ladder: raw → combined → ingested (monolithics+LIO) → registry",
          "(markers+hierarchy) → blocks → splats. `ledger obs` = observations this",
          "survey contributes to the sankofa ledger.",
          ""]
    text = "\n".join(L)
    if args.stdout:
        print(text)
    else:
        OUT.write_text(text)
        print(f"wrote {OUT} ({len(rows)} surveys)")


if __name__ == "__main__":
    main()
