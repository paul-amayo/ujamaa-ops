#!/usr/bin/env python3
"""Prod/experimental separation + per-agent prod-ready checklists.

Paul (2026-08-14): "separate in the folder the prod and experimental, now we
can quickly tell which of the surveys are prod ready for all the agent
servers ... each dataset now has a prod ready checklist for each of the
ujamaa agents."

Writes, per survey:
  <survey>/prod/prod.json   manifest: per-agent checklists, all DERIVED
  <survey>/prod/<asset>     symlinks to the blessed artifacts (relative when
                            inside the survey dir, absolute when external)
and one top-level PROD.md matrix.

THE SEPARATION RULE: agent servers mount <survey>/prod/ ONLY. Anything in a
survey dir that prod/ does not link is experimental by definition. Nothing
is physically moved — splat run config.yml self-paths break on relocation
(proved by the 2026-08-14 apr_2026_zed move).

Agents = the four adinkra panel agents (ujamaa/adinkra/server.py):
  tassili   fly-through recon: stage2 splats + verdicts + embedder + hierarchy
  bateleur  canopy / per-tree state: trajectory + hierarchy + instance registry
  sankofa   temporal ledger: survey in the one-datum ledger_v2, multi-epoch site
  azalai    routing: site geometry + planting rows + absolute georef
(Spoor and Hapi run on demo-synthetic data — no per-survey assets yet.)

Checks never run GPU work; they read artifacts on disk, including verdict
records distilled from run logs (automation/distill_containment_verdicts.py).
States are DERIVED, never hand-maintained. Re-run after any milestone and
commit the PROD.md diff.
"""
import json
import re
import os
import time
from pathlib import Path

CITRUS = Path("/home/paperspace/data/citrus_all")
KLAP = Path("/home/paperspace/data/klapmuts")
LEDGER_V2 = CITRUS / "sankofa_substrate/ledger_v2.json"
EMB_ROOT = Path("/home/paperspace/data/high/nerf")
TOPDOWN = Path("/home/paperspace/code/ujamaa/project/bateleur_orchard_topdown.json")
OUT_MD = Path("/home/paperspace/code/PROD.md")

STAGE2_NAME = "stage2_censusinit_fw2"
IOU_FLOOR = 0.80  # 04-canon containment floor

SURVEYS = [CITRUS / s for s in
           ("01_13B_Jackal", "02_13B_Jackal", "03_13B_Jackal",
            "04_13D_Jackal", "05_13D_Jackal")] + \
          [KLAP / s for s in
           ("apr_2026_zed", "dec_2025_a300", "dec_2025_ten_rows")]

# survey -> embedder experiment-name preference (canon first), then newest.
EMB_PREF = {
    "01_13B_Jackal": ["01_13B_c20cos20"],
    "02_13B_Jackal": ["02_13B_v1g"],
    "03_13B_Jackal": ["03_13B_c20cos20"],
    "04_13D_Jackal": ["04_13D_v3vocab1k"],
    "05_13D_Jackal": ["05_13D_v1g", "05_13D_c20cos20"],
    "apr_2026_zed": ["klap_v1g", "klapmuts_v1g"],
}
EMB_TAG = {"01_13B_Jackal": "01_13B", "02_13B_Jackal": "02_13B",
           "03_13B_Jackal": "03_13B", "04_13D_Jackal": "04_13D",
           "05_13D_Jackal": "05_13D", "apr_2026_zed": "klap"}

# Role notes (annotations only — states stay derived):
NOTES = {
    "02_13B_Jackal": "ledger control epoch — registry+ledger by design, no splat planned",
    "dec_2025_a300": "pilot mcap — georef + SAM3 ledger seed",
    "dec_2025_ten_rows": "Dec ten-rows — ingest chain in progress (task #10)",
}


def mono_dir(root: Path) -> Path:
    return root / "monolithics" if (root / "monolithics").is_dir() else root


def find_hierarchy(root: Path):
    cands = sorted(root.glob("scene_graph*/marker_hierarchy*.json"),
                   key=lambda p: p.stat().st_mtime)
    if not cands:
        return None, 0, 0
    try:
        d = json.load(open(cands[-1]))
        p = d.get("_provenance", {})
        return (cands[-1], p.get("n_objects") or len(d.get("objects", [])),
                p.get("n_rows") or len(d.get("rows", [])))
    except Exception:
        return cands[-1], 0, 0


def find_prod_cfg(root: Path):
    """Config with the most stage2 runs wins; else the one with splats.json."""
    best = (0, None)
    for cfg in sorted((root / "blocks_ns").glob("*/")) if (root / "blocks_ns").is_dir() else []:
        n = len(list(cfg.glob(f"block_*/splat_runs_*/{STAGE2_NAME}")))
        if n > best[0]:
            best = (n, cfg)
    if best[1] is None:
        for cfg in sorted((root / "blocks_ns").glob("*/")) if (root / "blocks_ns").is_dir() else []:
            if (cfg / "splats.json").exists():
                return cfg
    return best[1]


def stage2_state(cfg: Path):
    # canonical blocks are block_NNN exactly; suffixed dirs (block_001_L095_sky
    # etc.) are experiment variants and stay out of the prod denominator
    def canon(name):
        return re.fullmatch(r"block_\d+", name)
    blocks = sorted(d.name for d in cfg.glob("block_*") if d.is_dir() and canon(d.name))
    done = sorted({d.parent.parent.name for d in cfg.glob(f"block_*/splat_runs_*/{STAGE2_NAME}")
                   if canon(d.parent.parent.name)
                   and list(d.glob("high/*/nerfstudio_models*/*.ckpt"))})
    return blocks, done


def find_embedder(sid: str):
    for name in EMB_PREF.get(sid, []):
        p = EMB_ROOT / name / "ckpts/model_best.pth"
        if p.exists():
            return p, name + " (canon)"
    tag = EMB_TAG.get(sid)
    if tag:
        cands = sorted(EMB_ROOT.glob(f"{tag}*/ckpts/model_best.pth"),
                       key=lambda p: p.stat().st_mtime)
        if cands:
            return cands[-1], cands[-1].parent.parent.name + " (newest, no canon tag)"
    return None, "no embedder ckpt found"


def ledger_obs():
    if not LEDGER_V2.exists():
        return {}
    d = json.load(open(LEDGER_V2))
    recs = []
    for v in (d.values() if isinstance(d, dict) else d):
        recs.extend(v if isinstance(v, list) else [v])
    obs = {}
    for r in recs:
        s = r.get("source_survey")
        if s:
            obs[s] = obs.get(s, 0) + 1
    return obs


def site_of(sid: str) -> str:
    if "_13B_" in sid:
        return "13B"
    if "_13D_" in sid:
        return "13D"
    return "klapmuts"


def check(cid, ok, evidence):
    return {"id": cid, "ok": bool(ok), "evidence": evidence}


def survey_manifest(root: Path, obs: dict):
    sid = root.name
    md = mono_dir(root)
    kf = (md / "image_left_kf20cm.monolithic").exists()
    lio = any((md / n).exists() for n in
              ("transform_lio.monolithic", "zed_transform.monolithic",
               "ins_transform.monolithic"))
    hier, n_obj, n_rows = find_hierarchy(root)
    cfg = find_prod_cfg(root)
    agents = {}

    # ---- tassili -----------------------------------------------------------
    c = [check("kdomain", kf and lio,
               f"kf20cm={'y' if kf else 'MISSING'} lio_mono={'y' if lio else 'MISSING'} ({md.name})")]
    if cfg:
        blocks, done = stage2_state(cfg)
        c.append(check("blocks", bool(blocks), f"{cfg.name}: {len(blocks)} blocks"))
        c.append(check("stage2", blocks and len(done) == len(blocks),
                       f"{len(done)}/{len(blocks)} blocks have {STAGE2_NAME} ckpt"
                       + (f" (missing {sorted(set(blocks) - set(done))[:4]})" if set(blocks) - set(done) else "")))
        vfiles = sorted(cfg.glob("verdicts_*.json"))
        if vfiles:
            v = json.load(open(vfiles[0]))
            rec = v.get("blocks", {})
            passing = [b for b, r in rec.items() if r.get("tree_iou_min", 0) >= IOU_FLOOR]
            missing = sorted({b.split("_")[1] for b in blocks} - set(rec))
            c.append(check("verdicts", blocks and not missing and len(passing) == len(rec),
                           f"{len(rec)} recorded, {len(passing)} pass floor {IOU_FLOOR}"
                           + (f"; unrecorded blocks {missing[:5]}" if missing else "")
                           + (f"; failing {sorted(set(rec) - set(passing))}" if set(rec) - set(passing) else "")))
        else:
            c.append(check("verdicts", False, f"no verdicts_*.json in {cfg.name} — run containment verdicts"))
        c.append(check("registered", (cfg / "splats.json").exists(),
                       f"splats.json {'present' if (cfg / 'splats.json').exists() else 'MISSING — export + register for the viewer'}"))
    else:
        c.append(check("blocks", False, "no blocks_ns config with stage2 or splats.json"))
    emb, emb_ev = find_embedder(sid)
    c.append(check("embedder", emb is not None, emb_ev))
    c.append(check("hierarchy", hier is not None,
                   f"{n_obj} obj / {n_rows} rows ({hier.parent.name})" if hier else "no marker_hierarchy"))
    agents["tassili"] = c

    # ---- bateleur ----------------------------------------------------------
    c = [check("trajectory", lio, "lio/odom mono present" if lio else "no odometry mono"),
         check("hierarchy", hier is not None and n_rows > 0,
               f"{n_obj} obj / {n_rows} rows" if hier else "no marker_hierarchy")]
    gids = root / "sam3_v2/global_ids.json"
    q4 = root / "sam3_v2_q4/global_ids.json"
    c.append(check("registry", gids.exists(),
                   ("global_ids.json" + (" + Q4 reproduction verified" if q4.exists() else ""))
                   if gids.exists() else "no sam3_v2/global_ids.json"))
    td_ok, td_ev = False, "no bateleur topdown export"
    if TOPDOWN.exists():
        try:
            td = json.load(open(TOPDOWN))
            td_sid = td.get("survey") or td.get("source_survey")
            td_ok = td_sid == sid
            td_ev = (f"topdown export slot holds {td_sid or 'unlabelled survey'} "
                     f"(single-slot; re-export with export_bateleur_topdown.py {root})")
            if td_ok:
                td_ev = "topdown export current for this survey"
        except Exception:
            td_ev = "topdown export unreadable"
    c.append(check("topdown", td_ok, td_ev))
    agents["bateleur"] = c

    # ---- sankofa -----------------------------------------------------------
    n_obs = obs.get(sid, 0)
    site = site_of(sid)
    siblings = [s for s in obs if s != sid and isinstance(s, str) and site_of(s) == site]
    assoc = sorted((CITRUS / "sankofa_substrate").glob("assoc_*.npz"))
    assoc_hit = [a.name for a in assoc if sid.split("_")[0] in a.name]
    c = [check("in_ledger", n_obs > 0,
               f"{n_obs} observations in {LEDGER_V2.name}" if n_obs else
               f"not in {LEDGER_V2.name}"
               + (f" — association ready ({assoc_hit[0]}), ledger rebuild pending" if assoc_hit else "")),
         check("one_datum", LEDGER_V2.exists(), f"{LEDGER_V2.name} (single 03-datum)"),
         check("multi_epoch", n_obs > 0 and bool(siblings),
               f"site {site} epochs in ledger: {sorted(siblings + ([sid] if n_obs else []))}")]
    agents["sankofa"] = c

    # ---- azalai ------------------------------------------------------------
    site_json = next((p / "site.json" for p in (root, root.parent) if (p / "site.json").exists()), None)
    georef = next((n for n in ("gps.monolithic", "gnsscorr_raw.npz")
                   if (md / n).exists() or (root / n).exists()), None)
    c = [check("site_geometry", site_json is not None,
               str(site_json) if site_json else "no site.json (planting geometry)"),
         check("rows", n_rows > 0, f"{n_rows} planting rows" if n_rows else "no rows in hierarchy"),
         check("georef", georef is not None,
               f"{georef} present" if georef else
               ("no GNSS on ZED rig — inherit georef via census association (planned)"
                if sid == "apr_2026_zed" else "no absolute georef source"))]
    agents["azalai"] = c

    links = {}
    if cfg:
        links["blocks"] = cfg
    if hier:
        links["hierarchy.json"] = hier
    if gids.exists():
        links["global_ids.json"] = gids
    if emb:
        links["embedder.ckpt"] = emb
    if cfg and sorted(cfg.glob("verdicts_*.json")):
        links["verdicts.json"] = sorted(cfg.glob("verdicts_*.json"))[0]
    if (root / "kf_domain_cache.npz").exists():
        links["kf_domain_cache.npz"] = root / "kf_domain_cache.npz"
    if n_obs > 0:
        links["ledger.json"] = LEDGER_V2
    if site_json:
        links["site.json"] = site_json

    return {"survey": sid, "root": str(root), "note": NOTES.get(sid, ""),
            "prod_cfg": str(cfg) if cfg else None, "agents": agents,
            "links": {k: str(v) for k, v in links.items()}}


def write_prod_tree(root: Path, man: dict):
    prod = root / "prod"
    prod.mkdir(exist_ok=True)
    # idempotent: replace only symlinks we manage; never delete real files
    for name, target in man["links"].items():
        lnk = prod / name
        if lnk.is_symlink():
            lnk.unlink()
        elif lnk.exists():
            print(f"  WARN {lnk} exists and is not a symlink — left untouched")
            continue
        t = Path(target)
        try:
            rel = os.path.relpath(t, prod)
            use = rel if not rel.startswith("../../../") else t
        except ValueError:
            use = t
        lnk.symlink_to(use)
    man["generated"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    man["generator"] = "automation/build_prod_manifests.py"
    man["rule"] = ("agent servers mount this prod/ dir ONLY; everything not "
                   "linked here is experimental")
    (prod / "prod.json").write_text(json.dumps(man, indent=1))


def main():
    obs = ledger_obs()
    manifests = []
    for root in SURVEYS:
        if not root.is_dir():
            continue
        man = survey_manifest(root, obs)
        write_prod_tree(root, man)
        manifests.append(man)

    ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    L = ["# Prod readiness — surveys × UJAMAA agents", "",
         f"Generated {ts} by `automation/build_prod_manifests.py` — do not edit by hand.",
         "Each survey has `<root>/prod/` (symlinks to blessed artifacts + `prod.json`",
         "checklist). **Agent servers mount `prod/` only; everything else in a survey",
         "dir is experimental.** Nothing is physically moved — splat run configs",
         "self-reference absolute paths and break on relocation.",
         "",
         "Agents = adinkra panel agents (`ujamaa/adinkra/server.py`). Spoor and Hapi",
         "run on demo-synthetic data and have no per-survey assets yet.",
         "",
         "| survey | tassili | bateleur | sankofa | azalai | note |",
         "|---|---|---|---|---|---|"]

    def cell(checks):
        bad = [c["id"] for c in checks if not c["ok"]]
        return "**READY**" if not bad else f"{len(checks) - len(bad)}/{len(checks)} ✗ " + ",".join(bad)

    for m in manifests:
        L.append(f"| {m['survey']} | " +
                 " | ".join(cell(m["agents"][a]) for a in
                            ("tassili", "bateleur", "sankofa", "azalai")) +
                 f" | {m['note']} |")
    L += ["", "## Per-survey checklists", ""]
    for m in manifests:
        L.append(f"### {m['survey']}"
                 + (f" — {m['note']}" if m["note"] else ""))
        if m["prod_cfg"]:
            L.append(f"prod block config: `{m['prod_cfg']}`")
        for a in ("tassili", "bateleur", "sankofa", "azalai"):
            L.append(f"- **{a}**")
            for c in m["agents"][a]:
                L.append(f"  - {'[x]' if c['ok'] else '[ ]'} {c['id']}: {c['evidence']}")
        L.append("")
    OUT_MD.write_text("\n".join(L) + "\n")
    ready = sum(1 for m in manifests for a in m["agents"].values()
                if all(c["ok"] for c in a))
    print(f"wrote {OUT_MD} + {len(manifests)} prod/ trees; "
          f"{ready}/{len(manifests) * 4} survey-agent cells READY")


if __name__ == "__main__":
    main()
