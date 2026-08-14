#!/usr/bin/env python3
"""Physical prod/experimental separation + per-agent prod-ready checklists.

Paul (2026-08-14): "physical prod folders — those are unsafe to delete; the
experimental are safe to delete; keeps our data easily auditable. Subfolders
in prod are monos, tassili, sankofa, azalai ... the rest of the files with
this survey are experimental."

Target layout per survey (e.g. klapmuts/dec_2025_a300/):
  prod/
    monos/      source of truth: *.monolithic(+.index), bags/mcaps, rig.json,
                metadata, kf_domain_cache, gnss — the dataset itself
    tassili/    recon serving: kf_images, masks, supervision, the BLESSED
                blocks_ns config (canonical block_NNN only) with stage1/stage2
                runs + verdicts + splats.json, embedder ckpt copy
    bateleur/   per-tree state: scene_graph (canon), sam3_v2 instance registry
    sankofa/    temporal assets owned by this survey (e.g. sam3_ledger_v0);
                the cross-survey ledger lives site-level in sankofa_substrate
    azalai/     routing: site.json (planting geometry)
  experimental/ EVERYTHING else — superseded scene_graphs, non-blessed block
                configs, variant blocks (block_001_L095_sky…), ablation
                workspaces, dead jsons. SAFE TO DELETE WHOLESALE.

MIGRATION RULE: every move into prod/ leaves a symlink shim at the old path,
so existing absolute paths (config.yml self-paths, transforms.json frame
paths, tool defaults, night queues) keep resolving unchanged — renames are
atomic, nothing needs a rewrite, and the live viewer keeps serving. Moves
into experimental/ leave NO shim: anything still referencing them should
break loudly. Deleting experimental/ can never dangle prod (links point
into prod, never out of it).

prod-dir ≠ READY: prod holds the CURRENT BEST (unsafe to delete) even when
the checklist is not green — e.g. 01's served legacy splats sit in prod
while tassili still fails the stage2+verdict bar. READY = checklist green.

States are DERIVED from artifacts (stage2 ckpts, distilled verdict jsons,
ledger_v2, site.json, gnss). Re-run after milestones; commit the PROD.md
diff. Idempotent: already-migrated surveys classify cleanly to no-ops.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
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
AGENTS = ("monos", "tassili", "bateleur", "sankofa", "azalai")

SURVEYS = [CITRUS / s for s in
           ("01_13B_Jackal", "02_13B_Jackal", "03_13B_Jackal",
            "04_13D_Jackal", "05_13D_Jackal")] + \
          [KLAP / s for s in
           ("apr_2026_zed", "dec_2025_a300", "dec_2025_ten_rows")]

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

NOTES = {
    "02_13B_Jackal": "ledger control epoch — joined the splat rotation 2026-08-14 (stage2 contingent on painted semantics)",
    "dec_2025_a300": "pilot mcap — georef + SAM3 ledger seed",
    "dec_2025_ten_rows": "Dec ten-rows — in week rotation, gated on pose-domain verification (INS=ENU0 frame)",
}

# ---------- classification (top-level survey entries) -----------------------
# name-pattern -> prod agent subfolder; unmatched non-blessed entries go to
# experimental. blocks_ns and scene_graph* are handled specially below.
MONOS_PAT = [r".*\.monolithic(\.index)?$", r".*\.mcap$", r"combined\.bag$",
             r"monolithics$", r"rig\.json$", r"metadata\.yaml$",
             r"kf_domain_cache\.npz$", r"gnsscorr_raw\.npz$"]
TASSILI_PAT = [r"kf_images$", r"frames$", r"sky_masks$", r"fg_masks$",
               r"supervision.*$", r"scene\.json$"]
SANKOFA_PAT = [r"sam3_ledger_v0$"]
AZALAI_PAT = [r"site\.json$"]
BATELEUR_PAT = [r"sam3_v2$"]
KEEP_ROOT = {"prod", "experimental"}


def pat_agent(name: str):
    for pats, agent in ((MONOS_PAT, "monos"), (TASSILI_PAT, "tassili"),
                        (BATELEUR_PAT, "bateleur"), (SANKOFA_PAT, "sankofa"),
                        (AZALAI_PAT, "azalai")):
        if any(re.fullmatch(p, name) for p in pats):
            return agent
    return None


def mono_dir(root: Path) -> Path:
    for d in (root / "prod/monos/monolithics", root / "prod/monos",
              root / "monolithics"):
        if d.is_dir():
            return d
    return root


def canon_block(name: str):
    return re.fullmatch(r"block_\d+", name)


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
    """Blessed config = the one with canonical-block stage2 census-init runs.
    THE BAR IS THE RECIPE OF RECORD (two-stage + census init) — Paul
    2026-08-14: "if the prod pipeline is 2 stage with census init, that is
    the bar for prod". Legacy fleets (01 C-config, 03 dedup) do NOT qualify:
    they are re-trainable under the recipe and live in experimental until
    then. Searches prod/tassili/blocks_ns too so re-runs are stable."""
    best = (0, None)
    for base in (root / "blocks_ns", root / "prod/tassili/blocks_ns"):
        if not base.is_dir():
            continue
        for cfg in sorted(base.glob("*/")):
            if not cfg.is_dir() or cfg.is_symlink():
                continue
            n = len({d.parent.parent.name for d in cfg.glob(f"block_*/splat_runs_*/{STAGE2_NAME}")
                     if canon_block(d.parent.parent.name)})
            if n > best[0]:
                best = (n, cfg)
    return best[1].resolve() if best[1] else None


def stage2_state(cfg: Path):
    blocks = sorted(d.name for d in cfg.glob("block_*")
                    if d.is_dir() and canon_block(d.name))
    done = sorted({d.parent.parent.name for d in cfg.glob(f"block_*/splat_runs_*/{STAGE2_NAME}")
                   if canon_block(d.parent.parent.name)
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


# ---------- migration -------------------------------------------------------
def mv_shim(src: Path, dst: Path, plan, execute, shim=True):
    """Move src -> dst; leave a relative symlink shim at src when shim=True."""
    if src.resolve() == dst.resolve() or not src.exists():
        return
    plan.append(f"  {'mv+shim' if shim else 'mv     '} {src.name:40s} -> {os.path.relpath(dst, src.parent)}")
    if not execute:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_symlink():  # old flat-prod link or naming shim: re-link, no move
        src.unlink()
        return
    if dst.exists():  # never overwrite (e.g. a recreated _logs after cleanup)
        print(f"  WARN dst exists, left {src} in place: {dst}")
        return
    shutil.move(str(src), str(dst))
    if shim:
        src.symlink_to(os.path.relpath(dst, src.parent))


def migrate(root: Path, execute: bool):
    plan = [f"== {root} =="]
    prod, exp = root / "prod", root / "experimental"
    # retire the flat symlink prod/ from the previous design
    if prod.is_dir() and not any((prod / a).is_dir() for a in AGENTS):
        for lnk in prod.iterdir():
            if lnk.is_symlink():
                plan.append(f"  unlink  flat-prod {lnk.name}")
                if execute:
                    lnk.unlink()
    if execute:
        for a in AGENTS:
            (prod / a).mkdir(parents=True, exist_ok=True)
        exp.mkdir(exist_ok=True)

    hier, _, _ = find_hierarchy(root)
    canon_sg = hier.parent.name if hier else None
    cfg = find_prod_cfg(root)

    # frame dirs the blessed config actually trains/serves from are tassili
    # assets even when legacy-named (e.g. 03's stream-named images/) — claim
    # them by READING transforms.json, not by name pattern
    claimed = set()
    if cfg:
        for tj in sorted(cfg.glob("block_*/transforms.json"))[:1]:
            try:
                fr = json.load(open(tj))["frames"]
                for f in (fr[0], fr[-1]):
                    for key in ("file_path", "mask_path"):
                        p = f.get(key)
                        if p and Path(p).is_relative_to(root):
                            claimed.add(Path(p).relative_to(root).parts[0])
            except Exception:
                pass
    # self-heal: a claimed dir quarantined by an earlier run comes back
    for name in sorted(claimed):
        prev = exp / name
        if prev.exists() and not (root / name).exists():
            mv_shim(prev, prod / "tassili" / name, plan, execute, shim=False)
            if execute:
                (root / name).symlink_to(
                    os.path.relpath(prod / "tassili" / name, root))

    # reconcile: a previously-blessed config that no longer meets the bar
    # (e.g. after the bar tightened to stage2-census-init) demotes to
    # experimental, and its root shim goes with it
    for pc in sorted((prod / "tassili/blocks_ns").glob("*/")) if (prod / "tassili/blocks_ns").is_dir() else []:
        if cfg and pc.resolve() == cfg.resolve():
            continue
        mv_shim(pc, exp / "blocks_ns" / pc.name, plan, execute, shim=False)
        shim_at = root / "blocks_ns" / pc.name
        if shim_at.is_symlink():
            plan.append(f"  unlink  stale shim blocks_ns/{pc.name}")
            if execute:
                shim_at.unlink()

    for entry in sorted(root.iterdir()):
        n = entry.name
        if n in KEEP_ROOT or entry.is_symlink():
            continue
        if n.startswith("scene_graph"):
            dest = (prod / "bateleur" / n) if n == canon_sg else (exp / n)
            mv_shim(entry, dest, plan, execute, shim=(n == canon_sg))
        elif n == "blocks_ns":
            for c in sorted(entry.iterdir()):
                if c.is_symlink():
                    continue
                if cfg and c.name == cfg.name:
                    # split experiment-variant block dirs out first
                    for b in sorted(c.glob("block_*")):
                        if b.is_dir() and not canon_block(b.name):
                            mv_shim(b, exp / "blocks_ns" / c.name / b.name,
                                    plan, execute, shim=False)
                    mv_shim(c, prod / "tassili/blocks_ns" / c.name, plan, execute)
                else:
                    mv_shim(c, exp / "blocks_ns" / c.name, plan, execute, shim=False)
        else:
            agent = pat_agent(n) or ("tassili" if n in claimed else None)
            if agent:
                mv_shim(entry, prod / agent / n, plan, execute)
            else:
                mv_shim(entry, exp / n, plan, execute, shim=False)

    # blessed embedder: physical copy into prod/tassili (small ckpt)
    emb, _ = find_embedder(root.name)
    if emb and execute:
        dst = prod / "tassili/embedder" / emb.parent.parent.name / "model_best.pth"
        if not dst.exists() and emb.stat().st_size < 1 << 30:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(emb, dst)
            plan.append(f"  copy    embedder {emb.parent.parent.name} -> prod/tassili/embedder/")
    return plan


# ---------- checklists (read via root shims — unchanged semantics) ----------
def check(cid, ok, evidence):
    return {"id": cid, "ok": bool(ok), "evidence": evidence}


def survey_manifest(root: Path, obs: dict):
    sid = root.name
    md = mono_dir(root)
    kf = (md / "image_left_kf20cm.monolithic").exists() \
        or (root / "image_left_kf20cm.monolithic").exists()
    lio = any((d / n).exists() for d in (md, root) for n in
              ("transform_lio.monolithic", "zed_transform.monolithic",
               "ins_transform.monolithic"))
    hier, n_obj, n_rows = find_hierarchy(root)
    cfg = find_prod_cfg(root)
    agents = {}

    c = [check("kdomain", kf and lio,
               f"kf20cm={'y' if kf else 'MISSING'} lio_mono={'y' if lio else 'MISSING'}")]
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

    gids = next((p for p in (root / "prod/bateleur/sam3_v2/global_ids.json",
                             root / "sam3_v2/global_ids.json") if p.exists()), None)
    q4 = (root / "experimental/sam3_v2_q4/global_ids.json")
    q4 = q4 if q4.exists() else (root / "sam3_v2_q4/global_ids.json")
    c = [check("trajectory", lio, "lio/odom mono present" if lio else "no odometry mono"),
         check("hierarchy", hier is not None and n_rows > 0,
               f"{n_obj} obj / {n_rows} rows" if hier else "no marker_hierarchy"),
         check("registry", gids is not None,
               ("global_ids.json" + (" + Q4 reproduction verified" if q4.exists() else ""))
               if gids else "no sam3_v2/global_ids.json")]
    td_ok, td_ev = False, (f"no bateleur topdown export — run "
                           f"export_bateleur_topdown.py {root}")
    per_survey = TOPDOWN.parent / f"bateleur_orchard_topdown_{sid}.json"
    slot = per_survey if per_survey.exists() else TOPDOWN
    if slot.exists():
        try:
            td = json.load(open(slot))
            td_sid = td.get("survey") or td.get("source_survey")
            td_ok = td_sid == sid
            td_ev = (f"topdown export {slot.name}" if td_ok else
                     f"topdown slot holds {td_sid or 'unlabelled survey'} "
                     f"— run export_bateleur_topdown.py {root}")
        except Exception:
            td_ev = "topdown export unreadable"
    c.append(check("topdown", td_ok, td_ev))
    agents["bateleur"] = c

    n_obs = obs.get(sid, 0)
    site = site_of(sid)
    siblings = [s for s in obs if s != sid and isinstance(s, str) and site_of(s) == site]
    assoc = sorted((CITRUS / "sankofa_substrate").glob("assoc_*.npz"))
    assoc_hit = [a.name for a in assoc if sid.split("_")[0] in a.name]
    agents["sankofa"] = [
        check("in_ledger", n_obs > 0,
              f"{n_obs} observations in {LEDGER_V2.name}" if n_obs else
              f"not in {LEDGER_V2.name}"
              + (f" — association ready ({assoc_hit[0]}), ledger rebuild pending" if assoc_hit else "")),
        check("one_datum", LEDGER_V2.exists(), f"{LEDGER_V2.name} (single 03-datum)"),
        check("multi_epoch", n_obs > 0 and bool(siblings),
              f"site {site} epochs in ledger: {sorted(siblings + ([sid] if n_obs else []))}")]

    site_json = next((p for p in (root / "prod/azalai/site.json", root / "site.json",
                                  root.parent / "site.json") if p.exists()), None)
    georef = next((n for n in ("gps.monolithic", "gnsscorr_raw.npz")
                   if (md / n).exists() or (root / n).exists()
                   or (root / "prod/monos" / n).exists()), None)
    agents["azalai"] = [
        check("site_geometry", site_json is not None,
              str(site_json) if site_json else "no site.json (planting geometry)"),
        check("rows", n_rows > 0, f"{n_rows} planting rows" if n_rows else "no rows in hierarchy"),
        check("georef", georef is not None,
              f"{georef} present" if georef else
              ("no GNSS on ZED rig — inherit georef via census association (planned)"
               if sid == "apr_2026_zed" else "no absolute georef source"))]

    return {"survey": sid, "root": str(root), "note": NOTES.get(sid, ""),
            "prod_cfg": str(cfg) if cfg else None, "agents": agents}


def du_gb(path: Path) -> str:
    try:
        out = subprocess.run(["du", "-sBG", str(path)], capture_output=True,
                             text=True, timeout=600).stdout
        return out.split()[0]
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--migrate", action="store_true",
                    help="execute the physical migration (default: plan only)")
    ap.add_argument("--survey", help="limit to one survey id")
    args = ap.parse_args()

    roots = [r for r in SURVEYS if r.is_dir()
             and (not args.survey or r.name == args.survey)]
    for root in roots:
        for line in migrate(root, execute=args.migrate):
            print(line)
    if not args.migrate:
        print("\n(plan only — rerun with --migrate to execute)")
        return

    obs = ledger_obs()
    manifests = []
    for root in roots:
        man = survey_manifest(root, obs)
        man["generated"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
        man["generator"] = "automation/build_prod_manifests.py"
        man["rule"] = ("prod/ = unsafe to delete (source + current best); "
                       "experimental/ = safe to delete wholesale; shims at old "
                       "paths keep absolute references resolving")
        man["sizes"] = {"prod": du_gb(root / "prod"),
                        "experimental": du_gb(root / "experimental")}
        (root / "prod/prod.json").write_text(json.dumps(man, indent=1))
        manifests.append(man)

    ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    L = ["# Prod readiness — surveys × UJAMAA agents", "",
         f"Generated {ts} by `automation/build_prod_manifests.py` — do not edit by hand.",
         "",
         "Layout per survey: `prod/{monos,tassili,bateleur,sankofa,azalai}` =",
         "PHYSICAL folders, **unsafe to delete** (source data + current-best",
         "outputs). `experimental/` = everything else, **safe to delete",
         "wholesale**. Symlink shims at the old root paths keep existing",
         "absolute references resolving; deleting experimental/ can never",
         "dangle prod. prod-dir ≠ READY — prod holds the current best even",
         "when the checklist is not yet green.",
         "",
         "Agents = adinkra panel agents (`ujamaa/adinkra/server.py`). Spoor and",
         "Hapi run on demo-synthetic data — no per-survey assets yet.",
         "",
         "| survey | prod | experimental (deletable) | tassili | bateleur | sankofa | azalai | note |",
         "|---|---|---|---|---|---|---|---|"]

    def cell(checks):
        bad = [c["id"] for c in checks if not c["ok"]]
        return "**READY**" if not bad else f"{len(checks) - len(bad)}/{len(checks)} ✗ " + ",".join(bad)

    for m in manifests:
        L.append(f"| {m['survey']} | {m['sizes']['prod']} | {m['sizes']['experimental']} | " +
                 " | ".join(cell(m["agents"][a]) for a in
                            ("tassili", "bateleur", "sankofa", "azalai")) +
                 f" | {m['note']} |")
    L += ["", "## Per-survey checklists", ""]
    for m in manifests:
        L.append(f"### {m['survey']}" + (f" — {m['note']}" if m["note"] else ""))
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
    print(f"\nwrote {OUT_MD} + {len(manifests)} prod.json; "
          f"{ready}/{len(manifests) * 4} survey-agent cells READY")


if __name__ == "__main__":
    main()
