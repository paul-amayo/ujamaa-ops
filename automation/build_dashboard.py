#!/usr/bin/env python3
"""UJAMAA roadmap dashboard — one self-contained HTML page from live sources:
UJAMAA_ROADMAP.md (milestones), lab_notebook/*.md (experiment trail),
~/logs/*.log (queue state), git/disk (live risk probes).

Rerun any time:  python3 automation/build_dashboard.py
Output:          lab_notebook/dashboard.html
"""
import datetime
import html
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

# Roots resolve to the workstation when it is present, otherwise to whatever
# checkout this script lives in (e.g. the ujamaa-ops mirror on a laptop), so
# the same builder runs off-box. UJAMAA_CODE / UJAMAA_LOGS override both.
_WORKSTATION = Path("/home/paperspace/code")
CODE = Path(os.environ.get("UJAMAA_CODE") or
            (_WORKSTATION if _WORKSTATION.exists()
             else Path(__file__).resolve().parent.parent))
LOGS = Path(os.environ.get("UJAMAA_LOGS") or (CODE.parent / "logs"))
OUT = CODE / "lab_notebook" / "dashboard.html"
LAUNCH = datetime.date(2026, 12, 15)


def sh(cmd, cwd=None):
    try:
        return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True,
                              text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


# ---------- roadmap phases ----------
def parse_roadmap():
    txt = (CODE / "UJAMAA_ROADMAP.md").read_text()
    phases = []
    cur = None
    for line in txt.splitlines():
        m = re.match(r"^### (.+)$", line)
        if m:
            if cur:
                phases.append(cur)
            cur = {"name": m.group(1), "done": 0, "open": 0, "items": []}
            continue
        if cur is None:
            continue
        m = re.match(r"^- \[(x| )\] (.+)$", line)
        if m:
            done = m.group(1) == "x"
            cur["done" if done else "open"] += 1
            cur["items"].append((done, m.group(2)[:110]))
    if cur:
        phases.append(cur)
    return [p for p in phases if p["done"] + p["open"] > 0]


# ---------- notebook trail ----------
def parse_notebook():
    entries = []
    for f in sorted((CODE / "lab_notebook").glob("2026-*.md"), reverse=True):
        for line in f.read_text().splitlines():
            m = re.match(r"^## (\d{4}-\d{2}-[\d/]+) [·—-]+ ?(.*)$", line)
            if m:
                entries.append((m.group(1), m.group(2)))
        if len(entries) >= 8:
            break
    return entries[:8]


def parse_autonomy():
    """AUTONOMY: runs=N clean=N interventions=N debugged=N lines, per day.

    Defined in lab_notebook/TESTING.md section 5. One line per day (the last
    one on a given date wins, so a day can revise its own score)."""
    days = {}
    for f in sorted((CODE / "lab_notebook").glob("2026-*.md")):
        day = None
        for line in f.read_text().splitlines():
            m = re.match(r"^## (\d{4}-\d{2}-\d{2})", line)
            if m:
                day = m.group(1)
            a = re.match(r"^AUTONOMY:\s*(.*)$", line.strip())
            if a and day:
                kv = dict(re.findall(r"(\w+)=(\d+)", a.group(1)))
                days[day] = {k: int(v) for k, v in kv.items()}
    return sorted(days.items())[-14:]


def parse_tassili_ledger():
    """Open rows of the new-dataset-to-tassili table in TESTING.md."""
    src = CODE / "lab_notebook" / "TESTING.md"
    if not src.exists():
        return []
    rows, in_tab = [], False
    for line in src.read_text().splitlines():
        if line.startswith("| dataset |"):
            in_tab = True
            continue
        if in_tab:
            if not line.startswith("|"):
                break
            c = [x.strip() for x in line.strip("|").split("|")]
            if len(c) >= 7 and not set(c[0]) <= {"-", " "}:
                rows.append(c)
    return rows


# ---------- queue / log state ----------
def tail_state(path, patterns, n=400):
    p = LOGS / path
    if not p.exists():
        return []
    lines = p.read_text(errors="replace").splitlines()[-n:]
    out = []
    for ln in lines:
        if any(re.search(pat, ln) for pat in patterns):
            out.append(ln.strip()[:130])
    return out[-6:]


def queue_state():
    rows = []
    rows += tail_state("queue_tonight.log", [r"^QUEUE"])
    rows += tail_state("klap_pilot_de2.log", [r"=== STAGE", r"PILOT-DONE",
                                             r"Traceback", r"FATAL"])
    seen, dedup = set(), []
    for r in rows:
        if r not in seen:
            seen.add(r)
            dedup.append(r)
    return dedup




# ---------- the week GPU rotation (week_prod_queue_20260814.sh) ----------
# Ground truth from artifacts: stage2 ckpts + verdict json on disk, the
# pointer files, the queue log's SLOT marks. Replaces the tail of two retired
# queue logs that sat here until 2026-08-21 (it showed "PILOT-DONE" for days).
WEEK_LOG = LOGS / "week_prod_20260814.log"
WEEK_STATE = LOGS / "week_prod_20260814_state"
ROTATION = [("05_13D_Jackal", "/home/paperspace/data/citrus_all"),
            ("04_13D_Jackal", "/home/paperspace/data/citrus_all"),
            ("apr_2026_zed", "/home/paperspace/data/klapmuts"),
            ("01_13B_Jackal", "/home/paperspace/data/citrus_all"),
            ("03_13B_Jackal", "/home/paperspace/data/citrus_all"),
            ("02_13B_Jackal", "/home/paperspace/data/citrus_all")]
PASS_FLOOR = 0.80   # distill_containment_verdicts pass_rule: tree_iou_min >= 0.80


def _mark_ts(s):
    """'08-21 09:45:36' -> datetime (the log carries no year)."""
    try:
        return datetime.datetime.strptime(
            f"{datetime.date.today().year}-{s}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def week_queue():
    if not WEEK_LOG.exists():
        return None
    lines = WEEK_LOG.read_text(errors="replace").splitlines()
    slots, open_slot = [], None
    for ln in lines:
        m = re.match(r"\[(\d\d-\d\d \d\d:\d\d:\d\d)\] SLOT (\S+) (block_\d+) "
                     r"(OK|FAILED|\(round (\d+)\))", ln)
        if not m:
            continue
        ts, survey, block, kind, rnd = m.groups()
        t = _mark_ts(ts)
        if kind.startswith("(round"):
            open_slot = {"survey": survey, "block": block, "start": t,
                         "round": int(rnd)}
        elif open_slot and open_slot["survey"] == survey \
                and open_slot["block"] == block:
            open_slot.update(end=t, outcome=kind)
            slots.append(open_slot)
            open_slot = None
    tail40 = "\n".join(lines[-40:])
    stop = next((k for k in ("CIRCUIT-BREAKER", "ABORT-DISK", "WEEK-DONE",
                             "ABORT:") if k in tail40), None)
    alive = sh("ps -eo args | grep -c '[w]eek_prod_queue_20260814.sh'") \
        not in ("", "0")
    rnd_f = WEEK_STATE / "round"
    rnd = rnd_f.read_text().strip() if rnd_f.exists() else "?"
    wd = LOGS / "queue_watchdog.log"
    wd_lines = wd.read_text().strip().splitlines() if wd.exists() else []
    watchdog = wd_lines[-1] if wd_lines else "no action yet (queue never found dead)"
    cur = None
    if open_slot:
        s, b = open_slot["survey"], open_slot["block"]
        slog = LOGS / f"week_{s}_b{b[-3:]}.log"
        stage = ""
        if slog.exists():
            for ln in reversed(slog.read_text(errors="replace").splitlines()):
                if ln.startswith(f"[{b[-3:]}]"):
                    stage = re.sub(r"treelod=\S+", "treelod=on", ln)[:90]
                    break
        canary = (Path(dict(ROTATION)[s]) / s / "prod/tassili/blocks_ns/lio_row100"
                  / b / "canary/canary.jsonl")
        step = ""
        if canary.exists():
            last = canary.read_text().strip().splitlines()
            if last:
                try:
                    j = json.loads(last[-1])
                    step = f"step {j.get('step')} · train FG {j.get('train_fg')} dB"
                except Exception:
                    pass
        mins = (int((datetime.datetime.now() - open_slot["start"]).total_seconds() // 60)
                if open_slot["start"] else None)
        cur = dict(open_slot, stage=stage, step=step, mins=mins)
    rows = []
    for s, base in ROTATION:
        cfg = Path(base) / s / "prod/tassili/blocks_ns/lio_row100"
        blocks = sorted(d for d in cfg.glob("block_*") if re.search(r"block_\d+$", d.name))
        built = sum(1 for d in blocks if list(d.glob(
            "splat_runs_FEATFIX/stage2_censusinit_*/high/*/nerfstudio_models*/*.ckpt")))
        nxt_f = WEEK_STATE / f"{s}.next"
        nxt = None
        if nxt_f.exists() and nxt_f.read_text().strip().isdigit():
            nxt = int(nxt_f.read_text().strip())
        vj = cfg / "verdicts_censusinit_fw2.json"
        rec = npass = 0
        if vj.exists():
            try:
                vb = json.loads(vj.read_text()).get("blocks", {})
                rec = len(vb)
                npass = sum(1 for r in vb.values()
                            if (r.get("tree_iou_min") or 0) >= PASS_FLOOR)
            except Exception:
                pass
        # a FAILED marker is REAL only if the block still has no stage2 ckpt
        # and sits below the pointer; markers above a rewound pointer or on a
        # block that later trained OK are stale (the 08-20 march left 28)
        failed = sorted(int(p.name.split("block_")[1].split(".")[0])
                        for p in WEEK_STATE.glob(f"{s}.block_*.FAILED"))
        has_ck = {int(d.name[6:]) for d in blocks if list(d.glob(
            "splat_runs_FEATFIX/stage2_censusinit_*/high/*/nerfstudio_models*/*.ckpt"))}
        tried = [f for f in failed
                 if nxt is not None and f < nxt and f not in has_ck]
        oks = [x for x in slots if x["survey"] == s and x["outcome"] == "OK"]
        last_ok = max((x["end"] for x in oks if x["end"]), default=None)
        rows.append(dict(survey=s, n_blocks=len(blocks), built=built, next=nxt,
                         rec=rec, npass=npass, failed=tried,
                         stale_failed=len(failed) - len(tried), last_ok=last_ok,
                         oks=len(oks)))
    now = datetime.datetime.now()

    def window(hours):
        w = [x for x in slots if x["end"]
             and (now - x["end"]).total_seconds() < hours * 3600]
        ok = [x for x in w if x["outcome"] == "OK"]
        durs = sorted(int((x["end"] - x["start"]).total_seconds() // 60)
                      for x in ok if x["start"] and x["end"])
        return len(ok), len(w) - len(ok), (durs[len(durs) // 2] if durs else None)
    ok_day, fail_day, med = window(24)
    ok_12, fail_12, med_12 = window(12)
    return dict(alive=alive, stop=stop, round=rnd, watchdog=watchdog, cur=cur,
                rows=rows, recent=slots[-10:], ok_day=ok_day,
                fail_day=fail_day, med=med, ok_12=ok_12, fail_12=fail_12,
                med_12=med_12,
                gpu=sh("nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader"),
                disk=sh("df -h / | awk 'NR==2{print $4}'"))


def week_queue_html(q, esc):
    if q["stop"]:
        st, label = "serious", (f"STOPPED DELIBERATELY ({q['stop']}) — needs "
                                "diagnosis; the watchdog will not restart it")
    elif q["alive"]:
        st, label = "good", "running"
    else:
        st, label = "critical", ("DEAD — watchdog relaunches at :13/:43 unless "
                                 "a deliberate stop is logged")
    cur = q["cur"]
    if cur:
        cur_txt = (f"{esc(cur['survey'])} {esc(cur['block'])} (round {cur['round']})"
                   f" · {cur['mins']} min · {esc(cur['stage'])}"
                   + (f" · {esc(cur['step'])}" if cur["step"] else ""))
    else:
        cur_txt = "between slots"
    tp = f"12 h: {q['ok_12']} OK / {q['fail_12']} failed"
    if q["med_12"]:
        tp += f" (median OK slot {q['med_12']} min)"
    tp += f" · 24 h: {q['ok_day']} OK / {q['fail_day']} failed"
    head = (
        "<div class='wrap'><table>"
        f"<tr><td>rotation</td><td class='val'><span class='st st-{st}'>{ICON[st]} "
        f"{esc(label)}</span></td><td class='muted'>round {esc(q['round'])} · {tp}</td></tr>"
        f"<tr><td>current slot</td><td class='val' colspan='2'>{cur_txt}</td></tr>"
        f"<tr><td>GPU · disk</td><td class='val'>{esc(q['gpu'])}</td>"
        f"<td class='muted'>{esc(q['disk'])} free (ABORT-DISK floor 15G)</td></tr>"
        f"<tr><td>watchdog</td><td class='muted' colspan='2'>{esc(q['watchdog'])}</td></tr>"
        "</table></div>")
    trs = []
    for r in q["rows"]:
        pct = int(100 * r["built"] / r["n_blocks"]) if r["n_blocks"] else 0
        if r["next"] is None:
            nxt = "?"
        elif r["next"] >= r["n_blocks"]:
            nxt = "done"
        else:
            nxt = f"b{r['next']:03d}"
        failed = ", ".join(f"b{f:03d}" for f in r["failed"]) or "—"
        if r["stale_failed"]:
            failed += (f" <span class='muted small'>(+{r['stale_failed']} stale "
                       "markers: trained OK later or above the pointer)</span>")
        if r["rec"] and r["npass"] / r["rec"] >= 0.7:
            vst = "good"
        elif r["rec"] and r["npass"]:
            vst = "warning"
        elif r["rec"]:
            vst = "serious"
        else:
            vst = "warning"
        last_ok = r["last_ok"].strftime("%m-%d %H:%M") if r["last_ok"] else "—"
        trs.append(
            f"<tr><td>{esc(r['survey'])}</td><td class='val'>{r['built']}/{r['n_blocks']}</td>"
            f"<td style='min-width:120px'><div class='bar'><div class='fill' "
            f"style='width:{pct}%'></div></div></td>"
            f"<td class='val'>{esc(nxt)}</td><td class='val'>{r['oks']}</td>"
            f"<td class='muted'>{last_ok}</td>"
            f"<td><span class='st st-{vst}'>{ICON[vst]} {r['npass']}/{r['rec']}</span></td>"
            f"<td class='small'>{failed}</td></tr>")
    table = ("<div class='wrap'><table><tr class='muted'><td>survey</td>"
             "<td>stage2 built</td><td></td><td>next</td><td>OK slots</td>"
             "<td>last OK</td><td>verdict pass (tree IoU min &ge; 0.80)</td>"
             "<td>failed (tried, no ckpt)</td></tr>" + "".join(trs) + "</table></div>")
    hist = []
    for x in reversed(q["recent"]):
        d = (int((x["end"] - x["start"]).total_seconds() // 60)
             if x["start"] and x["end"] else "?")
        s = "good" if x["outcome"] == "OK" else "serious"
        when = x["end"].strftime("%m-%d %H:%M") if x["end"] else "?"
        hist.append(f"<div class='qline'><span class='st st-{s}'>{ICON[s]}</span> {when} "
                    f"{esc(x['survey'])} {esc(x['block'])} {esc(x['outcome'])} · "
                    f"{d} min · round {x['round']}</div>")
    return (head + table
            + "<div class='small muted' style='margin:.4rem 0 .2rem'>recent slots</div>"
            + "".join(hist))

# ---------- the science: parsed from lab_notebook/PILLARS.md ----------
# It used to be a Python literal in this file, which meant updating the
# science story was a CODE edit — so it drifted 25 notebook entries behind
# (2026-08-04). It now lives beside the notebook and is updated in the same
# commit as the day's entry; the freshness row below polices that.
def parse_pillars():
    src = CODE / "lab_notebook" / "PILLARS.md"
    if not src.exists():
        return []
    out, cur = [], None
    for line in src.read_text().splitlines():
        m = re.match(r"^## (.+?) — (.+)$", line)
        if m:
            if cur:
                out.append(cur)
            cur = dict(name=m.group(1), tag=m.group(2), question="",
                       solved=[], open=[], next="")
            continue
        if cur is None:
            continue
        m = re.match(r"^\*\*Q:\*\* (.+)$", line)
        if m:
            cur["question"] = m.group(1); continue
        m = re.match(r"^\*\*Next:\*\* (.+)$", line)
        if m:
            cur["next"] = m.group(1); continue
        m = re.match(r"^- \[(good|warning|serious|critical)\] (.+?)(?: — (.+))?$", line)
        if m:
            st, claim, ev = m.group(1), m.group(2), m.group(3) or ""
            (cur["solved"] if st == "good" else cur["open"]).append((claim, ev, st))
    if cur:
        out.append(cur)
    return out


PILLARS = parse_pillars()


# ---------- legacy scoreboard (kept for the risk table) ----------
SCOREBOARD = [
    ("Tree pointing (citrus b001, clean ckpt)", "95.8%", "good", "target met"),
    ("Row pointing", "99.1%", "good", "target met"),
    ("Fruit→fruit cross-level pointing", "F4 0.0 → S1 1.1 → S2 12.0%",
     "warning", "S3a queued: densification-surviving protection"),
    ("Fruit rendered radius (target 7.1)", "S2 4.58 (bimodal)", "warning",
     "protected core at word; bulk unprotected"),
    ("klapmuts census", "866 instances (pitch-split validated)", "good",
     "duplicates + far-left rows open"),
    ("klapmuts rows", "14 (RANSAC-init, 0.12 s)", "good",
     "CORAL optimiser: no-op@0/0, collapse@0.5 — bypassed"),
    ("Dec-2025 localisation", "pre-flight pending", "warning",
     "GPS-free; INS is local ENU (no datum)"),
]


# ---------- live risk probes ----------
def risks():
    out = []
    du = shutil.disk_usage("/")
    free_gb = du.free / 1e9
    out.append(("Disk", f"{free_gb:.0f} GB free",
                "good" if free_gb > 200 else
                "serious" if free_gb > 60 else "critical"))
    for name, d in [("high/", CODE / "high"),
                    ("aru_sil_core/src", CODE / "aru_sil_core" / "src")]:
        if not d.exists():
            # off-box build (ujamaa-ops mirror): no sub-repo to probe. Say so
            # rather than reporting a git count that was never taken.
            out.append((f"Uncommitted — {name}", "not in this checkout",
                        "unknown"))
            continue
        n = sh("git status --porcelain | wc -l", cwd=d)
        try:
            n = int(n)
        except ValueError:
            n = -1
        out.append((f"Uncommitted — {name}", f"{n} files",
                    "good" if n == 0 else "warning" if n < 8 else "serious"))
    reg = ("UNVERIFIED under K-indexed poses — gates 4D/Sankofa trust")
    out.append(("Citrus registries", reg, "serious"))
    out.append(("Single GPU", "all training serialised through one card",
                "warning"))
    out.append(("Fruit level (demo differentiator)",
                "gates not yet passed — S2 12% vs target", "serious"))
    out.append(("A200 staged-delete + src/config",
                "awaiting manual rm", "warning"))
    return out


# ---------- code health (computed live) ----------
# Scoping matters: whole-repo lint slanders vendored/legacy code and hides the
# active surface's real signal, so each scored unit declares its paths.
# Breakage = E9 (syntax) + F821 (undefined name)  — "this line cannot run".
# Hygiene  = F401 (unused import) + F841 (unused variable), per KLOC.
RUFF = Path.home() / ".local/bin/ruff"

HEALTH_UNITS = [
    dict(name="high/", root=CODE / "high", paths=["high", "tests"],
         suite="tests", note="training method (HiGH)"),
    dict(name="aru_sil_core/src — active", root=CODE / "aru_sil_core/src",
         paths=["scripts", "interfaces/rerun", "tests"], suite="tests",
         note="pipeline core: scripts + rerun/HiGH + tests"),
    dict(name="aru_sil_core/src — legacy+vendored", root=CODE / "aru_sil_core/src",
         paths=["interfaces/python", "integration", "thirdparty"], suite=None,
         note="teach_repeat ×2 + InstantSplat; quarantine candidate",
         legacy=True),
    dict(name="ujamaa/", root=CODE / "ujamaa", paths=["."], suite=None,
         note="design handoff bundle"),
    dict(name="nerf_new/ (nerfstudio fork)", root=CODE / "nerf_new",
         paths=["nerfstudio"], suite=None, no_f821=True,
         note="F821 skipped: jaxtyping shape symbols"),
    dict(name="automation/ (top-level repo)", root=CODE / "automation",
         paths=["."], suite=None,
         note="queue scripts + this dashboard; in code/ top-level repo"),
]


def _ruff_count(root, paths, select):
    if not RUFF.exists():
        return None
    targets = [str(root / p) for p in paths if (root / p).exists()]
    if not targets:
        return 0
    try:
        r = subprocess.run(
            [str(RUFF), "check", "--no-cache", "--isolated", "--select",
             select, "--output-format", "concise", *targets],
            capture_output=True, text=True, timeout=120)
        return sum(1 for ln in r.stdout.splitlines() if ".py:" in ln)
    except Exception:
        return None


def _unit_stats(u):
    root = u["root"]
    py = []
    if u.get("no_vcs"):
        for p in u["paths"]:
            py += [f for f in (root / p).rglob("*.py")]
    else:
        for f in sh("git ls-files '*.py'", cwd=root).splitlines():
            if any(f == p or f.startswith(p.rstrip("/") + "/") or p == "."
                   for p in u["paths"]):
                py.append(root / f)
    py = [f for f in py if Path(f).exists()]   # tracked-but-deleted files
    loc = 0
    for f in py:
        try:
            loc += sum(1 for _ in open(f, errors="replace"))
        except OSError:
            pass
    breakage = _ruff_count(root, u["paths"],
                           "E9" if u.get("no_f821") else "E9,F821")
    hygiene = _ruff_count(root, u["paths"], "F401,F841")
    tmp_refs = sum(1 for f in py
                   if "/tmp/" in open(f, errors="replace").read())
    big = sum(1 for f in py
              if sum(1 for _ in open(f, errors="replace")) > 600)
    dirty = age = 0
    if not u.get("no_vcs"):
        dirty = len(sh("git status --porcelain", cwd=root).splitlines())
        ts = sh("git log -1 --format=%ct", cwd=root)
        if ts.isdigit():
            age = (datetime.datetime.now()
                   - datetime.datetime.fromtimestamp(int(ts))).days
    tests = None
    if u.get("suite"):
        out = sh(f"python3 -m pytest {u['suite']} -q 2>&1 | tail -1", cwd=root)
        tests = out.strip()
    return dict(loc=loc, files=len(py), breakage=breakage, hygiene=hygiene,
                tmp_refs=tmp_refs, big=big, dirty=dirty, age=age, tests=tests)


def _score(u, s):
    sc = 100
    why = []
    if u.get("no_vcs"):
        sc -= 45
        why.append("no VCS -45")
    if s["breakage"]:
        d = min(30, 2 * s["breakage"])
        sc -= d
        why.append(f"breakage {s['breakage']} -{d}")
    kloc = max(s["loc"] / 1000, 0.001)
    hyg = (s["hygiene"] or 0) / kloc
    if hyg > 1:
        d = min(15, round(2 * hyg))
        sc -= d
        why.append(f"hygiene {hyg:.0f}/kloc -{d}")
    if u.get("suite"):
        if s["tests"] and "passed" in s["tests"] and "failed" not in s["tests"]:
            why.append("suite green")
        else:
            sc -= 30
            why.append("suite RED -30")
    elif not u.get("legacy") and not u.get("no_vcs"):
        sc -= 15
        why.append("no tests -15")
    if s["dirty"]:
        d = min(15, 3 * s["dirty"])
        sc -= d
        why.append(f"dirty {s['dirty']} -{d}")
    if s["dirty"] and s["age"] > 7:
        sc -= 10
        why.append(f"stale+dirty {s['age']}d -10")
    if s["tmp_refs"]:
        d = min(10, 2 * s["tmp_refs"])
        sc -= d
        why.append(f"/tmp refs {s['tmp_refs']} -{d}")
    if s["big"]:
        d = min(10, s["big"])
        sc -= d
        why.append(f">{s['big']} files over 600 loc -{d}")
    return max(0, sc), "; ".join(why)


def code_health():
    rows = []
    for u in HEALTH_UNITS:
        if not u["root"].exists():
            rows.append((u, dict(loc=0, files=0, breakage=None, hygiene=None,
                                 tmp_refs=0, big=0, dirty=0, age=0, tests=None,
                                 absent=True), None,
                         "not in this checkout — unmeasured", "unknown"))
            continue
        s = _unit_stats(u)
        score, why = _score(u, s)
        st = ("good" if score >= 85 else
              "warning" if score >= 65 else "serious")
        rows.append((u, s, score, why, st))
    return rows


HEALTH_FLAGS = [
    ("automation/ + lab_notebook/ now in code/ top-level repo (2026-07-31)",
     "was: no VCS at all (June-2026 regression class); commit-after-green "
     "now applies to queue scripts and the notebook too", "good"),
    ("teach_repeat duplicated wholesale (interfaces/python + integration/)",
     "66 undefined-name findings in dead legacy; quarantine or rm", "warning"),
    ("nerf_new fork: uncommitted local mods, last commit 94 days old",
     "the running training env drifts unversioned", "warning"),
    ("11 active scripts reference /tmp paths",
     "volatile-dependency class (killed the overnight resume once)", "warning"),
]


ICON = {"good": "&#9679;", "warning": "&#9650;", "serious": "&#9632;",
        "critical": "&#10006;", "unknown": "&#9675;"}
SNAME = {"good": "OK", "warning": "WATCH", "serious": "RISK",
         "critical": "CRITICAL", "unknown": "n/a"}


def narrative_freshness():
    """Is the hand-curated PILLARS block keeping up with the lab notebook?

    build_dashboard runs daily from cron, but PILLARS is written by hand —
    so the mechanical parts (metrics, risks, queue, trail) can be current
    while the science story silently rots. Surface the drift instead.
    """
    pill_day = sh("git log -1 --format=%cI -- lab_notebook/PILLARS.md",
                  cwd=CODE)[:10]
    nb_entries = []
    for f in sorted((CODE / "lab_notebook").glob("2026-*.md"), reverse=True):
        for line in f.read_text().splitlines():
            m = re.match(r"^## (\d{4}-\d{2}-\d{2})", line)
            if m:
                nb_entries.append(m.group(1))
        if nb_entries:
            break
    newest = max(nb_entries) if nb_entries else "—"
    behind = 0
    if pill_day and nb_entries:
        behind = sum(1 for d in nb_entries if d > pill_day)
    st = "good" if behind == 0 else "warning" if behind <= 4 else "serious"
    return pill_day or "?", newest, behind, st


def build():
    today = datetime.date.today()
    days = (LAUNCH - today).days
    phases = parse_roadmap()
    notes = parse_notebook()
    qs = queue_state()
    rk = risks()

    def esc(s):
        return html.escape(str(s))

    phase_html = ""
    for p in phases:
        tot = p["done"] + p["open"]
        pct = int(100 * p["done"] / tot) if tot else 0
        phase_html += f'''
        <div class="phase">
          <div class="phead"><span>{esc(p["name"])}</span>
            <span class="muted">{p["done"]}/{tot}</span></div>
          <div class="bar"><div class="fill" style="width:{pct}%"></div></div>
        </div>'''

    def prow(items):
        return "".join(
            f"<tr><td>{esc(k)}</td><td class='val'>{esc(v)}</td>"
            f"<td><span class='st st-{st}'>{ICON[st]} {SNAME[st]}</span></td></tr>"
            for k, v, st in items)
    pillar_html = ""
    for pl in PILLARS:
        rows = prow([(k, v, st) for k, v, st in pl["solved"]])
        rows += prow([(k, v, st) for k, v, st in pl["open"]])
        pillar_html += f"""
        <div class='pillar'>
          <h2>{esc(pl['name'])} <span class='muted tagline'>— {esc(pl['tag'])}</span></h2>
          <div class='q'>{esc(pl['question'])}</div>
          <div class='wrap'><table>{rows}</table></div>
          <div class='next'><b>Next:</b> {esc(pl['next'])}</div>
        </div>"""

    score_html = "".join(
        f'''<tr><td>{esc(k)}</td><td class="val">{esc(v)}</td>
        <td><span class="st st-{s}">{ICON[s]} {SNAME[s]}</span></td>
        <td class="muted">{esc(note)}</td></tr>'''
        for k, v, s, note in SCOREBOARD)

    risk_html = "".join(
        f'''<tr><td>{esc(k)}</td><td class="val">{esc(v)}</td>
        <td><span class="st st-{s}">{ICON[s]} {SNAME[s]}</span></td></tr>'''
        for k, v, s in rk)

    ch = code_health()
    health_html = "".join(
        f'''<tr><td>{esc(u["name"])}<div class="muted small">{esc(u["note"])}</div></td>
        <td class="val">{"—" if s.get("absent") else f'{s["loc"] / 1000:.1f}k'}</td>
        <td class="val">{esc(s["tests"] or "—")}</td>
        <td class="val">{"—" if s.get("absent") else (s["breakage"] if s["breakage"] is not None else "?")}</td>
        <td class="val">{"—" if s.get("absent") else f'{(s["hygiene"] or 0) / max(s["loc"] / 1000, .001):.0f}'}</td>
        <td class="val">{"—" if s.get("absent") else s["dirty"]}</td>
        <td><span class="st st-{st}">{ICON[st]} {SNAME[st] if score is None else score}</span></td></tr>
        <tr class="why"><td colspan="7" class="muted small">{esc(why)}</td></tr>'''
        for u, s, score, why, st in ch)
    flags_html = "".join(
        f'''<tr><td>{esc(k)}</td><td class="muted">{esc(v)}</td>
        <td><span class="st st-{s}">{ICON[s]} {SNAME[s]}</span></td></tr>'''
        for k, v, s in HEALTH_FLAGS)
    scored = [score for u, s, score, why, st in ch
              if not u.get("legacy") and score is not None]
    health_min = min(scored) if scored else "—"
    pill_day, nb_newest, nb_behind, nb_st = narrative_freshness()
    fresh_html = (
        f'<tr><td>Science narrative (PILLARS) vs lab notebook</td>'
        f'<td class="val">curated {esc(pill_day)} · newest entry {esc(nb_newest)}</td>'
        f'<td><span class="st st-{nb_st}">{ICON[nb_st]} '
        f'{"in sync" if nb_behind == 0 else str(nb_behind) + " entries ahead"}'
        f'</span></td></tr>')

    wq = week_queue()
    if wq:
        q_html = week_queue_html(wq, esc)
    else:
        q_html = ("".join(f"<div class='qline'>{esc(q)}</div>" for q in qs)
                  or "<div class='qline muted'>no active queue lines</div>")

    auto = parse_autonomy()
    if auto:
        arows = []
        for day, kv in reversed(auto):
            r, c = kv.get("runs", 0), kv.get("clean", 0)
            iv, dbg = kv.get("interventions", 0), kv.get("debugged", 0)
            pct = 100 * c / r if r else 0
            ast = ("good" if iv == 0 and dbg == 0 else
                   "warning" if iv + dbg <= 3 else "serious")
            arows.append(
                f'<tr><td class="muted">{esc(day)}</td>'
                f'<td class="val">{c}/{r} clean ({pct:.0f}%)</td>'
                f'<td class="val">{iv}</td><td class="val">{dbg}</td>'
                f'<td><span class="st st-{ast}">{ICON[ast]}</span></td></tr>')
        auto_html = ('<div class="wrap"><table><tr class="muted">'
                     '<td>day</td><td>runs clean</td><td>interventions</td>'
                     '<td>debugged</td><td></td></tr>'
                     + "".join(arows) + "</table></div>")
    else:
        auto_html = ('<div class="muted small">no AUTONOMY: lines in the '
                     'notebook yet — format in lab_notebook/TESTING.md §5</div>')

    ledger = parse_tassili_ledger()
    open_rows = [r for r in ledger if "open" in r[4] or "pending" in r[4]]
    led_html = ("".join(
        f'<tr><td>{esc(r[0])}</td><td class="muted">{esc(r[1])}</td>'
        f'<td class="val">{esc(r[4])}</td><td class="muted">{esc(r[6])}</td></tr>'
        for r in ledger) or "")
    led_html = (f'<div class="wrap"><table><tr class="muted"><td>dataset</td>'
                f'<td>landed</td><td>days to tassili</td><td>notes</td></tr>'
                f'{led_html}</table></div>') if ledger else ""

    notes_html = "".join(
        f"<tr><td class='muted'>{esc(d)}</td><td>{esc(t)}</td></tr>"
        for d, t in notes)

    page = f"""<meta charset="utf-8">
<title>UJAMAA — roadmap dashboard</title>
<style>
:root {{ --ink:#1a1a19; --muted:#6f6e66; --line:#e4e2da; --card:#faf9f5;
  --good:#1a7f37; --warning:#9a6700; --serious:#cf222e; --critical:#82071e;
  --accent:#4969ed; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --ink:#e8e6df; --muted:#a09e94; --line:#3a3934; --card:#232320;
    --good:#3fb950; --warning:#d29922; --serious:#f85149; --critical:#ff7b72; }}
  body {{ background:#191917; }} }}
body {{ font:15px/1.5 -apple-system,'Segoe UI',sans-serif; color:var(--ink);
  max-width:1080px; margin:2rem auto; padding:0 1.2rem; }}
h1 {{ font-size:1.5rem; margin-bottom:.2rem; }}
h2 {{ font-size:1.05rem; margin:1.6rem 0 .6rem; }}
.muted {{ color:var(--muted); }}
.hero {{ display:flex; gap:1rem; flex-wrap:wrap; margin:1rem 0; }}
.tile {{ background:var(--card); border:1px solid var(--line);
  border-radius:10px; padding:.8rem 1.1rem; min-width:150px; }}
.tile .n {{ font-size:1.6rem; font-weight:700; }}
.tile .l {{ font-size:.8rem; color:var(--muted); }}
.phase {{ margin:.45rem 0; }}
.phead {{ display:flex; justify-content:space-between; font-size:.9rem; }}
.bar {{ height:8px; background:var(--line); border-radius:4px; }}
.fill {{ height:8px; background:var(--accent); border-radius:4px; }}
table {{ border-collapse:collapse; width:100%; font-size:.88rem; }}
td {{ padding:.35rem .5rem; border-top:1px solid var(--line);
  vertical-align:top; }}
td.val {{ font-weight:600; white-space:nowrap; }}
.st {{ font-weight:700; font-size:.78rem; white-space:nowrap; }}
.st-good {{ color:var(--good); }} .st-warning {{ color:var(--warning); }}
.st-serious {{ color:var(--serious); }} .st-critical {{ color:var(--critical); }}
.st-unknown {{ color:var(--muted); }}
.qline {{ font:12px/1.6 ui-monospace,monospace; background:var(--card);
  border-left:3px solid var(--accent); padding:.15rem .6rem; margin:.15rem 0;
  overflow-x:auto; white-space:nowrap; }}
.wrap {{ overflow-x:auto; }}
.pillar {{ background:var(--card); border:1px solid var(--line);
  border-radius:12px; padding: .2rem 1rem .8rem; margin:1rem 0; }}
.pillar h2 {{ margin:.8rem 0 .3rem; }}
.tagline {{ font-weight:400; font-size:.85rem; }}
.q {{ font-style:italic; color:var(--muted); margin:.2rem 0 .6rem;
  border-left:3px solid var(--accent); padding-left:.6rem; }}
.next {{ font-size:.85rem; margin-top:.6rem; }}
.small {{ font-size:.78rem; }}
tr.why td {{ border-top:none; padding-top:0; }}
</style>
<h1>UJAMAA — roadmap dashboard</h1>
<div class="muted">generated {esc(datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))}
 · sources: UJAMAA_ROADMAP.md · lab_notebook · ~/logs · live git/disk</div>
<div class="hero">
  <div class="tile"><div class="n">{days}</div><div class="l">days to launch (Dec 15)</div></div>
  <div class="tile"><div class="n">{sum(p['done'] for p in phases)}/{sum(p['done']+p['open'] for p in phases)}</div><div class="l">roadmap items done</div></div>
  <div class="tile"><div class="n">{len([r for r in rk if r[2] in ('serious','critical')])}</div><div class="l">open big risks</div></div>
  <div class="tile"><div class="n">{health_min}</div><div class="l">code health (weakest scored unit)</div></div>
</div>
<h2>Milestone progress</h2>{phase_html}
{pillar_html}
<h2>Big risks</h2><div class="wrap"><table>{risk_html}</table></div>
<h2>Code health <span class="muted tagline">— computed at build time: ruff (breakage = E9 syntax + F821 undefined-name; hygiene = unused imports/vars per KLOC), pytest suites, git state</span></h2>
<div class="wrap"><table>
<tr class="muted"><td>unit</td><td>LOC</td><td>test suite</td><td>breakage</td><td>hyg/KLOC</td><td>dirty</td><td>score</td></tr>
{health_html}</table></div>
<div class="wrap"><table>{fresh_html}{flags_html}</table></div>
<h2>Pipeline autonomy <span class="muted tagline">— from AUTONOMY: lines in daily notebook entries (defs: lab_notebook/TESTING.md §5)</span></h2>
{auto_html}
<h2>New dataset &rarr; tassili <span class="muted tagline">— {len(open_rows)} open (ledger: TESTING.md §6)</span></h2>
{led_html}
<h2>Live queue <span class="muted tagline">— the week GPU rotation (automation/week_prod_queue_20260814.sh): pointers, stage2 ckpts and verdict json read from disk</span></h2>{q_html}
<h2>Experiment trail (lab notebook)</h2><div class="wrap"><table>{notes_html}</table></div>
"""
    OUT.write_text(page)
    print(f"-> {OUT}")


if __name__ == "__main__":
    build()
