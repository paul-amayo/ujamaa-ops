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
    # newest first: entries sit in file order, which is neither chronological
    # nor newest-first, so the trail used to show a month's OLDEST eight.
    entries.sort(key=lambda e: e[0][:10], reverse=True)
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


def esc(s):
    return html.escape(str(s))


def md(s):
    """Escape, then honour the **bold** the PILLARS.md source already uses."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", esc(s))


RANK = {"good": 0, "unknown": 0, "warning": 1, "serious": 2, "critical": 3}
# Phase 0 of the roadmap opens the week of Jul 13 — the programme clock starts there.
PROGRAMME_START = datetime.date(2026, 7, 13)

CSS = """
*,*::before,*::after { box-sizing:border-box; }
:root {
  --bg:#eceeec; --surface:#f7f9f8; --surface2:#e2e7e4; --ink:#12171a;
  --muted:#5b6660; --line:#d2d9d5; --hair:#dfe4e1;
  --accent:#1f4b6b; --accent-soft:#cddbe5;
  --good:#1c6b3f; --warning:#8a5a00; --serious:#b3261e; --critical:#7f1610;
  --unknown:#8b948f;
  --f-display:ui-serif,"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  --f-ui:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
  --f-mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark) {
  :root:not([data-theme="light"]) {
    --bg:#101413; --surface:#171c1a; --surface2:#212826; --ink:#e6ebe8;
    --muted:#94a29b; --line:#2c3432; --hair:#242b29;
    --accent:#79aed4; --accent-soft:#24384a;
    --good:#4cc97a; --warning:#e0a72e; --serious:#ff6b5e; --critical:#ff9a90;
    --unknown:#7d8a84;
  }
}
:root[data-theme="dark"] {
  --bg:#101413; --surface:#171c1a; --surface2:#212826; --ink:#e6ebe8;
  --muted:#94a29b; --line:#2c3432; --hair:#242b29;
  --accent:#79aed4; --accent-soft:#24384a;
  --good:#4cc97a; --warning:#e0a72e; --serious:#ff6b5e; --critical:#ff9a90;
  --unknown:#7d8a84;
}
html { -webkit-text-size-adjust:100%; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.55 var(--f-ui); font-variant-numeric:tabular-nums; }
a { color:inherit; }
:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }

/* ---------- sheet ---------- */
.sheet { display:grid; grid-template-columns:232px minmax(0,1fr); gap:2.75rem;
  max-width:1260px; margin:0 auto; padding:2.25rem 1.5rem 5rem; }
@media (max-width:900px) {
  .sheet { grid-template-columns:1fr; gap:1.25rem; padding-top:1.25rem; }
  .rail { position:static !important; }
}

/* ---------- left rail ---------- */
.rail { position:sticky; top:1.5rem; align-self:start; display:flex;
  flex-direction:column; gap:1.15rem; }
.brand { font:700 1.35rem/1 var(--f-display); letter-spacing:.01em; }
.brand span { display:block; font:500 .7rem/1.4 var(--f-mono);
  letter-spacing:.14em; text-transform:uppercase; color:var(--muted);
  margin-top:.4rem; }
.clock { border-top:2px solid var(--ink); padding-top:.6rem; }
.clock .n { font:700 2.1rem/1 var(--f-mono); letter-spacing:-.02em; }
.clock .l { font-size:.78rem; color:var(--muted); margin-top:.15rem; }
.track { height:5px; background:var(--surface2); margin-top:.7rem;
  position:relative; }
.track i { position:absolute; inset:0 auto 0 0; background:var(--accent);
  display:block; }
.track b { position:absolute; top:-3px; width:2px; height:11px;
  background:var(--ink); }
.tickrow { display:flex; justify-content:space-between;
  font:.66rem/1 var(--f-mono); color:var(--muted); margin-top:.35rem; }
nav { display:flex; flex-direction:column; border-top:1px solid var(--line); }
nav a { display:flex; align-items:center; gap:.55rem; padding:.34rem 0;
  font-size:.83rem; text-decoration:none; color:var(--muted);
  border-bottom:1px solid var(--hair); }
nav a:hover { color:var(--ink); }
nav a.on { color:var(--ink); font-weight:600; }
nav a .dot { width:7px; height:7px; flex:0 0 7px; border-radius:50%;
  background:var(--c,var(--good)); }
nav a .n { margin-left:auto; font:.7rem/1 var(--f-mono); color:var(--muted); }
.d-good{--c:var(--good)} .d-warning{--c:var(--warning)}
.d-serious{--c:var(--serious)} .d-critical{--c:var(--critical)}
.d-unknown{--c:var(--unknown)}

/* ---------- headings ---------- */
.eyebrow { font:600 .68rem/1 var(--f-mono); letter-spacing:.15em;
  text-transform:uppercase; color:var(--muted); }
.sec { margin:0 0 2.9rem; scroll-margin-top:1.25rem; }
.sec > h2 { font:600 1.3rem/1.25 var(--f-display); margin:.45rem 0 .3rem;
  text-wrap:balance; letter-spacing:.005em; }
.sec > .tagline { color:var(--muted); font-size:.85rem; max-width:74ch;
  margin-bottom:.9rem; }
h1 { font:600 1.15rem/1.3 var(--f-display); margin:0 0 .15rem; }

/* ---------- readout strip ---------- */
.readout { display:grid; grid-template-columns:repeat(4,minmax(0,1fr));
  border:1px solid var(--line); background:var(--surface); margin-bottom:2.4rem; }
@media (max-width:640px){ .readout { grid-template-columns:repeat(2,1fr); } }
.readout > div { padding:.85rem 1rem; border-left:1px solid var(--line); }
.readout > div:first-child { border-left:none; }
.readout .n { font:700 1.75rem/1.05 var(--f-mono); letter-spacing:-.02em; }
.readout .l { font-size:.75rem; color:var(--muted); margin-top:.3rem; }
.readout .n.warning { color:var(--warning); }
.readout .n.serious { color:var(--serious); }

/* ---------- triage ---------- */
.triage { border:1px solid var(--line); background:var(--surface); }
.triage .hd { display:flex; align-items:baseline; gap:.6rem; padding:.7rem 1rem;
  border-bottom:1px solid var(--line); background:var(--surface2); }
.triage .hd h2 { font:600 1rem/1 var(--f-display); margin:0; }
.item { display:grid; grid-template-columns:auto minmax(0,1fr);
  gap:.7rem; padding:.6rem 1rem .6rem .85rem; border-top:1px solid var(--hair);
  border-left:3px solid var(--c,var(--muted)); }
.item:first-of-type { border-top:none; }
.item .where { font:600 .68rem/1.5 var(--f-mono); letter-spacing:.06em;
  text-transform:uppercase; color:var(--muted); white-space:nowrap; }
.item .claim { font-weight:600; }
.item .ev { color:var(--muted); font-size:.85rem; max-width:80ch; }
.item a { text-decoration:none; border-bottom:1px solid var(--accent-soft); }
details.more > summary { cursor:pointer; padding:.55rem 1rem;
  font:600 .78rem/1 var(--f-mono); color:var(--muted);
  border-top:1px solid var(--line); }
details.more > summary:hover { color:var(--ink); }
details.more[open] > summary { color:var(--ink); }

/* ---------- phases ---------- */
.phase { display:grid; grid-template-columns:2.1rem minmax(0,1fr) auto;
  gap:.75rem; align-items:center; padding:.42rem 0;
  border-top:1px solid var(--hair); }
.phase .ix { font:600 .78rem/1 var(--f-mono); color:var(--muted); }
.phase .nm { font-size:.9rem; }
.phase .ct { font:600 .8rem/1 var(--f-mono); color:var(--muted); }
.phase.now .nm { font-weight:600; }
.phase.now .ix { color:var(--accent); }
.bar { height:6px; background:var(--surface2); }
.fill { height:6px; background:var(--accent); display:block; }
.phase .bar { margin-top:.3rem; }

/* ---------- pillar ledger ---------- */
.pillar { border:1px solid var(--line); background:var(--surface);
  padding:1.05rem 1.15rem 1.15rem; margin-bottom:1.15rem;
  scroll-margin-top:1.25rem; }
.pillar h2 { font:600 1.2rem/1.2 var(--f-display); margin:.2rem 0 .1rem; }
.pillar .tag { color:var(--muted); font-size:.85rem; }
.q { font-family:var(--f-display); font-style:italic; color:var(--muted);
  border-left:2px solid var(--accent); padding-left:.7rem; margin:.6rem 0 .9rem;
  max-width:76ch; }
.tally { display:flex; gap:.5rem; flex-wrap:wrap; margin-bottom:.2rem; }
.tally span { font:600 .68rem/1 var(--f-mono); letter-spacing:.06em;
  text-transform:uppercase; padding:.25rem .45rem; border:1px solid currentColor; }
.claimrow { border-left:3px solid var(--c,var(--muted)); padding:.5rem .8rem;
  border-top:1px solid var(--hair); }
.claimrow:first-of-type { border-top:none; }
.claimrow .t { font-weight:600; font-size:.92rem; }
.claimrow .e { color:var(--muted); font-size:.85rem; max-width:82ch;
  margin-top:.15rem; }
.next { margin-top:.9rem; padding-top:.7rem; border-top:1px solid var(--line);
  font-size:.87rem; max-width:82ch; }
.next b { font:600 .68rem/1 var(--f-mono); letter-spacing:.1em;
  text-transform:uppercase; color:var(--muted); display:block;
  margin-bottom:.25rem; }
.s-good{--c:var(--good)} .s-warning{--c:var(--warning)}
.s-serious{--c:var(--serious)} .s-critical{--c:var(--critical)}
.s-unknown{--c:var(--unknown)}
body.openonly .claimrow.s-good { display:none; }
.toggle { font:600 .72rem/1 var(--f-mono); letter-spacing:.06em;
  text-transform:uppercase; color:var(--muted); background:none;
  border:1px solid var(--line); padding:.4rem .6rem; cursor:pointer; }
.toggle:hover { color:var(--ink); }
body.openonly .toggle { color:var(--ink); border-color:var(--accent); }

/* ---------- tables ---------- */
.wrap { overflow-x:auto; border:1px solid var(--line); background:var(--surface); }
table { border-collapse:collapse; width:100%; font-size:.86rem; }
td { padding:.44rem .7rem; border-top:1px solid var(--hair);
  vertical-align:top; }
tr:first-child td { border-top:none; }
td.val { font-family:var(--f-mono); font-weight:600; white-space:nowrap; }
tr.head td { font:600 .68rem/1 var(--f-mono); letter-spacing:.1em;
  text-transform:uppercase; color:var(--muted); background:var(--surface2);
  border-top:none; padding-top:.55rem; padding-bottom:.55rem; }
tr.why td { border-top:none; padding-top:0; }
.muted { color:var(--muted); }
.small { font-size:.78rem; }
.st { font:700 .68rem/1 var(--f-mono); letter-spacing:.07em; white-space:nowrap; }
.st-good{color:var(--good)} .st-warning{color:var(--warning)}
.st-serious{color:var(--serious)} .st-critical{color:var(--critical)}
.st-unknown{color:var(--unknown)}
.absent td { opacity:.55; }

/* ---------- queue ---------- */
.qline { font:12px/1.7 var(--f-mono); background:var(--surface);
  border:1px solid var(--line); border-left:3px solid var(--accent);
  padding:.2rem .7rem; margin:.2rem 0; overflow-x:auto; white-space:nowrap; }

/* ---------- autonomy ---------- */
.auto { display:grid; grid-template-columns:auto minmax(90px,1fr) auto auto;
  gap:.6rem .8rem; align-items:center; border:1px solid var(--line);
  background:var(--surface); padding:.8rem 1rem; }
.auto .d { font:.78rem/1 var(--f-mono); color:var(--muted); }
.auto .p { font:600 .78rem/1 var(--f-mono); white-space:nowrap; }
.auto .x { font:.75rem/1 var(--f-mono); color:var(--muted); white-space:nowrap; }

/* ---------- trail ---------- */
.trail { border-left:2px solid var(--line); padding-left:1rem; }
.tr { position:relative; padding:.42rem 0; border-top:1px solid var(--hair); }
.tr:first-child { border-top:none; }
.tr::before { content:""; position:absolute; left:-1.32rem; top:.85rem;
  width:7px; height:7px; background:var(--accent); border-radius:50%; }
.tr .d { font:.72rem/1 var(--f-mono); color:var(--muted); }
.tr .t { font-size:.9rem; max-width:84ch; }
footer { color:var(--muted); font-size:.78rem; border-top:1px solid var(--line);
  padding-top:.8rem; margin-top:2rem; }
@media print {
  .rail nav, .toggle { display:none; }
  .sheet { grid-template-columns:1fr; }
  body { background:#fff; }
}
@media (prefers-reduced-motion:reduce) { * { scroll-behavior:auto !important; } }
"""

JS = """
(function () {
  var links = [].slice.call(document.querySelectorAll('nav a'));
  var secs = links.map(function (a) {
    return document.getElementById(a.getAttribute('href').slice(1));
  });
  function spy() {
    var best = 0, top = 120;
    secs.forEach(function (s, i) {
      if (s && s.getBoundingClientRect().top <= top) best = i;
    });
    links.forEach(function (a, i) { a.classList.toggle('on', i === best); });
  }
  addEventListener('scroll', spy, { passive: true });
  spy();
  var t = document.getElementById('openonly');
  if (t) t.addEventListener('click', function () {
    var on = document.body.classList.toggle('openonly');
    t.textContent = on ? 'Show settled claims' : 'Open items only';
    t.setAttribute('aria-pressed', on ? 'true' : 'false');
  });
})();
"""


def worst(sts):
    """Worst severity in a list of status strings (for the nav dots)."""
    o = "good"
    for s in sts:
        if RANK.get(s, 0) > RANK.get(o, 0):
            o = s
    return o


def attention(pillars, rk, ch, fresh, wq):
    """Everything the page already knows is not-green, in one triage list.

    The statuses were always computed per-section; nothing collected them, so
    'what needs me today' meant scrolling the whole sheet."""
    out = []
    for i, pl in enumerate(pillars):
        for claim, ev, st in pl["open"]:
            out.append(dict(st=st, where=pl["name"], href=f"#p{i}",
                            claim=claim, ev=ev))
    for k, v, st in rk:
        if RANK.get(st, 0) > 0:
            out.append(dict(st=st, where="Risk", href="#risks", claim=k, ev=v))
    for u, s, score, why, st in ch:
        if score is not None and RANK.get(st, 0) > 0:
            out.append(dict(st=st, where="Code health", href="#health",
                            claim=f'{u["name"]} scores {score}', ev=why))
    pill_day, nb_newest, nb_behind, nb_st = fresh
    if nb_behind:
        out.append(dict(st=nb_st, where="Narrative", href="#health",
                        claim=f"PILLARS is {nb_behind} notebook entries behind",
                        ev=f"curated {pill_day} · newest entry {nb_newest}"))
    if wq:
        if wq["stop"]:
            out.append(dict(st="serious", where="Queue", href="#queue",
                            claim=f"rotation stopped ({wq['stop']})",
                            ev="needs diagnosis; the watchdog will not restart it"))
        elif not wq["alive"]:
            out.append(dict(st="critical", where="Queue", href="#queue",
                            claim="rotation is dead",
                            ev="watchdog relaunches at :13/:43 unless a deliberate "
                               "stop is logged"))
    out.sort(key=lambda d: -RANK.get(d["st"], 0))
    return out


def item_html(d):
    return (f'<div class="item s-{d["st"]}">'
            f'<div class="where">{esc(d["where"])}</div>'
            f'<div><div class="claim"><a href="{d["href"]}">{md(d["claim"])}</a></div>'
            f'<div class="ev">{md(d["ev"])}</div></div></div>')


def build():
    today = datetime.date.today()
    days = (LAUNCH - today).days
    phases = parse_roadmap()
    notes = parse_notebook()
    qs = queue_state()
    rk = risks()
    ch = code_health()
    fresh = narrative_freshness()
    wq = week_queue()
    att = attention(PILLARS, rk, ch, fresh, wq)

    # ---- programme clock ----
    span = (LAUNCH - PROGRAMME_START).days
    elapsed = max(0, min(span, (today - PROGRAMME_START).days))
    pct_time = int(100 * elapsed / span) if span else 0
    done_items = sum(p["done"] for p in phases)
    all_items = sum(p["done"] + p["open"] for p in phases)
    pct_work = int(100 * done_items / all_items) if all_items else 0

    # ---- phases ----
    now_ix = next((i for i, p in enumerate(phases) if p["open"]), len(phases) - 1)
    phase_html = ""
    for i, p in enumerate(phases):
        tot = p["done"] + p["open"]
        pct = int(100 * p["done"] / tot) if tot else 0
        m = re.match(r"^Phase (\S+) — (.+)$", p["name"])
        ix, nm = (f"P{m.group(1)}", m.group(2)) if m else ("\u00b7", p["name"])
        phase_html += (
            f'<div class="phase{" now" if i == now_ix else ""}">'
            f'<div class="ix">{esc(ix)}</div>'
            f'<div><div class="nm">{esc(nm)}</div>'
            f'<div class="bar"><i class="fill" style="width:{pct}%"></i></div></div>'
            f'<div class="ct">{p["done"]}/{tot}</div></div>')

    # ---- pillars ----
    pillar_html = ""
    pillar_nav = []
    for i, pl in enumerate(PILLARS):
        rows = ""
        for claim, ev, st in pl["open"] + pl["solved"]:
            rows += (f'<div class="claimrow s-{st}"><div class="t">{md(claim)}</div>'
                     + (f'<div class="e">{md(ev)}</div>' if ev else "")
                     + "</div>")
        sts = [st for _, _, st in pl["open"]]
        nopen = len(sts)
        w = worst(sts)
        tally = f'<span class="st-good">{len(pl["solved"])} settled</span>'
        for label, key in (("watch", "warning"), ("risk", "serious"),
                           ("critical", "critical")):
            n = sum(1 for s in sts if s == key)
            if n:
                tally += f'<span class="st-{key}">{n} {label}</span>'
        pillar_html += (
            f'<section class="pillar" id="p{i}">'
            f'<div class="eyebrow">{esc(pl["tag"])}</div>'
            f'<h2>{esc(pl["name"])}</h2>'
            f'<div class="q">{esc(pl["question"])}</div>'
            f'<div class="tally">{tally}</div>{rows}'
            + (f'<div class="next"><b>Next</b>{esc(pl["next"])}</div>'
               if pl["next"] else "")
            + "</section>")
        pillar_nav.append((f"p{i}", pl["name"].split(" ")[0], w, nopen or ""))

    # ---- risks / health ----
    risk_html = "".join(
        f'<tr><td>{esc(k)}</td><td class="val">{esc(v)}</td>'
        f'<td><span class="st st-{s}">{ICON[s]} {SNAME[s]}</span></td></tr>'
        for k, v, s in rk)
    DASH = "\u2014"
    hrows = []
    for u, s, score, why, st in ch:
        absent = s.get("absent")
        loc = DASH if absent else f"{s['loc'] / 1000:.1f}k"
        brk = DASH if absent else (
            s["breakage"] if s["breakage"] is not None else "?")
        hyg = DASH if absent else (
            f"{(s['hygiene'] or 0) / max(s['loc'] / 1000, .001):.0f}")
        drt = DASH if absent else s["dirty"]
        badge = SNAME[st] if score is None else score
        cls = ' class="absent"' if absent else ""
        hrows.append(
            f'<tr{cls}><td>{esc(u["name"])}'
            f'<div class="muted small">{esc(u["note"])}</div></td>'
            f'<td class="val">{loc}</td>'
            f'<td class="val">{esc(s["tests"] or DASH)}</td>'
            f'<td class="val">{brk}</td><td class="val">{hyg}</td>'
            f'<td class="val">{drt}</td>'
            f'<td><span class="st st-{st}">{ICON[st]} {badge}</span></td></tr>'
            f'<tr class="why"><td colspan="7" class="muted small">'
            f'{esc(why)}</td></tr>')
    health_html = "".join(hrows)
    flags_html = "".join(
        f'<tr><td>{esc(k)}</td><td class="muted">{esc(v)}</td>'
        f'<td><span class="st st-{s}">{ICON[s]} {SNAME[s]}</span></td></tr>'
        for k, v, s in HEALTH_FLAGS)
    scored = [score for u, s, score, why, st in ch
              if not u.get("legacy") and score is not None]
    health_min = min(scored) if scored else "—"
    pill_day, nb_newest, nb_behind, nb_st = fresh
    fresh_html = (
        f'<tr><td>Science narrative (PILLARS) vs lab notebook</td>'
        f'<td class="val">curated {esc(pill_day)} · newest entry {esc(nb_newest)}</td>'
        f'<td><span class="st st-{nb_st}">{ICON[nb_st]} '
        f'{"in sync" if nb_behind == 0 else str(nb_behind) + " entries ahead"}'
        f'</span></td></tr>')

    # ---- queue ----
    if wq:
        q_html = week_queue_html(wq, esc)
        q_st = "serious" if wq["stop"] else "good" if wq["alive"] else "critical"
    else:
        q_html = ("".join(f'<div class="qline">{esc(q)}</div>' for q in qs)
                  or '<div class="qline muted">no active queue lines — '
                     '~/logs is not readable from this checkout</div>')
        q_st = "unknown"

    # ---- autonomy ----
    auto = parse_autonomy()
    if auto:
        cells = ""
        for day, kv in reversed(auto):
            r, c = kv.get("runs", 0), kv.get("clean", 0)
            iv, dbg = kv.get("interventions", 0), kv.get("debugged", 0)
            pct = 100 * c / r if r else 0
            ast = ("good" if iv == 0 and dbg == 0 else
                   "warning" if iv + dbg <= 3 else "serious")
            cells += (
                f'<div class="d">{esc(day)}</div>'
                f'<div class="bar"><i class="fill" style="width:{pct:.0f}%;'
                f'background:var(--{ast})"></i></div>'
                f'<div class="p">{c}/{r} clean</div>'
                f'<div class="x">{iv} interventions · {dbg} debugged</div>')
        auto_html = f'<div class="auto">{cells}</div>'
    else:
        auto_html = ('<div class="muted small">no AUTONOMY: lines in the '
                     'notebook yet — format in lab_notebook/TESTING.md §5</div>')

    # ---- dataset ledger ----
    ledger = parse_tassili_ledger()
    open_rows = [r for r in ledger if "open" in r[4] or "pending" in r[4]]
    led_html = ""
    if ledger:
        body = "".join(
            f'<tr><td>{esc(r[0])}</td><td class="muted">{esc(r[1])}</td>'
            f'<td class="val">{esc(r[4])}</td><td class="muted">{esc(r[6])}</td></tr>'
            for r in ledger)
        led_html = ('<div class="wrap"><table><tr class="head"><td>dataset</td>'
                    '<td>landed</td><td>days to tassili</td><td>notes</td></tr>'
                    f"{body}</table></div>")

    trail_html = "".join(
        f'<div class="tr"><div class="d">{esc(d)}</div>'
        f'<div class="t">{esc(t)}</div></div>' for d, t in notes)

    # ---- triage band ----
    hot = [d for d in att if RANK.get(d["st"], 0) >= 2]
    warm = [d for d in att if RANK.get(d["st"], 0) == 1]
    triage = "".join(item_html(d) for d in hot)
    if warm:
        triage += ('<details class="more"><summary>'
                   f'+ {len(warm)} watch items</summary>'
                   + "".join(item_html(d) for d in warm) + "</details>")
    if not att:
        triage = '<div class="item s-good"><div class="where">clear</div>' \
                 '<div><div class="claim">nothing above watch level</div></div></div>'

    n_bigrisk = len([r for r in rk if RANK.get(r[2], 0) >= 2])
    nav = [("attention", "Needs attention", worst([d["st"] for d in att]),
            len(hot) or ""),
           ("schedule", "Schedule", "good", f"{done_items}/{all_items}"),
           ("risks", "Big risks", worst([r[2] for r in rk]), n_bigrisk or "")]
    nav += pillar_nav
    nav += [("queue", "Live queue", q_st, ""),
            ("health", "Code health", worst([st for _, _, _, _, st in ch]),
             health_min),
            ("autonomy", "Autonomy", "good", ""),
            ("datasets", "Datasets", "warning" if open_rows else "good",
             len(open_rows) or ""),
            ("trail", "Trail", "good", "")]
    nav_html = "".join(
        f'<a href="#{i}"><span class="dot d-{s}"></span>{esc(t)}'
        f'<span class="n">{esc(n)}</span></a>' for i, t, s, n in nav)

    page = f"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>UJAMAA Programme Sheet</title>
<style>{CSS}</style>
<div class="sheet">
<aside class="rail">
  <div class="brand">UJAMAA<span>programme sheet</span></div>
  <div class="clock">
    <div class="n">{days}</div>
    <div class="l">days to launch · {LAUNCH:%d %b %Y}</div>
    <div class="track"><i style="width:{pct_work}%"></i>
      <b style="left:{pct_time}%"></b></div>
    <div class="tickrow"><span>{pct_work}% of items</span>
      <span>{pct_time}% of calendar</span></div>
  </div>
  <nav>{nav_html}</nav>
</aside>

<main>
  <h1>Digital-twin orchard programme</h1>
  <div class="muted small">generated {esc(today.strftime('%Y-%m-%d'))} {esc(datetime.datetime.now().strftime('%H:%M'))}
   · roadmap · lab notebook · ~/logs · live git &amp; disk</div>

  <div class="readout">
    <div><div class="n">{done_items}<span class="muted">/{all_items}</span></div>
      <div class="l">roadmap items done</div></div>
    <div><div class="n {"serious" if len(hot) else ""}">{len(hot)}</div>
      <div class="l">items at risk or worse</div></div>
    <div><div class="n {"warning" if warm else ""}">{len(warm)}</div>
      <div class="l">on watch</div></div>
    <div><div class="n">{health_min}</div>
      <div class="l">code health, weakest unit</div></div>
  </div>

  <section class="sec" id="attention">
    <div class="triage">
      <div class="hd"><h2>Needs attention</h2>
        <span class="muted small">every non-green claim on this sheet, worst first</span></div>
      {triage}
    </div>
  </section>

  <section class="sec" id="schedule">
    <div class="eyebrow">Roadmap</div>
    <h2>Schedule</h2>
    <div class="tagline">Phases run in order; the marked phase is the first with
      open items. The rail's tick compares work done against calendar spent.</div>
    {phase_html}
  </section>

  <section class="sec" id="risks">
    <div class="eyebrow">Programme</div>
    <h2>Big risks</h2>
    <div class="tagline">Live probes (disk, working trees) alongside the
      standing structural risks.</div>
    <div class="wrap"><table>{risk_html}</table></div>
  </section>

  <div style="display:flex;justify-content:space-between;align-items:baseline;gap:1rem;margin-bottom:.7rem">
    <div><div class="eyebrow">Research pillars</div>
      <h2 style="font:600 1.3rem/1.25 var(--f-display);margin:.45rem 0 0">Claim ledger</h2></div>
    <button class="toggle" id="openonly" aria-pressed="false">Open items only</button>
  </div>
  {pillar_html}

  <section class="sec" id="queue">
    <div class="eyebrow">Automation</div>
    <h2>Live queue</h2>
    <div class="tagline">The week GPU rotation (automation/week_prod_queue_20260814.sh):
      pointers, stage2 checkpoints and verdict json read from disk.</div>
    {q_html}
  </section>

  <section class="sec" id="health">
    <div class="eyebrow">Engineering</div>
    <h2>Code health</h2>
    <div class="tagline">Computed at build time: ruff breakage (E9 syntax + F821
      undefined name), hygiene (unused imports and variables per KLOC), pytest
      suites, git state.</div>
    <div class="wrap"><table>
    <tr class="head"><td>unit</td><td>LOC</td><td>test suite</td><td>breakage</td>
      <td>hyg/KLOC</td><td>dirty</td><td>score</td></tr>
    {health_html}</table></div>
    <div class="wrap" style="margin-top:.8rem"><table>{fresh_html}{flags_html}</table></div>
  </section>

  <section class="sec" id="autonomy">
    <div class="eyebrow">Pipeline</div>
    <h2>Autonomy</h2>
    <div class="tagline">From AUTONOMY: lines in daily notebook entries; bar length
      is the share of runs that finished clean (definitions: lab_notebook/TESTING.md §5).</div>
    {auto_html}
  </section>

  <section class="sec" id="datasets">
    <div class="eyebrow">Ingest</div>
    <h2>New dataset &rarr; tassili</h2>
    <div class="tagline">{len(open_rows)} open (ledger: TESTING.md §6).</div>
    {led_html}
  </section>

  <section class="sec" id="trail">
    <div class="eyebrow">Lab notebook</div>
    <h2>Experiment trail</h2>
    <div class="trail">{trail_html}</div>
  </section>

  <footer>Rebuild: <code>python3 automation/build_dashboard.py</code> ·
    roots follow UJAMAA_CODE / UJAMAA_LOGS, defaulting to the workstation.</footer>
</main>
</div>
<script>{JS}</script>
"""
    OUT.write_text(page)
    print(f"-> {OUT}")


if __name__ == "__main__":
    build()
