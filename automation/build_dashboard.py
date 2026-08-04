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
import re
import shutil
import subprocess
from pathlib import Path

CODE = Path("/home/paperspace/code")
LOGS = Path("/home/paperspace/logs")
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



# ---------- the science: pillars, questions, and how they're being solved ----
# Curated alongside the notebook — this IS the scientific narrative.
PILLARS = [
 dict(name="Tassili", tag="survey → queryable hierarchical splat",
  question="Can a farm survey become a 3D field you can QUERY BY LEVEL — "
           "section, row, tree, fruit — from one hyperbolic embedding "
           "(depth == radius on the Lorentz manifold)?",
  solved=[
    ("Tree / row levels separate in the rendered field",
     "95.8% / 99.1% cross-level pointing, IoU 0.702", "good"),
    ("Background capture cured (v16 masked loss)",
     "row pointing 15% → 89.9%", "good"),
    ("Checkpoint corruption root-caused",
     "opacity resets poison saves on 3000-step boundaries (5–7× loss)", "good"),
  ],
  open=[
    ("FRUIT level collapses — the demo differentiator",
     "scarcity ~1% of gradient; label conflict from unfired frames; "
     "S-ladder: F4 0.0% → S1 1.1% → S2 12.0% pointing (protected class works "
     "where it fires; coverage was reset by densification)", "serious"),
    ("Vocabulary never chosen for separation",
     "tree-word cosine p90 = 0.90, max 0.9993", "warning"),
  ],
  next="S3a (queued): supervision tally rides gsplat strategy_state so "
       "protection survives densification — gates: fruit radius ≈7, "
       "cross-level pointing."),
 dict(name="Bateleur", tag="top-down farm state — instances & rows",
  question="Can a trustworthy PLANT REGISTRY (instances, rows, counts) be "
           "built from noisy per-frame detections + LiDAR, and transfer "
           "across sites?",
  solved=[
    ("Citrus registry + QA method",
     "census audit exposed 24% undercount (112 vs 85); v4 NMS recluster "
     "canonical", "good"),
    ("klapmuts census (2nd site, new crop type)",
     "866 instances; median size 0.99 m = measured pitch; 72% of points "
     "plant-scale", "good"),
    ("Row structure solved at the RANSAC init",
     "14 clean rows in 0.12 s once dir gate ±10° and thr = spacing/2; "
     "CORAL optimiser shown no-op (λ=β=0) or destructive (0.5)", "good"),
    ("Transfer methodology: 9+ silent site/hardware constants externalised",
     "rig.json + site.json per dataset — the paper's transfer section", "good"),
  ],
  open=[
    ("Two-sided duplicates & far-side rows in census",
     "~128 two-plant merges; rows beyond 5 m gate unmapped", "warning"),
    ("Citrus registries under K-indexed poses",
     "clustered with stream-indexed poses — unverified", "serious"),
  ],
  next="Fleet tonight: 10 klapmuts row-block splats; then registry "
       "re-verification (Q4)."),
 dict(name="Sankofa", tag="the tree ledger across time",
  question="Is it the SAME plant across epochs — without a shared datum — "
           "and what changed biologically?",
  solved=[
    ("Absolute WGS84 association (citrus)",
     "01/03/04 associated, 99% match; latency calibrated (+300 ms)", "good"),
    ("Systematic 1.4 m common-mode shift found & corroborated two ways",
     "correcting it: 219→272 pairs at a TIGHTER gate → semantic pose-graph "
     "lead (trees as loop-closure landmarks)", "good"),
    ("Ledger v0", "677 observations / 402 canonical trees / 275 multi-epoch",
     "good"),
  ],
  open=[
    ("GPS-free epochs (klapmuts Dec-2025)",
     "INS is LOCAL ENU — datum unrecorded (the citrus ENU trap again); "
     "cross-season appearance shift is large (cover/floor/fruiting all "
     "changed)", "serious"),
    ("Biological metrics per tree per epoch",
     "ledger compares a structure PROXY, not phenology", "warning"),
  ],
  next="Localisation pre-flight: DINOv2 retrieval Dec→April; decision "
       "image-first vs structure-first (bag lattice = landmarks)."),
 dict(name="Adinkra", tag="natural-language query over the twin",
  question="Can plain language select geometry — 'the fruiting trees in row "
           "7' — through the hierarchical embedding?",
  solved=[
    ("Relevancy formula hardened",
     "negatives-free scoring + geodesic interpolation walk (steps=4) — "
     "committed", "good"),
    ("Query server live", "port 8002 against served splats", "good"),
  ],
  open=[
    ("Level collapse in queries",
     "same word can bleed across fruit/tree/row when a level is weak — "
     "gated on Tassili's fruit fix", "warning"),
  ],
  next="Re-validate viewer relevancy after S3a; wire klapmuts fleet into "
       "the query server (farm #2 demo)."),
 dict(name="Spoor · Azalai · Hapi", tag="design-stage pillars",
  question="Deliberately 'coming soon' in the demo — scope control is the "
           "plan, not a failure (roadmap).",
  solved=[], open=[], next="Design mockups only until the slice ships."),
]


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
        "critical": "&#10006;"}
SNAME = {"good": "OK", "warning": "WATCH", "serious": "RISK",
         "critical": "CRITICAL"}


def narrative_freshness():
    """Is the hand-curated PILLARS block keeping up with the lab notebook?

    build_dashboard runs daily from cron, but PILLARS is written by hand —
    so the mechanical parts (metrics, risks, queue, trail) can be current
    while the science story silently rots. Surface the drift instead.
    """
    src = CODE / "automation" / "build_dashboard.py"
    pill_day = sh("git log -1 --format=%cI -- automation/build_dashboard.py",
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
        <td class="val">{s["loc"] / 1000:.1f}k</td>
        <td class="val">{esc(s["tests"] or "—")}</td>
        <td class="val">{s["breakage"] if s["breakage"] is not None else "?"}</td>
        <td class="val">{(s["hygiene"] or 0) / max(s["loc"] / 1000, .001):.0f}</td>
        <td class="val">{s["dirty"]}</td>
        <td><span class="st st-{st}">{ICON[st]} {score}</span></td></tr>
        <tr class="why"><td colspan="7" class="muted small">{esc(why)}</td></tr>'''
        for u, s, score, why, st in ch)
    flags_html = "".join(
        f'''<tr><td>{esc(k)}</td><td class="muted">{esc(v)}</td>
        <td><span class="st st-{s}">{ICON[s]} {SNAME[s]}</span></td></tr>'''
        for k, v, s in HEALTH_FLAGS)
    scored = [score for u, s, score, why, st in ch if not u.get("legacy")]
    health_min = min(scored) if scored else 0
    pill_day, nb_newest, nb_behind, nb_st = narrative_freshness()
    fresh_html = (
        f'<tr><td>Science narrative (PILLARS) vs lab notebook</td>'
        f'<td class="val">curated {esc(pill_day)} · newest entry {esc(nb_newest)}</td>'
        f'<td><span class="st st-{nb_st}">{ICON[nb_st]} '
        f'{"in sync" if nb_behind == 0 else str(nb_behind) + " entries ahead"}'
        f'</span></td></tr>')

    q_html = ("".join(f"<div class='qline'>{esc(q)}</div>" for q in qs)
              or "<div class='qline muted'>no active queue lines</div>")

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
<h2>Live queue</h2>{q_html}
<h2>Experiment trail (lab notebook)</h2><div class="wrap"><table>{notes_html}</table></div>
"""
    OUT.write_text(page)
    print(f"-> {OUT}")


if __name__ == "__main__":
    build()
