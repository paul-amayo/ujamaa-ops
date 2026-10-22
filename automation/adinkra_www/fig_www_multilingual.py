#!/usr/bin/env python3
"""Figure: the Adinkra x Way With Words multilingual field-interaction evaluation (r1 -> r3).

Two steps, so the numbers come from the run records and the drawing needs no box:
  stats (on the A100, reads the frozen run dirs):
    ssh paperspace@184.105.3.25 'python3 - stats' < fig_www_multilingual.py > www_fig_numbers.json
  plot (anywhere with matplotlib):
    python3 fig_www_multilingual.py plot www_fig_numbers.json <out_stem>

Panel (d) verdicts are hand-read from the r3 passA answers (Claude, 20 Oct 2026); they are
not native-speaker validated, which is the review asked of WWW.
"""
import glob, json, re, statistics, sys
from collections import Counter, defaultdict

RUNS = {"r1": "/home/paperspace/data/experimental_adinkra_www_20260827",
        "r3": "/home/paperspace/data/experimental_adinkra_www_20260827_r3"}
LANGS = ["en", "af", "xh", "zu", "sw", "ha", "ar", "am"]
NAMES = {"en": "English", "af": "Afrikaans", "xh": "isiXhosa", "zu": "isiZulu", "sw": "Kiswahili",
         "ha": "Hausa", "ar": "Arabic", "am": "Amharic"}
SENTINEL = "non-JSON"   # panel_agents parse-failure fallback: raw scratchpad passed to the user (r1)


def stats():
    out = {}
    for run, d in RUNS.items():
        rows = [json.loads(l) for f in sorted(glob.glob(d + "/pass*.jsonl")) for l in open(f)]
        a = [r for r in rows if r["tag"] == "passA"]
        lat = defaultdict(list)
        for r in a:
            lat[r["lang"]].append(r["secs"])
        per = {}
        for lg in LANGS:
            rl = [r for r in rows if r["lang"] == lg]
            if not rl:
                continue
            leak = sum(SENTINEL in json.dumps(r) for r in rl)
            fail = sum(r.get("action") == "no_answer" for r in rl)
            per[lg] = {"n": len(rl), "leaked": leak, "no_answer": fail, "clean": len(rl) - leak - fail,
                       "passA_median_s": round(statistics.median(lat[lg]), 1), "passA_n": len(lat[lg])}
        out[run] = {"dir": d, "queries": len(rows), "returned": sum(r.get("ok", False) for r in rows),
                    "empty": sum(r.get("empty", False) for r in rows),
                    "leaked": sum(v["leaked"] for v in per.values()),
                    "no_answer": sum(v["no_answer"] for v in per.values()),
                    "items": len({r["id"] for r in rows}), "actions": Counter(r.get("action") for r in rows),
                    "per_lang": per}
    json.dump(out, sys.stdout, indent=1)


# (d) hand-read verdicts, r3 passA. category -> colour; text shown in the cell
VERDICTS = {
    "B3 citrus\n“Any plants missing\nalong row 7?”": {
        "en": ("ok", "No: 13 =\ncensus"), "af": ("fail", "no answer"), "xh": ("empty", "“…”"),
        "zu": ("wrong", "Yes: reads\n13 of 107\nas missing"), "sw": ("empty", "“…”"),
        "ha": ("partial", "13 plants,\nno verdict"), "ar": ("ok", "No: all 13\nlisted"), "am": ("ok", "No: 13\nconfirmed")},
    "B3 Klapmuts\n“Any plants missing\nalong row 7?”": {
        "en": ("ok", "No: 48 =\ncensus"), "af": ("fail", "no answer"), "xh": ("wrong", "Maybe: ID\ngaps"),
        "zu": ("wrong", "Yes: lists\nother rows"), "sw": ("wrong", "Yes: ID\ngaps"), "ha": ("fail", "no answer"),
        "ar": ("partial", "48 plants,\nno verdict"), "am": ("partial", "48 plants,\nno verdict")},
    "A3 citrus\n“Show me the plant\nnearest the gate”\n(no gate in the data)": {
        "en": ("fail", "no answer"), "af": ("fail", "no answer"), "xh": ("fail", "no answer"),
        "zu": ("fail", "no answer"), "sw": ("wrong", "to Obj_1\n(gate\nassumed)"), "ha": ("fail", "no answer"),
        "ar": ("wrong", "to Obj_1\n(gate\nassumed)"), "am": ("wrong", "to Obj_1\n(gate\nassumed)")},
}


def plot(numbers, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    N = json.load(open(numbers))
    r1, r3 = N["r1"], N["r3"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlesize": 9.5, "axes.titleweight": "bold",
                         "axes.titlelocation": "left"})
    INK, DIM, C1, C3 = "#1f2a1c", "#7a8076", "#c9a77c", "#2f6b45"
    CAT = {"ok": "#3f8f5a", "partial": "#c8d6a8", "wrong": "#d9822b", "empty": "#b9b2a8", "fail": "#e8e2d6"}
    fig = plt.figure(figsize=(7.16, 7.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.36, 1.0, 1.15], hspace=0.58, wspace=0.28,
                          left=0.07, right=0.97, top=0.965, bottom=0.03)

    # (a) setup
    ax = fig.add_subplot(gs[0, :]); ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 30)
    ax.set_title("(a) A farm worker’s question, in their language, answered from the survey")
    boxes = [(1, "Question, typed", "38 items in user voice: retrieve, inspect,\ncompare, vague, out of scope, act\n"
              + " · ".join(LANGS), "#efe9dc"),
             (35, "Adinkra (Gemma-4 12B)", "grounded on the farm’s own survey:\ncitrus 107 trees, 8 rows\nKlapmuts 517 plants, 17 rows", "#dfe9dc"),
             (69, "Panel action", "answer · goto · highlight · locate\n· explain · or an honest\n“no answer” in the asker’s language", "#efe9dc")]
    for x, head, body, fc in boxes:
        ax.add_patch(FancyBboxPatch((x, 2), 30, 24, boxstyle="round,pad=0.4,rounding_size=1.5", fc=fc, ec="#b8b2a4", lw=0.8))
        ax.text(x + 15, 21.5, head, ha="center", va="center", weight="bold", color=INK, fontsize=8.8)
        ax.text(x + 15, 10.5, body, ha="center", va="center", color=INK, fontsize=7.0, linespacing=1.35)
    for x in (31.6, 65.6):
        ax.annotate("", xy=(x + 3, 14), xytext=(x, 14), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))

    # (b) latency
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("(b) Time to answer, median per language")
    xs = range(len(LANGS)); w = 0.38
    v1 = [r1["per_lang"].get(l, {}).get("passA_median_s") for l in LANGS]
    v3 = [r3["per_lang"][l]["passA_median_s"] for l in LANGS]
    ax.bar([x - w / 2 for x, v in zip(xs, v1) if v], [v for v in v1 if v], w, color=C1, label=f"r1: 5 languages, {r1['queries']} queries")
    ax.bar([x + w / 2 for x in xs], v3, w, color=C3, label=f"r3: 8 languages, {r3['queries']} queries")
    for x, v in zip(xs, v3):
        ax.text(x + w / 2, v + 0.8, f"{v:.0f}", ha="center", fontsize=7, color=C3)
    for x, v in zip(xs, v1):
        if v:
            ax.text(x - w / 2, v + 0.8, f"{v:.0f}", ha="center", fontsize=7, color="#8a6d45")
    ax.set_xticks(list(xs)); ax.set_xticklabels(LANGS); ax.set_ylabel("seconds (first pass, n ≈ 46–48)")
    ax.set_ylim(0, 62); ax.legend(frameon=False, fontsize=6.8, loc="upper right", bbox_to_anchor=(1.04, 1.0), handlelength=1)
    dx = [round(100 * (1 - r3["per_lang"][l]["passA_median_s"] / r1["per_lang"][l]["passA_median_s"])) for l in ("xh", "zu")]
    ax.text(4.45, 47, f"r3 adds the row structure to\nthe context: isiXhosa falls {dx[0]}%,\nisiZulu {dx[1]}%", fontsize=6.8, color=DIM, ha="left", va="top")

    # (c) what the asker received
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title("(c) What the asker received")
    cols = [("clean", "an answer or action", C3),
            ("leaked", f"raw model scratchpad: r1 {r1['leaked']}/{r1['queries']}, r3 {r3['leaked']}/{r3['queries']}", "#c0392b"),
            ("no_answer", f"honest “no answer”: r3 {r3['no_answer']}/{r3['queries']}", "#c9c2b4")]
    ylab, yy = [], 0
    for run, R in (("r1", r1), ("r3", r3)):
        for l in LANGS:
            p = R["per_lang"].get(l)
            if not p:
                continue
            left = 0
            for k, _, c in cols:
                f = p[k] / p["n"]
                ax.barh(yy, f, left=left, color=c, height=0.78, lw=0)
                left += f
            ylab.append((yy, f"{run} {l}")); yy -= 1
        yy -= 0.7
    ax.set_yticks([y for y, _ in ylab]); ax.set_yticklabels([t for _, t in ylab], fontsize=6.6)
    ax.set_xlim(0, 1); ax.set_xticks([0, .5, 1]); ax.set_xticklabels(["0", "50%", "100%"])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=lab) for _, lab, c in cols], frameon=False, fontsize=6.6,
              loc="upper left", bbox_to_anchor=(-0.12, -0.08), ncol=1, handlelength=1)

    # (d) verdict grid
    ax = fig.add_subplot(gs[2, :])
    ax.set_title("(d) Ill-defined questions reach different conclusions by language (r3)")
    rows = list(VERDICTS)
    for i, q in enumerate(rows):
        for j, l in enumerate(LANGS):
            cat, txt = VERDICTS[q][l]
            ax.add_patch(plt.Rectangle((j + 0.04, -i - 0.96), 0.92, 0.92, fc=CAT[cat], ec="white"))
            ax.text(j + 0.5, -i - 0.5, txt, ha="center", va="center", fontsize=6.3, linespacing=1.1,
                    color="white" if cat in ("ok", "wrong") else INK)
        ax.text(-0.08, -i - 0.5, q, ha="right", va="center", fontsize=7, linespacing=1.2, color=INK)
    for j, l in enumerate(LANGS):
        ax.text(j + 0.5, 0.12, NAMES[l], ha="center", va="bottom", fontsize=7.2, weight="bold", color=INK)
    ax.set_xlim(-1.75, len(LANGS)); ax.set_ylim(-len(rows) - 0.55, 0.5); ax.set_axis_off()
    key = [("ok", "right, matches the data", -1.75), ("partial", "a fact, no verdict", 0.45), ("wrong", "invented definition or assumption", 2.25),
           ("empty", "no content", 5.4), ("fail", "honest failure", 6.6)]
    for c, lab, x in key:
        ax.add_patch(plt.Rectangle((x, -len(rows) - 0.48), 0.22, 0.22, fc=CAT[c], ec="#b8b2a4", lw=0.4))
        ax.text(x + 0.3, -len(rows) - 0.37, lab, fontsize=6.8, va="center", color=INK)

    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    print(stem)


if __name__ == "__main__":
    if sys.argv[1] == "stats":
        stats()
    else:
        plot(sys.argv[2], sys.argv[3])
