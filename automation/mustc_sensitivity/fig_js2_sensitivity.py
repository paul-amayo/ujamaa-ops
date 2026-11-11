#!/usr/bin/env python3
"""JS-2 figure (SENSITIVITY, NOT ADOPTED): the numbers behind items 31, 33 and 30 on both MuST-C plots.

Reads only sensitivity_table_plot198.json, sensitivity_table_plot178.json (the tables of record of this test) and
spotcheck_a100_plot178.json (one post hoc point, labelled so).  Writes one PNG.

Box: /mnt/shared2/must-c/venv/bin/python fig_js2_sensitivity.py --out <png>
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
A = ap.parse_args()
S198 = "/mnt/shared2/must-c/plot198/census/sensitivity_20261112"; S178 = "/mnt/shared2/must-c/plot178/census/sensitivity_20261112"
T = {"198": json.load(open(f"{S198}/sensitivity_table_plot198.json")), "178": json.load(open(f"{S178}/sensitivity_table_plot178.json"))}
SPOT178 = json.load(open(f"{S178}/spotcheck_a100_plot178.json"))

# palette: reference slots 1 and 2 (validated all-pairs in the reference palette); ink and surface from the same reference
C = {"198": "#2a78d6", "178": "#eb6834"}; MK = {"198": "o", "178": "D"}
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.4), gridspec_kw={"width_ratios": [1.55, 1.0, 1.0]})

# ---------------------------------------------------------------- (a) item 31
ax = axs[0]; x0 = 0; xt, xl, groups, names = [], [], [], []
for p in ("198", "178"):
    i31 = T[p]["columns"]["item 31"]["T7 plant clause"]["item31"]["per_stop"]
    two_g = T[p]["columns"]["as registered"]["T7 plant clause"]["two_g_m"] * 100
    days = sorted({s.split("_")[0] for s in i31})
    xs_plot = []
    for d in days:
        st = sorted(s for s in i31 if s.startswith(d)); xs = list(range(x0, x0 + len(st)))
        fl = i31[st[0]]["m2d_per_plant_abs_diff_p95_m"] * 100
        ax.hlines(fl, xs[0] - 0.4, xs[-1] + 0.4, color=INK2, lw=2.0, zorder=2)
        ax.text(xs[-1] + 0.45, fl, f"{fl:.1f}", va="center", ha="left", fontsize=7, color=INK2)
        for xx, s in zip(xs, st):
            v = i31[s]["per_plant_absdiff_p95_m"] * 100; inside = i31[s]["within_floor"]
            ax.scatter(xx, v, s=46, marker=MK[p], color=C[p], edgecolor=SURF, linewidth=1.0, zorder=4)
            xt.append(xx); xl.append(s.split("_")[1].replace("pos", "p")); names.append((p, s))
            if inside:
                ax.annotate("within", (xx, v), xytext=(0, -11), textcoords="offset points", ha="center", fontsize=6.5, color=INK2)
        groups.append(((xs[0] + xs[-1]) / 2, d)); xs_plot += xs; x0 = xs[-1] + 2
    ax.hlines(two_g, xs_plot[0] - 0.5, xs_plot[-1] + 0.5, color=INK, lw=1.1, ls=(0, (4, 3)), zorder=3)
    ax.text(xs_plot[0] - 0.5, two_g + 0.35, f"2 g as registered, {two_g:.1f} cm" + (" (g capped at 8 cm)" if p == "178" else ""), fontsize=7, color=INK, ha="left", va="bottom")
    ax.text((xs_plot[0] + xs_plot[-1]) / 2, 21.6, f"plot {p}", ha="center", fontsize=9, color=C[p], fontweight="bold")
    x0 += 1
ph = SPOT178["post_hoc"]["t7_230616_vs_230613_laser_across_harvest"]["per_stop"]["230616_pos00"]
xi = names.index(("178", "230616_pos00"))
ax.scatter(xt[xi], ph["p95_without_inside_m"] * 100, s=46, marker="D", facecolor="none", edgecolor=C["178"], linewidth=1.3, zorder=4)
ax.vlines(xt[xi], ph["p95_without_inside_m"] * 100, ph["p95_all_m"] * 100, color=MUTED, lw=0.8, ls=":", zorder=3)
ax.annotate(f"post hoc, not a rule (dotted, hollow):\n230616 p00 without the {ph['n_inside_SB7_150623_dilated_0.10']} plants harvested on\n15 June, after its laser date 230613: {ph['p95_all_m'] * 100:.1f} -> {ph['p95_without_inside_m'] * 100:.1f} cm",
            (xt[xi], ph["p95_all_m"] * 100), xytext=(xt[xi] - 0.4, 19.9), textcoords="data", fontsize=6.5, color=INK2, va="center", ha="right",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkB=4))
ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=6.5)
for gx, d in groups:
    ax.text(gx, -0.115, d, transform=ax.get_xaxis_transform(), ha="center", fontsize=7.5, color=INK2)
ax.set_ylim(0, 22.5); ax.set_ylabel("per-plant |h95 splat - h95 laser|, p95 over plants (cm)")
ax.set_title("(a) Item 31, T7 plant clause: each stop against the M2d floor\n(solid bars, UAV1 vs laser per date) instead of 2 g (dashed)", fontsize=9, loc="left", color=INK)
ax.legend(handles=[Line2D([], [], color=INK2, lw=2.0, label="M2d per-plant floor (item 31)"), Line2D([], [], color=INK, lw=1.1, ls=(0, (4, 3)), label="2 g (as registered)")],
          loc="upper left", fontsize=7, frameon=False, bbox_to_anchor=(0.0, 0.93))

# ---------------------------------------------------------------- (b) item 33
ax = axs[1]; LASER_N = {}
for p in ("198", "178"):
    sp = json.load(open(f"{S198 if p == '198' else S178}/spotcheck_a100_plot{p}.json"))["t3_numbers"]
    rr = sorted(sp["p95_by_radius_m"], key=float); x = [float(r) * 100 for r in rr]; y = [sp["p95_by_radius_m"][r] * 100 for r in rr]
    ax.plot(x, y, color=C[p], lw=2.0, marker=MK[p], ms=6, mec=SURF, mew=1.0, zorder=4, label=f"plot {p}, splat, 230613")
    for xx, yy, r in zip(x, y, rr):
        if r in ("0.150", "0.200"):
            ax.annotate(f"n {sp['n_by_radius'][r]}", (xx, yy), xytext=(5, -9 if p == "178" else 4), textcoords="offset points", fontsize=6.5, color=INK2)
    ax.scatter(15, sp["laser_pooled_p95_m"] * 100, marker="s", s=40, facecolor="none", edgecolor=C[p], linewidth=1.3, zorder=5)
    LASER_N[p] = sp["laser_n"]
ax.axhline(9.5, color=INK, lw=1.1, ls=(0, (4, 3)), zorder=3); ax.text(8.15, 9.62, "cap 9.5 cm", fontsize=7, color=INK, ha="left", va="bottom")
ax.plot([8, 20.5], [8, 20.5], color=MUTED, lw=0.8, ls=":", zorder=2); ax.text(20.7, 17.0, "p95 = match radius\n(censoring limit)", fontsize=6.5, color=MUTED, ha="right", va="bottom")
ax.axvline(15, color=MUTED, lw=0.8, zorder=1); ax.text(15.15, 5.4, "item 33 reads\nat 0.15 m", fontsize=6.5, color=INK2)
ax.set_xlim(8, 20.8); ax.set_ylim(5, 21); ax.set_xlabel("match radius (cm)"); ax.set_ylabel("same-day position difference, p95 (cm)")
ax.set_title("(b) Item 33, T3 association: 230613 same-day p95 against\nthe match radius; as registered read at 0.15 and 0.20 m together", fontsize=9, loc="left", color=INK)
h_, l_ = ax.get_legend_handles_labels()
h_ += [Line2D([], [], marker="s", ls="", mfc="none", mec=C[p], mew=1.3, ms=6) for p in ("198", "178")]
l_ += [f"laser two-scanner, pooled, 0.15 m cap, plot {p} (n {LASER_N[p]})" for p in ("198", "178")]
ax.legend(h_, l_, loc="upper left", fontsize=6.8, frameon=False)

# ---------------------------------------------------------------- (c) item 30
ax = axs[2]; pos = 0; ticks, labels = [], []
for p in ("198", "178"):
    c30 = T[p]["columns"]["item 30"]
    for sens in ("splat", "laser"):
        row = [r for r in c30 if "specificity" in r and (("splat" in r and "laser" not in r) if sens == "splat" else "laser" in r)][0]
        i = c30[row]["item30"]
        fr, (fl, fh) = i["false_removal_rate"], i["false_removal_wilson95"]; sd = i["same_day_rate"]; sl, sh = i["same_day_wilson95"]
        ax.errorbar(pos - 0.13, fr, yerr=[[fr - fl], [fh - fr]], fmt=MK[p], color=C[p], ms=6, mec=SURF, mew=1.0, elinewidth=1.4, capsize=0, zorder=4)
        ax.errorbar(pos + 0.13, sd, yerr=[[sd - sl], [sh - sd]], fmt=MK[p], color=C[p], mfc=SURF, ms=6, mew=1.3, elinewidth=1.4, capsize=0, zorder=4)
        ax.annotate(f"{i['n_removed_called']}/{i['n_candidates']}", (pos - 0.13, fr), xytext=(-6, 0), textcoords="offset points", ha="right", va="center", fontsize=6.5, color=INK2)
        cnt = i["same_day_counts"]
        ax.annotate(f"{cnt['n_unmatched']}/{cnt['n_in']}", (pos + 0.13, sd), xytext=(6, 0), textcoords="offset points", ha="left", va="center", fontsize=6.5, color=INK2)
        ax.annotate(i["verdict"].split(" (")[0], (pos, 0.98), ha="center", va="top", fontsize=6.5, color=INK, xycoords=("data", "axes fraction"))
        ticks.append(pos); labels.append(f"{sens}\nplot {p}"); pos += 1
    pos += 0.4
ax.axhline(0.5, color=INK, lw=1.1, ls=(0, (4, 3)), zorder=3); ax.text(pos - 0.55, 0.51, "0.5", fontsize=7, color=INK, ha="right", va="bottom")
ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=7.5); ax.set_ylim(0, 1.0); ax.set_xlim(-0.6, pos - 0.5)
ax.set_ylabel("rate (95 % Wilson interval)")
ax.set_title("(c) Item 30, T5 specificity: false-removal rate (filled)\nbeside the same-day rate (hollow); top line = item-30 word", fontsize=9, loc="left", color=INK)
ax.legend(handles=[Line2D([], [], marker="o", color=INK2, ls="", ms=6, label="false-removal rate, unchanged on-row plants"),
                   Line2D([], [], marker="o", color=INK2, mfc=SURF, ls="", ms=6, mew=1.3, label="same-day false-change rate"),
                   Line2D([], [], color=INK, lw=1.1, ls=(0, (4, 3)), label="item 30: no verdict above 0.5")],
          loc="upper right", fontsize=6.8, frameon=False, bbox_to_anchor=(1.0, 0.93))
fig.suptitle("MuST-C JS-2: SENSITIVITY, NOT ADOPTED. The numbers behind items 31, 33 and 30 (verdicts of record = as registered)", fontsize=10.5, x=0.01, ha="left", color=INK)
fig.tight_layout(rect=(0, 0.02, 1, 0.95)); fig.subplots_adjust(bottom=0.16)
fig.savefig(A.out, dpi=200)
print("wrote", A.out)
