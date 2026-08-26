"""Fruit diet check — the pre/post gate for mask-driven densification.

From an interaction census W ([entities x gaussians], labels incl. fruit ids
>= FRUIT_ID_BASE), reports per fruit entity how much of ITS pixel blend weight
comes from gaussians whose own supervised diet is mostly fruit. That share is
the ceiling on the rendered fruit component at that entity's pixels — measured
2026-08-26 on 05 b000 at 8-23% pre-densify (rendered norm pinned at tree
radius), which is why the mask-driven split pass (densify_block.sh) exists.

usage: fruit_diet_check.py <interaction_W.npz> [--thresh 0.5]
"""
import argparse

import numpy as np

FRUIT_ID_BASE = 10000   # must match scripts/compile_supervision.py:FRUIT_ID_BASE

ap = argparse.ArgumentParser()
ap.add_argument("w_npz")
ap.add_argument("--thresh", type=float, default=0.5)
a = ap.parse_args()

z = np.load(a.w_npz, allow_pickle=True)
W, lab = z["W"], [int(x) for x in z["labels"]]
fr = np.array([l >= FRUIT_ID_BASE for l in lab])
if not fr.any():
    print("DIET: no fruit entities in this census")
    raise SystemExit(0)
tot = W.sum(0)
sup = tot > 0
share = np.zeros_like(tot)
share[sup] = W[fr].sum(0)[sup] / tot[sup]

print(f"DIET: {int((share > a.thresh).sum())} gaussians with diet > "
      f"{int(a.thresh*100)}% fruit ({int((share > 0.9).sum())} > 90%)")
worst = 100.0
for i, l in enumerate(lab):
    if l < FRUIT_ID_BASE:
        continue
    w = W[i]
    t = w.sum()
    pct = 100 * w[share > a.thresh].sum() / t if t > 0 else 0.0
    worst = min(worst, pct)
    print(f"DIET: fruit {l}  blend-from-fruit-diet {pct:5.1f}%  "
          f"(px-weight {t:.0f}, top-gaussian diet {share[w.argmax()]:.2f})")
print(f"DIET-MIN: {worst:.1f}")
