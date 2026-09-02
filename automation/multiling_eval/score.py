#!/usr/bin/env python3
"""Score the anchor runs -> per-language table + the decomposition.

Belebele: accuracy. FLORES: chrF, local implementation (character n-grams
n=1..6, beta=2, whitespace stripped, macro-averaged over sentences) —
labelled as such; comparable within this run, not against published tables.

  python3 score.py <run_dir with *.jsonl>
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ARMS = {"gemma-3-12b-pt": "g3", "AfriqueGemma-12B": "afrique",
        "gemma-4-12b#bf16": "g4pt", "gemma-4-12b-it": "g4"}
LANGS = ["en", "af", "xh", "zu", "sw", "ha", "ar", "am"]


def ngrams(s, n):
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(hyp, ref, max_n=6, beta=2.0):
    h = "".join(hyp.split())
    r = "".join(ref.split())
    if not h or not r:
        return 0.0
    ps, rs = [], []
    for n in range(1, max_n + 1):
        hn, rn = ngrams(h, n), ngrams(r, n)
        if not hn or not rn:
            continue
        overlap = sum((hn & rn).values())
        ps.append(overlap / max(sum(hn.values()), 1))
        rs.append(overlap / max(sum(rn.values()), 1))
    if not ps:
        return 0.0
    p, rc = sum(ps) / len(ps), sum(rs) / len(rs)
    if p + rc == 0:
        return 0.0
    b2 = beta * beta
    return 100 * (1 + b2) * p * rc / (b2 * p + rc)


def arm_of(model):
    for k, v in ARMS.items():
        if k in model:
            return v
    return model


def main():
    run_dir = Path(sys.argv[1])
    acc = defaultdict(list)     # (arm, lang) -> [bool]
    chrfs = defaultdict(list)   # (arm, lang, direction) -> [f]
    for f in run_dir.glob("*.jsonl"):
        for ln in open(f):
            d = json.loads(ln)
            arm = arm_of(d["model"])
            if d["task"] == "belebele":
                acc[(arm, d["lang"])].append(d["ok"])
            else:
                direction = str(d["idx"]).split(":")[0]
                chrfs[(arm, d["lang"], direction)].append(chrf(d["hyp"], d["ref"]))

    ARMLIST = ["g3", "afrique", "g4pt", "g4"]

    def table(title, get, langs):
        print(f"\n=== {title} ===")
        print("lang   " + "".join(f"{a:>9}" for a in ARMLIST)
              + "     adapt    gen_pt     gener")
        for lg in langs:
            v = {a: get(a, lg) for a in ARMLIST}
            if all(x is None for x in v.values()):
                continue
            row = f"{lg:5}"
            for a in ARMLIST:
                row += f"{v[a]:9.1f}" if v[a] is not None else f"{'—':>9}"
            if v["g3"] is not None:
                for other in ("afrique", "g4pt", "g4"):
                    row += (f"  {v[other] - v['g3']:+8.1f}"
                            if v[other] is not None else f"  {'—':>8}")
            print(row)

    table("Belebele accuracy (%)",
          lambda a, lg: (100 * sum(acc[(a, lg)]) / len(acc[(a, lg)])
                         if acc[(a, lg)] else None), LANGS)
    for d in ["en-xx", "xx-en"]:
        table(f"FLORES chrF {d} (local impl)",
              lambda a, lg, d=d: (sum(chrfs[(a, lg, d)]) / len(chrfs[(a, lg, d)])
                                  if chrfs[(a, lg, d)] else None),
              [l for l in LANGS if l != "en"])
    print("\nadapt  = afrique - g3 (base vs base, identical raw few-shot, both Q4)")
    print("gen_pt = g4pt - g3 (base vs base, identical protocol; g4pt is bf16 vs")
    print("         g3 Q4 — precision favours g4pt slightly)")
    print("gener  = g4-it - g3 — CROSS-PLANE (chat template, deployed plane);")
    print("         indicative only.")


if __name__ == "__main__":
    main()
