#!/usr/bin/env python3
"""Distill containment sweep logs into a machine-readable verdicts file.

The censusinit verdict step (automation/censusinit_block.sh step 5) prints
human-greppable lines and saves figs but records no json; prod-readiness
checks (automation/build_prod_manifests.py) need a machine-readable record.
This distills a sweep log's containment_eval lines into
blocks_ns/<cfg>/verdicts_censusinit_fw2.json with full provenance.

Line format parsed (containment_eval.py output):
  [002 kf_000188.png] TREE 14  "pollock": thr 0.99 IoU 0.920 prec 0.938 rec 0.980

--inject adds blocks whose verdicts were recorded outside the sweep log
(e.g. individual session runs with figs); pass a json literal.

One-shot usage that produced the 05 file (2026-08-14):
  python3 automation/distill_containment_verdicts.py \
    --log /home/paperspace/logs/containment_sweep_05.log \
    --out /home/paperspace/data/citrus_all/05_13D_Jackal/blocks_ns/lio_row100/verdicts_censusinit_fw2.json \
    --inject '{"000": {...mixed-pose 0.498...}, "001": {...0.958...}}'

FOLLOW-UP (open): censusinit_block.sh should append verdicts here directly
so future blocks never need distillation.
"""
import argparse
import json
import re
import time
from pathlib import Path

PAT = re.compile(
    r"\[(\d{3}) (kf_\d+\.png)\] (TREE|ROW) \d+\s+\"([a-z]+)\": "
    r"thr [\d.]+ IoU ([\d.]+) prec ([\d.]+) rec ([\d.]+)")
# containment_eval's RAW stdout has no [NNN frame] prefix (the sweep script
# added those) — bare lines attach to --block-id/--frame when given
PAT_BARE = re.compile(
    r"(TREE|ROW) \d+\s+\"([a-z]+)\": "
    r"thr [\d.]+ IoU ([\d.]+) prec ([\d.]+) rec ([\d.]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config-label", default="stage2_censusinit_fw2")
    ap.add_argument("--inject", default="{}",
                    help="json literal: extra block records from session runs")
    ap.add_argument("--merge", action="store_true",
                    help="update blocks into an existing --out file instead of "
                         "overwriting it (per-block append from the week queue)")
    ap.add_argument("--block-id", default=None,
                    help="block id (e.g. 001) for UNPREFIXED containment_eval "
                         "lines — required to parse raw per-block output")
    ap.add_argument("--frame", default=None,
                    help="frame name recorded with --block-id entries")
    args = ap.parse_args()

    blocks = {}
    for line in Path(args.log).read_text(errors="replace").splitlines():
        m = PAT.search(line)
        if m:
            b, frame, level, word, iou = m.groups()[:5]
        elif args.block_id:
            mb = PAT_BARE.search(line)
            if not mb:
                continue
            level, word, iou = mb.groups()[:3]
            b, frame = args.block_id, args.frame or "?"
        else:
            continue
        rec_ = blocks.setdefault(b, {"frame": frame, "trees": {}, "rows": {},
                                     "source": f"distilled from {args.log}"})
        rec_["trees" if level == "TREE" else "rows"][word] = float(iou)
    for b, rec_ in blocks.items():
        vals = list(rec_["trees"].values())
        if vals:
            rec_["tree_iou_min"], rec_["tree_iou_max"] = min(vals), max(vals)

    if not blocks and not json.loads(args.inject):
        print(f"PARSE-ZERO: no verdict lines matched in {args.log} — refusing "
              f"to write (pass --block-id for raw containment_eval output)")
        raise SystemExit(2)

    for b, rec_ in json.loads(args.inject).items():
        blocks[b] = rec_

    if args.merge and Path(args.out).exists():
        prev = json.load(open(args.out)).get("blocks", {})
        prev.update(blocks)
        blocks = prev

    out = {
        "config": args.config_label,
        "metric": ("containment tree IoU — containment_eval.py best_mask "
                   "threshold sweep, GT = compiled supervision masks"),
        "pass_rule": "tree_iou_min >= 0.80 (04-canon floor)",
        "distilled": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "distiller": "automation/distill_containment_verdicts.py",
        "blocks": dict(sorted(blocks.items())),
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    n_pass = sum(1 for r in blocks.values()
                 if r.get("tree_iou_min", 0) >= 0.80)
    print(f"wrote {args.out}: {len(blocks)} blocks recorded, {n_pass} pass")


if __name__ == "__main__":
    main()
