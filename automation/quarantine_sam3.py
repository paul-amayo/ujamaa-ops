#!/usr/bin/env python3
"""Quarantine each survey's SAM3 products so the week queue regenerates them.

WHY (Paul, 2026-08-18): the stored SAM3 masks are the oldest layer in every
survey — 01's are from 06 Jul, 03's 10 Jul, 04's 11 Jul, 05's 13 Jul — and the
whole semantic stack (markers -> hierarchy -> embedder -> seeded features ->
containment) is built on them. Re-running image mode on 05 at the asset's OWN
recorded settings (prompt 'tree', conf 0.3, min_area 400) produced 12.4x more
masks over block_006's span than the stored file holds: 408 vs 33, and 74 of 79
frames non-empty vs 22. The stored asset is not reproducible from its own
parameters, so it is quarantined rather than trusted.

WHAT MOVES, and why it is not just sam3_v2:
the pipeline's regeneration gates are existence tests on scene_graph, not on
the masks —
    [1/7]  if [ ! -f $ROOT/scene_graph/markers_v2.monolithic ]   SAM3 + markers
    [3/7]  if [ ! -f $ROOT/scene_graph/marker_hierarchy.json ]   semantic + hierarchy
so moving sam3_v2 alone would cache-hit on the old markers and never re-run
SAM3 at all. Both directories move together.

Moves, never deletes: everything lands under
<survey>/experimental/quarantine_<stamp>_sam3/, which is the tree that is safe
to delete once the regenerated assets are verified.

  usage: quarantine_sam3.py [--apply] [--stamp YYYYMMDD] [--survey <dir>]
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOTS = [Path('/home/paperspace/data/citrus_all'), Path('/home/paperspace/data/klapmuts')]
TARGETS = [('prod/bateleur/sam3_v2', 'SAM3 masks, clips.json, global_ids.json'),
           ('prod/bateleur/scene_graph', 'markers_v2.monolithic, marker_hierarchy.json')]


def du(p: Path):
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())


def surveys():
    for r in ROOTS:
        if not r.is_dir():
            continue
        for d in sorted(r.iterdir()):
            if d.is_dir() and (d / 'prod').is_dir():
                yield d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--stamp', default=datetime.now().strftime('%Y%m%d'))
    ap.add_argument('--survey')
    args = ap.parse_args()

    moved = total = 0
    for s in surveys():
        if args.survey and s != Path(args.survey.rstrip('/')):
            continue
        qdir = s / 'experimental' / f'quarantine_{args.stamp}_sam3'
        receipt = {'quarantined_at': datetime.now().isoformat(timespec='seconds'),
                   'reason': 'SAM3 asset not reproducible from its recorded '
                             'settings (05 block_006 span: 33 stored vs 408 on '
                             'a fresh run at conf 0.3); regenerate via the queue',
                   'items': []}
        rows = []
        for rel, what in TARGETS:
            src = s / rel
            if not src.exists():
                rows.append(f"    {rel:<28} absent")
                continue
            size = du(src)
            n = len(list(src.rglob('*'))) if src.is_dir() else 1
            rows.append(f"    {rel:<28} {size/1e6:8.1f} MB  {n:5d} entries  ({what})")
            receipt['items'].append({'from': str(src), 'to': str(qdir / Path(rel).name),
                                     'bytes': size, 'entries': n})
            if args.apply:
                qdir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(qdir / Path(rel).name))
            moved += 1
            total += size
        if rows:
            print(f"{s.name}")
            print('\n'.join(rows))
        if args.apply and receipt['items']:
            (qdir / 'quarantine_receipt.json').write_text(json.dumps(receipt, indent=1))

    verb = 'moved' if args.apply else 'would move'
    print(f"\n{verb} {moved} directories, {total/1e9:.2f} GB total")
    if not args.apply:
        print('dry run — re-run with --apply')
    else:
        print('regeneration: the pipeline rebuilds SAM3+markers at [1/7] and '
              'semantic+hierarchy at [3/7]; every downstream artifact '
              '(supervision, embedder, seeded features) is stale until re-run')
    return 0


if __name__ == '__main__':
    sys.exit(main())
