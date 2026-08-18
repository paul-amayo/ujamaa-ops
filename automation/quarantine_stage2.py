#!/usr/bin/env python3
"""Quarantine every stage2 artifact so the queue re-seeds from scratch.

WHY (Paul, 2026-08-18: "the stage 2 should all disappear"): the SAM3 masks,
markers, hierarchy and embedders were all quarantined today, and every stage2
checkpoint was seeded against the pair that is now gone. Features seeded under
one embedder read ~0 through another (latent cosine -0.12 though
decode-equivalent), so these checkpoints would score as noise against the
regenerated stack — measured this morning on 05: blocks 005-008 read
0.001-0.013 on trees until re-seeded, then 0.89-0.90.

MOVED per block (all derived from the quarantined semantic stack):
    splat_runs_FEATFIX/   stage2 runs, interaction census, provenance
    stage2_init/          zero-feature seed ckpt
    stage2_init_census/   census-majority feature init
    semantic_v2_B/        compiled colour-PNG supervision source
    supervision/          per-block compiled ids (word/level tables)
    splats/*stage2*       exported plys + features
and per config: verdicts_censusinit_fw2*.json (they record stage2 outcomes),
splats.json / index.json (they point at the exported stage2 plys).

KEPT: splat_runs_STAGE1 (geometry+appearance, independent of semantics),
transforms*.json, colmap_pass_*, init_lidar.ply, clip_cache_*. Stage1 is
expensive and untouched by any of this — re-seeding is ~7 min/block, retraining
stage1 would be ~25.

  usage: quarantine_stage2.py [--apply] [--stamp YYYYMMDD] [--survey <dir>]
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOTS = [Path('/home/paperspace/data/citrus_all'), Path('/home/paperspace/data/klapmuts')]
CFG = 'lio_row100'
BLOCK_ITEMS = ('splat_runs_FEATFIX', 'stage2_init', 'stage2_init_census',
               'semantic_v2_B', 'supervision')


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

    grand = 0
    for s in surveys():
        if args.survey and s != Path(args.survey.rstrip('/')):
            continue
        cfgd = s / 'prod/tassili/blocks_ns' / CFG
        if not cfgd.is_dir():
            continue
        qroot = s / 'experimental' / f'quarantine_{args.stamp}_stage2'
        receipt = {'quarantined_at': datetime.now().isoformat(timespec='seconds'),
                   'reason': 'seeded against the SAM3/hierarchy/embedder stack '
                             'quarantined 2026-08-18; would score as noise '
                             'against the regenerated pair',
                   'kept': 'splat_runs_STAGE1, transforms*.json, colmap_pass_*, '
                           'init_lidar.ply',
                   'items': []}
        moved = size = nblocks = 0
        for blk in sorted(cfgd.glob('block_*')):
            if not blk.is_dir() or not blk.name[6:].isdigit():
                continue
            hits = [blk / n for n in BLOCK_ITEMS if (blk / n).exists()]
            hits += [p for p in blk.glob('splats/*stage2*')]
            if not hits:
                continue
            nblocks += 1
            for src in hits:
                sz = du(src)
                size += sz
                moved += 1
                rel = src.relative_to(cfgd)
                receipt['items'].append({'from': str(src), 'bytes': sz})
                if args.apply:
                    dst = qroot / CFG / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
        # config-level records of stage2 outcomes
        for pat in ('verdicts_censusinit_fw2*.json', 'splats.json', 'index.json'):
            for src in sorted(cfgd.glob(pat)):
                sz = du(src)
                size += sz
                moved += 1
                receipt['items'].append({'from': str(src), 'bytes': sz})
                if args.apply:
                    dst = qroot / CFG / src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
        if moved:
            print(f"{s.name:<20} {moved:4d} items over {nblocks:3d} blocks  "
                  f"{size/1e9:6.2f} GB")
            grand += size
            if args.apply:
                qroot.mkdir(parents=True, exist_ok=True)
                (qroot / 'quarantine_receipt.json').write_text(json.dumps(receipt, indent=1))

    verb = 'moved' if args.apply else 'would move'
    print(f"\n{verb} {grand/1e9:.2f} GB of stage2 artifacts")
    if not args.apply:
        print('dry run — re-run with --apply')
    return 0


if __name__ == '__main__':
    sys.exit(main())
