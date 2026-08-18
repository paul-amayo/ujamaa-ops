#!/usr/bin/env python3
"""Stage each survey's LIVE embedder into prod/bateleur/embedder/ with provenance.

WHY (Paul, 2026-08-18): "embedders need to be kept at prod/bateleur". The
embedder belongs to the survey, alongside the SAM3 masks and scene_graph it
was trained from — not in a shared scratch tree.

The trainer writes to /home/paperspace/data/high/nerf/<exp>/ckpts/model_best.pth.
Before this, a COPY also sat in prod/tassili/embedder/, and every one of those
copies was stale relative to the live file (05 by four days, 02 by two, 03 by
one; 01/04 had none). A stale embedder is the most destructive artifact in the
pipeline: features seeded under one embedder read ~0 through another
(latent cosine -0.12 though decode-equivalent), which is what zeroed 05's
blocks 005-009.

So this stages by CONTENT, records sha256 + source mtime, and refuses to
silently overwrite a differing file without saying so.

  usage: stage_embedders.py [--apply] [--survey <dir>]
"""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOTS = [Path('/home/paperspace/data/citrus_all'), Path('/home/paperspace/data/klapmuts')]
TRAIN_OUT = Path('/home/paperspace/data/high/nerf')


def sha256(p: Path, chunk=1 << 20):
    h = hashlib.sha256()
    with p.open('rb') as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()


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
    ap.add_argument('--survey')
    args = ap.parse_args()

    staged = skipped = missing = 0
    for s in surveys():
        if args.survey and s != Path(args.survey.rstrip('/')):
            continue
        exp = f"{s.name.replace('_Jackal', '')}_v1g"
        live = TRAIN_OUT / exp / 'ckpts' / 'model_best.pth'
        dst_dir = s / 'prod' / 'bateleur' / 'embedder' / exp
        dst = dst_dir / 'model_best.pth'

        if not live.is_file():
            alt = sorted(TRAIN_OUT.glob(f"{s.name.replace('_Jackal','')}*/ckpts/model_best.pth"))
            print(f"{s.name:22s} NO live embedder for {exp}"
                  f"{'  (alternatives: ' + ', '.join(p.parents[1].name for p in alt) + ')' if alt else ''}")
            missing += 1
            continue

        lsha = sha256(live)
        if dst.is_file() and sha256(dst) == lsha:
            print(f"{s.name:22s} up to date  {exp}")
            skipped += 1
            continue

        state = 'DIFFERS — restaging' if dst.is_file() else 'staging'
        print(f"{s.name:22s} {state}  {exp}  ({live.stat().st_size/1e6:.0f} MB, "
              f"live {datetime.fromtimestamp(live.stat().st_mtime):%Y-%m-%d %H:%M})")
        if args.apply:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(live, dst)
            (dst_dir / 'embedder_provenance.json').write_text(json.dumps({
                'experiment': exp,
                'source': str(live),
                'sha256': lsha,
                'source_mtime': datetime.fromtimestamp(live.stat().st_mtime).isoformat(timespec='seconds'),
                'staged_at': datetime.now().isoformat(timespec='seconds'),
                'note': 'canonical embedder for this survey; scored features must '
                        'be seeded with THIS checkpoint (a retrain orthogonalises '
                        'every already-seeded block)',
            }, indent=1))
        staged += 1

        # flag the legacy staged copy — stale copies are what poison a run
        old = s / 'prod' / 'tassili' / 'embedder' / exp / 'model_best.pth'
        if old.is_file():
            same = sha256(old) == lsha
            print(f"{'':22s}   legacy copy at prod/tassili/embedder "
                  f"({'identical' if same else 'STALE'}, "
                  f"{datetime.fromtimestamp(old.stat().st_mtime):%Y-%m-%d %H:%M}) — retire it")

    verb = 'staged' if args.apply else 'would stage'
    print(f"\n{verb} {staged}; {skipped} already current; {missing} with no live embedder")
    if not args.apply:
        print("dry run — re-run with --apply")
    return 0


if __name__ == '__main__':
    sys.exit(main())
