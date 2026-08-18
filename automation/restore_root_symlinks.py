#!/usr/bin/env python3
"""Undo remove_root_symlinks.sh from the audit's restore manifest.

Recreates each survey-root symlink exactly as recorded (same name, same
target string, so relative links stay relative). Never overwrites a real
file or directory — if real data now sits at that path it is reported and
skipped, since that is the copy that matters.

  usage: restore_root_symlinks.py [--survey <dir>] [--dry-run]
"""
import argparse
import json
import os
import sys
from pathlib import Path

MANIFEST = Path('/home/paperspace/logs/root_symlink_manifest.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=str(MANIFEST))
    ap.add_argument('--survey', help='restore only this survey dir')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    m = json.loads(Path(args.manifest).read_text())
    made = present = blocked = 0
    for sdir, links in m['surveys'].items():
        if args.survey and Path(sdir) != Path(args.survey.rstrip('/')):
            continue
        print(f"=== {Path(sdir).name} ({len(links)} links)")
        for l in links:
            p = Path(sdir) / l['path']
            if p.is_symlink():
                present += 1
                continue
            if p.exists():
                print(f"  BLOCKED {l['path']}: real data sits here now — left alone")
                blocked += 1
                continue
            if args.dry_run:
                print(f"  would link {l['path']} -> {l['target']}")
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(l['target'], p)
                print(f"  linked {l['path']} -> {l['target']}")
            made += 1
    verb = 'would restore' if args.dry_run else 'restored'
    print(f"\n{verb} {made}; {present} already present; {blocked} blocked by real data")
    return 0


if __name__ == '__main__':
    sys.exit(main())
