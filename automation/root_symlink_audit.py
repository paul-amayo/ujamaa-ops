#!/usr/bin/env python3
"""Inventory EVERY symlink under each survey and write a RESTORE MANIFEST.

WHY (Paul, 2026-08-18): prod assets must physically live in prod/. The
survey root — and the shimmed paths beneath it — is where poisoned and stale
artifacts accumulated, and the links let those stale paths keep resolving
silently. Removing them makes every stale reference fail loudly instead of
quietly reading whatever the shim points at.

The pipeline WILL break. That is the point: each break names a call site
with staleness embedded in it.

Removing a symlink never deletes its target, so this loses no data — only
reachability. This manifest records every link (path relative to the survey,
plus the literal target string, so relative links stay relative) and is what
makes the experiment reversible: restore_root_symlinks.py puts them all back.

Read-only. Writes one JSON.
"""
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOTS = [Path('/home/paperspace/data/citrus_all'), Path('/home/paperspace/data/klapmuts')]
OUT = Path('/home/paperspace/logs/root_symlink_manifest.json')


def surveys():
    for r in ROOTS:
        if not r.is_dir():
            continue
        for d in sorted(r.iterdir()):
            if d.is_dir() and (d / 'prod').is_dir():
                yield d


def walk_links(s: Path):
    for dirpath, dirnames, filenames in os.walk(s, followlinks=False):
        for n in list(dirnames) + list(filenames):
            p = Path(dirpath) / n
            if p.is_symlink():
                yield p


def main():
    manifest = {'created': datetime.now().isoformat(timespec='seconds'),
                'note': 'every symlink under each survey; restore with '
                        'automation/restore_root_symlinks.py',
                'surveys': {}}
    tot = 0
    for s in surveys():
        links = []
        for p in sorted(walk_links(s)):
            tgt = os.readlink(p)
            resolved = Path(os.path.realpath(p))
            links.append({
                'path': str(p.relative_to(s)),      # depth-aware
                'target': tgt,                      # literal, keeps relatives relative
                'resolved': str(resolved),
                'exists': resolved.exists(),
                'depth': len(p.relative_to(s).parts),
                'into_prod': '/prod/' in str(resolved) or str(resolved).endswith('/prod'),
                'outside_survey': not str(resolved).startswith(str(s)),
            })
        manifest['surveys'][str(s)] = links
        tot += len(links)
        d = Counter(l['depth'] for l in links)
        print(f"{s.name:22s} {len(links):5d} links   "
              f"root {d.get(1,0):3d}  deeper {sum(v for k,v in d.items() if k>1):5d}   "
              f"dangling {sum(1 for l in links if not l['exists']):3d}   "
              f"point outside survey {sum(1 for l in links if l['outside_survey']):3d}")
    OUT.write_text(json.dumps(manifest, indent=1))
    print(f"\n{tot} symlinks across {len(manifest['surveys'])} surveys")
    print(f"restore manifest -> {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
