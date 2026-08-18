#!/usr/bin/env python3
"""Unified-pipeline asset audit: per step, per survey — what it needs, what it
creates, and whether that exists.

Written 2026-08-18, right after all 3,222 survey symlinks were removed so that
prod assets can only be addressed where they physically live. The pipeline
still addresses everything as $ROOT/<asset>, so this reports BOTH:

  ROOT   does the path the pipeline actually uses exist?
  PHYS   where does the data physically live, and is it there?

The physical location comes from the symlink restore manifest
(/home/paperspace/logs/root_symlink_manifest.json), which recorded every
link's target before removal — i.e. the ground truth of where each logical
asset was really stored.

Steps and asset paths are taken from run_unified_pipeline.sh, not from memory.
"""
import json
import sys
from pathlib import Path

MANIFEST = Path('/home/paperspace/logs/root_symlink_manifest.json')
CONFIG = 'lio_row100'
HIGH_NERF = Path('/home/paperspace/data/high/nerf')

# (step label, asset path relative to survey root, what it is)
STEPS = [
    ('0/7   pose',        'transform_lio.monolithic',           'file'),
    ('0b/7  kf mono',     'image_left_kf20cm.monolithic',       'file'),
    ('0b/7  kf index',    'image_left_kf20cm.monolithic.index', 'file'),
    ('0c/7  kf pngs',     'kf_images',                          'dir:kf_*.png'),
    ('1/7   laser',       'laser.monolithic',                   'file'),
    ('1/7   sam3',        'sam3_v2/clips.json',                 'file'),
    ('1/7   global ids',  'sam3_v2/global_ids.json',            'file'),
    ('1/7   markers',     'scene_graph/markers_v2.monolithic',  'file'),
    ('3/7   semantic',    'filtered_semantic_v2.monolithic',    'file'),
    ('3/7   hierarchy',   'scene_graph/marker_hierarchy.json',  'file'),
    ('3b/7  fg masks',    'fg_masks',                           'dir:*.png'),
    ('3b/7  sky masks',   'sky_masks',                          'dir:*.png'),
    ('4/7   partition',   f'blocks/{CONFIG}/blocks.json',       'file'),
    ('4/7   blocks_ns',   f'blocks_ns/{CONFIG}',                'dir:block_*'),
]


def probe(p: Path, kind: str):
    """(exists, detail) — detail counts members for dirs, MB for files."""
    if kind.startswith('dir:'):
        pat = kind.split(':', 1)[1]
        if not p.is_dir():
            return False, '-'
        n = len(list(p.glob(pat)))
        return n > 0, f'{n} {pat}'
    if not p.is_file():
        return False, '-'
    mb = p.stat().st_size / 1e6
    return True, (f'{mb:.0f} MB' if mb >= 1 else f'{p.stat().st_size} B')


def main():
    man = json.loads(MANIFEST.read_text())['surveys']
    surveys = sorted(man)
    for s in surveys:
        S = Path(s)
        # logical asset -> physical target, from the pre-removal manifest
        phys = {l['path']: l['resolved'] for l in man[s]}
        print(f"\n{'='*104}\n{S.name}\n{'='*104}")
        print(f"  {'step':<20} {'asset':<36} {'ROOT':<6} {'PHYSICAL':<9} where / detail")
        for label, rel, kind in STEPS:
            root_p = S / rel
            r_ok, r_det = probe(root_p, kind)
            # resolve physical: exact match, else nearest recorded ancestor
            target = phys.get(rel)
            if target is None:
                for anc in Path(rel).parents:
                    if str(anc) in phys:
                        target = str(Path(phys[str(anc)]) / Path(rel).relative_to(anc))
                        break
            if target is None:
                p_ok, p_det, where = r_ok, r_det, '(no link recorded — real path)'
            else:
                p_ok, p_det = probe(Path(target), kind)
                where = str(Path(target).relative_to(S)) if str(target).startswith(str(S)) else target
            print(f"  {label:<20} {rel:<36} {'yes' if r_ok else 'NO ':<6} "
                  f"{'yes' if p_ok else 'NO ':<9} {where}  [{p_det if p_ok else r_det}]")
        tag = S.name.replace('_Jackal', '')
        ck = HIGH_NERF / f'{tag}_v1g' / 'ckpts' / 'model_best.pth'
        alt = sorted(HIGH_NERF.glob(f'{tag}*/ckpts/model_best.pth'))
        e_ok, e_det = probe(ck, 'file')
        print(f"  {'5b/7  embedder':<20} {f'{tag}_v1g/ckpts/model_best.pth':<36} "
              f"{'-':<6} {'yes' if e_ok else 'NO ':<9} "
              f"{ck if e_ok else ('alternatives: ' + ', '.join(p.parents[1].name for p in alt) if alt else 'none')}"
              f"  [{e_det}]")
    return 0


if __name__ == '__main__':
    sys.exit(main())
