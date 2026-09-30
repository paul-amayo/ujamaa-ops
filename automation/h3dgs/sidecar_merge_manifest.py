"""Merged supervision manifest for a chunk side-car: the union of the owning blocks' word tables (the words the
side-car's identity field was trained against). render_service resolves 'tree N' through this table; without it
every query clears. usage: merge_manifest.py <survey> <chunk sidecar dir> [level=trees_only]"""
import json, sys, glob
from pathlib import Path
SV, C = sys.argv[1], Path(sys.argv[2]); LEVEL = sys.argv[3] if len(sys.argv) > 3 else "trees_only"
out = C / "supervision" / LEVEL / "manifest.json"
if out.exists(): sys.exit(f"{out} exists; not overwriting")
wt, src, conflicts = {}, {}, []
for m in sorted(glob.glob(f"/home/paperspace/data/citrus_all/{SV}/prod/tassili/blocks_ns/lio_row100/block_*/supervision/{LEVEL}/manifest.json")):
    d = json.load(open(m)); b = m.split("blocks_ns/lio_row100/")[1].split("/")[0]
    for k, w in d.get("word_table", {}).items():
        if k in wt and wt[k] != w: conflicts.append((k, wt[k], w, b))
        wt.setdefault(k, w); src.setdefault(k, b)
    base = d.get("fruit_id_base", base if "base" in dir() else None); level_table = d.get("level_table")
assert not conflicts, conflicts[:5]
json.dump({"schema": "merged word table for a chunk side-car (2026-09-30)", "level": LEVEL, "word_table": wt,
           "fruit_id_base": base, "level_table": level_table, "source_blocks": sorted(set(src.values())),
           "note": "union of the owning blocks' manifests; words verified identical wherever an id appears in more than one block"}, open(out, "w"), indent=1)
print(f"{out}: {len(wt)} ids from {len(set(src.values()))} blocks; tree 90 -> {wt.get('90')!r}, tree 5 -> {wt.get('5')!r}")
