#!/usr/bin/env python3
"""zip_bad_chunks.py — find which download chunks of a zip are corrupt (2026-10-14: FruitNeRF_Real.zip md5 mismatch after the Range bug):
reads every member (zipfile checks each CRC), and maps the members that fail to the byte span they occupy in the archive (local header ..
end of compressed data) -> the chunk indices (chunk = --chunk-mib) that overlap a bad member. Prints the indices for zenodo_repair_chunks.py.
system python3:  zip_bad_chunks.py <zip> [--chunk-mib 64] [--out bad.json]"""
import argparse, json, zipfile, zlib
ap = argparse.ArgumentParser(); ap.add_argument('zip'); ap.add_argument('--chunk-mib', type=int, default=64); ap.add_argument('--out', default=''); a = ap.parse_args()
CH = a.chunk_mib << 20; z = zipfile.ZipFile(a.zip); infos = sorted(z.infolist(), key=lambda i: i.header_offset); bad = []; n = 0
for k, zi in enumerate(infos):
    if zi.is_dir(): continue
    n += 1
    try:
        with z.open(zi) as f:
            while f.read(1 << 24): pass
    except Exception as e:
        end = infos[k + 1].header_offset if k + 1 < len(infos) else zi.header_offset + 30 + len(zi.filename) + len(zi.extra) + zi.compress_size
        bad.append({'name': zi.filename, 'start': zi.header_offset, 'end': end, 'err': str(e)[:80]})
chunks = sorted({c for b in bad for c in range(b['start'] // CH, (b['end'] - 1) // CH + 1)})
print(f'[zipcheck] {n} members read, {len(bad)} fail their CRC; bad chunks ({len(chunks)}): {chunks}', flush=True)
for b in bad[:12]: print(f"[zipcheck]   {b['name']} bytes {b['start']}-{b['end']}: {b['err']}", flush=True)
if a.out: json.dump({'chunk_bytes': CH, 'bad_chunks': chunks, 'bad_members': bad}, open(a.out, 'w'), indent=1)
