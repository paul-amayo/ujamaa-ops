#!/usr/bin/env python3
"""zenodo_repair_chunks.py — re-fetch given chunks of a Zenodo file straight into an existing (corrupt) local copy, in place (2026-10-14):
for each chunk index i, download bytes [i*CH, (i+1)*CH-1] with a Range request (206 + Content-Range checked, writes capped — the fixes of
zenodo_parallel_get.py), into memory, then write them at offset i*CH of the local file. N connections; retries; md5 of the result vs Zenodo's.
system python3:  zenodo_repair_chunks.py <record id> <file name> <local file> --chunks 0 1 2 ... [--chunk-mib 64] [--conn 12]"""
import argparse, hashlib, json, os, threading, time, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument('record'); ap.add_argument('name'); ap.add_argument('local'); ap.add_argument('--chunks', nargs='+', type=int, required=True)
ap.add_argument('--chunk-mib', type=int, default=64); ap.add_argument('--conn', type=int, default=12); a = ap.parse_args()
URL = f'https://zenodo.org/api/records/{a.record}/files/{a.name}/content'; CH = a.chunk_mib << 20; total = os.path.getsize(a.local)
meta = json.load(urllib.request.urlopen(f'https://zenodo.org/api/records/{a.record}', timeout=60)); want = [f for f in meta['files'] if f['key'] == a.name][0]
assert want['size'] == total, f'size {total} vs zenodo {want["size"]}'
todo = list(a.chunks); lock = threading.Lock(); done = []; failed = []; fh_lock = threading.Lock()
def fetch(i):
    lo, hi = i * CH, min(total, (i + 1) * CH) - 1; need = hi - lo + 1
    for attempt in range(10):
        try:
            buf = bytearray(); r = urllib.request.Request(URL, headers={'Range': f'bytes={lo}-{hi}'})
            with urllib.request.urlopen(r, timeout=120) as resp:
                cr = resp.headers.get('Content-Range', '')
                if resp.status != 206 or not cr.startswith(f'bytes {lo}-'): raise IOError(f'Range not honoured ({resp.status}, {cr!r})')
                while len(buf) < need:
                    b = resp.read(min(1 << 18, need - len(buf)))
                    if not b: break
                    buf += b
            if len(buf) != need: raise IOError(f'short read {len(buf)} / {need}')
            with fh_lock:
                with open(a.local, 'r+b') as f: f.seek(lo); f.write(buf)
            return True
        except Exception as e:
            time.sleep(2 + 3 * attempt)
    return False
def worker():
    while True:
        with lock:
            if not todo: return
            i = todo.pop(0)
        ok = fetch(i)
        with lock: (done if ok else failed).append(i)
t0 = time.time(); ts = [threading.Thread(target=worker, daemon=True) for _ in range(a.conn)]; [t.start() for t in ts]
while any(t.is_alive() for t in ts):
    time.sleep(30); print(f'[repair] {len(done)} / {len(a.chunks)} chunks rewritten, {len(failed)} failed, {(time.time() - t0) / 60:.1f} min', flush=True)
h = hashlib.md5()
with open(a.local, 'rb') as f:
    while True:
        b = f.read(1 << 24)
        if not b: break
        h.update(b)
print(f"[repair] done: {len(done)} rewritten, failed {sorted(failed)}; md5 {h.hexdigest()} vs zenodo {want['checksum']} -> {'MATCH' if 'md5:' + h.hexdigest() == want['checksum'] else 'MISMATCH'}", flush=True)
