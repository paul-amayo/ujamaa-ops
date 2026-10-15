#!/usr/bin/env python3
"""zenodo_parallel_get.py — download one Zenodo file over many parallel byte-range connections (2026-10-13). Zenodo throttles each
connection to tens of KB/s (FruitNeRF_Real.zip: 8-20 KB/s single-stream, i.e. weeks for 19 GB) but honours Range on its API endpoint
(/api/records/<id>/files/<name>/content -> 206), so N connections give ~N x the throughput. Resumable: every chunk is its own file under
<out>.parts/ and is skipped when complete; the chunks are concatenated into <out> at the end and the parts removed only after the size
check. Prints the aggregate rate every 30 s.
system python3:  zenodo_parallel_get.py <record id> <file name> <out path> [--conn 32] [--chunk-mib 64]"""
import argparse, os, sys, time, threading, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument('record'); ap.add_argument('name'); ap.add_argument('out'); ap.add_argument('--conn', type=int, default=32); ap.add_argument('--chunk-mib', type=int, default=64)
a = ap.parse_args(); URL = f'https://zenodo.org/api/records/{a.record}/files/{a.name}/content'; CH = a.chunk_mib * 1048576
req = urllib.request.Request(URL, method='HEAD'); total = int(urllib.request.urlopen(req, timeout=60).headers['Content-Length'])
parts = f'{a.out}.parts'; os.makedirs(parts, exist_ok=True); n = (total + CH - 1) // CH
todo = [i for i in range(n) if not (os.path.exists(f'{parts}/{i:05d}') and os.path.getsize(f'{parts}/{i:05d}') == min(CH, total - i * CH))]
print(f'[zenodo] {a.name}: {total / 1e9:.2f} GB in {n} chunks of {a.chunk_mib} MiB, {len(todo)} to fetch, {a.conn} connections', flush=True)
lock = threading.Lock(); got = [0]; fails = [0]
def worker():
    while True:
        with lock:
            if not todo: return
            i = todo.pop(0)
        lo, hi = i * CH, min(total, (i + 1) * CH) - 1; tmp = f'{parts}/{i:05d}.tmp'
        for attempt in range(8):
            try:
                need = hi - lo + 1; have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
                if have > need: have = 0   # (2026-10-14) an oversized tmp is garbage: rewritten from scratch below ('wb'), never appended to
                if have < need:
                    r = urllib.request.Request(URL, headers={'Range': f'bytes={lo + have}-{hi}'})
                    with urllib.request.urlopen(r, timeout=120) as resp:
                        # (2026-10-14) Zenodo sometimes ignores Range and answers 200 with the WHOLE file: before this check that stream was
                        # appended to the chunk's tmp (11 tmps grew to 2.1-2.7 GB each, the last 12 chunks never completed, 27 GB of disk)
                        cr = resp.headers.get('Content-Range', '')
                        if resp.status != 206 or not cr.startswith(f'bytes {lo + have}-'):
                            raise IOError(f'Range not honoured (status {resp.status}, Content-Range {cr!r})')
                        with open(tmp, 'ab' if have else 'wb') as f:
                            left = need - have
                            while left > 0:
                                b = resp.read(min(1 << 18, left))
                                if not b: break
                                f.write(b); left -= len(b)
                                with lock: got[0] += len(b)
                if os.path.getsize(tmp) == hi - lo + 1: os.replace(tmp, f'{parts}/{i:05d}'); break
            except Exception as e:
                with lock: fails[0] += 1
                time.sleep(2 + 3 * attempt)
        else: print(f'[zenodo] chunk {i} FAILED after 8 attempts', flush=True)
ts = [threading.Thread(target=worker, daemon=True) for _ in range(a.conn)]; [t.start() for t in ts]; t0 = time.time(); last = 0
while any(t.is_alive() for t in ts):
    time.sleep(30); done = sum(os.path.getsize(f'{parts}/{f}') for f in os.listdir(parts)); el = time.time() - t0
    print(f'[zenodo] {done / 1e9:.2f} / {total / 1e9:.2f} GB, {(got[0] - last) / 30 / 1e6:.2f} MB/s now, {got[0] / el / 1e6:.2f} MB/s mean, {fails[0]} retries, ETA {(total - done) / max(got[0] / el, 1) / 3600:.1f} h', flush=True); last = got[0]
with open(a.out, 'wb') as out:
    for i in range(n):
        p = f'{parts}/{i:05d}'
        if not os.path.exists(p): sys.exit(f'[zenodo] missing chunk {i}; rerun to resume')
        with open(p, 'rb') as f: out.write(f.read())
if os.path.getsize(a.out) == total:
    for f in os.listdir(parts): os.remove(f'{parts}/{f}')
    os.rmdir(parts); print(f'[zenodo] done: {a.out} ({total} bytes) in {(time.time() - t0) / 60:.0f} min', flush=True)
else: sys.exit(f'[zenodo] size mismatch {os.path.getsize(a.out)} vs {total}')
