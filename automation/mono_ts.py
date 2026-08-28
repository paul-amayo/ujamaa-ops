"""Timestamps of any pbImage monolithic — header-only walk (pattern from
gps_from_mono.py, inlined because that module imports pb2 at top level).
pbImage: width(1) height(2) timestamp(3) varints, image_data(4) length-delimited."""
import sys
import numpy as np


def read_varint(f):
    result = shift = n = 0
    while True:
        b = f.read(1)
        if not b:
            return None, 0
        n += 1
        v = b[0]
        result |= (v & 0x7F) << shift
        if not v & 0x80:
            return result, n
        shift += 7


def mono_timestamps(path):
    out = []
    with open(path, "rb") as f:
        while True:
            size, _ = read_varint(f)
            if size is None:
                break
            start = f.tell()
            ts = None
            while f.tell() < start + size:
                tag, _ = read_varint(f)
                if tag is None:
                    break
                field, wire = tag >> 3, tag & 7
                if wire == 0:
                    val, _ = read_varint(f)
                    if field == 3:
                        ts = val
                elif wire == 2:
                    ln, _ = read_varint(f)
                    f.seek(ln, 1)
                    if field == 4:
                        break
                else:
                    f.seek(8 if wire == 1 else 4, 1)
            f.seek(start + size)
            out.append(ts)
    return np.array(out, dtype=np.int64)


def normalize_ms(t):
    t = int(t)
    while t > 10 ** 14:  # ns/us -> ms
        t //= 1000
    return t


if __name__ == "__main__":
    for p in sys.argv[1:]:
        ts = np.array([normalize_ms(t) for t in mono_timestamps(p)])
        dt = np.diff(ts)
        print(f"{p.split('/')[-1]}: {len(ts)} frames, {(ts[-1]-ts[0])/60000:.1f} min, "
              f"median dt {np.median(dt):.0f} ms -> {1000/np.median(dt):.2f} fps, "
              f"p5/p95 dt {np.percentile(dt,5):.0f}/{np.percentile(dt,95):.0f} ms")
