#!/usr/bin/env python3
"""Undistort SAM3 id maps of an image-only (phone) survey into the H3DGS training camera (UJAMAA 2026-10-04; Paul: "undistort
the masks and reseed the cabbage"). The survey's SfM self-calibrated an OPENCV camera (<survey>/sparse/0/cameras.bin) and H3DGS's
preprocess (colmap image_undistorter) wrote the training images as a PINHOLE camera with the same focal/centre; the SAM3 maps
stayed on the raw frames. Per output pixel (u, v) of the PINHOLE camera: normalise, apply the OPENCV distortion, sample the
raw map there (NEAREST - ids are labels), outside the raw frame -> 65535 (unlabelled). Non-PNG files (manifest.json) copied.
usage: undistort_id_maps.py <raw cameras.bin> <pinhole cameras.bin> <src map dir> <dst map dir> [--check-raw <raw rgb> --check-rect <rectified rgb>]"""
import argparse, os, shutil, struct, glob
import numpy as np
from PIL import Image
def cam(p):
    f = open(p, 'rb'); n = struct.unpack('<Q', f.read(8))[0]; cid, model, w, h = struct.unpack('<iiQQ', f.read(24))
    k = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12}[model]; return model, int(w), int(h), struct.unpack('<' + 'd' * k, f.read(8 * k))
ap = argparse.ArgumentParser(); ap.add_argument('raw_cam'); ap.add_argument('pin_cam'); ap.add_argument('src'); ap.add_argument('dst')
ap.add_argument('--check-raw'); ap.add_argument('--check-rect'); a = ap.parse_args()
m0, W0, H0, p0 = cam(a.raw_cam); m1, W1, H1, p1_ = cam(a.pin_cam)
assert m0 == 4 and m1 == 1, f'expected OPENCV -> PINHOLE, got {m0} -> {m1}'
fx, fy, cx, cy, k1, k2, q1, q2 = p0; Fx, Fy, Cx, Cy = p1_
v, u = np.mgrid[0:H1, 0:W1].astype(np.float64)
x = (u - Cx) / Fx; y = (v - Cy) / Fy; r2 = x * x + y * y; rad = 1 + k1 * r2 + k2 * r2 * r2
xd = x * rad + 2 * q1 * x * y + q2 * (r2 + 2 * x * x); yd = y * rad + q1 * (r2 + 2 * y * y) + 2 * q2 * x * y
us = np.rint(fx * xd + cx).astype(np.int64); vs = np.rint(fy * yd + cy).astype(np.int64); ok = (us >= 0) & (us < W0) & (vs >= 0) & (vs < H0)
usc, vsc = np.clip(us, 0, W0 - 1), np.clip(vs, 0, H0 - 1)
print(f'[undistort] OPENCV k1 {k1:.5f} k2 {k2:.5f} p1 {q1:.5f} p2 {q2:.5f} -> PINHOLE {W1}x{H1}; {100 * (~ok).mean():.2f}% of output pixels fall outside the raw frame')
if a.check_raw and a.check_rect:
    raw = np.asarray(Image.open(a.check_raw).convert('L')).astype(np.float32); rect = np.asarray(Image.open(a.check_rect).convert('L')).astype(np.float32)
    warped = np.where(ok, raw[vsc, usc], 0)
    def ncc(p, q):
        gp = np.hypot(*np.gradient(p)); gq = np.hypot(*np.gradient(q)); m = ok & (rect > 0)
        gp = gp[m] - gp[m].mean(); gq = gq[m] - gq[m].mean(); return float((gp * gq).sum() / np.sqrt((gp * gp).sum() * (gq * gq).sum()))
    print(f'[undistort] check: edge NCC raw vs training image {ncc(raw, rect):.3f} -> warped raw vs training image {ncc(warped, rect):.3f}; mean abs diff {np.abs(raw - rect)[ok].mean():.1f} -> {np.abs(warped - rect)[ok].mean():.1f}')
os.makedirs(a.dst, exist_ok=True); n = 0
for f in sorted(glob.glob(os.path.join(a.src, '*'))):
    out = os.path.join(a.dst, os.path.basename(f))
    if f.endswith('.png'):
        m = np.array(Image.open(f)); assert m.shape[:2] == (H0, W0), (f, m.shape)
        o = m[vsc, usc]; o[~ok] = 65535 if m.dtype == np.uint16 else 0; Image.fromarray(o).save(out); n += 1
    elif os.path.isfile(f): shutil.copy2(f, out)
print(f'[undistort] {n} maps -> {a.dst}')
