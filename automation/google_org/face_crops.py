"""Collect the highest-scoring YuNet detections of a segment as crops, to judge real faces vs foliage false positives."""
import sys, cv2, numpy as np
M = "/Users/paulamayo/data/for_a100/google_org_2026-11/_models/face_detection_yunet_2023mar.onnx"
v, t0, dur, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
cap = cv2.VideoCapture(v); fps = cap.get(cv2.CAP_PROP_FPS) or 30; W, H = int(cap.get(3)), int(cap.get(4))
det = cv2.FaceDetectorYN.create(M, "", (W, H), 0.6, 0.3, 5000); cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
best = []; n = 0
while n < dur * fps:
    ok, fr = cap.read()
    if not ok: break
    if n % 15 == 0:
        _, f = det.detect(fr)
        for r in (f if f is not None else []):
            x, y, w, h = [int(a) for a in r[:4]]; s = float(r[14]); m = max(w, h)
            c = fr[max(0, y - m // 2):y + h + m // 2, max(0, x - m // 2):x + w + m // 2]
            if c.size: best.append((s, round(t0 + n / fps, 1), cv2.resize(c, (96, 96))))
    n += 1
best.sort(key=lambda b: -b[0]); best = best[:24]
tiles = [cv2.putText(c.copy(), f"{s:.2f} {t}", (2, 12), 0, 0.35, (0, 255, 255), 1) for s, t, c in best] or [np.zeros((96, 96, 3), np.uint8)]
while len(tiles) % 8: tiles.append(np.zeros((96, 96, 3), np.uint8))
cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)])); print(out, len(best), "top score", best[0][0] if best else None)
