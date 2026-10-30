"""Scan phone-video segments for faces (OpenCV YuNet) at a few fps; report which segments need blurring.
usage: face_scan.py <video> <start_s> <dur_s> [fps]"""
import sys, cv2
M = "/Users/paulamayo/data/for_a100/google_org_2026-11/_models/face_detection_yunet_2023mar.onnx"
v, t0, dur = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); fps = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
cap = cv2.VideoCapture(v); src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
det = cv2.FaceDetectorYN.create(M, "", (W, H), 0.6, 0.3, 5000)
hits = []; step = max(1, int(round(src_fps / fps)))
cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000); n = 0; i0 = int(t0 * src_fps)
while n < dur * src_fps:
    ok, fr = cap.read()
    if not ok: break
    if n % step == 0:
        _, f = det.detect(fr)
        if f is not None:
            for r in f: hits.append((round((i0 + n) / src_fps, 2), int(r[2]), round(float(r[14]), 2)))
    n += 1
print(v.split("/")[-1], f"{t0}-{t0 + dur}s", f"{W}x{H}", "faces:", len(hits), hits[:6])
