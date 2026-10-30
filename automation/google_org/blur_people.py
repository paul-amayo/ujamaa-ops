#!/usr/bin/env python3
"""Blur faces in phone field footage for the Google.org reel (Paul, 30 Oct 2026: "blur faces").

A face detector alone was not enough on this footage: YuNet found the frontal faces but fired on foliage at the same scores
and missed people seen side-on or far away. So: SAM3 image mode with two prompts per frame, "person" and "human face".
  - every "human face" mask is blurred (dilated);
  - every "person" mask has its HEAD blurred: the top 30 % of the mask's bounding box, intersected with the dilated mask
    and widened slightly, so a turned or distant head is covered even when no face is visible.
Masks are computed every STRIDE-th frame and held for the frames in between (people move little at 30 fps), then each
blurred region is held for +/- HOLD frames so a missed detection does not let a face flash through.
Decode and encode go through ffmpeg pipes (no OpenCV writer): H.264 crf 17, yuv420p, same fps, no audio.

sam3 pixi env:  blur_people.py <in.mp4> <out.mp4> [stride=2] [conf=0.35]
Writes <out>.json with per-frame region counts, and <out>_check.jpg with the frames that had the most blurred regions."""
import json, subprocess, sys
import numpy as np, torch, cv2
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

src, dst = sys.argv[1], sys.argv[2]
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 2
CONF = float(sys.argv[4]) if len(sys.argv) > 4 else 0.35
HOLD = 6

info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                           "stream=width,height,r_frame_rate,nb_frames", "-of", "json", src]))["streams"][0]
W, H = int(info["width"]), int(info["height"]); num, den = (int(x) for x in info["r_frame_rate"].split("/")); fps = num / den

torch.set_grad_enabled(False)
model = build_sam3_image_model(); proc = Sam3Processor(model)

def regions(frame_rgb):
    """-> uint8 mask [H, W] of pixels to blur in this frame"""
    img = Image.fromarray(frame_rgb); state = proc.set_image(img); out = np.zeros((H, W), np.uint8)
    for prompt in ("human face", "person"):
        r = proc.set_text_prompt(state=state, prompt=prompt)
        ms, sc = r.get("masks"), r.get("scores")
        if ms is None: continue
        ms = ms.detach().cpu().numpy() if torch.is_tensor(ms) else np.asarray(ms)
        sc = sc.detach().cpu().numpy() if torch.is_tensor(sc) else np.asarray(sc)
        for m, s in zip(ms.reshape(-1, *ms.shape[-2:]), sc.reshape(-1)):
            if s < CONF: continue
            m = (m > 0.5).astype(np.uint8)
            if m.shape != (H, W): m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.nonzero(m)
            if len(ys) < 30: continue
            k = max(9, int(0.25 * max(xs.max() - xs.min(), ys.max() - ys.min())) | 1)
            md = cv2.dilate(m, np.ones((k, k), np.uint8))
            if prompt == "human face":
                out |= md
            else:   # the head: top 30 % of the person's box, inside the dilated mask, widened by half the box width
                y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max(); hy = y0 + int(0.30 * (y1 - y0 + 1))
                bw = x1 - x0 + 1; head = np.zeros_like(m); head[max(0, y0 - k):hy, max(0, x0 - bw // 4):min(W, x1 + bw // 4 + 1)] = 1
                out |= (head & md)
    return out

def frames_of(path):
    d = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE); fsz = W * H * 3
    while True:
        buf = d.stdout.read(fsz)
        if len(buf) < fsz: break
        yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    d.wait()
# pass 1: masks every STRIDE-th frame, kept at quarter resolution (a 40 s 4K clip would not fit in memory at full size)
S4 = (W // 4, H // 4); masks = []; last = np.zeros((S4[1], S4[0]), np.uint8)
for i, fr in enumerate(frames_of(src)):
    if i % STRIDE == 0: last = cv2.resize(regions(fr), S4, interpolation=cv2.INTER_NEAREST)
    masks.append(last)
    if i % 60 == 0: print(f"[blur] {src.split('/')[-1]}: pass 1 frame {i}", flush=True)
n = len(masks)
held = []   # each region held for +/- HOLD frames
for j in range(n):
    m = np.zeros_like(masks[0])
    for k in range(max(0, j - HOLD), min(n, j + HOLD + 1)): m |= masks[k]
    held.append(m)
# pass 2: blur and encode
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{num}/{den}",
                        "-i", "-", "-c:v", "libx264", "-crf", "17", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst],
                       stdin=subprocess.PIPE)
counts = []
for j, fr in enumerate(frames_of(src)):
    m = cv2.resize(held[j], (W, H), interpolation=cv2.INTER_NEAREST) if j < n else np.zeros((H, W), np.uint8)
    out = fr
    if m.any():
        nlab, lab, stats, _ = cv2.connectedComponentsWithStats(m); out = fr.copy()
        for c in range(1, nlab):
            x, y, w, h, a = stats[c]; sig = max(8, int(0.35 * max(w, h)))
            roi = out[y:y + h, x:x + w]; bl = cv2.GaussianBlur(roi, (0, 0), sig); sel = (lab[y:y + h, x:x + w] == c)[..., None]
            out[y:y + h, x:x + w] = np.where(sel, bl, roi)
        counts.append(int(nlab - 1))
    else: counts.append(0)
    enc.stdin.write(np.ascontiguousarray(out).tobytes())
enc.stdin.close(); enc.wait()
json.dump({"src": src, "frames": n, "fps": fps, "stride": STRIDE, "conf": CONF, "hold": HOLD,
           "frames_with_blur": int(sum(1 for c in counts if c)), "regions_per_frame": counts}, open(dst + ".json", "w"))
top = sorted(range(n), key=lambda j: -counts[j])[:6]
if counts and max(counts) > 0:
    cap = subprocess.run(["ffmpeg", "-v", "error", "-i", dst, "-vf", "select='" + "+".join(f"eq(n\\,{j})" for j in sorted(top)) + "',scale=-2:480,tile=6x1",
                          "-frames:v", "1", "-y", dst.rsplit(".", 1)[0] + "_check.jpg"])
print(f"[blur] {dst}: {n} frames, {sum(1 for c in counts if c)} with blurred regions (max {max(counts) if counts else 0} per frame)", flush=True)
