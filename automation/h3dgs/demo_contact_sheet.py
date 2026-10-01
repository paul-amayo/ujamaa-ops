"""Contact sheet for an 'ask the orchard' reel: one frame per script segment (the last hold frame for answered questions,
the middle frame otherwise), labelled, beside the mp4 as contact_sheet.png (dashboard session, 2026-10-01: "keep each reel's
contact_sheet.png beside its mp4"). Output frame index = path index + the holds inserted before it (sidecar_demo_overlay.py
writes `hold` extra frames right after each answered segment's hold_at frame).
  python demo_contact_sheet.py <demo dir with demo_script.json + frames/>  [--cols 3]"""
import argparse, json
from pathlib import Path
import cv2, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('demo_dir'); ap.add_argument('--cols', type=int, default=3); a = ap.parse_args()
D = Path(a.demo_dir); frames = sorted((D / 'frames').glob('*.png')); scr = json.load(open(D / 'demo_script.json'))
segs = scr['segments']; holds = [(s['hold_at'], s.get('hold', 0)) for s in segs if s.get('answer') and s.get('hold_at', -1) >= 0]
def out_index(p, after_hold=False):
    k = p + sum(h for at, h in holds if at < p)
    if after_hold: k += sum(h for at, h in holds if at == p)
    return min(k, len(frames) - 1)
picks = []
for s in segs:
    if s.get('answer') and s.get('hold_at', -1) >= 0: picks.append((out_index(s['hold_at'], True), f"{s['mode']}: {s['question'][:38]} -> {s['answer'][:30]}"))
    else: picks.append((out_index((s['lo'] + s['hi']) // 2), f"{s['mode']}: {s['question'][:48] or '(plain)'}"))
tiles = []
for i, lab in picks:
    im = cv2.imread(str(frames[i])); cv2.putText(im, f'#{i} {lab}', (10, im.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA); cv2.putText(im, f'#{i} {lab}', (10, im.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA); tiles.append(im)
h, w = tiles[0].shape[:2]; rows = (len(tiles) + a.cols - 1) // a.cols; sheet = np.zeros((rows * (h + 8), a.cols * (w + 8), 3), np.uint8)
for k, t in enumerate(tiles): r, c = divmod(k, a.cols); sheet[r * (h + 8):r * (h + 8) + h, c * (w + 8):c * (w + 8) + w] = t
cv2.imwrite(str(D / 'contact_sheet.png'), sheet); print(f'[sheet] {D.name}: {len(tiles)} segments of {len(frames)} frames -> {D / "contact_sheet.png"} ({sheet.shape[1]}x{sheet.shape[0]})', flush=True)
