"""Compose the query A/B: legacy vs corrected (glref) seeds, no-query and tree-78 frames, same
keyframe (kf_001703, block 21) through the real render-service path. Reads q_<tag>_*.jpg."""
import glob
from PIL import Image, ImageDraw
tiles = []
for tag in ("legacy", "glref"):
    none = f"/home/paperspace/logs/q_{tag}_none.jpg"; q = sorted(glob.glob(f"/home/paperspace/logs/q_{tag}_tree*.jpg"))
    if not q: continue
    tiles.append((f"{tag}: no query", none)); tiles.append((f"{tag}: query tree 78", q[-1]))
W, H = 640, 360
c = Image.new("RGB", (W * 2 + 10, (H + 24) * (len(tiles) // 2) + 10), (20, 20, 20)); d = ImageDraw.Draw(c)
for i, (label, path) in enumerate(tiles):
    x, y = (i % 2) * (W + 10), (i // 2) * (H + 34)
    c.paste(Image.open(path).convert("RGB").resize((W, H)), (x, y + 24)); d.text((x + 6, y + 6), label, fill=(235, 235, 225))
c.save("/home/paperspace/logs/query_ab.png"); print(f"[ab] wrote query_ab.png ({len(tiles)} tiles)")
