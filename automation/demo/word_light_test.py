"""Light a QUERY (any {"tree"|"row"|"fruit"|"q": ...} JSON) through the live stream at a walk pose, no registry needed:
plain vs lit frame, changed pixels. usage: word_light_test.py <survey> '<query json>' [pose_idx]   (nerf_new env)"""
import json, sys, asyncio, urllib.request
import numpy as np
from PIL import Image
import websockets
SURVEY, Q = sys.argv[1], json.loads(sys.argv[2]); IDX = int(sys.argv[3]) if len(sys.argv) > 3 else None; OUT = "/tmp/appcheck"
get = lambda u: json.loads(urllib.request.urlopen(u, timeout=30).read())
recs = get("http://127.0.0.1:8031/scene/trajectory?stride=1")["frames"]
def c2w_of(r):
    m = np.array(r["matrix"], float); return m.reshape(4, 4) if m.size == 16 else np.vstack([m.reshape(3, 4), [0, 0, 0, 1]])
j = IDX if IDX is not None else len(recs) // 2; c2w = c2w_of(recs[j]); print("pose idx", j, recs[j].get("image_name"))
W, H, fovy = 720, 1280, 2 * np.arctan(960 / 1560)   # portrait phone, trained-camera fovy
async def go():
    async with websockets.connect("ws://127.0.0.1:8024/ws", max_size=None) as ws:
        for tag, q in (("plain", {"t": "query"}), ("lit", {"t": "query", **Q})):
            await ws.send(json.dumps(q)); print(tag, (await ws.recv())[:200])
            for seq in (1, 2):
                await ws.send(json.dumps({"t": "pose", "seq": seq, "c2w": c2w.flatten().tolist(), "w": W, "h": H, "fovy": float(fovy), "image_name": recs[j].get("image_name")}))
                data = await ws.recv()
                while isinstance(data, str): data = await ws.recv()
            open(f"{OUT}/wlight_{SURVEY}_{tag}.jpg", "wb").write(data[20:])
asyncio.run(go())
a = Image.open(f"{OUT}/wlight_{SURVEY}_plain.jpg"); b = Image.open(f"{OUT}/wlight_{SURVEY}_lit.jpg")
s = Image.new("RGB", (a.width * 2, a.height)); s.paste(a, (0, 0)); s.paste(b, (a.width, 0)); s.save(f"{OUT}/wlight_{SURVEY}_strip.jpg", quality=85)
diff = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16)).sum(2); print("changed px", int((diff > 60).sum()), "of", diff.size)
