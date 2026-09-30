"""Light one tree through the real stream: query on :8024 (H3DGS + identity side-car), a pose from the walk
near the tree, plain vs lit frame side by side."""
import json, sys, asyncio, urllib.request
import numpy as np
from PIL import Image
import websockets
SURVEY, TREE = sys.argv[1], int(sys.argv[2]); OUT = "/tmp/appcheck"
get = lambda u: json.loads(urllib.request.urlopen(u, timeout=30).read())
reg = get(f"http://127.0.0.1:8011/api/registry?survey={SURVEY}")
pl = next(o for o in reg["objects"] if o["n"] == TREE); print("tree", TREE, "xyz", pl["xyz"], "fruit", pl.get("fruit"))
traj = get("http://127.0.0.1:8031/scene/trajectory")
recs = traj["frames"]   # /scene/trajectory: {"frames": [{"image_idx", "centre", "matrix": 16 floats (row-major c2w)}, ...]}
def c2w_of(r):
    m = np.array(r["matrix"], float); return m.reshape(4, 4) if m.size == 16 else np.vstack([m.reshape(3, 4), [0, 0, 0, 1]])
P = np.array([c2w_of(r)[:3, 3] for r in recs]); t = np.array(pl["xyz"])
d = np.linalg.norm(P - t, axis=1); order = np.argsort(d)
# a pose ~5 m before the closest one along the walk so the tree is in view, not underfoot
i = int(order[0]); j = max(0, i - 25); c2w = c2w_of(recs[j]); print("pose idx", j, "dist to tree", round(float(np.linalg.norm(c2w[:3, 3] - t)), 2), "m")
W, H, fovy = 1280, 720, 1.194
async def go():
    async with websockets.connect("ws://127.0.0.1:8024/ws", max_size=None) as ws:
        for tag, q in (("plain", {"t": "query"}), ("lit", {"t": "query", "tree": TREE})):
            await ws.send(json.dumps(q)); print(tag, await ws.recv())
            for seq in (1, 2):
                await ws.send(json.dumps({"t": "pose", "seq": seq, "c2w": c2w.flatten().tolist(), "w": W, "h": H, "fovy": fovy}))
                data = await ws.recv()
                while isinstance(data, str): data = await ws.recv()
            open(f"{OUT}/light_{SURVEY}_{TREE}_{tag}.jpg", "wb").write(data[20:])
asyncio.run(go())
a = Image.open(f"{OUT}/light_{SURVEY}_{TREE}_plain.jpg"); b = Image.open(f"{OUT}/light_{SURVEY}_{TREE}_lit.jpg")
s = Image.new("RGB", (a.width * 2, a.height)); s.paste(a, (0, 0)); s.paste(b, (a.width, 0)); s.save(f"{OUT}/light_{SURVEY}_{TREE}_strip.jpg", quality=85)
diff = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16)).sum(2); print("changed px", int((diff > 60).sum()), "of", diff.size)
