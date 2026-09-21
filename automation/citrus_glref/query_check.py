"""Containment-query check through the render service ws: render block 21's kf_001703 with no
query and with an object-addressed tree query; measure how many pixels the highlight fires and
save both frames. usage: query_check.py <port> <tag>"""
import asyncio, json, sys, urllib.request
import numpy as np, websockets
from PIL import Image
from pathlib import Path
port, tag = sys.argv[1], sys.argv[2]
BD = "/home/paperspace/data/citrus_all/05_13D_Jackal/prod/tassili/blocks_ns/lio_row100/block_021"
fr = {Path(f["file_path"]).name: f for f in json.load(open(f"{BD}/transforms_cv_pre_glfix.json"))["frames"]}
M = np.array(fr["kf_001703.png"]["transform_matrix"]).flatten().tolist()
H = json.load(open("/home/paperspace/data/citrus_all/05_13D_Jackal/prod/bateleur/scene_graph/marker_hierarchy.json"))
cam = np.array(fr["kf_001703.png"]["transform_matrix"])[:3, 3]
objs = H["objects"]; d = [np.linalg.norm(np.array(o["xyz"]) - cam) for o in objs]
tree = int(objs[int(np.argmin(d))]["id"]); print(f"[query] target: hierarchy object {tree} ({min(d):.1f} m from kf_001703 camera), row {objs[int(np.argmin(d))]['row_id']}")
async def frame(ws, q):
    await ws.send(json.dumps(q))
    for k in range(40):
        await ws.send(json.dumps({"t":"pose","seq":k+1,"c2w":M,"w":640,"h":360,"fovy":0.99,"block":21}))
        d = await asyncio.wait_for(ws.recv(), timeout=120)
        while isinstance(d, str): d = await asyncio.wait_for(ws.recv(), timeout=120)
        if len(d) > 15000: return d[20:]
        await asyncio.sleep(1.5)
    return d[20:]
async def main():
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws", max_size=2**24) as ws:
        a = await frame(ws, {"t":"query","q":""})
        b = await frame(ws, {"t":"query","tree":tree})
        await ws.send(json.dumps({"t":"query","q":""}))   # leave the service clean
    open(f"/home/paperspace/logs/q_{tag}_none.jpg","wb").write(a); open(f"/home/paperspace/logs/q_{tag}_tree{tree}.jpg","wb").write(b)
    A = np.asarray(Image.open(f"/home/paperspace/logs/q_{tag}_none.jpg").convert("RGB")).astype(int)
    B = np.asarray(Image.open(f"/home/paperspace/logs/q_{tag}_tree{tree}.jpg").convert("RGB")).astype(int)
    diff = np.abs(A - B).sum(-1) > 60
    print(f"[query:{tag}] tree {tree} on block 21: highlighted pixels {100*diff.mean():.1f}% of frame ({diff.sum():,}); wrote q_{tag}_*.jpg")
asyncio.run(main())
