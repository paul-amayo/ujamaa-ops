#!/usr/bin/env python3
"""Live check of hier_render_service with NATIVE identity (UJAMAA, 2026-10-03): send a recorded keyframe's pose in
the wire convention (OpenCV c2w, LIO world = R_W^T @ the chunk's COLMAP c2w) plus queries, save each returned frame.
h3dgs env: python native_ws_check.py --port 8036 --proj <h3dgs project> --chunk 1_0 --frame kf_002411.png
             --queries tree:36 row:3@0.88 none --out <dir>"""
import argparse, asyncio, json, math, os, struct, sys, time, urllib.request
import numpy as np, websockets
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess")
from read_write_model import read_images_binary, read_cameras_binary, qvec2rotmat
ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, default=8036); ap.add_argument('--proj', required=True); ap.add_argument('--chunk', required=True)
ap.add_argument('--frame', required=True); ap.add_argument('--queries', nargs='+', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
_meta = json.load(open(f"{a.proj}/export_meta.json")); R_W = np.array(_meta.get("world_rotation_to_zup") or _meta["world_rotation_lio_to_h3dgs"])   # both export-meta key names, as hier_render_service
src = f"{a.proj}/camera_calibration/chunks/{a.chunk}/sparse/0"
cam = list(read_cameras_binary(f"{src}/cameras.bin").values())[0]; W, H = int(cam.width), int(cam.height); fx, fy, cx, cy = cam.params[:4]
im = next(v for v in read_images_binary(f"{src}/images.bin").values() if v.name == a.frame)
w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec
wire = (R_W.T @ np.linalg.inv(w2c)).flatten().tolist()
pose = {"t": "pose", "c2w": wire, "w": W, "h": H, "fovy": 2 * math.atan(H / (2 * fy)), "primx": cx / W, "primy": cy / H, "tau": 3, "quality": 90}
async def main():
    async with websockets.connect(f"ws://127.0.0.1:{a.port}/ws", max_size=2**25) as ws:
        for k, qs in enumerate(a.queries):
            if qs == "none": q = {"t": "query"}
            else:
                kind, rest = qs.split(":"); oid, cut = (rest.split("@") + [""])[:2]
                q = {"t": "query", kind: int(oid), **({"cut": float(cut)} if cut else {})}
            await ws.send(json.dumps(q)); ack = json.loads(await ws.recv())
            t0 = time.time(); await ws.send(json.dumps({**pose, "seq": k + 1})); raw = await ws.recv(); dt = (time.time() - t0) * 1000
            seq, w, h, _, _, rms, tms = struct.unpack("<IHHHHff", raw[:20])
            fn = os.path.join(a.out, f"ws_{a.frame[:-4]}_{qs.replace(':', '').replace('@', '_cut')}.jpg"); open(fn, "wb").write(raw[20:])
            print(f"[ws] {qs}: ack ok={ack.get('ok')} native={ack.get('native')} reason={ack.get('reason', '')} | frame seq {seq} {w}x{h}, render {rms:.0f} ms, total {tms:.0f} ms, round trip {dt:.0f} ms -> {fn}", flush=True)
    print("[ws] healthz:", json.loads(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/healthz").read()))
asyncio.run(main())
