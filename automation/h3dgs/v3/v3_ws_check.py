#!/usr/bin/env python3
"""Live check of v3_render_service (recipe v3, 2026-10-08), the measurement the serving item of plans/recipe_v3_splatfacto.md asks for:
send recorded keyframes' poses in the wire convention (OpenCV c2w, LIO frame = R_W^T @ the chunk's COLMAP c2w) NAMED (a recorded replay:
the camera's own bilateral grid) and UNNAMED (a free pose: nearest-camera grid blend), receive the JPEG frames, PSNR against the rectified
photo (a named trained view should land near splat_bilateral_score.py's grid-applied number minus JPEG), render / total ms; then queries
(tree:N, row:N, fruit:N, q:word) on the first frame -> lit pixel counts + PNGs for a look. Also 20 sequential poses for the frame rate.
h3dgs env (websockets):  python v3_ws_check.py --port 8046 --proj <h3dgs project> --chunk 1_0 --frames kf_a.png ... [--queries tree:5 row:6] --out <dir>"""
import argparse, asyncio, io, json, math, os, struct, sys, time, urllib.request
import numpy as np, websockets
from PIL import Image
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess")
from read_write_model import read_images_binary, read_cameras_binary, qvec2rotmat
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8046); ap.add_argument("--proj", required=True); ap.add_argument("--chunk", required=True)
ap.add_argument("--frames", nargs="+", required=True); ap.add_argument("--queries", nargs="*", default=[]); ap.add_argument("--out", required=True); ap.add_argument("--rate-n", type=int, default=20)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
meta = json.load(open(f"{a.proj}/export_meta.json")); R_W = np.array(meta.get("world_rotation_to_zup") or meta["world_rotation_lio_to_h3dgs"])
src = f"{a.proj}/camera_calibration/chunks/{a.chunk}/sparse/0"; cam = list(read_cameras_binary(f"{src}/cameras.bin").values())[0]
W, H = int(cam.width), int(cam.height); fx, fy, cx, cy = cam.params[:4]; ims = {v.name: v for v in read_images_binary(f"{src}/images.bin").values()}
TEST = {l.strip() for l in open(f"{src}/test.txt") if l.strip()}
def pose(name, named):
    im = ims[name]; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec
    p = {"t": "pose", "c2w": (R_W.T @ np.linalg.inv(w2c)).flatten().tolist(), "w": W, "h": H, "fovy": 2 * math.atan(H / (2 * fy)), "primx": cx / W, "primy": cy / H, "quality": 90}
    if named: p["image_name"] = name
    return p
def photo(name): return np.asarray(Image.open(f"{a.proj}/camera_calibration/rectified/images/{name}").convert("RGB"), np.float32) / 255
def psnr(x, y): return 10 * math.log10(1 / max(float(((x - y) ** 2).mean()), 1e-10))
print("[v3-check] healthz:", urllib.request.urlopen(f"http://127.0.0.1:{a.port}/healthz", timeout=10).read().decode()[:400], flush=True)
async def main():
    async with websockets.connect(f"ws://127.0.0.1:{a.port}/ws", max_size=2**25) as ws:
        async def frame(p, seq):
            t0 = time.time(); await ws.send(json.dumps({**p, "seq": seq})); raw = await ws.recv(); dt = (time.time() - t0) * 1000
            _, w, h, _, _, rms, tms = struct.unpack("<IHHHHff", raw[:20]); img = np.asarray(Image.open(io.BytesIO(raw[20:])).convert("RGB"), np.float32) / 255
            return img, rms, tms, dt
        seq = 0; rows = []
        for name in a.frames:
            if name not in ims: print(f"[v3-check] {name}: not in the chunk"); continue
            g = photo(name); r = {"frame": name, "split": "held-out" if name in TEST else "trained"}
            for named in (True, False):
                seq += 1; img, rms, tms, dt = await frame(pose(name, named), seq); r["named" if named else "free"] = round(psnr(img, g), 2); r[("named" if named else "free") + "_ms"] = (round(rms, 1), round(tms, 1), round(dt, 1))
                Image.fromarray((img * 255).astype(np.uint8)).save(f"{a.out}/{name[:-4]}_{'named' if named else 'free'}.png")
            rows.append(r); print(f"[v3-check] {name} ({r['split']}): PSNR named {r['named']} / free {r['free']} dB; render/total/round-trip ms named {r['named_ms']} free {r['free_ms']}", flush=True)
        if rows:
            for s in ("trained", "held-out"):
                v = [x for x in rows if x["split"] == s]
                if v: print(f"[v3-check] {s}: {len(v)} frames, named median {np.median([x['named'] for x in v]):.2f} / free median {np.median([x['free'] for x in v]):.2f} dB", flush=True)
        base = None
        for qs in a.queries:
            if not rows: break
            kind, rest = qs.split(":", 1); oid, cut = (rest.split("@") + [""])[:2]
            q = {"t": "query", kind: (oid if kind == "q" else int(oid)), **({"cut": float(cut)} if cut else {})}
            await ws.send(json.dumps(q)); ack = json.loads(await ws.recv())
            seq += 1; img, rms, tms, dt = await frame(pose(a.frames[0], True), seq)
            if base is None:
                await ws.send(json.dumps({"t": "query"})); await ws.recv(); seq += 1; base, _, _, _ = await frame(pose(a.frames[0], True), seq)
                await ws.send(json.dumps(q)); await ws.recv()
            lit = int((np.abs(img - base).max(-1) > 0.08).sum()); Image.fromarray((img * 255).astype(np.uint8)).save(f"{a.out}/{a.frames[0][:-4]}_{kind}{oid}.png")
            print(f"[v3-check] query {qs}: ack ok={ack.get('ok')} word={ack.get('object')} competitors={ack.get('competitors')} reason={ack.get('reason', '')} -> lit px {lit} ({100 * lit / (W * H):.1f}% of frame), render {rms:.1f} ms total {tms:.1f} ms", flush=True)
        await ws.send(json.dumps({"t": "query"})); await ws.recv()
        if rows and a.rate_n:
            t = []; p = pose(a.frames[0], False)
            for i in range(a.rate_n):
                seq += 1; _, rms, tms, dt = await frame(p, seq); t.append((rms, tms, dt))
            t = np.array(t); print(f"[v3-check] {a.rate_n} sequential free poses: render {np.median(t[:, 0]):.1f} ms, total {np.median(t[:, 1]):.1f} ms, round trip {np.median(t[:, 2]):.1f} ms median -> {1000 / np.median(t[:, 2]):.1f} fps", flush=True)
        json.dump(rows, open(f"{a.out}/psnr.json", "w"), indent=1)
asyncio.run(main())
