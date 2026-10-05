"""Offline capture of a Tassili walk for the demo video (1 Oct): the survey must be staged (POST /api/stage); frames
are rendered through the live stream (:8024, H3DGS + identity side-car) at every Nth trajectory record, so the clip
is exactly what the stage shows, at full rate rather than the headless browser's 6-11 fps.

  usage: capture_walk.py <survey> <out_dir> [--stride 2] [--from 0] [--to -1] [--w 1280 --h 720 --fovy 1.194]
                         [--light tree:90@120 ...]   light tree 90 from record 120 onward (fruit:10001@40 for fruit)
                         [--fps 24]                  assemble <out_dir>.mp4 with ffmpeg
Portrait surveys: pass --w 720 --h 1280."""
import argparse, json, asyncio, subprocess, time, urllib.request
from pathlib import Path
import numpy as np, websockets
ap = argparse.ArgumentParser(); ap.add_argument("survey"); ap.add_argument("out"); ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--from", dest="i0", type=int, default=0); ap.add_argument("--to", dest="i1", type=int, default=-1)
ap.add_argument("--w", type=int, default=1280); ap.add_argument("--h", type=int, default=720); ap.add_argument("--fovy", type=float, default=1.194)
ap.add_argument("--light", action="append", default=[]); ap.add_argument("--fps", type=int, default=24); ap.add_argument("--api", default="http://127.0.0.1:8011")
ap.add_argument("--light-near", action="append", default=[], help="tree:5[:40] — light tree 5 from 40 records before the walk's nearest point to it")
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
get = lambda u: json.loads(urllib.request.urlopen(u, timeout=60).read())
st = get(f"{a.api}/api/stage"); assert st.get("survey") == a.survey and st.get("state") == "ready", f"stage {a.survey} first: {st}"
fr = get("http://127.0.0.1:8031/scene/trajectory?stride=1")["frames"]; i1 = len(fr) if a.i1 < 0 else a.i1   # the endpoint strides 5x by default
recs = fr[a.i0:i1:a.stride]; print(f"{len(recs)} frames from {len(fr)} records (stride {a.stride})")
lights = []   # (record index, query dict)
for spec in a.light:
    what, at = spec.split("@"); kind, oid = what.split(":"); lights.append((int(at), {"t": "query", kind: int(oid)}))
for spec in a.light_near:   # record chosen from the registry: the walk's closest approach to the plant, minus a lead-in
    parts = spec.split(":"); kind, oid, lead = parts[0], int(parts[1]), int(parts[2]) if len(parts) > 2 else 40
    pl = next(o for o in get(f"{a.api}/api/registry?survey={a.survey}")["objects"] if o["n"] == oid)
    P = np.array([np.array(r["matrix"], float).reshape(4, 4)[:3, 3] if len(r["matrix"]) == 16 else np.array(r["matrix"], float).reshape(3, 4)[:, 3] for r in fr])
    near = int(np.argmin(np.linalg.norm(P - np.array(pl["xyz"]), axis=1))); at = max(a.i0, near - lead)
    lights.append((at, {"t": "query", kind: oid})); print(f"light {kind} {oid}: nearest record {near}, lit from {at}")
lights.sort()
def c2w_of(r):
    m = np.array(r["matrix"], float); return m.reshape(4, 4) if m.size == 16 else np.vstack([m.reshape(3, 4), [0, 0, 0, 1]])
async def go():
    async with websockets.connect("ws://127.0.0.1:8024/ws", max_size=None) as ws:
        await ws.send(json.dumps({"t": "query"})); await ws.recv()
        li = 0; t0 = time.time()
        for n, r in enumerate(recs):
            idx = a.i0 + n * a.stride
            while li < len(lights) and lights[li][0] <= idx:
                await ws.send(json.dumps(lights[li][1])); ack = await ws.recv(); print(f"  record {idx}: {lights[li][1]} -> {ack[:80]}"); li += 1
                await asyncio.sleep(2.0)   # the side-car applies a new query on its next frame; give the block a beat
            await ws.send(json.dumps({"t": "pose", "seq": n, "c2w": c2w_of(r).flatten().tolist(), "w": a.w, "h": a.h, "fovy": a.fovy, "quality": 92,
                                      "image_name": r.get("image_name")}))   # that view's own trained exposure (HIER_EXPOSURE=nearest exact match)
            data = await ws.recv()
            while isinstance(data, str): data = await ws.recv()
            (out / f"f_{n:05d}.jpg").write_bytes(data[20:])
            if n % 50 == 0: print(f"  {n}/{len(recs)} frames, {(time.time() - t0):.0f} s", flush=True)
asyncio.run(go())
mp4 = str(out) + ".mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(out / "f_%05d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", mp4], check=True)
print("wrote", mp4)
