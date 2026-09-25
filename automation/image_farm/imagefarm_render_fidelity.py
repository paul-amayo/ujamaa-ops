"""Is the viewer's renderer faithful? Fetch frames from the :8004 render service at recorded keyframe poses with the
TRAINING camera geometry (portrait 1080x1920, fovy from transforms.json), compare with the photos (PSNR), and save
photo | service render pairs. Run with the nerf_new pixi python.
  python imagefarm_render_fidelity.py <segment_dir> [frame indices...]"""
import asyncio, json, struct, sys, io, math, urllib.request, numpy as np
from pathlib import Path
from PIL import Image
import websockets
SEG = Path(sys.argv[1]); IDX = [int(x) for x in sys.argv[2:]] or [40, 95, 150]
t = json.load(open(SEG / "transforms.json")); fovy = 2 * math.atan(t["h"] / (2 * t["fl_y"])); W, H = t["w"], t["h"]
traj = {f["image_name"]: f for f in json.load(urllib.request.urlopen("http://127.0.0.1:8001/scene/trajectory?stride=1"))["frames"]}
async def fetch(fr):
    async with websockets.connect("ws://127.0.0.1:8004/ws", max_size=None) as ws:
        await ws.send(json.dumps({"t": "pose", "seq": 1, "c2w": fr["matrix"], "w": W, "h": H, "fovy": fovy, "quality": 95, "sky": [0, 0, 0], "block": fr.get("block", 0)}))
        while True:
            m = await ws.recv()
            if isinstance(m, (bytes, bytearray)):
                seq, w, h, nb, flags, rms, tms = struct.unpack("<IHHHHff", m[:20]); return Image.open(io.BytesIO(m[20:])).convert("RGB"), rms
def psnr(a, b):
    d = (np.asarray(a, np.float32) - np.asarray(b, np.float32)) / 255.0; return 10 * math.log10(1.0 / max(float((d ** 2).mean()), 1e-12))
tiles = []
for i in IDX:
    name = f"image_{i}.png"; fr = traj[name]; photo = Image.open(SEG / "images" / name).convert("RGB")
    ren, rms = asyncio.run(fetch(fr)); held = (i % 10 == 0)
    print(f"{name}: {'HELD-OUT' if held else 'training'} view, service render {rms:.0f} ms, PSNR vs photo {psnr(ren, photo):.2f} dB")
    tiles.append(np.concatenate([np.asarray(photo), np.full((H, 6, 3), 255, np.uint8), np.asarray(ren.resize((W, H)))], 1))
sheet = np.concatenate([np.concatenate([tl, np.full((6, tl.shape[1], 3), 255, np.uint8)], 0) for tl in tiles], 0)
out = f"/home/paperspace/logs/{SEG.name}_render_fidelity.jpg"; Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2)).save(out, quality=88); print("wrote", out, "(photo | service render, half size)")
EOF_MARK = None
