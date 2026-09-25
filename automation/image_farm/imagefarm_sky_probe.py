"""Fetch the same view twice from the :8004 render service — sky composite blue vs black — and measure how much of the
frame the model leaves uncovered (where the two differ). Saves a side-by-side. Run with the nerf_new pixi python.
  python imagefarm_sky_probe.py [trajectory index] [out.png]"""
import asyncio, json, struct, sys, io, urllib.request, numpy as np
from PIL import Image
import websockets
IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 90; OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/paperspace/logs/imagefarm_sky_probe.png"
traj = json.load(urllib.request.urlopen("http://127.0.0.1:8001/scene/trajectory?stride=1"))["frames"]
fr = traj[min(IDX, len(traj) - 1)]; print("view", fr["image_name"], "block", fr.get("block"))
async def fetch(sky):
    async with websockets.connect("ws://127.0.0.1:8004/ws", max_size=None) as ws:
        await ws.send(json.dumps({"t": "pose", "seq": 1, "c2w": fr["matrix"], "w": int(__import__("os").environ.get("PW", 1080)), "h": int(__import__("os").environ.get("PH", 1920)), "fovy": float(__import__("os").environ.get("PF", 1.10)), "quality": 95, "sky": sky, "block": fr.get("block", 0)}))
        while True:
            m = await ws.recv()
            if isinstance(m, (bytes, bytearray)):
                seq, w, h, nb, flags, rms, tms = struct.unpack("<IHHHHff", m[:20]); return Image.open(io.BytesIO(m[20:])).convert("RGB"), rms, nb
blue, rms, nb = asyncio.run(fetch([135, 178, 235])); black, _, _ = asyncio.run(fetch([0, 0, 0]))
a, b = np.asarray(blue, np.int16), np.asarray(black, np.int16); d = np.abs(a - b).max(-1)
unc = d > 12; print(f"render {rms:.0f} ms, blocks {nb}; pixels where the sky colour shows through (alpha < 1): {unc.mean()*100:.1f} % of the frame; strongly (alpha < 0.5): {(d > 60).mean()*100:.1f} %")
mask = Image.fromarray((unc * 255).astype(np.uint8)).convert("RGB")
sheet = Image.new("RGB", (blue.width * 3 + 20, blue.height), (30, 30, 30)); sheet.paste(blue, (0, 0)); sheet.paste(black, (blue.width + 10, 0)); sheet.paste(mask, (2 * blue.width + 20, 0))
sheet = sheet.resize((sheet.width // 2, sheet.height // 2)); sheet.save(OUT); print("wrote", OUT)
