"""Seam profile: for several block boundaries of 05_13D, render the +/-6 keyframes around the
hand-off with BOTH neighbouring checkpoints (owner hint) via the live render service and score
PSNR against the ground-truth photo. Shows how far edge degradation extends into each block and
whether a PSNR-argmax owner assignment would beat the fixed keyframe-index hand-off."""
import asyncio, json, io, sys, urllib.request
import numpy as np, websockets
from PIL import Image
from pathlib import Path
B = Path("/home/paperspace/data/citrus_all/05_13D_Jackal/prod/tassili/blocks_ns/lio_row100")
KF = Path("/home/paperspace/data/citrus_all/05_13D_Jackal/prod/scratch_sam3")
traj = json.load(urllib.request.urlopen("http://127.0.0.1:8001/scene/trajectory?stride=1"))["frames"]
own = [(f["image_name"], f["block"], f["matrix"]) for f in traj if f.get("block") is not None]
bounds = [i for i in range(1, len(own)) if own[i][1] != own[i-1][1]]
pick = bounds[::max(1, len(bounds)//6)][:6]
W, H = 480, 270
def psnr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); m = ((a-b)**2).mean(); return 99.0 if m == 0 else 10*np.log10(255**2/m)
async def render(ws, c2w, blk):
    for k in range(30):
        await ws.send(json.dumps({"t":"pose","seq":k+1,"c2w":c2w,"w":W,"h":H,"fovy":0.99,"block":int(blk)}))
        d = await asyncio.wait_for(ws.recv(), timeout=120)
        while isinstance(d, str): d = await asyncio.wait_for(ws.recv(), timeout=120)
        if len(d) > 12000: return Image.open(io.BytesIO(d[20:])).convert("RGB")
        await asyncio.sleep(1.2)
    return None
async def main():
    out = []
    for bi in pick:
        A, Bk = own[bi-1][1], own[bi][1]
        for off in range(-6, 6):
            j = bi + off
            if not (0 <= j < len(own)): continue
            name, owner, m = own[j]; gt = Image.open(KF/name).convert("RGB").resize((W, H))
            row = {"boundary": f"{A}->{Bk}", "off": off, "name": name, "owner": owner}
            for tag, blk in (("A", A), ("B", Bk)):
                for attempt in range(3):
                    try:
                        async with websockets.connect("ws://127.0.0.1:8004/ws", max_size=2**24) as ws:
                            im = await render(ws, m, blk); break
                    except Exception: im = None; await asyncio.sleep(5)
                row[tag] = round(psnr(gt, im), 2) if im is not None else None
            out.append(row); print(f"[seam] {row['boundary']:>7} off {off:+d} {name} owner {owner}: PSNR A({A})={row['A']} B({Bk})={row['B']}", flush=True)
    json.dump(out, open("/home/paperspace/logs/boundary_profile.json", "w"))
    # summary: fixed hand-off vs argmax
    fixed = [r["A"] if r["off"] < 0 else r["B"] for r in out if r["A"] and r["B"]]
    best = [max(r["A"], r["B"]) for r in out if r["A"] and r["B"]]
    edge = [r for r in out if r["A"] and r["B"] and abs(r["off"]) <= 1]
    print(f"[seam] frames {len(fixed)}: fixed hand-off mean PSNR {np.mean(fixed):.2f} -> argmax owner {np.mean(best):.2f} (+{np.mean(best)-np.mean(fixed):.2f} dB); owner would switch on {sum(1 for r in out if r['A'] and r['B'] and ((r['off']<0) != (r['A']>=r['B'])))}/{len(fixed)} frames")
    print(f"[seam] at the seam (|off|<=1): mean of the RENDERED owner {np.mean([r['A'] if r['off']<0 else r['B'] for r in edge]):.2f} dB vs frames 5-6 away {np.mean([r['A'] if r['off']<0 else r['B'] for r in out if r['A'] and r['B'] and abs(r['off'])>=5]):.2f} dB")
asyncio.run(main())
