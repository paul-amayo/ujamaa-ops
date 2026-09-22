"""Seam profile v2: true checkpoint FOV (fovy from transforms.json), +/-20 keyframes (4 m) around three
block hand-offs, each keyframe rendered by BOTH neighbours through the live service (owner hint path,
exactly what the browser replay sends).  Also: mean-RGB jump between the two checkpoints on the seam
frame (appearance mismatch), and a GT | A | B strip for the demo-hero seam 21->22."""
import asyncio, json, io, math, urllib.request
import numpy as np, websockets
from PIL import Image, ImageDraw
from pathlib import Path
KF = Path("/home/paperspace/data/citrus_all/05_13D_Jackal/prod/scratch_sam3")
BL = Path("/home/paperspace/data/citrus_all/05_13D_Jackal/prod/tassili/blocks_ns/lio_row100")
t21 = json.load(open(BL/"block_021/transforms.json")); FOVY = 2*math.atan(t21["h"]/2/t21["fl_y"])
traj = json.load(urllib.request.urlopen("http://127.0.0.1:8001/scene/trajectory?stride=1"))["frames"]
own = [(f["image_name"], f["block"], f["matrix"]) for f in traj if f.get("block") is not None]
bounds = {f"{own[i-1][1]}->{own[i][1]}": i for i in range(1, len(own)) if own[i][1] != own[i-1][1]}
import sys
SEAMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["21->22", "35->36", "7->8"]; SPAN = 20; W, H = 640, 360
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
def psnr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); m = ((a-b)**2).mean(); return 99.0 if m == 0 else 10*np.log10(255**2/m)
async def render(ws, c2w, blk):
    for k in range(40):
        await ws.send(json.dumps({"t":"pose","seq":k+1,"c2w":c2w,"w":W,"h":H,"fovy":FOVY,"block":int(blk)}))
        d = await asyncio.wait_for(ws.recv(), timeout=120)
        while isinstance(d, str): d = await asyncio.wait_for(ws.recv(), timeout=120)
        if len(d) > 15000: return Image.open(io.BytesIO(d[20:])).convert("RGB")
        await asyncio.sleep(1.0)
    return None
async def main():
    out, strip = [], {}
    async with websockets.connect("ws://127.0.0.1:8004/ws", max_size=2**24) as ws:
        for sname in SEAMS:
            bi = bounds[sname]; A, Bk = [int(x) for x in sname.split("->")]
            for off in range(-SPAN, SPAN):
                j = bi + off
                if not (0 <= j < len(own)) or own[j][1] not in (A, Bk): continue
                name, owner, m = own[j]; gt = Image.open(KF/name).convert("RGB").resize((W, H))
                imA = await render(ws, m, A); imB = await render(ws, m, Bk)
                row = {"seam": sname, "off": off, "name": name, "owner": owner,
                       "A": round(psnr(gt, imA), 2) if imA else None, "B": round(psnr(gt, imB), 2) if imB else None,
                       "meanA": np.asarray(imA, float).mean((0,1)).round(1).tolist() if imA else None,
                       "meanB": np.asarray(imB, float).mean((0,1)).round(1).tolist() if imB else None,
                       "meanGT": np.asarray(gt, float).mean((0,1)).round(1).tolist()}
                out.append(row); print(f"[seam2] {sname:>7} off {off:+3d} {name} owner {owner}: A={row['A']} B={row['B']}", flush=True)
                if sname == SEAMS[0] and off in (-4, -1, 0, 2, 5, 10) and imA and imB: strip[off] = (gt, imA, imB, row)
    json.dump(out, open(f"/home/paperspace/logs/seam_profile2{TAG}.json", "w"))
    # strip
    offs = sorted(strip); pad = 4; th = 22; sw, sh = 426, 240
    canvas = Image.new("RGB", (3*(sw+pad)+pad, len(offs)*(sh+th+pad)+pad+th), (20,20,20)); d = ImageDraw.Draw(canvas)
    for c, lab in enumerate(["ground-truth photo", "rendered by block 21 (before seam)", "rendered by block 22 (after seam)"]): d.text((pad + c*(sw+pad), 4), lab, fill=(255,255,255))
    for r, off in enumerate(offs):
        gt, imA, imB, row = strip[off]; y = th + pad + r*(sh+th+pad)
        d.text((pad, y), f"keyframe {row['name']}  off {off:+d} ({abs(off)*0.2:.1f} m {'before' if off<0 else 'after'} the hand-off)  owner block {row['owner']}   PSNR: block21 {row['A']} dB   block22 {row['B']} dB", fill=(255,230,120))
        for c, im in enumerate((gt, imA, imB)): canvas.paste(im.resize((sw, sh)), (pad + c*(sw+pad), y+th))
    canvas.save(f"/home/paperspace/logs/seam_strip{TAG}.png")
    # summary
    ok = [r for r in out if r["A"] and r["B"]]
    print("\n[seam2] per-offset mean over seams (owner*):")
    for off in range(-SPAN, SPAN, 1):
        rs = [r for r in ok if r["off"] == off]
        if not rs: continue
        a = np.mean([r["A"] for r in rs]); b = np.mean([r["B"] for r in rs]); nb = sum(1 for r in rs if (r["B"] > r["A"]) == (off < 0))
        print(f"[seam2]  off {off:+3d} ({off*0.2:+.1f} m) n={len(rs)} A={a:5.2f}{'*' if off<0 else ' '} B={b:5.2f}{'*' if off>=0 else ' '} neighbour-better {nb}/{len(rs)}")
    for r in ok:
        if r["off"] == 0:
            dA = np.array(r["meanA"]) - np.array(r["meanGT"]); dB = np.array(r["meanB"]) - np.array(r["meanGT"])
            print(f"[seam2] seam {r['seam']} frame {r['name']}: mean RGB minus GT  A={dA.round(1).tolist()}  B={dB.round(1).tolist()}  A-B={(np.array(r['meanA'])-np.array(r['meanB'])).round(1).tolist()}")
asyncio.run(main())
