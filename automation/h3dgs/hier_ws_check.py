"""Protocol + pose check of the hierarchy backend: send recorded keyframe poses (exactly what the browser
replay sends) to ws://127.0.0.1:8006/ws, decode the JPEG, score PSNR against the photo, save a strip."""
import asyncio, io, json, struct, sys, time, urllib.request
import numpy as np, websockets
from PIL import Image, ImageDraw
KF = "/home/paperspace/data/citrus_all/05_13D_Jackal/prod/scratch_sam3"
names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["kf_001411.png", "kf_001947.png", "kf_002360.png", "kf_000621.png"]
traj = {f["image_name"]: f["matrix"] for f in json.load(urllib.request.urlopen("http://127.0.0.1:8001/scene/trajectory?stride=1"))["frames"]}
# the poses the blocks/hierarchy were TRAINED on (block transforms, OpenGL c2w -> OpenCV wire convention)
from pathlib import Path
trained = {}
for tj in Path("/home/paperspace/data/citrus_all/05_13D_Jackal/prod/tassili/blocks_ns/lio_row100").glob("block_*/transforms.json"):
    for f in json.load(open(tj))["frames"]:
        trained[Path(f["file_path"]).name] = (np.array(f["transform_matrix"]) @ np.diag([1., -1, -1, 1])).flatten().tolist()
# the poses the HIERARCHY was trained on: the chunk's bundle-adjusted COLMAP poses (z-up frame) -> wire (LIO) frame
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess")
from read_write_model import read_images_binary, qvec2rotmat
PROJ = Path("/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs")
R_W = np.array(json.load(open(PROJ / "export_meta.json"))["world_rotation_lio_to_h3dgs"])
chunkpose = {}
for cdir in sorted((PROJ / "camera_calibration/chunks").glob("*_*")):
    for im in read_images_binary(str(cdir / "sparse/0/images.bin")).values():
        w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec
        chunkpose.setdefault(im.name, (R_W.T @ np.linalg.inv(w2c)).flatten().tolist())
W, H = 1280, 720
def psnr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); return 10 * np.log10(255 ** 2 / max(((a - b) ** 2).mean(), 1e-9))
async def main():
    rows = []
    async with websockets.connect("ws://127.0.0.1:8006/ws", max_size=2 ** 25) as ws:
        for tau, src in ((0.0, "wire"), (0.0, "trained"), (0.0, "chunk")):
            for k, n in enumerate(names):
                t0 = time.time()
                await ws.send(json.dumps({"t": "pose", "seq": k + 1, "c2w": {"wire": traj, "trained": trained, "chunk": chunkpose}[src][n], "w": W, "h": H, "fovy": 1.1939, "quality": 90, "tau": tau, "primx": 647.198/1280, "primy": 354.643/720}))
                d = await asyncio.wait_for(ws.recv(), timeout=120)
                while isinstance(d, str): d = await asyncio.wait_for(ws.recv(), timeout=120)
                seq, w, h, nb, fl, render_ms, total_ms = struct.unpack("<IHHHHff", d[:20])
                im = Image.open(io.BytesIO(d[20:])).convert("RGB"); gt = Image.open(f"{KF}/{n}").convert("RGB")
                rows.append((tau, n, psnr(im, gt), render_ms, total_ms, (time.time() - t0) * 1000, len(d) / 1024, im, src))
                print(f"[hier-ws] tau {tau:.0f} {src:>7} pose {n}: PSNR {rows[-1][2]:.2f} dB, render {render_ms:.0f} ms, server total {total_ms:.0f} ms, round trip {rows[-1][5]:.0f} ms, {rows[-1][6]:.0f} kB", flush=True)
    pad, th, sw, sh = 4, 22, 480, 270
    r0 = [r for r in rows if r[0] == 0.0 and r[8] == "chunk"]
    canvas = Image.new("RGB", (2 * (sw + pad) + pad, len(r0) * (sh + th + pad) + pad + th), (20, 20, 20)); d = ImageDraw.Draw(canvas)
    for c, lab in enumerate(["ground-truth photo", "hierarchy backend :8006 (tau 0, chunk-BA pose), streamed JPEG"]): d.text((pad + c * (sw + pad), 4), lab, fill=(255, 255, 255))
    for i, (tau, n, p, rm, tm, rt, kb, im, src) in enumerate(r0):
        y = th + pad + i * (sh + th + pad)
        d.text((pad, y), f"{n}  PSNR {p:.2f} dB  render {rm:.0f} ms  round trip {rt:.0f} ms", fill=(255, 230, 120))
        canvas.paste(Image.open(f"{KF}/{n}").convert("RGB").resize((sw, sh)), (pad, y + th)); canvas.paste(im.resize((sw, sh)), (pad + sw + pad, y + th))
    canvas.save("/home/paperspace/logs/hier_ws_check.png"); print("[hier-ws] DONE")
asyncio.run(main())
