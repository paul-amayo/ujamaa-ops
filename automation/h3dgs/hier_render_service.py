"""Hierarchical-3DGS render backend for the Tassili viewer.

Same wire protocol as render_service.py (pose JSON in, 20-byte header + JPEG out), but the scene is ONE
merged H3DGS hierarchy for the whole survey instead of one nerfstudio checkpoint per block — no owner
hint, no hand-off, level of detail by `tau`. Runs in the `h3dgs` conda env (torch 2.4 + the hierarchy
rasterizer), so it is a separate process on its own port; the client selects it with
`?stream_url=ws://<host>:8006/ws`.

Pose: the wire c2w is OpenCV camera-to-world in the survey (LIO) world; the hierarchy was trained in the
export's z-up frame, so c2w_h = R_W @ c2w (R_W from export_meta.json). Cells whose chunk is not merged yet
are filled with the coarse scaffold's gaussians (rendered like the skybox: always, as leaves).

Env: HIER (merged .hier), SCAFFOLD (scaffold/point_cloud/iteration_30000), META (export_meta.json),
CHUNKS_DIR + MERGED_CHUNKS (names, for the scaffold fill), TAU (default 3), PORT (8006), HIER_FILL (1).
Containment queries are not available on this backend: {"t":"query"} gets a query_ack with ok:false.
"""
from __future__ import annotations
import asyncio, json, math, os, struct, sys, threading, time
import numpy as np, torch, cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians")
from argparse import ArgumentParser
from arguments import PipelineParams
from scene import GaussianModel
from gaussian_renderer import render_post
from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

HIER = os.environ["HIER"]; SCAFFOLD = os.environ["SCAFFOLD"]; META = json.load(open(os.environ["META"]))
CHUNKS_DIR = os.environ.get("CHUNKS_DIR", ""); MERGED = os.environ.get("MERGED_CHUNKS", "").split()
TAU = float(os.environ.get("TAU", "3")); PORT = int(os.environ.get("PORT", "8006")); FILL = os.environ.get("HIER_FILL", "1") == "1"
COMPACT = os.environ.get("HIER_COMPACT", "1") == "1"   # fp16 attributes, subset-only gathers: ~4x less VRAM than render_post
R_W = np.array(META.get("world_rotation_to_zup") or META["world_rotation_lio_to_h3dgs"], dtype=np.float64)

CH = None
if COMPACT:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hier_compact import CompactHierarchy
    fill_boxes = []
    if FILL and CHUNKS_DIR and MERGED:
        fill_boxes = [(np.loadtxt(f"{CHUNKS_DIR}/{c}/center.txt"), np.loadtxt(f"{CHUNKS_DIR}/{c}/extent.txt")) for c in MERGED]
    CH = CompactHierarchy(HIER, SCAFFOLD, fill_boxes or None)
    N_HIER, n_fill, N = CH.n_hier, CH.fill, CH.N
    print(f"[hier] compact: {N_HIER} hierarchy nodes, skybox {CH.skybox}, scaffold fill {n_fill}, resident {CH.gpu_gib():.2f} GiB, tau {TAU}", flush=True)
else:
    _pp_parser = ArgumentParser(); pipe = PipelineParams(_pp_parser).extract(_pp_parser.parse_args([]))
    gaussians = GaussianModel(3); gaussians.active_sh_degree = 3
    gaussians.create_from_hier(HIER, 1.0, SCAFFOLD)
    N_HIER = gaussians._xyz.size(0) - gaussians.skybox_points
    n_fill = 0
    if FILL and CHUNKS_DIR and MERGED:
        # scaffold scene gaussians (after its skybox) that lie outside every merged chunk's cell -> appended
        # after the skybox and counted as "skybox points" so render_post always draws them as leaves.
        xyz, f_dc, f_ex, op, sc, ro = gaussians.load_ply_file(SCAFFOLD + "/point_cloud.ply", 1)
        with open(SCAFFOLD + "/pc_info.txt") as f: nsky = int(f.readline())
        xyz = torch.from_numpy(xyz).float()[nsky:]; keep = torch.ones(xyz.size(0), dtype=torch.bool)
        for c in MERGED:
            cc = np.loadtxt(f"{CHUNKS_DIR}/{c}/center.txt"); ee = np.loadtxt(f"{CHUNKS_DIR}/{c}/extent.txt")
            inside = (torch.abs(xyz[:, 0] - cc[0]) <= ee[0] / 2) & (torch.abs(xyz[:, 1] - cc[1]) <= ee[1] / 2)
            keep &= ~inside
        sel = keep.nonzero().flatten() + nsky
        if sel.numel():
            dc = torch.from_numpy(f_dc).permute(0, 2, 1).float()[sel]; ex = torch.from_numpy(f_ex).permute(0, 2, 1).float()[sel]
            filler = torch.zeros(sel.numel(), 15, 3); filler[:, :3, :] = ex
            with torch.no_grad():
                gaussians._xyz = torch.nn.Parameter(torch.cat((gaussians._xyz, xyz[sel - nsky].cuda())))
                gaussians._features_dc = torch.nn.Parameter(torch.cat((gaussians._features_dc, dc.cuda())))
                gaussians._features_rest = torch.nn.Parameter(torch.cat((gaussians._features_rest, filler.cuda())))
                gaussians._opacity = torch.nn.Parameter(torch.cat((gaussians._opacity, torch.sigmoid(torch.from_numpy(op).float()[sel]).cuda())))
                gaussians._scaling = torch.nn.Parameter(torch.cat((gaussians._scaling, torch.from_numpy(sc).float()[sel].cuda())))
                gaussians._rotation = torch.nn.Parameter(torch.cat((gaussians._rotation, torch.from_numpy(ro).float()[sel].cuda())))
            n_fill = int(sel.numel()); gaussians.skybox_points += n_fill
    N = gaussians._xyz.size(0)
    ri, pi_, nri = (torch.zeros(N).int().cuda() for _ in range(3)); iw = torch.zeros(N).float().cuda(); ns = torch.zeros(N).int().cuda()
    print(f"[hier] loaded {HIER}: {N_HIER} hierarchy nodes, skybox {gaussians.skybox_points - n_fill}, scaffold fill {n_fill}, tau {TAU}", flush=True)
GPU_LOCK = threading.Lock()
app = FastAPI()


class Cam:
    def __init__(self, c2w_h: np.ndarray, W: int, H: int, fovy: float, primx: float = 0.5, primy: float = 0.5):
        w2c = np.linalg.inv(c2w_h)
        R, T = c2w_h[:3, :3], w2c[:3, 3]
        self.image_width, self.image_height = W, H
        self.FoVy = fovy; self.FoVx = 2 * math.atan(math.tan(fovy / 2) * W / H)
        self.image_name = "live"
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).float().transpose(0, 1).cuda()
        self.projection_matrix = torch.tensor(getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy, primx=primx, primy=primy)).float().transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


@torch.no_grad()
def _render(msg: dict) -> bytes:
    t0 = time.time()
    c2w = np.asarray(msg["c2w"], dtype=np.float64).reshape(4, 4)
    W, H = int(msg.get("w", 1280)), int(msg.get("h", 720)); fovy = float(msg.get("fovy") or 1.194)
    tau = float(msg.get("tau", TAU)); quality = int(msg.get("quality", 90))
    # optional normalised principal point (recorded-camera replays); the browser's virtual camera is centred
    cam = Cam(R_W @ c2w, W, H, fovy, float(msg.get("primx", 0.5)), float(msg.get("primy", 0.5)))
    t1 = time.time()
    if COMPACT:
        im, n = CH.render(cam, tau)
    else:
        thr = (2 * (tau + 0.5)) * math.tan(cam.FoVx * 0.5) / (0.5 * W)
        n = expand_to_size(gaussians.nodes, gaussians.boxes, thr, cam.camera_center, torch.zeros(3), ri, pi_, nri)
        idx = ri[:n].int().contiguous(); nidx = nri[:n].contiguous()
        get_interpolation_weights(nidx, thr, gaussians.nodes, gaussians.boxes, cam.camera_center.cpu(), torch.zeros(3), iw, ns)
        im = render_post(cam, gaussians, pipe, torch.zeros(3, device="cuda"), render_indices=idx, parent_indices=pi_,
                         interpolation_weights=iw, num_node_kids=ns, use_trained_exp=False)["render"]
    torch.cuda.synchronize(); render_ms = (time.time() - t1) * 1000
    frame = (torch.clamp(im, 0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy()
    ok, buf = cv2.imencode(".jpg", frame[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    header = struct.pack("<IHHHHff", int(msg.get("seq", 0)) & 0xFFFFFFFF, W, H, 1, 0, render_ms, (time.time() - t0) * 1000)
    return header + buf.tobytes()


@app.get("/healthz")
def healthz():
    return {"ok": True, "backend": "h3dgs", "hier": HIER, "nodes": N_HIER, "compact": COMPACT, "resident_gib": round(CH.gpu_gib(), 2) if CH else None,
            "skybox_and_fill": int(CH.tail if CH else gaussians.skybox_points), "fill": n_fill,
            "tau": TAU, "merged_chunks": MERGED, "free_gib": round(torch.cuda.mem_get_info()[0] / 2**30, 2)}


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    await sock.accept()
    pending = None; have = asyncio.Event()

    async def receiver():
        nonlocal pending
        while True:
            msg = json.loads(await sock.receive_text())
            if msg.get("t") == "pose":
                pending = msg; have.set()          # newest pose wins
            elif msg.get("t") == "query":
                await sock.send_text(json.dumps({"t": "query_ack", "ok": False, "reason": "hierarchy backend renders colour only (no feature field)"}))
            elif msg.get("t") == "tau":
                global TAU; TAU = float(msg.get("v", TAU))
    rx = asyncio.ensure_future(receiver())
    loop = asyncio.get_event_loop()
    try:
        while True:
            await have.wait(); have.clear()
            msg, pending = pending, None
            def job(m=msg):
                with GPU_LOCK: return _render(m)
            await sock.send_bytes(await loop.run_in_executor(None, job))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        rx.cancel()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
