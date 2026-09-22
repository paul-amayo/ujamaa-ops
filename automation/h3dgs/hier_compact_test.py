"""Parity + footprint test: CompactHierarchy vs H3DGS render_post on the same hierarchy and camera.
usage: python hier_compact_test.py <hier> [tau] [--no-ref]   (run from the h3dgs repo dir, h3dgs env)"""
import sys, time, math, json, urllib.request, numpy as np, torch
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians"); sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/interfaces/splat_viewer")
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from hier_compact import CompactHierarchy
hier = sys.argv[1]; tau = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0; ref = "--no-ref" not in sys.argv
OUT = "/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs/output"; SC = f"{OUT}/scaffold/point_cloud/iteration_30000"
R_W = np.array(json.load(open("/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs/export_meta.json"))["world_rotation_lio_to_h3dgs"])
c2w = R_W @ np.array(next(f["matrix"] for f in json.load(urllib.request.urlopen("http://127.0.0.1:8001/scene/trajectory?stride=1"))["frames"] if f["image_name"] == "kf_001411.png")).reshape(4, 4)
W, H, fovy = 1280, 720, 1.194; w2c = np.linalg.inv(c2w)
class Cam: pass
cam = Cam(); cam.image_width, cam.image_height, cam.FoVy = W, H, fovy; cam.FoVx = 2 * math.atan(math.tan(fovy / 2) * W / H); cam.image_name = "x"
cam.world_view_transform = torch.tensor(getWorld2View2(c2w[:3, :3], w2c[:3, 3])).float().transpose(0, 1).cuda()
cam.projection_matrix = torch.tensor(getProjectionMatrix(0.01, 100.0, cam.FoVx, cam.FoVy, 0.5, 0.5)).float().transpose(0, 1).cuda()
cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0); cam.camera_center = cam.world_view_transform.inverse()[3, :3]
def T(): torch.cuda.synchronize(); return time.time()
torch.cuda.reset_peak_memory_stats(); t0 = time.time()
ch = CompactHierarchy(hier, SC); t1 = time.time()
print(f"[compact] loaded N={ch.N} (hier {ch.n_hier} + tail {ch.tail}) in {t1-t0:.0f}s, resident {ch.gpu_gib():.2f} GiB, torch peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
for _ in range(2): imc, n = ch.render(cam, tau)
t2 = T(); imc, n = ch.render(cam, tau); t3 = T()
print(f"[compact] tau {tau}: {n} nodes, render {1000*(t3-t2):.0f} ms, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB, mean {imc.mean().item():.4f}", flush=True)
if ref:
    from argparse import ArgumentParser
    from arguments import PipelineParams
    from scene import GaussianModel
    from gaussian_renderer import render_post
    from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights
    pp = ArgumentParser(); pipe = PipelineParams(pp).extract(pp.parse_args([]))
    torch.cuda.reset_peak_memory_stats()
    g = GaussianModel(3); g.active_sh_degree = 3; g.create_from_hier(hier, 1.0, SC); N = g._xyz.size(0)
    ri, pi_, nri = (torch.zeros(N).int().cuda() for _ in range(3)); iw = torch.zeros(N).float().cuda(); ns = torch.zeros(N).int().cuda()
    thr = (2 * (tau + 0.5)) * math.tan(cam.FoVx * 0.5) / (0.5 * W)
    def ref_render():
        n = expand_to_size(g.nodes, g.boxes, thr, cam.camera_center, torch.zeros(3), ri, pi_, nri)
        idx = ri[:n].int().contiguous(); nidx = nri[:n].contiguous()
        get_interpolation_weights(nidx, thr, g.nodes, g.boxes, cam.camera_center.cpu(), torch.zeros(3), iw, ns)
        with torch.no_grad():
            return render_post(cam, g, pipe, torch.zeros(3, device="cuda"), render_indices=idx, parent_indices=pi_, interpolation_weights=iw, num_node_kids=ns, use_trained_exp=False)["render"].clamp(0, 1)
    ref_render(); t4 = T(); imr = ref_render(); t5 = T()
    mse = ((imc - imr) ** 2).mean().item(); print(f"[ref] render_post {1000*(t5-t4):.0f} ms, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB; compact-vs-ref PSNR {10*math.log10(1/max(mse,1e-12)):.2f} dB, max abs diff {(imc-imr).abs().max().item():.4f}", flush=True)
    from PIL import Image
    Image.fromarray((torch.cat([imr, imc], 2).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save("/home/paperspace/logs/hier_compact_vs_ref.png")
print("[compact] DONE", flush=True)
