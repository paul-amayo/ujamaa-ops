import sys, time, math, json, urllib.request, numpy as np, torch, faulthandler
faulthandler.dump_traceback_later(int(sys.argv[3]) if len(sys.argv) > 3 else 90, exit=True)
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians")
from argparse import ArgumentParser
from arguments import PipelineParams
from scene import GaussianModel
from gaussian_renderer import render_post
from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
hier, tau = sys.argv[1], float(sys.argv[2])
OUT = "/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs/output"
pp = ArgumentParser(); pipe = PipelineParams(pp).extract(pp.parse_args([]))
def T(): torch.cuda.synchronize(); return time.time()
g = GaussianModel(3); g.active_sh_degree = 3; g.create_from_hier(hier, 1.0, "" if len(sys.argv) > 4 and sys.argv[4] == "nosky" else f"{OUT}/scaffold/point_cloud/iteration_30000"); N = g._xyz.size(0)
ri, pi_, nri = (torch.zeros(N).int().cuda() for _ in range(3)); iw = torch.zeros(N).float().cuda(); ns = torch.zeros(N).int().cuda()
R_W = np.array(json.load(open("/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs/export_meta.json"))["world_rotation_lio_to_h3dgs"])
c2w = R_W @ np.array(next(f["matrix"] for f in json.load(urllib.request.urlopen("http://127.0.0.1:8001/scene/trajectory?stride=1"))["frames"] if f["image_name"] == "kf_001411.png")).reshape(4, 4)
W, H, fovy = 1280, 720, 1.194; w2c = np.linalg.inv(c2w)
class Cam: pass
cam = Cam(); cam.image_width, cam.image_height, cam.FoVy = W, H, fovy; cam.FoVx = 2 * math.atan(math.tan(fovy / 2) * W / H); cam.image_name = "x"
cam.world_view_transform = torch.tensor(getWorld2View2(c2w[:3, :3], w2c[:3, 3])).float().transpose(0, 1).cuda()
cam.projection_matrix = torch.tensor(getProjectionMatrix(0.01, 100.0, cam.FoVx, cam.FoVy, 0.5, 0.5)).float().transpose(0, 1).cuda()
cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0); cam.camera_center = cam.world_view_transform.inverse()[3, :3]
thr = (2 * (tau + 0.5)) * math.tan(cam.FoVx * 0.5) / (0.5 * W)
t1 = T(); n = expand_to_size(g.nodes, g.boxes, thr, cam.camera_center, torch.zeros(3), ri, pi_, nri)
LIM = int(sys.argv[5]) if len(sys.argv) > 5 else n; n = min(n, LIM); idx = ri[:n].int().contiguous(); nidx = nri[:n].contiguous()
get_interpolation_weights(nidx, thr, g.nodes, g.boxes, cam.camera_center.cpu(), torch.zeros(3), iw, ns); t2 = T()
# rough tile-coverage estimate of the selected set: projected radius from the largest scale
with torch.no_grad():
    sel = idx.long(); xyz = g.get_xyz[sel]; sc = g.get_scaling[sel].max(1).values
    p = (torch.cat([xyz, torch.ones(len(sel), 1, device="cuda")], 1) @ cam.world_view_transform); z = p[:, 2].clamp(min=0.01)
    fx = W / (2 * math.tan(cam.FoVx / 2)); r_px = 3 * sc * fx / z; infront = p[:, 2] > 0.01
    tiles = ((2 * r_px / 16) ** 2).clamp(max=3600)[infront]
    print(f"[probe] N={N} tau={tau}: selected {n} ({infront.sum().item()} in front), select+weights {t2-t1:.2f}s, est. tile instances {tiles.sum().item():.3g} (2^31 = 2.15e9), nodes with r>500px: {(r_px[infront] > 500).sum().item()}", flush=True)
    t3 = T(); im = render_post(cam, g, pipe, torch.zeros(3, device="cuda"), render_indices=idx, parent_indices=pi_, interpolation_weights=iw, num_node_kids=ns, use_trained_exp=False)["render"]; t4 = T()
    print(f"[probe] render_post {t4-t3:.2f}s mean {im.mean().item():.3f} peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
