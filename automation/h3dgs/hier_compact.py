"""VRAM-bounded renderer for a merged H3DGS hierarchy (2026-09-22).

H3DGS's `render_post` keeps every gaussian attribute in fp32 on the GPU and, per frame, materialises N-sized
temporaries (`get_features` concatenates all SH coefficients: 26 M nodes -> 5 GB per frame), which is why the
reference path needs ~18 GB for a 4-chunk merge and would not fit a whole survey beside training. This class
loads the hierarchy straight from the file into GPU tensors with positions in fp32 and everything else in
fp16, and per frame gathers ONLY the selected nodes (and their parents) before calling the same rasteriser
with the same interpolation as `render_post`'s python path. Skybox (and optional scaffold fill) gaussians are
appended as an always-rendered tail, exactly like `create_from_hier` + `render_post` do.

Precision: positions fp32; SH / log-scales / rotations / opacities fp16 (colour has 8-bit output, scales are
stored as logs so fp16 covers 1e-15..1e5 m). Parity vs render_post is measured by hier_compact_test.py.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_hierarchy._C import load_hierarchy, expand_to_size, get_interpolation_weights
from plyfile import PlyData


def _load_ply(path: str):
    """Scaffold point cloud as numpy arrays (same fields as GaussianModel.load_ply_file with degree 1)."""
    v = PlyData.read(path)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)               # (P,3)
    names = sorted([p.name for p in v.properties if p.name.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))
    rest = np.stack([v[n] for n in names], 1).astype(np.float32) if names else np.zeros((len(xyz), 0), np.float32)
    op = v["opacity"].astype(np.float32)
    sc = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float32)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
    return xyz, dc, rest, op, sc, rot


class CompactHierarchy:
    def __init__(self, hier_path: str, scaffold_dir: str = "", fill_outside: list | None = None, device: str = "cuda"):
        xyz, shs, alpha, scales, rots, nodes, boxes = load_hierarchy(hier_path)    # CPU fp32; shs (P,16,3)
        self.n_hier = xyz.size(0)
        tail_xyz, tail_sh, tail_alpha, tail_sc, tail_rot = [], [], [], [], []
        self.skybox = 0; self.fill = 0
        if scaffold_dir:
            sx, sdc, srest, sop, ssc, srot = _load_ply(scaffold_dir + "/point_cloud.ply")
            with open(scaffold_dir + "/pc_info.txt") as f:
                nsky = int(f.readline())
            # SH layout of the ply: f_rest is (P, 3*K) stored coefficient-major (all R coeffs, then G, then B)
            K = srest.shape[1] // 3 if srest.shape[1] else 0
            def sh16(idx):
                out = np.zeros((len(idx), 16, 3), np.float32)
                out[:, 0, :] = sdc[idx]
                if K:
                    r = srest[idx].reshape(len(idx), 3, K).transpose(0, 2, 1)   # (n, K, 3)
                    out[:, 1:1 + min(K, 15), :] = r[:, :15, :]
                return out
            sky = np.arange(nsky)
            tail_xyz.append(sx[sky]); tail_sh.append(sh16(sky)); tail_alpha.append(1 / (1 + np.exp(-sop[sky])))
            tail_sc.append(ssc[sky]); tail_rot.append(srot[sky]); self.skybox = nsky
            if fill_outside:
                keep = np.ones(len(sx) - nsky, bool); scene = sx[nsky:]
                for c, e in fill_outside:
                    keep &= ~((np.abs(scene[:, 0] - c[0]) <= e[0] / 2) & (np.abs(scene[:, 1] - c[1]) <= e[1] / 2))
                sel = np.nonzero(keep)[0] + nsky
                if sel.size:
                    tail_xyz.append(sx[sel]); tail_sh.append(sh16(sel)); tail_alpha.append(1 / (1 + np.exp(-sop[sel])))
                    tail_sc.append(ssc[sel]); tail_rot.append(srot[sel]); self.fill = int(sel.size)
        self.tail = self.skybox + self.fill
        dev = torch.device(device)
        cat = lambda a, b: torch.cat([a, torch.from_numpy(np.concatenate(b))]) if b else a
        self.xyz = cat(xyz, tail_xyz).float().to(dev)
        self.sh = cat(shs, tail_sh).half().to(dev)
        self.alpha = cat(alpha.reshape(-1), [t.reshape(-1) for t in tail_alpha]).half().to(dev)
        self.logscale = cat(scales, tail_sc).half().to(dev)
        self.rot = cat(rots, tail_rot).half().to(dev)
        self.nodes = nodes.to(dev); self.boxes = boxes.to(dev)
        N = self.xyz.size(0); self.N = N
        self.ri, self.pi, self.nri = (torch.zeros(N, dtype=torch.int32, device=dev) for _ in range(3))
        self.iw = torch.zeros(N, dtype=torch.float32, device=dev); self.ns = torch.zeros(N, dtype=torch.int32, device=dev)
        self.tail_idx = torch.arange(N - self.tail, N, device=dev)
        del xyz, shs, alpha, scales, rots

    def gpu_gib(self) -> float:
        return sum(t.numel() * t.element_size() for t in (self.xyz, self.sh, self.alpha, self.logscale, self.rot, self.nodes, self.boxes, self.ri, self.pi, self.nri, self.iw, self.ns)) / 2**30

    @torch.no_grad()
    def render(self, cam, tau: float, bg=None, sh_degree: int = 3):
        """cam: object with image_width/height, FoVx/FoVy, world_view_transform, full_proj_transform, camera_center (cuda)."""
        W, H = int(cam.image_width), int(cam.image_height)
        thr = (2 * (tau + 0.5)) * math.tan(cam.FoVx * 0.5) / (0.5 * W)
        n = expand_to_size(self.nodes, self.boxes, thr, cam.camera_center, torch.zeros(3), self.ri, self.pi, self.nri)
        idx = self.ri[:n].int().contiguous(); nidx = self.nri[:n].contiguous()
        get_interpolation_weights(nidx, thr, self.nodes, self.boxes, cam.camera_center.cpu(), torch.zeros(3), self.iw, self.ns)
        c = idx.long(); p = self.pi[:n].long(); w = self.iw[:n].unsqueeze(1); wi = 1.0 - w
        means = w * self.xyz[c] + wi * self.xyz[p]
        scales = w * torch.exp(self.logscale[c].float()) + wi * torch.exp(self.logscale[p].float())
        sh = w.unsqueeze(2) * self.sh[c].float() + wi.unsqueeze(2) * self.sh[p].float()
        rc = torch.nn.functional.normalize(self.rot[c].float()); rp = torch.nn.functional.normalize(self.rot[p].float())
        rp[(rc * rp).sum(1) < 0] *= -1
        rot = w * rc + wi * rp
        op = (w * self.alpha[c].float().unsqueeze(1) + wi * self.alpha[p].float().unsqueeze(1))
        t = self.tail_idx
        means = torch.cat([means, self.xyz[t]]); scales = torch.cat([scales, torch.exp(self.logscale[t].float())])
        sh = torch.cat([sh, self.sh[t].float()]); rot = torch.cat([rot, torch.nn.functional.normalize(self.rot[t].float())])
        op = torch.cat([op, self.alpha[t].float().unsqueeze(1)])
        self.iw[n:n + self.tail] = 1.0; self.ns[n:n + self.tail] = 1
        means2D = torch.zeros_like(means)
        settings = GaussianRasterizationSettings(
            image_height=H, image_width=W, tanfovx=math.tan(cam.FoVx * 0.5), tanfovy=math.tan(cam.FoVy * 0.5),
            bg=bg if bg is not None else torch.zeros(3, device=means.device), scale_modifier=1.0,
            viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform, sh_degree=sh_degree,
            campos=cam.camera_center, prefiltered=False, debug=False,
            render_indices=torch.Tensor([]).int(), parent_indices=torch.Tensor([]).int(),
            interpolation_weights=self.iw, num_node_kids=self.ns, do_depth=False)
        image, radii, _ = GaussianRasterizer(raster_settings=settings)(
            means3D=means, means2D=means2D, shs=sh, colors_precomp=None, opacities=op, scales=scales, rotations=rot, cov3D_precomp=None)
        return image.clamp(0, 1), int(n)
