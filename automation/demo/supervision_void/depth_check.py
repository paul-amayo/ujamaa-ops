# 3 depth_check.py (h3dgs env): depth of the laser points each detection lifted (cluster_tree_instances cache: same 1.5 m depth-back,
#   10 m gates) vs the tree centres, via a LIO -> H3DGS similarity fitted on the scene-graph trees
import json, sys, numpy as np
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians/preprocess')
from read_write_model import read_images_binary, read_cameras_binary, qvec2rotmat
S = '/home/paperspace/data/citrus_all/01_13B_Jackal'; G = f'{S}/prod/bateleur/sam3_v2'; src = f'{S}/experimental/h3dgs/camera_calibration/chunks/3_1/sparse/0'
ims = {v.name: v for v in read_images_binary(f'{src}/images.bin').values()}; cam0 = list(read_cameras_binary(f'{src}/cameras.bin').values())[0]; fx, fy, cx, cy = cam0.params[:4]
objs = {o['id']: o for o in json.load(open(f'{S}/prod/bateleur/scene_graph/marker_hierarchy.json'))['objects']}
T = {int(k): np.array(v) for k, v in json.load(open('/home/paperspace/logs/demo_chunks/01_3_1_720/demo_path.json'))['trees'].items()}
st = {int(k): v for k, v in json.load(open(f'{G}/global_ids.json'))['stats'].items()}
ids = [t for t in T if t in st]; A = np.array([st[t]['world_centroid'] for t in ids]); B = np.array([T[t] for t in ids])   # LIO cluster centroids -> H3DGS tree positions
ma, mb = A.mean(0), B.mean(0); U, s, Vt = np.linalg.svd((B - mb).T @ (A - ma)); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); R = U @ D @ Vt
sc = (s * np.diag(D)).sum() / ((A - ma) ** 2).sum(); t = mb - sc * R @ ma; res = np.linalg.norm((sc * (R @ A.T)).T + t - B, axis=1)
print(f'LIO -> H3DGS similarity from {len(ids)} trees: scale {sc:.4f}, residual median {np.median(res):.3f} max {res.max():.3f} (scene units)')
z = np.load(f'{G}/_collect_cache_v3_db1.5_md10.0_citrus_jackal.npz'); keys = z['det_keys']; srcs = z['sources']; pts = z['pts']
idx = {(int(a), int(b)): i for i, (a, b) in enumerate(keys)}
def cam(kf):
    im = ims[f'kf_{kf:06d}.png']; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec; return w2c
def depth_uv(Pw, w2c):
    p = (w2c[:3, :3] @ Pw.T).T + w2c[:3, 3]; return p[:, 2] / sc, fx * p[:, 0] / p[:, 2] + cx, fy * p[:, 1] / p[:, 2] + cy   # depth back in metres (LIO units)
for kf in (3112, 3113, 3115):
    w2c = cam(kf); d, u, v = depth_uv(np.array([T[x] for x in (146, 148, 156, 157)]), w2c)
    print(f'kf_{kf}: tree centres (id, row, depth m, u px): ' + ', '.join(f'{x} (row {objs[x]["row_id"]}, {dd:.1f} m, u {uu:.0f})' for x, dd, uu in zip((146, 148, 156, 157), d, u)))
for oid, kf in [(1617, 3112), (1632, 3113), (1660, 3115)]:
    i = idx[(15, oid)]; P = pts[srcs == i]; Ph = (sc * (R @ P.T)).T + t; d, u, v = depth_uv(Ph, cam(kf))
    print(f'  centre-tree mask oid {oid} on kf_{kf}: {len(P)} lifted pts, depth {d.min():.1f}-{d.max():.1f} m (median {np.median(d):.1f}), u {u.min():.0f}-{u.max():.0f}, v {v.min():.0f}-{v.max():.0f}')
