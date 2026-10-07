"""Serve April lane 2 (h3dgs_e2s + native identity) on the live demo site's klapmuts-apr26 survey (Paul 2026-10-06: "just add the
3d to the live demo website and then freeze"). The site's April registry (sankofa_demo/klapmuts_registry_apr_2026_v5.json) is the
grow-bag ledger: ids 0..911 in the ledger's April frame. The tassili page lights `tree: <registry id>` and tests "in the 3D section"
by registry position vs the walk, so (1) the wire frame must be the ledger frame and (2) the render service's tree ids must be
registry ids:
  frame  T_lane->ledger = (x, y, z) -> (y, -x, z) after the rigid fit of the lane-LO cameras onto the same frames' April export
         poses; export_meta_ledger.json carries world_rotation_to_zup = T_ledger->lane (hier_render_service: c2w_h = R_W @ c2w);
         lio_image_poses.json (the walk = the 197 training poses, OpenCV c2w) is written in the ledger frame
  ids    manifest_v5ids.json: registry id -> the identity's word, for registry bags and scene-graph objects (bateleur
         marker_hierarchy, the ids the cell was supervised with) that are mutual nearest neighbours within 0.5 m in the ledger
         frame (the ledger's own association rule); text_bank_v5ids.npz is built from it (build_text_bank.py, nerf_new env)
  h3dgs env: apr_lane_stage.py"""
import json, re, sys
import numpy as np
from scipy.spatial import cKDTree
K = "/home/paperspace/data/klapmuts/apr_2026_zed"; LD = f"{K}/experimental/lane2_apr_full"; P = f"{LD}/h3dgs_e2s"; N = f"{P}/native/cell_trees"; GL = np.diag([1.0, -1, -1, 1])
def tf(p): return {f["file_path"].split("/")[-1]: np.array(f["transform_matrix"], float) @ GL for f in json.load(open(p))["frames"]}
aw, alo, aex = tf(f"{LD}/transforms_ref_lo.json"), tf(f"{LD}/transforms_lo.json"), tf(f"{LD}/transforms_lo_export.json"); an = sorted(aw)
X = np.array([alo[n][:3, 3] for n in an]); Y = np.array([aex[n][:3, 3] for n in an]); mx, my = X.mean(0), Y.mean(0)
U, _, Vt = np.linalg.svd((X - mx).T @ (Y - my)); S = np.eye(3); S[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ S @ U.T; t = my - R @ mx   # rigid (scale 1.0001 dropped: 2.5 mm over 25 m)
Rz = np.array([[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]]); T = np.eye(4); T[:3, :3] = Rz @ R; T[:3, 3] = Rz @ t; Ti = np.linalg.inv(T)
res = np.linalg.norm((X @ R.T + t) - Y, axis=1); print(f"[stage] lane -> export rigid fit: residual median {np.median(res) * 100:.1f} cm, max {res.max() * 100:.1f} cm")
meta = json.load(open(f"{P}/export_meta.json")); meta["world_rotation_to_zup"] = Ti.tolist(); meta["wire_frame"] = "klapmuts ledger April frame ((y, -x) of the April H3DGS export); world_rotation_to_zup = T_ledger->lane (apr_lane_stage.py)"
json.dump(meta, open(f"{P}/export_meta_ledger.json", "w"), indent=1)
walk = json.load(open(f"{LD}/lio_image_poses.json"))
if "ledger" not in json.dumps(walk[0].get("frame", "")):
    for e in walk: e["transform"] = (T @ np.array(e["transform"])).tolist(); e["frame"] = "ledger"
    json.dump(walk, open(f"{LD}/lio_image_poses.json", "w"), indent=1)
C = np.array([e["transform"] for e in walk])[:, :2, 3]
R5 = json.load(open("/home/paperspace/data/sankofa_demo/klapmuts_registry_apr_2026_v5.json")); B = np.array([o["xyz"] for o in R5["objects"]], float)[:, :2]; bid = np.array([o["id"] for o in R5["objects"]])
print(f"[stage] walk: {len(walk)} poses in the ledger frame; camera to nearest registry bag median {np.median(cKDTree(B).query(C)[0]):.2f} m")
H = json.load(open(f"{K}/prod/bateleur/scene_graph/marker_hierarchy.json")); RW = np.array(json.load(open(f"{K}/experimental/h3dgs/export_meta.json"))["world_rotation_to_zup"], float)
Xm = np.array([o["xyz"] for o in H["objects"]], float); E = Xm @ RW[:3, :3].T + RW[:3, 3]; M = np.c_[E[:, 1], -E[:, 0]]; mid = np.array([o["id"] for o in H["objects"]])
wt = {str(k): v for k, v in json.load(open(f"{N}/supervision/trees_only/manifest.json"))["word_table"].items()}
d1, j1 = cKDTree(M).query(B); d2, j2 = cKDTree(B).query(M); pairs = [(int(bid[i]), int(mid[j1[i]])) for i in range(len(B)) if d1[i] <= 0.5 and j2[j1[i]] == i and str(int(mid[j1[i]])) in wt]
man = json.load(open(f"{N}/supervision/trees_only/manifest.json")); man["word_table"] = {str(b): wt[str(m)] for b, m in pairs}; man["level_table"] = {str(b): "tree" for b, _ in pairs}
man["registry_id_map"] = {str(b): m for b, m in pairs}; man["note"] = "word_table keyed by the site's April registry ids (klapmuts_registry_apr_2026_v5.json): mutual nearest neighbours within 0.5 m (ledger frame) between registry bags and the scene-graph objects the cell was supervised with (apr_lane_stage.py)"
json.dump(man, open(f"{N}/manifest_v5ids.json", "w"), indent=1)
near = np.where(cKDTree(C).query(np.c_[B])[0] < 3.0)[0]; lane_ids = set(int(bid[i]) for i in near); ok = [b for b, _ in pairs if b in lane_ids]
print(f"[stage] registry ids with an identity word: {len(pairs)} of {len(B)} bags; along lane 2 (within 3 m of the walk): {len(ok)} of {len(lane_ids)}")
json.dump({"T_lane_to_ledger": T.tolist(), "pairs": pairs, "lane_ids": sorted(lane_ids), "lightable_lane_ids": sorted(ok)}, open(f"{N}/stage_ids.json", "w"), indent=1)
