"""Where did an April-only bag come from? For ledger bag B[i]: the April keyframes 1.5-6 m before it that
look at it, with the SAM3 'grow bag' masks of those frames drawn (COCO RLE from prod/bateleur/sam3_growbag)
and the ledger position projected (circle), plus the December view of the same spot through the STPG edge.
usage: bag_provenance.py <B ids...>   (ground height: lidar per-bag ground if bag_heights_*.npz exist, else provisional)"""
import json, sys, glob
import numpy as np, cv2
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts"); import survey_paths as sp
from offline_vtr import monos
K = "/home/paperspace/data/klapmuts"; D = K + "/dec_2025_ten_rows"; A = K + "/apr_2026_zed"; O = D + "/experimental/azalai/stpg_dec_apr_20261021"
L = json.load(open(D + "/experimental/sankofa/klapmuts_ledger_v5.json")); B = np.array(L["B_positions"])
r = L["registration_A_to_B"]; th = np.radians(r["yaw_deg"]); Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); tt = np.array(r["t"])
Adec = (np.array(L["A_positions_apr_frame"]) - tt) @ Rz
l2c = sp.load_l2c(D); R = l2c[:3, :3]; Kmat, W, H = monos.intrinsics(A)
tsA, apr = monos.keyframe_poses(A); tsD, dec = monos.keyframe_poses(D)
g = json.load(open(O + "/lo_superpoint/stpg.json"))
edges = {s["k"]: (s["j"], dec[s["j"]] @ np.asarray(s["T_teach_repeat"]), s["n_inliers"]) for s in g["spatial"] if s["T_teach_repeat"] is not None and s["n_inliers"] >= 20}
stA, stD = monos.survey_streams(A), monos.survey_streams(D); kfA, kfD = monos.image_stamps(stA["kf"]), monos.image_stamps(stD["kf"])
# ground height in the laser world: lidar per-bag ground when available, else the sensor height minus 1.2 m
def ground_z(survey, bag_xy, fallback):
    f = O + f"/bag_heights_{survey}.npz"
    try:
        z = np.load(f); G = z["ground"]; bags = z["bags"]; i = int(np.argmin(np.linalg.norm(bags - bag_xy, axis=1)))
        return float(G[i]) if np.isfinite(G[i]) else float(np.nanmedian(G))
    except Exception:
        return fallback
# masks
clips = json.load(open(A + "/prod/bateleur/sam3_growbag/clips.json"))["clips"]
def rle_decode(rle):
    h, w = rle["size"]; s = rle["counts"]; counts = []; i = 0
    while i < len(s):                      # COCO compressed RLE
        x = 0; k = 0; more = True
        while more:
            c = ord(s[i]) - 48; x |= (c & 0x1F) << (5 * k); more = bool(c & 0x20); i += 1; k += 1
            if not more and (c & 0x10): x |= -1 << (5 * k)
        if len(counts) > 2: x += counts[-2]
        counts.append(x)
    m = np.zeros(h * w, np.uint8); pos = 0; v = 0
    for c in counts:
        m[pos:pos + c] = v; pos += c; v = 1 - v
    return m.reshape((w, h)).T
def masks_for(kf):
    c = next((c for c in clips if c["start_idx"] <= kf <= c["end_idx"]), None)
    if c is None: return []
    fe = json.load(open(A + f"/prod/bateleur/sam3_growbag/clip_{c['clip_id']:03d}/frame_entries.json"))
    local = kf - c["start_idx"]
    return fe["frame_entries"].get(str(local), [])
def project(c2w, p):
    v = np.linalg.inv(c2w) @ np.array([*p, 1.0])
    if v[2] <= 0.3: return None, v
    u = Kmat @ v[:3]; return (int(u[0] / u[2]), int(u[1] / u[2])), v
tiles = []
for bid in [int(x) for x in sys.argv[1:]]:
    gzA = ground_z("apr", B[bid], -2.6)
    pA = R @ np.array([B[bid, 0], B[bid, 1], gzA])
    cands = []
    for k in np.flatnonzero(~np.isnan(apr[:, 0, 0])):
        u, v = project(apr[k], pA)
        if u is not None and 1.5 <= v[2] <= 6.0 and abs(v[0]) <= v[2] * 0.75: cands.append((v[2], k, u))
    cands.sort()
    picks = [cands[i] for i in np.linspace(0, len(cands) - 1, min(3, len(cands))).astype(int)] if cands else []
    print(f"bag {bid} at April frame ({B[bid,0]:.2f}, {B[bid,1]:.2f}), ground z {gzA:.2f}: {len(cands)} viewing keyframes, showing {[(round(z,1), int(k)) for z,k,u in picks]}")
    for z, k, u in picks:
        img = next(monos.iter_images(stA["kf"], [kfA[k]], gray=False))[2]
        ents = masks_for(k); n_near = 0
        for e in ents:
            m = rle_decode(e["rle"]); cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            inside = m[min(H - 1, max(0, u[1])), min(W - 1, max(0, u[0]))] > 0
            col = (0, 0, 255) if inside else (0, 165, 255); n_near += inside
            cv2.drawContours(img, cnts, -1, col, 2)
        cv2.circle(img, u, 40, (0, 220, 255), 3, cv2.LINE_AA)
        # December view through the STPG edge of this keyframe (or the nearest localised one)
        kk = k if k in edges else min(edges, key=lambda q: abs(q - k))
        j, est, ni = edges[kk]; p_dec = (est @ np.linalg.inv(apr[kk]) @ np.array([*pA, 1.0]))[:3]
        imd = next(monos.iter_images(stD["kf"], [kfD[j]], gray=False))[2]; ud, vd = project(dec[j], p_dec)
        if ud is not None and 0 <= ud[0] < W and 0 <= ud[1] < H: cv2.circle(imd, ud, 40, (0, 220, 255), 3, cv2.LINE_AA)
        # the nearest December bag to the carried point
        dn = np.linalg.norm(Adec - (np.linalg.inv(R) @ p_dec)[:2], axis=1); jn = int(dn.argmin())
        cv = np.hstack([imd, img]); cv2.rectangle(cv, (0, 0), (2 * W, 34), (0, 0, 0), -1)
        cv2.putText(cv, f"April bag {bid}: April kf {k} at {z:.1f} m (right; grow-bag masks: red = contains the projected position, orange = others, {len(ents)} in frame)  |  December kf {j} via edge of kf {kk} ({ni} inl.), nearest Dec bag {jn} at {dn[jn]:.2f} m", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv)
sheet = np.vstack(tiles); cv2.imwrite("/tmp/bag_provenance.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]); print("tiles", len(tiles))
