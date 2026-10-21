"""Vet the candidate bag changes (moved mid-row, April-only on a row, December-only on a row) with the SAM3
grow-bag masks of both surveys. For each candidate spot, in each survey: the keyframes that look at the spot
(1.5-6 m ahead, within +-37 deg), the ledger position projected with the full lidar->camera extrinsic at the
lidar ground height, and whether a grow-bag mask covers it (25 px tolerance). A bag is 'present' in a survey
when >= 2 viewing frames have a mask there and the hit rate is >= 0.3.
Verdicts: April-only -> added (present Apr, absent Dec) | association miss (present in both) | artefact (absent Apr).
December-only -> removed (present Dec, absent Apr) | association miss | artefact. Moved -> moved (Dec present at
Dec spot & absent at Apr spot; Apr present at Apr spot & absent at Dec spot) | not moved (both spots in both) | unclear.
Writes vet_changes.csv and one evidence tile per bag (best frame per survey) into vet_tiles/."""
import json, sys, csv
import numpy as np, cv2
from pathlib import Path
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts"); import survey_paths as sp
from offline_vtr import monos
K = "/home/paperspace/data/klapmuts"; D = K + "/dec_2025_ten_rows"; A = K + "/apr_2026_zed"; O = D + "/experimental/azalai/stpg_dec_apr_20261021"
L = json.load(open(D + "/experimental/sankofa/klapmuts_ledger_v5.json")); r = L["registration_A_to_B"]
th = np.radians(r["yaw_deg"]); Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); tt = np.array(r["t"])
Aapr = np.array(L["A_positions_apr_frame"]); B = np.array(L["B_positions"]); Adec = (Aapr - tt) @ Rz
apr_to_dec = lambda p: (p - tt) @ Rz
l2c = sp.load_l2c(D); Kmat, W, H = monos.intrinsics(A)
hd = np.load(O + "/bag_heights_dec.npz"); ha = np.load(O + "/bag_heights_apr.npz")
gmed = {"dec": float(np.nanmedian(hd["ground"])), "apr": float(np.nanmedian(ha["ground"]))}
def ground(survey, xy):
    z = hd if survey == "dec" else ha; bags = z["bags"]; i = int(np.argmin(np.linalg.norm(bags - xy, axis=1)))
    g = z["ground"][i]; return float(g) if np.isfinite(g) and np.linalg.norm(bags[i] - xy) < 1.0 else gmed[survey]
S = {"dec": {"root": D, "c2w": monos.keyframe_poses(D)[1]}, "apr": {"root": A, "c2w": monos.keyframe_poses(A)[1]}}
for s in S.values():
    s["streams"] = monos.survey_streams(s["root"]); s["kf_ts"] = monos.image_stamps(s["streams"]["kf"])
    s["clips"] = json.load(open(s["root"] + "/prod/bateleur/sam3_growbag/clips.json"))["clips"]; s["fe_cache"] = {}
def rle_decode(rle):
    h, w = rle["size"]; st = rle["counts"]; counts = []; i = 0
    while i < len(st):
        x = 0; k = 0; more = True
        while more:
            c = ord(st[i]) - 48; x |= (c & 0x1F) << (5 * k); more = bool(c & 0x20); i += 1; k += 1
            if not more and (c & 0x10): x |= -1 << (5 * k)
        if len(counts) > 2: x += counts[-2]
        counts.append(x)
    m = np.zeros(h * w, np.uint8); pos = 0; v = 0
    for c in counts:
        m[pos:pos + c] = v; pos += c; v = 1 - v
    return m.reshape((w, h)).T
def masks_for(s, kf):
    c = next((c for c in s["clips"] if c["start_idx"] <= kf <= c["end_idx"]), None)
    if c is None: return []
    if c["clip_id"] not in s["fe_cache"]:
        s["fe_cache"][c["clip_id"]] = json.load(open(s["root"] + f"/prod/bateleur/sam3_growbag/clip_{c['clip_id']:03d}/frame_entries.json"))["frame_entries"]
    return s["fe_cache"][c["clip_id"]].get(str(kf - c["start_idx"]), [])
def project(c2w, p):
    v = np.linalg.inv(c2w) @ np.array([*p, 1.0])
    if v[2] <= 0.3: return None, v
    u = Kmat @ v[:3]; return (int(u[0] / u[2]), int(u[1] / u[2])), v
def presence(survey, xy, want_tile=False):
    """-> (n_view, n_hit, best (kf, u, hit)) for the laser-world xy in this survey."""
    s = S[survey]; p = (l2c @ np.array([xy[0], xy[1], ground(survey, xy), 1.0]))[:3]; c2w = s["c2w"]
    views = []
    for k in np.flatnonzero(~np.isnan(c2w[:, 0, 0])):
        u, v = project(c2w[k], p)
        if u is None or not (1.5 <= v[2] <= 6.0) or abs(v[0]) > v[2] * 0.75 or not (0 <= u[0] < W and 0 <= u[1] < H): continue
        views.append((abs(v[2] - 3.0), int(k), u))
    views.sort(); views = views[:12]
    hits = []; best = None
    for _, k, u in views:
        hit = False
        for e in masks_for(s, k):
            m = rle_decode(e["rle"]); y0, y1 = max(0, u[1] - 25), min(H, u[1] + 26); x0, x1 = max(0, u[0] - 25), min(W, u[0] + 26)
            if m[y0:y1, x0:x1].any(): hit = True; break
        hits.append(hit)
        if best is None or (hit and not best[2]): best = (k, u, hit)
    return len(views), int(sum(hits)), best
def tile(survey, best, label):
    s = S[survey]; k, u, hit = best
    img = next(monos.iter_images(s["streams"]["kf"], [s["kf_ts"][k]], gray=False))[2]
    for e in masks_for(s, k):
        m = rle_decode(e["rle"]); cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(img, cnts, -1, (0, 165, 255), 2)
    cv2.circle(img, u, 40, (0, 255, 0) if hit else (0, 0, 255), 3, cv2.LINE_AA)
    x0 = max(0, min(W - 640, u[0] - 320)); y0 = max(0, min(H - 480, u[1] - 300)); c = cv2.resize(img[y0:y0 + 480, x0:x0 + 640], (800, 600))
    cv2.rectangle(c, (0, 0), (800, 30), (0, 0, 0), -1); cv2.putText(c, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA); return c
out = Path(O + "/vet_tiles"); out.mkdir(exist_ok=True)
recs = list(csv.DictReader(open(O + "/change_ledger_dec_apr.csv")))
cands = [r_ for r_ in recs if r_["verdict"] == "candidate change" and r_["change"] in ("moved", "april_only", "december_only")]
print("candidates:", len(cands), flush=True)
rows = []
for n, r_ in enumerate(cands):
    kind = r_["change"]
    if kind == "april_only":
        j = int(r_["apr_id"]); pa = presence("apr", B[j]); pd = presence("dec", apr_to_dec(B[j]))
        pres_a = pa[1] >= 2 and pa[1] / max(1, pa[0]) >= 0.3; pres_d = pd[1] >= 2 and pd[1] / max(1, pd[0]) >= 0.3
        verdict = "added" if pres_a and not pres_d else ("association miss (bag in both)" if pres_a and pres_d else "artefact (no April mask)")
        tiles = [("dec", pd[2], f"Dec at the spot: masks {pd[1]}/{pd[0]} frames"), ("apr", pa[2], f"Apr bag {j}: masks {pa[1]}/{pa[0]} frames")]
        rows.append(dict(change=kind, dec_id="", apr_id=j, dec_view=pd[0], dec_hit=pd[1], apr_view=pa[0], apr_hit=pa[1], verdict=verdict))
    elif kind == "december_only":
        i = int(r_["dec_id"]); pd = presence("dec", Adec[i]); pa = presence("apr", Aapr[i])
        pres_a = pa[1] >= 2 and pa[1] / max(1, pa[0]) >= 0.3; pres_d = pd[1] >= 2 and pd[1] / max(1, pd[0]) >= 0.3
        verdict = "removed" if pres_d and not pres_a else ("association miss (bag in both)" if pres_a and pres_d else "artefact (no December mask)")
        tiles = [("dec", pd[2], f"Dec bag {i}: masks {pd[1]}/{pd[0]} frames"), ("apr", pa[2], f"Apr at the spot: masks {pa[1]}/{pa[0]} frames")]
        rows.append(dict(change=kind, dec_id=i, apr_id="", dec_view=pd[0], dec_hit=pd[1], apr_view=pa[0], apr_hit=pa[1], verdict=verdict))
    else:
        i, j = int(r_["dec_id"]), int(r_["apr_id"])
        d_at_d = presence("dec", Adec[i]); a_at_a = presence("apr", B[j]); d_at_a = presence("dec", apr_to_dec(B[j])); a_at_d = presence("apr", Aapr[i])
        P = lambda x: x[1] >= 2 and x[1] / max(1, x[0]) >= 0.3
        verdict = "moved" if P(d_at_d) and P(a_at_a) and not P(a_at_d) and not P(d_at_a) else ("not moved (both spots in both)" if P(a_at_d) and P(d_at_a) else "unclear")
        tiles = [("dec", d_at_d[2], f"Dec bag {i} at Dec spot: {d_at_d[1]}/{d_at_d[0]}; Apr spot in Dec: {d_at_a[1]}/{d_at_a[0]}"), ("apr", a_at_a[2], f"Apr bag {j} at Apr spot: {a_at_a[1]}/{a_at_a[0]}; Dec spot in Apr: {a_at_d[1]}/{a_at_d[0]}")]
        rows.append(dict(change=kind, dec_id=i, apr_id=j, dec_view=d_at_d[0], dec_hit=d_at_d[1], apr_view=a_at_a[0], apr_hit=a_at_a[1], verdict=verdict))
    ims = [tile(sv, b, lab) for sv, b, lab in tiles if b is not None]
    if ims:
        sheet = np.hstack(ims); cv2.rectangle(sheet, (0, 570), (sheet.shape[1], 600), (0, 0, 0), -1)
        cv2.putText(sheet, f"{kind} -> {verdict}", (8, 592), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{kind}_{r_['dec_id'] or 'x'}_{r_['apr_id'] or 'x'}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(n + 1, kind, rows[-1]["verdict"], flush=True)
with open(O + "/vet_changes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
from collections import Counter
print(Counter((r_["change"], r_["verdict"]) for r_ in rows))
