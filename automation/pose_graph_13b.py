#!/usr/bin/env python3
"""P7 — semantic pose graph over 13B surveys (trees as loop-closure landmarks).

Level 1 (rigid): each survey s gets an SE(2) node T_s = (tx, ty, theta); the
WGS84-anchored survey (03) is gauge-fixed. Canonical trees are free 2D
landmarks L_j. Observations are the RAW per-survey EN positions — rebuilt by
adding back the datum shift ledger_v2 recorded removing — so the recorded
shifts are ground truth the graph must recover blind:
  residual_{sj} = R(theta_s) p_{sj} + t_s - L_j     (robust soft_l1)

Success (roadmap lead, opened 2026-07-23): the ~1.4 m 01<->03 common-mode
offset comes out as T_01's translation, not a post-hoc scalar; association
residuals match/beat the post-hoc result (272 pairs @ 0.56 m); 02 (9 sparse
legs) lands consistently. A re-association pass (NN at the 1.5 m gate on
graph-corrected positions) then closes the "jointly optimise poses +
associations" loop once.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

SUB = Path("/home/paperspace/data/citrus_all/sankofa_substrate")
ANCHOR = "03_13B_Jackal"
GATE = 1.5

led = json.loads((SUB / "ledger_v2.json").read_text())
shifts = {s: np.array(v["en_13b_shift_removed_m_EN"])
          for s, v in led["meta"].get("datum_correction", {}).items()}

# raw per-survey EN observations: en_13b_frame had shift REMOVED -> add back
obs = []          # (survey_idx, land_idx, p_raw[2])
surveys, lands = {}, {}
for o in led["observations"]:
    if o["farm"] != "13B" or o.get("en_13b_frame") in (None, "None"):
        continue
    s = o["source_survey"]
    p = np.asarray(o["en_13b_frame"], float)[:2] + shifts.get(s, np.zeros(2))
    si = surveys.setdefault(s, len(surveys))
    li = lands.setdefault(o["canonical_id"], len(lands))
    obs.append((si, li, p))
S, L, N = len(surveys), len(lands), len(obs)
names = sorted(surveys, key=surveys.get)
print(f"[graph] {N} observations, {S} surveys {names}, {L} landmarks")
ai = surveys[ANCHOR]

# variables: for each non-anchor survey (tx, ty, theta), then landmarks (2L)
free = [s for s in range(S) if s != ai]
fpos = {s: i for i, s in enumerate(free)}
n_pose = 3 * len(free)


def unpack(x):
    T = np.zeros((S, 3))
    for s, i in fpos.items():
        T[s] = x[3 * i:3 * i + 3]
    Lxy = x[n_pose:].reshape(L, 2)
    return T, Lxy


def residuals(x):
    T, Lxy = unpack(x)
    r = np.empty((N, 2))
    for k, (si, li, p) in enumerate(obs):
        tx, ty, th = T[si]
        c, s_ = np.cos(th), np.sin(th)
        r[k, 0] = c * p[0] - s_ * p[1] + tx - Lxy[li, 0]
        r[k, 1] = s_ * p[0] + c * p[1] + ty - Lxy[li, 1]
    return r.ravel()


# init: landmarks at mean of raw obs; poses at identity
L0 = np.zeros((L, 2))
cnt = np.zeros(L)
for si, li, p in obs:
    L0[li] += p
    cnt[li] += 1
L0 /= cnt[:, None]
x0 = np.concatenate([np.zeros(n_pose), L0.ravel()])

spar = lil_matrix((2 * N, x0.size), dtype=bool)
for k, (si, li, p) in enumerate(obs):
    if si != ai:
        spar[2 * k:2 * k + 2, 3 * fpos[si]:3 * fpos[si] + 3] = True
    spar[2 * k:2 * k + 2, n_pose + 2 * li:n_pose + 2 * li + 2] = True

r0 = residuals(x0).reshape(N, 2)
print(f"[graph] pre-fit residual median {np.median(np.linalg.norm(r0, axis=1)):.3f} m")
sol = least_squares(residuals, x0, loss="soft_l1", f_scale=0.7,
                    jac_sparsity=spar, x_scale="jac", verbose=0)
T, Lxy = unpack(sol.x)
r1 = np.linalg.norm(residuals(sol.x).reshape(N, 2), axis=1)
print(f"[graph] post-fit residual median {np.median(r1):.3f} m  "
      f"p90 {np.percentile(r1, 90):.3f} m")

print("\n[graph] recovered survey nodes (vs ledger's recorded datum shift):")
result = {}
for s in names:
    si = surveys[s]
    tx, ty, th = T[si]
    gt = -shifts.get(s, np.zeros(2))  # graph must UNDO the added-back shift
    rec = {"t_recovered_EN": [round(tx, 3), round(ty, 3)],
           "theta_deg": round(np.degrees(th), 4),
           "t_expected_EN": [round(gt[0], 3), round(gt[1], 3)],
           "err_m": round(float(np.hypot(tx - gt[0], ty - gt[1])), 3)}
    result[s] = rec
    tag = "ANCHOR (fixed)" if si == ai else f"expected {rec['t_expected_EN']}, err {rec['err_m']} m"
    print(f"  {s}: t=({tx:+.3f}, {ty:+.3f}) m  theta={np.degrees(th):+.4f} deg   {tag}")

# per-survey residuals + 02 inheritance quality
per = {}
for s in names:
    si = surveys[s]
    m = np.array([k for k, (a, _, _) in enumerate(obs) if a == si])
    per[s] = {"n_obs": int(len(m)), "median_res_m": round(float(np.median(r1[m])), 3)}
    print(f"  {s}: {len(m)} obs, median residual {per[s]['median_res_m']} m")

# one re-association pass: NN at the gate between graph-corrected surveys
def corrected(s):
    si = surveys[s]
    tx, ty, th = T[si]
    c, s_ = np.cos(th), np.sin(th)
    out = {}
    for a, li, p in obs:
        if a != si:
            continue
        out[li] = np.array([c * p[0] - s_ * p[1] + tx, s_ * p[0] + c * p[1] + ty])
    return out


reassoc = {}
for a in range(S):
    for b in range(a + 1, S):
        sa, sb = names[a], names[b]
        A, B = corrected(sa), corrected(sb)
        pa = np.array(list(A.values()))
        pb = np.array(list(B.values()))
        d = np.linalg.norm(pa[:, None] - pb[None], axis=2)
        nn = d.argmin(1)
        dd = d[np.arange(len(pa)), nn]
        # 1-1: keep mutual nearest under gate
        nnb = d.argmin(0)
        mutual = [i for i in range(len(pa)) if dd[i] < GATE and nnb[nn[i]] == i]
        med = float(np.median(dd[mutual])) if mutual else float("nan")
        reassoc[f"{sa[:2]}<->{sb[:2]}"] = {"pairs": len(mutual), "median_m": round(med, 3)}
        print(f"  re-assoc {sa[:2]}<->{sb[:2]}: {len(mutual)} mutual pairs @ {med:.3f} m median (gate {GATE})")

# ---- Level 2: per-(survey,row) nodes — is there intra-survey structure? ----
# Legs by proxy: a row visit is a leg. Each (survey,row) with >=3 obs gets its
# own SE(2), chained to the survey's neighbouring rows by a weak smoothness
# prior (0.3 m / 0.3 deg) and initialised at the rigid solution. If this
# barely beats rigid, the residual is association/centroid scatter and the
# 1.4 m story is PURE datum; a big drop means real intra-survey pose error.
row_of = {}
for o in led["observations"]:
    if o["farm"] != "13B" or o.get("en_13b_frame") in (None, "None"):
        continue
    li = lands[o["canonical_id"]]
    row_of[(surveys[o["source_survey"]], li)] = o.get("row_id")

seg_obs = {}
for k, (si, li, p) in enumerate(obs):
    r = row_of.get((si, li))
    seg_obs.setdefault((si, r if r is not None else -99), []).append(k)
segs = sorted(k for k, v in seg_obs.items() if len(v) >= 3)
spos = {sg: i for i, sg in enumerate(segs)}
small = [k for sg, v in seg_obs.items() if sg not in spos for k in v]
print(f"\n[legs] {len(segs)} (survey,row) nodes; {len(small)} obs on thin rows ride the rigid node")

nseg = 3 * len(segs)


def leg_residuals(x):
    Ts = x[:nseg].reshape(-1, 3)
    Lxy2 = x[nseg:].reshape(L, 2)
    rr = []
    for k, (si, li, p) in enumerate(obs):
        sg = (si, row_of.get((si, li), -99))
        if sg in spos:
            tx, ty, th = Ts[spos[sg]]
        else:
            tx, ty, th = T[si]
        c, s_ = np.cos(th), np.sin(th)
        rr.append([c * p[0] - s_ * p[1] + tx - Lxy2[li, 0],
                   s_ * p[0] + c * p[1] + ty - Lxy2[li, 1]])
    # smoothness between adjacent rows of the same survey
    for (si, r), i in spos.items():
        nxt = (si, (r if isinstance(r, int) else -99) + 1)
        if nxt in spos:
            j = spos[nxt]
            rr.append([(Ts[i, 0] - Ts[j, 0]) / 0.3, (Ts[i, 1] - Ts[j, 1]) / 0.3])
            rr.append([(Ts[i, 2] - Ts[j, 2]) / np.radians(0.3), 0.0])
    # weak prior tying each leg to its survey's rigid solution
    for (si, r), i in spos.items():
        rr.append([(Ts[i, 0] - T[si, 0]) / 1.0, (Ts[i, 1] - T[si, 1]) / 1.0])
    return np.asarray(rr).ravel()


x0l = np.concatenate([np.stack([T[si] for si, r in segs]).ravel(), Lxy.ravel()])
soll = least_squares(leg_residuals, x0l, loss="soft_l1", f_scale=0.7,
                     x_scale="jac", verbose=0, max_nfev=60)
rl = np.linalg.norm(leg_residuals(soll.x)[:2 * N].reshape(N, 2), axis=1)
Tsl = soll.x[:nseg].reshape(-1, 3)
mag = np.linalg.norm(Tsl[:, :2] - np.stack([T[si] for si, r in segs])[:, :2], axis=1)
print(f"[legs] post-fit residual median {np.median(rl):.3f} m (rigid was "
      f"{np.median(r1):.3f} m); leg corrections vs rigid: median "
      f"{np.median(mag):.3f} m, p90 {np.percentile(mag, 90):.3f} m, max {mag.max():.3f} m")

out = {"schema": "pose_graph_13b/v1-rigid", "anchor": ANCHOR,
       "legs": {"n_nodes": len(segs),
                "median_res_m": round(float(np.median(rl)), 3),
                "leg_correction_median_m": round(float(np.median(mag)), 3),
                "leg_correction_p90_m": round(float(np.percentile(mag, 90)), 3)},
       "n_obs": N, "n_landmarks": L,
       "post_fit_median_res_m": round(float(np.median(r1)), 3),
       "surveys": result, "per_survey": per, "reassoc": reassoc,
       "landmarks_EN": {cid: [round(float(Lxy[li, 0]), 3), round(float(Lxy[li, 1]), 3)]
                        for cid, li in lands.items()}}
(SUB / "pose_graph_13b_v1.json").write_text(json.dumps(out, indent=1))
print(f"\n[graph] wrote {SUB/'pose_graph_13b_v1.json'}")
