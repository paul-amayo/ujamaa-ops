"""dec_clock_check.py (2026-10-22) — December ten-rows image-vs-INS clock offset: the Dec KISS-ICP SuperPoint teach keyframes (lo_superpoint/dec.npz) rigidly fitted to ins_enu_abs.npz over an offset scan."""
import numpy as np
D = '/home/paperspace/data/klapmuts/dec_2025_ten_rows'; d = np.load('/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/azalai/stpg_dec_apr_20261021/lo_superpoint/dec.npz', allow_pickle=True); z = np.load(D + '/experimental/ins_enu_abs.npz')
ts, c2w = d['ts_ms'].astype(float), d['c2w']; iT, iP = z['ts_ms'].astype(float), z['xyz']
print(f"dec keyframes {len(ts)} ts {ts[0]:.0f}..{ts[-1]:.0f}; INS {len(iP)} ts {iT[0]:.0f}..{iT[-1]:.0f}; teach survey {str(d['survey'])}")
def at(q):
    out = np.full((len(q), 2), np.nan); ok = (q >= iT[0]) & (q <= iT[-1])
    for k in range(2): out[ok, k] = np.interp(q[ok], iT, iP[:, k])
    return out
def rigid2(A, B):
    ca, cb = A.mean(0), B.mean(0); H = (A - ca).T @ (B - cb); U, S, Vt = np.linalg.svd(H); R = Vt.T @ U.T
    if np.linalg.det(R) < 0: Vt[1] *= -1; R = Vt.T @ U.T
    return R, cb - R @ ca
P = c2w[:, [0, 2], 3]; g0 = np.isfinite(P).all(1); res = []
for off in range(-12000, 12001, 100):
    Q = at(ts + off); g = g0 & np.isfinite(Q).all(1); R, t = rigid2(P[g], Q[g]); r = np.linalg.norm(P[g] @ R.T + t - Q[g], axis=1); res.append((off, np.median(r), np.percentile(r, 90), g.sum()))
b = min(res, key=lambda r: r[1]); z0 = [r for r in res if r[0] == 0][0]
print(f"Dec KISS-ICP teach keyframes vs Dec INS: at 0 ms median {z0[1]:.2f} m; best {b[0]:+d} ms median {b[1]:.2f} m p90 {b[2]:.2f} ({b[3]} kf)")
print('scan:', ' '.join(f"{r[0]}:{r[1]:.1f}" for r in res[::10]))
