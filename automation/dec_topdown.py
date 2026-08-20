"""TOP-DOWN map of Dec klapmuts pots + berries (Paul 2026-08-09).
Fuses: SAM3 ledgers (sam3_ledger_v0, 290 frames) + native ZED depth (mcap)
+ ZED odometry (mcap) + rig.json intrinsics. Streaming: depth matched to the
ledger frames' color timestamps; masks unprojected at their median depth;
optical->body (ROS) -> world via odom. Pots NMS-clustered at 0.4 m.
Sanity: pots must form two lines flanking the trajectory."""
import json
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BAG = Path('/home/paperspace/data/klapmuts/dec_2025_a300')
LED = BAG / 'sam3_ledger_v0'
K = json.load(open('/home/paperspace/data/klapmuts/rig.json'))['intrinsics']
FX, FY, CX, CY = K['fx'], K['fy'], K['cx'], K['cy']
OUT = '/home/paperspace/code/lab_notebook/figs/dec_topdown_pots_berries.png'

# ---- pass 1: color ts (index->ts), odom, depth ts index ----
color_ts, odom = [], []
with AnyReader([BAG]) as reader:
    cc = [c for c in reader.connections if c.topic == '/zed/zed_node/left/image_rect_color']
    for c, t, raw in reader.messages(connections=cc):
        m = reader.deserialize(raw, c.msgtype)
        color_ts.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
    oc = [c for c in reader.connections if c.topic == '/zed/zed_node/odom']
    for c, t, raw in reader.messages(connections=oc):
        m = reader.deserialize(raw, c.msgtype)
        p, q = m.pose.pose.position, m.pose.pose.orientation
        odom.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                     [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]))
color_ts = np.array(color_ts)
ots = np.array([o[0] for o in odom])
opos = np.array([o[1] for o in odom])
oquat = np.array([o[2] for o in odom])
print(f'{len(color_ts)} color frames, {len(odom)} odom')

# ledger frames -> needed color ts
need = {}
for f in sorted(LED.glob('dec_*.npz')):
    idx = int(f.stem.split('_')[1])
    if idx < len(color_ts):
        need[round(color_ts[idx], 3)] = (idx, f)
print(f'{len(need)} ledger frames mapped to timestamps')

R_bc = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])   # optical->body

def world_points(ts, uvd):
    i = np.argmin(np.abs(ots - ts))
    Rwb = R.from_quat(oquat[i]).as_matrix()
    t = opos[i]
    out = []
    for u, v, d in uvd:
        pc = np.array([(u - CX) / FX * d, (v - CY) / FY * d, d])
        out.append(Rwb @ (R_bc @ pc) + t)
    return out

pots_w, berries_w, traj = [], [], opos[:, :2]
with AnyReader([BAG]) as reader:
    dc = [c for c in reader.connections if c.topic == '/zed/zed_node/depth/depth_registered']
    for c, t, raw in reader.messages(connections=dc):
        m = reader.deserialize(raw, c.msgtype)
        ts = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        hit = None
        for key in (round(ts, 3),):
            if key in need:
                hit = need.pop(key)
        if hit is None:
            close = [k for k in need if abs(k - ts) < 0.06]
            if close:
                hit = need.pop(close[0])
        if hit is None:
            continue
        idx, f = hit
        depth = np.frombuffer(m.data, np.float32).reshape(m.height, m.width)
        z = np.load(f)
        for kind, sink, dmax in (('pot', pots_w, 12.0), ('berry', berries_w, 5.0)):
            ms, sc = z[f'{kind}_masks'], z[f'{kind}_scores']
            uvd = []
            for mi in range(ms.shape[0]):
                mask = ms[mi]
                dm = depth[mask]
                dm = dm[np.isfinite(dm) & (dm > 0.3) & (dm < dmax)]
                if len(dm) < 8:
                    continue
                ys, xs = np.where(mask)
                uvd.append((xs.mean(), ys.mean(), float(np.median(dm))))
            for p in world_points(ts, uvd):
                sink.append((p[0], p[1], float(sc[0]) if len(sc) else 0.5))
print(f'raw: {len(pots_w)} pot obs, {len(berries_w)} berry obs; unmatched ledger frames: {len(need)}')

def nms_cluster(obs, r):
    obs = sorted(obs, key=lambda o: -o[2])
    kept = []
    for x, y, s in obs:
        if all((x - kx) ** 2 + (y - ky) ** 2 > r * r for kx, ky, _, _ in kept):
            kept.append((x, y, s, 1))
        else:
            for j, (kx, ky, ks, kn) in enumerate(kept):
                if (x - kx) ** 2 + (y - ky) ** 2 <= r * r:
                    kept[j] = (kx, ky, ks, kn + 1)
                    break
    return kept

pot_c = nms_cluster(pots_w, 0.45)
pot_solid = [p for p in pot_c if p[3] >= 2]
print(f'pot clusters: {len(pot_c)} total, {len(pot_solid)} with >=2 observations')

fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
ax.plot(traj[:, 0], traj[:, 1], '-', color='#888', lw=1.5, label=f'trajectory ({len(traj)} poses, 44 m)')
if berries_w:
    b = np.array(berries_w)
    ax.scatter(b[:, 0], b[:, 1], s=6, c='#d62728', alpha=0.45, label=f'berry obs ({len(b)})')
if pot_c:
    p = np.array([(x, y, n) for x, y, _, n in pot_c])
    ax.scatter(p[:, 0], p[:, 1], s=18 + 6 * np.minimum(p[:, 2], 10), facecolors='none',
               edgecolors='#1f77b4', lw=1.4, label=f'pot clusters ({len(pot_c)}; {len(pot_solid)} seen 2+ frames)')
ax.set_aspect('equal')
ax.set_xlabel('x (m, odom frame)')
ax.set_ylabel('y (m)')
ax.legend(loc='upper right', fontsize=9)
ax.set_title('DEC-2025 klapmuts top-down — plant-pot clusters + berry observations\n'
             '(SAM3 ledgers x native ZED depth x ZED odometry; sanity: pot lines should flank the trajectory)',
             loc='left', fontsize=11)
fig.tight_layout()
fig.savefig(OUT, bbox_inches='tight')
print(f'SAVED {OUT}')
