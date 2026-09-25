"""Pull the INS odometry (header stamp, log/receive time, position, orientation, pose covariance diagonal) and the
header-vs-log stamps of the ZED left images and the ZED odometry straight from the ten_rows mcap, to settle the ~3 s
INS-vs-image stamp offset seen in tenrows_colmap_vs_ins_dt.py and to read the INS position quality (covariance) per
leg. Writes prod/tassili/ten_rows_ins_stamps.npz. Run with nerf_new python3.10 (rosbags)."""
import sys, time
from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader
R = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows"); BAG = R / "prod/monos/rosbag2_2025_12_03-13_22_24_0.mcap"
T = {"ins": "/a300_00047/sensors/ins_0/odom", "left": "/zed/zed_node/left/image_rect_color", "zed": "/zed/zed_node/odom", "gnss": "/a300_00047/sensors/ins_0/fixposition/fpa/gnsscorr"}
ins, left, zed, gnss = [], [], [], []; t0 = time.time(); n = 0
with AnyReader([BAG]) as reader:
    conns = [c for c in reader.connections if c.topic in T.values()]
    for conn, log_ts, raw in reader.messages(connections=conns):
        n += 1
        if conn.topic == T["gnss"]: gnss.append(log_ts); continue
        msg = reader.deserialize(raw, conn.msgtype); h = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
        if conn.topic == T["left"]:
            if len(left) % 20 == 0 or True: left.append((h, log_ts))
        elif conn.topic == T["ins"]:
            p, q = msg.pose.pose.position, msg.pose.pose.orientation; cov = np.asarray(msg.pose.covariance, np.float64).reshape(6, 6)
            tw = msg.twist.twist.linear
            ins.append((h, log_ts, p.x, p.y, p.z, q.w, q.x, q.y, q.z, cov[0, 0], cov[1, 1], cov[2, 2], cov[5, 5], tw.x, tw.y, tw.z, str(getattr(msg, "child_frame_id", ""))[:0] and 0 or 0))
        elif conn.topic == T["zed"]:
            p = msg.pose.pose.position; zed.append((h, log_ts, p.x, p.y, p.z))
        if n % 20000 == 0: print(f"[stamps] {n} msgs: ins {len(ins)} left {len(left)} zed {len(zed)} gnss {len(gnss)} ({time.time() - t0:.0f}s)", flush=True)
ins = np.array(ins, np.float64); left = np.array(left, np.float64); zed = np.array(zed, np.float64); gnss = np.array(gnss, np.float64)
np.savez(R / "prod/tassili/ten_rows_ins_stamps.npz", ins=ins, left=left, zed=zed, gnss=gnss)
def off(a): d = (a[:, 0] - a[:, 1]) / 1e9; return f"header-log median {np.median(d):+.3f} s (p10 {np.percentile(d, 10):+.3f}, p90 {np.percentile(d, 90):+.3f})"
print(f"[stamps] INS {len(ins)}: {off(ins)}; rate {len(ins) / ((ins[-1, 1] - ins[0, 1]) / 1e9):.1f} Hz; pos cov sqrt median x {np.sqrt(np.median(ins[:, 9])):.3f} y {np.sqrt(np.median(ins[:, 10])):.3f} z {np.sqrt(np.median(ins[:, 11])):.3f} m, yaw {np.degrees(np.sqrt(np.median(ins[:, 12]))):.2f} deg; "
      f"cov x p90 {np.sqrt(np.percentile(ins[:, 9], 90)):.3f} max {np.sqrt(ins[:, 9].max()):.3f} m", flush=True)
print(f"[stamps] left images {len(left)}: {off(left)}; ZED odom {len(zed)}: {off(zed)}; gnsscorr {len(gnss)} msgs, rate {len(gnss) / ((gnss[-1] - gnss[0]) / 1e9):.2f} Hz", flush=True)
print(f"[stamps] INS header vs left header at the same log time: INS header lead = {np.median((ins[:, 0] - ins[:, 1])) / 1e9 - np.median(left[:, 0] - left[:, 1]) / 1e9:+.3f} s")
print(f"[stamps] done in {time.time() - t0:.0f}s -> prod/tassili/ten_rows_ins_stamps.npz", flush=True)
