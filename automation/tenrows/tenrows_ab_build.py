"""A/B arm builder for one ten_rows block: keyframes (existing block) vs ALL full-stream
frames in the same time span, both evaluated on the SAME held-out keyframes.
- test = every 8th keyframe of the block (by K); val = test.
- kf arm: train = remaining keyframes.
- full arm: train = every image_left frame whose timestamp lies in [first kf, last kf],
  EXCLUDING frames inside each test keyframe's window (midpoint to prev kf .. midpoint to
  next kf) so no near-duplicate of a test view (67 ms / ~2 cm apart) leaks into training.
- full-arm poses: the SAME formula as survey_paths.poses_at_keyframes (prefix-composed
  laser-frame stream, slerp partial, l2c conjugation), evaluated at image timestamps;
  asserted equal to the block's transform_matrix at the keyframe timestamps.
- init: the kf block's init_lidar.ply is copied (same world, same init => fair)."""
from PIL import Image  # noqa: PIL before the binding
import sys, json, shutil, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts")
from survey_paths import _read_increments, _slerp_partial, load_l2c, monos
import aru_nerf_interface as a
BLK = sys.argv[1] if len(sys.argv) > 1 else "block_013"; TEST_EVERY = 8
R = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows"); MD = R/"prod/monos/monolithics"
BD = R/"prod/tassili/blocks_ns/lio_row100"/BLK; FD = BD.parent/f"{BLK}_full"; PNG = R/"prod/scratch_full"/BLK; PNG.mkdir(parents=True, exist_ok=True)
tj = json.loads((BD/"transforms.json").read_text()); names = [Path(f["file_path"]).name for f in tj["frames"]]
kidx = {e["K"]: e for e in json.loads((MD/"kf_index.json").read_text())}
Ks = [int(n.split("_")[1].split(".")[0]) for n in names]; kf_ts = np.array([kidx[K]["ts_ms"] for K in Ks], float)
test = names[::TEST_EVERY]; train_kf = [n for n in names if n not in test]
tj["train_filenames"], tj["val_filenames"], tj["test_filenames"] = train_kf, list(test), list(test)
(BD/"transforms.json").write_text(json.dumps(tj, indent=2))
# poses at arbitrary timestamps, same formula as poses_at_keyframes
incs = _read_increments(monos(MD)/"transform_lio.monolithic"); inc_ts = np.asarray([t for t, _ in incs], np.int64)
absolute = np.empty((len(incs), 4, 4)); absolute[0] = np.eye(4)
for k in range(1, len(incs)): absolute[k] = absolute[k-1] @ incs[k][1]
l2c = load_l2c(MD); l2c_inv = np.linalg.inv(l2c)
def pose_at(t):
    j = int(np.searchsorted(inc_ts, t, side="left"))
    if j <= 0 or j >= len(incs): return None
    ratio = float(inc_ts[j] - t) / float(inc_ts[j] - inc_ts[j-1]); partial = _slerp_partial(incs[j][1], ratio)
    return l2c @ (absolute[j-1] @ (incs[j][1] @ np.linalg.inv(partial))) @ l2c_inv
FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV c2w -> OpenGL c2w (block transforms.json contract)
err = max(np.abs(pose_at(int(kf_ts[i])) @ FLIP - np.array(tj["frames"][i]["transform_matrix"])).max() for i in range(len(names)))
print(f"[ab] {BLK}: {len(names)} kf, test {len(test)} (every {TEST_EVERY}th), train-kf {len(train_kf)}; pose-formula replication max|diff| = {err:.2e}")
assert err < 1e-6
rdr = a.ImageMonoReader(image_mono=str(MD/"image_left.monolithic")); n = rdr.num_images()
its = np.array([rdr.timestamp(i) for i in range(n)], float)
span = np.where((its >= kf_ts[0]) & (its <= kf_ts[-1]))[0]
# exclusion windows around test keyframes
ex = np.zeros(n, bool)
for i, nm in enumerate(names):
    if nm not in test: continue
    lo = (kf_ts[i-1] + kf_ts[i]) / 2 if i > 0 else kf_ts[i] - 1; hi = (kf_ts[i] + kf_ts[i+1]) / 2 if i + 1 < len(names) else kf_ts[i] + 1
    ex |= (its >= lo) & (its <= hi)
train_full = [i for i in span if not ex[i]]
frames = []
for i in train_full:
    T = pose_at(int(its[i]))
    if T is None: continue
    f = PNG/f"img_{i:06d}.png"
    if not f.exists(): Image.fromarray(np.asarray(rdr.read_index(i))[:, :, :3]).save(f, compress_level=1)
    frames.append({"file_path": str(f), "transform_matrix": (T @ FLIP).tolist()})
kf_by_name = {Path(fr["file_path"]).name: fr for fr in tj["frames"]}
for nm in test: frames.append(dict(kf_by_name[nm]))
ftj = {k: v for k, v in tj.items() if k not in ("frames", "train_filenames", "val_filenames", "test_filenames")}
ftj["frames"] = frames; ftj["train_filenames"] = [Path(fr["file_path"]).name for fr in frames[:len(frames)-len(test)]]
ftj["val_filenames"] = ftj["test_filenames"] = list(test); ftj["ply_file_path"] = "init_lidar.ply"
FD.mkdir(exist_ok=True); (FD/"transforms.json").write_text(json.dumps(ftj, indent=2))
dt = np.diff(its[train_full]); print(f"[ab] full arm: span {len(span)} images ({(kf_ts[-1]-kf_ts[0])/1e3:.1f} s), excluded {int(ex[span].sum())} near test views, train-full {len(train_full)} frames (median dt {np.median(dt):.0f} ms), test {len(test)} kf; wrote {FD}/transforms.json")
print(f"[ab] frame ratio full/kf = {len(train_full)/len(train_kf):.2f}x")
