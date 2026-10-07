"""April lane 2 relevancy, the regular native-identity chain on the lane H3DGS model (Paul 2026-10-06: "we have a survey embedder,
we would just be creating supervision and doing the second stage ... I don't see how this is different from what we do
regularly"). The survey's keyframes ARE frames of the lane stream (kf 190 / 402 stamps = the first / last lane frames), and
their supervision exists (prod/tassili/blocks_ns/lio_row100 block_001..005 supervision/trees_only, the maps the April blocks'
stage 2 used with bateleur/embedder/apr_2026_zed_v1g + scene_graph/marker_hierarchy.json). So: the keyframes inside the lane
window get the lane model's placed pose of their stream frame and the census runs over them as for any H3DGS cell.
  mode kfmap (nerf_new python3.10 + aru_py_logger): <lane>/kf_map.json {kf_XXXXXX.png: f_XXXXX.png} by exact stamp
  mode cell  (h3dgs env): <cell>/supervision/trees_only (links + merged manifest), <cell>/census_src (sparse/0 with the
        keyframe cameras in the lane model's frame, images = links to the stream frames), split_names.json (held out:
        every 10th keyframe), frames.json (the held-out keyframes + 8 training ones, for scoring)
usage: apr_lane_native_prep.py kfmap <lane> | cell <lane> <project> <cell dir>"""
import json, os, re, shutil, sys
from pathlib import Path
import numpy as np
mode, LD = sys.argv[1], Path(sys.argv[2]); S = Path("/home/paperspace/data/klapmuts/apr_2026_zed"); BL = S / "prod/tassili/blocks_ns/lio_row100"
if mode == "kfmap":
    sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-cpython-310/lib"); import aru_py_logger
    st = {n: int(v) for n, v in json.load(open(LD / "stamps.json")).items()}; by_t = {v: n for n, v in st.items()}; w = json.load(open(LD / "window.json"))
    lg = aru_py_logger.MonoImageLogger(str(S / "prod/monos/image_left_kf20cm.monolithic"), False); i = 0; out = {}
    while not lg.end_of_file():
        img, ts = lg.read_from_file()
        if img is None or getattr(img, "size", 0) == 0: break
        if w["kf_first"] <= i <= w["kf_last"]:
            if int(ts) in by_t: out[f"kf_{i:06d}.png"] = by_t[int(ts)]
        if i > w["kf_last"]: break
        i += 1
    json.dump(out, open(LD / "kf_map.json", "w"), indent=1); print(f"[native-prep] kf_map: {len(out)} keyframes kf {w['kf_first']}..{w['kf_last']} matched to stream frames by exact stamp", flush=True)
else:
    sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess"); from read_write_model import read_model, write_model, Image as CImage, rotmat2qvec
    PROJ, CELL = LD / sys.argv[3], Path(sys.argv[4]); SUP = CELL / "supervision/trees_only"; SRC = CELL / "census_src"; SP = SRC / "sparse/0"
    for d in (SUP, SP, SRC / "images"): d.mkdir(parents=True, exist_ok=True)
    km = json.load(open(LD / "kf_map.json")); GL = np.diag([1.0, -1, -1, 1])
    pose = {f["file_path"].split("/")[-1]: np.array(f["transform_matrix"], float) @ GL for f in json.load(open(LD / "transforms_ref_lo.json"))["frames"]}   # placed lane cameras (all window frames)
    blocks = sorted(BL.glob("block_0*")); src = {}; wt, lt = {}, {}; man0 = None
    for b in blocks:
        md = b / "supervision/trees_only"
        if not (md / "manifest.json").exists(): continue
        m = json.load(open(md / "manifest.json")); man0 = man0 or m
        for k, v in m.get("word_table", {}).items():
            assert wt.get(k, v) == v, f"word clash {k}: {wt[k]} vs {v} ({b.name})"; wt[k] = v
        lt.update(m.get("level_table", {}))
        for p in md.glob("kf_*.png"): src.setdefault(p.name, p)
    kfs = sorted(k for k in km if k in src and km[k] in pose); held = set(kfs[::10]); train = [k for k in kfs if k not in held]
    for k in kfs:
        for d, tgt in ((SUP / k, src[k]), (SRC / "images" / k, LD / "images" / km[k])):
            if not d.exists(): os.symlink(tgt, d)
    man = dict(man0); man["word_table"] = wt; man["level_table"] = lt; man["note"] = f"merged from {len(blocks)} lio_row100 blocks' trees_only manifests for April lane 2 keyframes (apr_lane_native_prep.py)"
    json.dump(man, open(SUP / "manifest.json", "w"), indent=1)
    cams, ims_src, _ = read_model(str(PROJ / "camera_calibration/chunks/lane/sparse/0"), ".bin"); ims = {}
    for i, k in enumerate(kfs, 1):
        w2c = np.linalg.inv(pose[km[k]]); ims[i] = CImage(id=i, qvec=rotmat2qvec(w2c[:3, :3]), tvec=w2c[:3, 3], camera_id=list(cams)[0], name=k, xys=np.zeros((0, 2)), point3D_ids=np.zeros((0,), np.int64))
    write_model(cams, ims, {}, str(SP), ".bin"); (SP / "test.txt").write_text("\n".join(sorted(held)) + "\n")
    for f in ("points3D.ply",):
        if (PROJ / "camera_calibration/chunks/lane/sparse/0" / f).exists() and not (SP / f).exists(): os.symlink(PROJ / "camera_calibration/chunks/lane/sparse/0" / f, SP / f)
    for f in ("center.txt", "extent.txt"): shutil.copy2(PROJ / "camera_calibration/chunks/lane" / f, SRC / f)
    json.dump({"source": "April lane 2 keyframes in the lane window; held out = every 10th keyframe", "train": train, "eval": sorted(held)}, open(CELL / "split_names.json", "w"), indent=1)
    tr8 = [str(s[len(s) // 2]) for s in np.array_split(np.array(train), 8)]; fr = sorted(set(tr8) | held)
    json.dump({"frames": fr, "info": {n: {"split": "eval" if n in held else "train"} for n in fr}}, open(CELL / "frames.json", "w"), indent=1)
    print(f"[native-prep] cell {CELL}: {len(kfs)} keyframes with supervision + a placed pose ({len(train)} census, {len(held)} held out); manifest {len(wt)} words from {len(blocks)} blocks; scoring frames {len(fr)}", flush=True)
