"""SAM3-in-the-loop few-shot pilot on citrus 05 tree 72 (Paul 2026-10-06: "few shot rather than full finetuning", "nothing hand
labeled"). Labels and scores come from multi-view consistency on the 10 fps native clip (fruit3d_t72: 68 frames in two sighting
windows, LIO pose at each stamp, tree mask, the canonical stock detections).
  track   SAM3 video (detect + track, text 'fruit') per sighting window on a fixed-size crop that follows the tree centroid
          (2x upscale, like the canonical recipe) -> tracks.json {track: [(frame, u, v, area, prob)]} in full-frame px
  tri     bearing-only triangulation of every track with >= MIN_VIEWS frames (LIO c2w, OpenCV axes = the convention
          fruit3d_cluster.py measured) -> reprojection residual, parallax, depth; confirmed = static 3D orange
  score   stock recall per confirmed orange-view (canonical detections.json, centroid within R_MATCH px), stock precision proxy
          (detections matched to a confirmed orange), tracker-only views = candidate misses
  exemplar SAM3 image model on the canonical crop: text 'fruit' alone (reproduces stock) vs text + K positive exemplar boxes
          taken from confirmed oranges the stock detector already found in that frame; recall measured on the OTHER confirmed
          orange-views of the frame (the exemplars never count), K = 1, 3, 5
sam3 pixi env: pilot_t72.py [track|tri|score|exemplar|all]  (outputs in <workdir>/fewshot/)"""
import json, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
W = Path(os.environ.get("FRUIT3D_WORK", "/home/paperspace/data/citrus_all/05_13D_Jackal/prod/scratch_sam3/fruit3d_t72")); OUT = W / "fewshot"; OUT.mkdir(exist_ok=True)
meta = json.load(open(W / "meta.json")); FR = meta["frames"]; names = [r["name"] for r in FR]; stock = {r["name"]: r["detections"] for r in json.load(open(W / "detections.json"))}
root = Path(meta["root"]); T = json.load(open(next(root.glob("prod/tassili/blocks_ns/*/block_*/transforms.json")))); K = np.array([[T["fl_x"], 0, T["cx"]], [0, T["fl_y"], T["cy"]], [0, 0, 1]]); Kinv = np.linalg.inv(K)
H3 = json.load(open(root / "prod/bateleur/scene_graph/marker_hierarchy.json")); CENT = np.array(next(o["xyz"] for o in H3["objects"] if o["id"] == meta["tree"]))
POSE = {r["name"]: np.array(r["pose"], float) for r in FR};
if os.environ.get("POSES", "lio") == "refined":   # trained keyframe pose (H3DGS export, globally BA'd) x LIO relative motion kf -> native frame; everything then lives in the export frame
    sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess"); from read_write_model import read_images_binary, qvec2rotmat
    _P = root / "experimental/h3dgs_expo"; _meta = json.load(open(_P / "export_meta.json")); _RW = np.array(_meta.get("world_rotation_to_zup") or _meta["world_rotation_lio_to_h3dgs"], float)
    _ims = {im.name: im for im in read_images_binary(str(_P / "camera_calibration/aligned/sparse/0/images.bin")).values()}; _TR = {}
    for r in FR:
        if r["is_kf"]:
            im = _ims[f"kf_{r['donor_kf']:06d}.png"]; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec; _TR[r["donor_kf"]] = (np.linalg.inv(w2c), np.array(r["pose"], float))
    POSE = {r["name"]: _TR[r["donor_kf"]][0] @ np.linalg.inv(_TR[r["donor_kf"]][1]) @ np.array(r["pose"], float) for r in FR}
    CENT = (_RW @ np.array([*CENT, 1.0]))[:3]; print(f"[poses] refined: {len(_TR)} trained keyframe poses x LIO relative motion; tree centroid in the export frame {CENT.round(2).tolist()}", flush=True)
ts = np.array([r["ts_ms"] for r in FR]); cut = int(np.where(np.diff(ts) > 500)[0][0]) + 1; WINDOWS = [list(range(0, cut)), list(range(cut, len(FR)))]
PAD, UP, OVERLAP, R_MATCH, MIN_VIEWS = 24, 2.0, 0.5, 12.0, 3; SUF = ("_refined" if os.environ.get("POSES", "lio") == "refined" else "") + ("_kfonly" if os.environ.get("KF_ONLY") else "")
def tmask(n): return np.array(Image.open(W / "masks" / f"{n}.png")) > 127
def canon_crop(n):   # fruit3d_detect.py's crop: tree bbox + pad, 2x
    t = tmask(n); ys, xs = np.where(t); H, Wd = t.shape; y0, y1 = max(0, ys.min() - PAD), min(H, ys.max() + PAD); x0, x1 = max(0, xs.min() - PAD), min(Wd, xs.max() + PAD)
    rgb = Image.open(W / "frames" / f"{n}.png").convert("RGB"); return rgb.crop((x0, y0, x1, y1)).resize((int((x1 - x0) * UP), int((y1 - y0) * UP)), Image.BILINEAR), (x0, y0, x1, y1), t
def dets_from(masks, scores, box, t):   # masks on the upscaled crop -> full-frame centroids, parent gate as the canonical recipe
    x0, y0, x1, y1 = box; cw, ch = x1 - x0, y1 - y0; out = []
    for oi, m in enumerate(masks):
        arr = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m); arr = arr.squeeze()
        small = np.array(Image.fromarray(arr.astype(np.uint8) * 255).resize((cw, ch), Image.BILINEAR)) > 127; fm = np.zeros(t.shape, bool); fm[y0:y1, x0:x1] = small; area = int(fm.sum())
        if area == 0 or (fm & t).sum() / area < OVERLAP: continue
        fy, fx = np.where(fm); out.append({"u": float(fx.mean()), "v": float(fy.mean()), "area": area, "score": float(scores[oi]) if oi < len(scores) else 0.0})
    return out
def track():
    import torch; from sam3.model_builder import build_sam3_video_predictor
    pred = build_sam3_video_predictor(gpus_to_use=[0]); tracks = {}; t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    # the video model's own detector thresholds (model_builder hard-codes new-object 0.7 / detection 0.5; the canonical image recipe keeps everything >= 0.5,
    # and only 131 of tree 72's 534 stock detections score >= 0.7, so at 0.7 the tracker carried a quarter of what the image recipe finds)
    pred.model.new_det_thresh = float(os.environ.get("NEW_DET", "0.5")); pred.model.score_threshold_detection = float(os.environ.get("SCORE_DET", "0.5")); print(f"[track] thresholds: new object {pred.model.new_det_thresh}, detection {pred.model.score_threshold_detection}", flush=True)
    for wi, idx in enumerate(WINDOWS):
        bb = [np.where(tmask(names[i])) for i in idx]; cw = int(max(b[1].max() - b[1].min() for b in bb) + 2 * PAD); ch = int(max(b[0].max() - b[0].min() for b in bb) + 2 * PAD); imgs, offs = [], []
        for i, b in zip(idx, bb):   # fixed-size window on the tree centroid, clamped to the frame
            cy, cx = b[0].mean(), b[1].mean(); x0 = int(np.clip(cx - cw / 2, 0, 1280 - cw)); y0 = int(np.clip(cy - ch / 2, 0, 720 - ch))
            imgs.append(Image.open(W / "frames" / f"{names[i]}.png").convert("RGB").crop((x0, y0, x0 + cw, y0 + ch)).resize((int(cw * UP), int(ch * UP)), Image.BILINEAR)); offs.append((x0, y0))
        sid = pred.handle_request(request=dict(type="start_session", resource_path=imgs))["session_id"]
        pred.handle_request(request=dict(type="add_prompt", session_id=sid, frame_index=0, text="fruit"))
        for resp in pred.handle_stream_request(request=dict(type="propagate_in_video", session_id=sid)):
            fi, o = resp["frame_index"], resp["outputs"]; x0, y0 = offs[fi]; t = tmask(names[idx[fi]])
            for oid, m, p in zip(o["out_obj_ids"].tolist(), o["out_binary_masks"], o["out_probs"].tolist()):
                if not m.any(): continue
                ys, xs = np.where(m); u, v = x0 + xs.mean() / UP, y0 + ys.mean() / UP; area = m.sum() / UP ** 2
                tracks.setdefault(f"w{wi}_{oid}", []).append({"f": idx[fi], "name": names[idx[fi]], "u": float(u), "v": float(v), "area": float(area), "prob": float(p), "in_tree": bool(t[min(719, int(v)), min(1279, int(u))])})
        pred.handle_request(request=dict(type="close_session", session_id=sid)); print(f"[track] window {wi}: {len(idx)} frames, crop {cw}x{ch} @2x, tracks so far {len(tracks)}", flush=True)
    json.dump(tracks, open(OUT / "tracks.json", "w"), indent=1)
    L = [len(v) for v in tracks.values()]; print(f"[track] {len(tracks)} tracks in {time.time() - t0:.0f}s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB; views per track median {np.median(L):.0f} max {max(L)}; tracks with >= {MIN_VIEWS} views: {sum(l >= MIN_VIEWS for l in L)}", flush=True)
def tri():
    tracks = json.load(open(OUT / "tracks.json")); out = {}
    for tid, obs in tracks.items():
        obs = [o for o in obs if o["in_tree"] and (not os.environ.get("KF_ONLY") or FR[o["f"]]["is_kf"])]   # KF_ONLY=1: keyframe views only (trained poses exact at their stamps)
        if len(obs) < MIN_VIEWS: continue
        A = np.zeros((3, 3)); b = np.zeros(3); rays = []
        for o in obs:
            P = POSE[o["name"]]; c = P[:3, 3]; d = P[:3, :3] @ (Kinv @ np.array([o["u"], o["v"], 1.0])); d /= np.linalg.norm(d); M = np.eye(3) - np.outer(d, d); A += M; b += M @ c; rays.append((c, d))
        X = np.linalg.solve(A + 1e-9 * np.eye(3), b); err, depth = [], []
        for o in obs:
            P = POSE[o["name"]]; Xc = P[:3, :3].T @ (X - P[:3, 3]); depth.append(Xc[2]); uv = K @ Xc; err.append(float(np.hypot(uv[0] / uv[2] - o["u"], uv[1] / uv[2] - o["v"])) if Xc[2] > 0.05 else 1e9)
        D = np.array([r[1] for r in rays]); par = float(np.degrees(np.arccos(np.clip((D @ D.T).min(), -1, 1))))
        out[tid] = {"X": X.tolist(), "n_views": len(obs), "reproj_med_px": float(np.median(err)), "reproj_p90_px": float(np.percentile(err, 90)), "parallax_deg": par, "depth_min": float(min(depth)), "depth_max": float(max(depth)), "dist_centroid": float(np.linalg.norm(X - CENT)), "prob_med": float(np.median([o["prob"] for o in obs])), "views": [o["f"] for o in obs]}
    json.dump(out, open(OUT / f"tri{SUF}.json", "w"), indent=1)
    e = np.array([v["reproj_med_px"] for v in out.values()]); p = np.array([v["parallax_deg"] for v in out.values()]); dc = np.array([v["dist_centroid"] for v in out.values()])
    print(f"[tri] {len(out)} tracks with >= {MIN_VIEWS} in-tree views; reprojection median px: p25 {np.percentile(e, 25):.1f} p50 {np.median(e):.1f} p75 {np.percentile(e, 75):.1f} p90 {np.percentile(e, 90):.1f}; parallax deg p50 {np.median(p):.1f}; dist to centroid p50 {np.median(dc):.2f} m", flush=True)
    for thr in (4, 6, 8, 12): print(f"[tri]   confirmed at reproj <= {thr} px & parallax >= 2 deg & 0.5 < depth < 12 & within 6 m: {sum(1 for v in out.values() if ok(v, thr))}", flush=True)
def ok(v, thr=6.0): return v["reproj_med_px"] <= thr and v["parallax_deg"] >= 2 and v["depth_min"] > 0.5 and v["depth_max"] < 12 and v["dist_centroid"] < 6
def confirmed(thr=6.0):
    tracks = json.load(open(OUT / "tracks.json")); tri_ = json.load(open(OUT / f"tri{SUF}.json")); C = {}
    for tid, v in tri_.items():
        if ok(v, thr): C[tid] = [o for o in tracks[tid] if o["in_tree"]]
    return C
def near(dets, u, v): return any(np.hypot(d["u"] - u, d["v"] - v) <= R_MATCH for d in dets)
def score(thr=6.0):
    C = confirmed(thr); views = [(tid, o) for tid, obs in C.items() for o in obs]; hit = sum(near(stock[o["name"]], o["u"], o["v"]) for _, o in views)
    per_frame = {}
    for tid, o in views: per_frame.setdefault(o["name"], []).append((o["u"], o["v"]))
    n_stock = sum(len(stock[n]) for n in names); matched = sum(1 for n in names for d in stock[n] if any(np.hypot(d["u"] - u, d["v"] - v) <= R_MATCH for u, v in per_frame.get(n, [])))
    per_orange = [np.mean([near(stock[o["name"]], o["u"], o["v"]) for o in obs]) for obs in C.values()]
    res = {"thr_px": thr, "confirmed_oranges": len(C), "orange_views": len(views), "stock_recall_views": hit / max(len(views), 1), "oranges_never_found_by_stock": int(sum(r == 0 for r in per_orange)), "stock_detections": n_stock, "stock_matched_to_confirmed": matched, "stock_precision_proxy": matched / max(n_stock, 1)}
    json.dump(res, open(OUT / f"score{SUF}_thr{thr:g}.json", "w"), indent=1); print("[score]", json.dumps(res), flush=True); return C
def exemplar(thr=6.0):
    import torch; from sam3.model_builder import build_sam3_image_model; from sam3.model.sam3_image_processor import Sam3Processor
    C = confirmed(thr); proc = Sam3Processor(build_sam3_image_model()); by_frame = {}
    for tid, obs in C.items():
        for o in obs: by_frame.setdefault(o["name"], []).append((tid, o))
    rows = []
    for n in names:
        if n not in by_frame: continue
        crop, box, t = canon_crop(n); x0, y0, x1, y1 = box
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            st = proc.set_image(crop); r = proc.set_text_prompt(state=st, prompt="fruit")
        base = dets_from(r.get("masks", []), r.get("scores", []).float().cpu().numpy() if hasattr(r.get("scores"), "cpu") else [], box, t)
        found = sorted([(tid, o) for tid, o in by_frame[n] if near(base, o["u"], o["v"])], key=lambda x: -x[1]["prob"])   # exemplar candidates: confirmed oranges the detector already finds here
        for Kx in (1, 3, 5):
            ex = found[:Kx]
            if len(ex) < Kx: continue
            test = [(tid, o) for tid, o in by_frame[n] if tid not in {e[0] for e in ex}]
            if not test: continue
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                proc.reset_all_prompts(st); r2 = proc.set_text_prompt(state=st, prompt="fruit")
                for _, o in ex:   # box = [cx, cy, w, h] normalised to the crop; size from the track mask area (circle)
                    s = 2 * np.sqrt(o["area"] / np.pi) + 4; r2 = proc.add_geometric_prompt(box=[(o["u"] - x0) / (x1 - x0), (o["v"] - y0) / (y1 - y0), s / (x1 - x0), s / (y1 - y0)], label=True, state=st)
            ex_d = dets_from(r2.get("masks", []), r2.get("scores", []).float().cpu().numpy() if hasattr(r2.get("scores"), "cpu") else [], box, t)
            allc = [(o["u"], o["v"]) for _, o in by_frame[n]]
            rows.append({"name": n, "K": Kx, "n_test": len(test), "base_hit": sum(near(base, o["u"], o["v"]) for _, o in test), "ex_hit": sum(near(ex_d, o["u"], o["v"]) for _, o in test),
                         "base_n": len(base), "ex_n": len(ex_d), "base_unmatched": sum(1 for d in base if not any(np.hypot(d["u"] - u, d["v"] - v) <= R_MATCH for u, v in allc)), "ex_unmatched": sum(1 for d in ex_d if not any(np.hypot(d["u"] - u, d["v"] - v) <= R_MATCH for u, v in allc))})
    json.dump(rows, open(OUT / f"exemplar{SUF}_thr{thr:g}.json", "w"), indent=1)
    for Kx in (1, 3, 5):
        R = [r for r in rows if r["K"] == Kx]
        if not R: print(f"[exemplar] K={Kx}: no frame has {Kx} found exemplars + a test orange"); continue
        nt = sum(r["n_test"] for r in R); print(f"[exemplar] K={Kx}: {len(R)} frames, {nt} test orange-views: recall text-only {sum(r['base_hit'] for r in R) / nt:.3f} -> text+{Kx} exemplars {sum(r['ex_hit'] for r in R) / nt:.3f}; detections per frame {np.mean([r['base_n'] for r in R]):.1f} -> {np.mean([r['ex_n'] for r in R]):.1f}, of which not on any confirmed orange {np.mean([r['base_unmatched'] for r in R]):.1f} -> {np.mean([r['ex_unmatched'] for r in R]):.1f}", flush=True)
stage = sys.argv[1] if len(sys.argv) > 1 else "all"; thr = float(os.environ.get("REPROJ_PX", "6"))
if stage in ("track", "all"): track()
if stage in ("tri", "all"): tri()
if stage in ("score", "all"): score(thr)
if stage in ("exemplar", "all"): exemplar(thr)
