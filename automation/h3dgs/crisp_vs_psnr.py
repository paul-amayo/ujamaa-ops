"""Does per-view image crispness explain held-out PSNR? ten_rows (H3DGS LiDAR survey) vs citrus 05 (H3DGS), same evaluator.
Per test view: non-sky Laplacian variance (crispness), gradient anisotropy |Gx|/|Gy| (motion-blur proxy), saturated fraction;
correlate with the view's PSNR; PSNR by crispness tercile. Plus a native-pixel GT|render crop figure for both surveys."""
import json, numpy as np, cv2
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
S = {"ten_rows": dict(proj=Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/h3dgs_tr_lidar"),
                      sky=Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows/prod/tassili/sky_masks")),
     "citrus_05": dict(proj=Path("/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs"),
                       sky=Path("/home/paperspace/data/citrus_all/05_13D_Jackal/prod/tassili/sky_masks"))}
crops = {}
for tag, d in S.items():
    rows = [r for r in json.load(open(d["proj"] / "output/eval_compact/scores.json")) if r["tau"] == 3.0]
    key = next(k for k in ("psnr_nosky", "psnr_fg", "psnr") if sum(r.get(k) is not None for r in rows) > len(rows) // 2)
    rows = [r for r in rows if r.get(key) is not None]
    lap, aniso, sat, ps, names = [], [], [], [], []
    for r in rows:
        f = d["proj"] / "camera_calibration/rectified/images" / r["name"]
        g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if g is None: continue
        m = np.ones(g.shape, bool)
        sm = d["sky"] / r["name"]
        if sm.exists():
            s = cv2.imread(str(sm), cv2.IMREAD_GRAYSCALE); s = cv2.dilate((s > 0).astype(np.uint8), np.ones((15, 15), np.uint8)); m = s == 0
        if m.sum() < 5000: continue
        L = cv2.Laplacian(g, cv2.CV_64F); gx = np.abs(cv2.Sobel(g, cv2.CV_64F, 1, 0)); gy = np.abs(cv2.Sobel(g, cv2.CV_64F, 0, 1))
        lap.append(L[m].var()); aniso.append(gx[m].mean() / max(gy[m].mean(), 1e-6)); sat.append((g[m] >= 250).mean()); ps.append(r[key]); names.append(r["name"])
    lap, aniso, sat, ps = map(np.array, (lap, aniso, sat, ps))
    q = np.percentile(lap, [33.3, 66.7]); t = np.digitize(lap, q)
    print(f"\n== {tag}: {len(ps)} held-out views scored on {key} (tau 3) ==")
    print(f"  crispness (non-sky Laplacian var): median {np.median(lap):.0f}  p10 {np.percentile(lap,10):.0f}  p90 {np.percentile(lap,90):.0f}")
    print(f"  anisotropy |Gx|/|Gy| median {np.median(aniso):.2f} (p10 {np.percentile(aniso,10):.2f}, p90 {np.percentile(aniso,90):.2f});  saturated non-sky px median {np.median(sat)*100:.2f}%")
    print(f"  PSNR median {np.median(ps):.2f};  Pearson r(crispness, PSNR) = {pearsonr(lap, ps)[0]:+.2f}, Spearman {spearmanr(lap, ps)[0]:+.2f};  r(anisotropy, PSNR) = {pearsonr(aniso, ps)[0]:+.2f}")
    for k, lab in enumerate(["blurriest third", "middle third", "sharpest third"]):
        print(f"    {lab:15s}: n={int((t==k).sum()):3d}  crispness {np.median(lap[t==k]):5.0f}  PSNR mean {ps[t==k].mean():.2f}  median {np.median(ps[t==k]):.2f}")
    # crops: pick 2 of the saved renders (render_3) closest to the survey's median PSNR
    saved = {p.name for p in (d["proj"] / "output/eval_compact/render_3").glob("*.png")}
    cand = sorted([(abs(p - np.median(ps)), n, p) for n, p in zip(names, ps) if n in saved])[:2]
    crops[tag] = [(n, p, d["proj"] / "camera_calibration/rectified/images" / n, d["proj"] / "output/eval_compact/render_3" / n) for _, n, p in cand]
# figure: native pixels, 420x300 window at lower-centre (foreground), GT | render per view, two views per survey
W, H = 420, 300; tiles = []
for tag, lst in crops.items():
    for n, p, gt, rd in lst:
        a = cv2.imread(str(gt)); b = cv2.imread(str(rd)); h, w = a.shape[:2]; x0, y0 = w // 2 - W // 2, int(h * 0.62) - H // 2
        ca, cb = a[y0:y0 + H, x0:x0 + W].copy(), b[y0:y0 + H, x0:x0 + W].copy()
        for im, lab in ((ca, f"{tag} {n} GT"), (cb, f"render {p:.1f} dB")):
            cv2.rectangle(im, (0, 0), (W, 22), (0, 0, 0), -1); cv2.putText(im, lab, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.concatenate([ca, np.full((H, 4, 3), 255, np.uint8), cb], 1))
sep = np.full((6, tiles[0].shape[1], 3), 255, np.uint8)
grid = np.concatenate([np.concatenate([tiles[0], sep, tiles[1]], 0), np.full((tiles[0].shape[0] * 2 + 6, 10, 3), 255, np.uint8), np.concatenate([tiles[2], sep, tiles[3]], 0)], 1)
out = "/home/paperspace/logs/gt_vs_render_native_crops.png"; cv2.imwrite(out, grid); print("\nwrote", out, grid.shape)
