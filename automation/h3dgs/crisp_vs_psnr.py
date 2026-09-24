"""Does per-view image crispness explain held-out PSNR?  Per test view: non-sky Laplacian variance of the GT frame
(crispness), gradient anisotropy |Gx|/|Gy| (motion-blur proxy), saturated fraction; correlated with that view's score
from the compact evaluator (tau 3 when present, else the smallest tau scored); PSNR by crispness tercile.
Plus a native-pixel GT | render crop figure from the evaluator's saved renders.
  python crisp_vs_psnr.py [--out fig.png] [tag=<proj_dir>,<scores.json rel. to proj>,<sky_mask_dir> ...]
Defaults: ten_rows LiDAR survey vs citrus 05 (both merged hierarchies, output/eval_compact)."""
import argparse, json, numpy as np, cv2
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
ap = argparse.ArgumentParser(); ap.add_argument("sets", nargs="*"); ap.add_argument("--out", default="/home/paperspace/logs/gt_vs_render_native_crops.png")
ap.add_argument("--crop", type=int, nargs=2, default=[420, 300]); ap.add_argument("--per_set", type=int, default=2, help="views per set in the crop figure")
a = ap.parse_args()
DEFAULT = ["ten_rows=/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/h3dgs_tr_lidar,output/eval_compact/scores.json,/home/paperspace/data/klapmuts/dec_2025_ten_rows/prod/tassili/sky_masks",
           "citrus_05=/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs,output/eval_compact/scores.json,/home/paperspace/data/citrus_all/05_13D_Jackal/prod/tassili/sky_masks"]
crops = []
for spec in (a.sets or DEFAULT):
    tag, rest = spec.split("=", 1); parts = rest.split(","); proj = Path(parts[0]); scores = proj / parts[1]; sky = Path(parts[2])
    allrows = json.load(open(scores)); taus = sorted(set(r["tau"] for r in allrows)); tau = 3.0 if 3.0 in taus else taus[0]
    rows = [r for r in allrows if r["tau"] == tau]; rdir = scores.parent / f"render_{tau:g}"
    key = next(k for k in ("psnr_nosky", "psnr_fg", "psnr") if sum(r.get(k) is not None for r in rows) > len(rows) // 2)
    rows = [r for r in rows if r.get(key) is not None]
    lap, aniso, sat, ps, names = [], [], [], [], []
    for r in rows:
        g = cv2.imread(str(proj / "camera_calibration/rectified/images" / r["name"]), cv2.IMREAD_GRAYSCALE)
        if g is None: continue
        m = np.ones(g.shape, bool); sm = sky / r["name"]
        if sm.exists():
            s = cv2.imread(str(sm), cv2.IMREAD_GRAYSCALE); s = cv2.dilate((s > 0).astype(np.uint8), np.ones((15, 15), np.uint8)); m = s == 0
        if m.sum() < 5000: continue
        L = cv2.Laplacian(g, cv2.CV_64F); gx = np.abs(cv2.Sobel(g, cv2.CV_64F, 1, 0)); gy = np.abs(cv2.Sobel(g, cv2.CV_64F, 0, 1))
        lap.append(L[m].var()); aniso.append(gx[m].mean() / max(gy[m].mean(), 1e-6)); sat.append((g[m] >= 250).mean()); ps.append(r[key]); names.append(r["name"])
    lap, aniso, sat, ps = map(np.array, (lap, aniso, sat, ps))
    q = np.percentile(lap, [33.3, 66.7]); t = np.digitize(lap, q)
    print(f"\n== {tag}: {len(ps)} held-out views scored on {key} (tau {tau:g}), {scores.relative_to(proj)} ==")
    print(f"  crispness (non-sky Laplacian var): median {np.median(lap):.0f}  p10 {np.percentile(lap,10):.0f}  p90 {np.percentile(lap,90):.0f}")
    print(f"  anisotropy |Gx|/|Gy| median {np.median(aniso):.2f} (p10 {np.percentile(aniso,10):.2f}, p90 {np.percentile(aniso,90):.2f});  saturated non-sky px median {np.median(sat)*100:.2f}%")
    print(f"  PSNR median {np.median(ps):.2f};  Pearson r(crispness, PSNR) = {pearsonr(lap, ps)[0]:+.2f}, Spearman {spearmanr(lap, ps)[0]:+.2f};  r(anisotropy, PSNR) = {pearsonr(aniso, ps)[0]:+.2f}")
    for k, lab in enumerate(["blurriest third", "middle third", "sharpest third"]):
        print(f"    {lab:15s}: n={int((t==k).sum()):3d}  crispness {np.median(lap[t==k]):5.0f}  PSNR mean {ps[t==k].mean():.2f}  median {np.median(ps[t==k]):.2f}")
    saved = {p.name for p in rdir.glob("*.png")} if rdir.exists() else set()
    cand = sorted([(abs(p - np.median(ps)), n, p) for n, p in zip(names, ps) if n in saved])[:a.per_set]
    crops += [(tag, n, p, proj / "camera_calibration/rectified/images" / n, rdir / n) for _, n, p in cand]
if not crops: raise SystemExit("no saved renders for the crop figure")
W, H = a.crop; tiles = []
for tag, n, p, gt, rd in crops:
    im0 = cv2.imread(str(gt)); im1 = cv2.imread(str(rd)); h, w = im0.shape[:2]; x0, y0 = w // 2 - W // 2, int(h * 0.62) - H // 2
    ca, cb = im0[y0:y0 + H, x0:x0 + W].copy(), im1[y0:y0 + H, x0:x0 + W].copy()
    for im, lab in ((ca, f"{tag} {n} GT"), (cb, f"render {p:.1f} dB")):
        cv2.rectangle(im, (0, 0), (W, 22), (0, 0, 0), -1); cv2.putText(im, lab, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(np.concatenate([ca, np.full((H, 4, 3), 255, np.uint8), cb], 1))
per = a.per_set; cols = []   # one column per set, per_set rows
for c in range(len(tiles) // per):
    col = tiles[c * per:(c + 1) * per]; cols.append(np.concatenate(sum([[x, np.full((6, x.shape[1], 3), 255, np.uint8)] for x in col], [])[:-1], 0))
grid = np.concatenate(sum([[c, np.full((c.shape[0], 10, 3), 255, np.uint8)] for c in cols], [])[:-1], 1)
cv2.imwrite(a.out, grid); print("\nwrote", a.out, grid.shape)
