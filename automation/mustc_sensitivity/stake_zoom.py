#!/usr/bin/env python3
"""JS-2 (12 Nov 2026): full-resolution crops of the UAV1 orthos around the stake-like laser instances of plot 178 that the
stake rule flags outside SB7_150623 (design item 25 of plans/mustc_jirs.md), so their identity can be read from imagery.

For each target (laser date, instance id) and each ortho date: a 0.6 x 0.6 m crop at the ortho's 1 mm pixels (shown x2),
with the instance at its raw laser position (cyan circle, 5 cm radius) and at the position moved into that ortho's own frame
(yellow circle, 5 cm radius): laser + T(laser date -> its paired ortho, georef_prep/frames_prep_178.json: soil texture when
accepted, else the two-pass plant NN) + ortho-to-ortho offset (grey-soil phase correlation).  A laser date without a measured
link gets the raw position only and a 10 cm circle.  The laser canopy-height model of the same window is appended.
Reads only; writes PNGs and stake_zoom.json to --out-dir.

Box: /mnt/shared2/must-c/venv/bin/python stake_zoom.py --out-dir <dir>
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
PLOT = "178"
CENSUS = f"/mnt/shared2/must-c/plot{PLOT}/census"
TARGETS = [  # (label, laser date, id, ortho dates)
    ("230621_id41_in_SB7_080723", "230621", 41, ["230614", "230623", "230708"]),
    ("230525_id4_in_SB7_240623", "230525", 4, ["230517", "230526", "230602"]),
    ("reference_rod_230621_id1_in_SB7_150623", "230621", 1, ["230614", "230623", "230708"]),   # a rod already identified on 4 Nov: what a rod looks like
]
HALF = 0.30


def ortho_path(date):
    h = sorted(glob.glob(f"/mnt/shared2/must-c/plot{PLOT}/plot-wise/plot{PLOT}/*/raster_data/UAV1-RGB/{date}.tif"))
    return h[0] if h else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", required=True); a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    Pp = json.load(open(f"{CENSUS}/georef_prep/frames_prep_178.json"))
    o2ref = {"230602": (0.0, 0.0)}   # ortho date -> offset to ADD to its coordinates to land on the 230602 ortho
    for k, v in Pp["ortho_pairs"].items():
        o2ref[k.split("->")[0]] = (v["grey_soil"]["dE_mm"] / 1e3, v["grey_soil"]["dN_mm"] / 1e3)
    l2o = {}
    for v in Pp["laser_to_ortho"].values():
        q = v["soil_texture"] if v["soil_texture"].get("accepted") else v["nn_two_pass"]
        l2o[v["laser_date"]] = {"ortho": v["ortho_date"], "dE": q["dE_mm"] / 1e3, "dN": q["dN_mm"] / 1e3, "how": "soil texture" if v["soil_texture"].get("accepted") else "two-pass plant NN (soil texture ambiguous)"}
    try:
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except Exception:
        fnt = ImageFont.load_default()
    rec = {"targets": {}, "l2o": l2o, "ortho_to_230602": o2ref}
    for label, ld, iid, odates in TARGETS:
        J = json.load(open(f"{CENSUS}/lmi_{ld}.json")); p = [q for q in J["plants"] if q["id"] == iid][0]
        x, y = p["x"], p["y"]
        panels = []; tr = {"laser_date": ld, "id": iid, "x": x, "y": y, "h95_m": p["height_p95"], "area_cm2": p["area_cm2"], "row": p["row"], "in_polygon": p["in_polygon"], "orthos": {}}
        for od in odates:
            path = ortho_path(od)
            if path is None:
                tr["orthos"][od] = {"error": "no ortho"}; continue
            im = Image.open(path); t = im.tag_v2; tp, sc = t[33922], t[33550]
            X0, Y0, sx, sy = float(tp[3]), float(tp[4]), float(sc[0]), float(sc[1])
            moved = None
            if ld in l2o and l2o[ld]["ortho"] in o2ref and od in o2ref:
                po = l2o[ld]["ortho"]
                moved = (x + l2o[ld]["dE"] + o2ref[po][0] - o2ref[od][0], y + l2o[ld]["dN"] + o2ref[po][1] - o2ref[od][1])
            c0, r0 = int(round((x - HALF - X0) / sx)), int(round((Y0 - (y + HALF)) / sy)); n = int(round(2 * HALF / sx))
            crop = im.crop((c0, r0, c0 + n, r0 + n)).convert("RGB").resize((2 * n, 2 * n), Image.LANCZOS)
            d = ImageDraw.Draw(crop)

            def circ(px, py, rad_m, col):
                cx, cy = (px - (X0 + c0 * sx)) / sx * 2, ((Y0 - r0 * sy) - py) / sy * 2; rr = rad_m / sx * 2
                d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=col, width=3)
            circ(x, y, 0.05 if moved else 0.10, (0, 220, 255))
            if moved:
                circ(moved[0], moved[1], 0.05, (255, 230, 0))
            d.text((8, 6), f"UAV1 ortho {od}; laser {ld} id {iid}", fill=(255, 255, 255), font=fnt)
            d.text((8, 30), "cyan = raw laser position" + ("; yellow = moved into this ortho's frame (5 cm circles)" if moved else " (10 cm circle; no measured laser-to-ortho link for this date)"), fill=(255, 255, 255), font=fnt)
            d.line((8, crop.size[1] - 14, 8 + 0.10 / sx * 2, crop.size[1] - 14), fill=(255, 255, 255), width=4); d.text((8, crop.size[1] - 42), "10 cm; N up, E right", fill=(255, 255, 255), font=fnt)
            crop.save(f"{a.out_dir}/zoom_{label}_ortho{od}.jpg", quality=92)
            panels.append(crop); tr["orthos"][od] = {"path": path, "moved_xy": moved, "pixel_m": sx, "link": (l2o.get(ld) or {}).get("how")}
        # laser CHM of the same window
        z = np.load(f"{CENSUS}/lmi_{ld}.npz"); X0c, Y0c, c, nx, ny = [float(v) for v in z["raster_1cm_meta"]]; chm = z["chm_1cm"]
        i0, j0 = int((x - HALF - X0c) / c), int((y - HALF - Y0c) / c); m = int(round(2 * HALF / c))
        sub = np.zeros((m, m)); a0, b0 = max(i0, 0), max(j0, 0); a1, b1 = min(i0 + m, chm.shape[0]), min(j0 + m, chm.shape[1])
        if a1 > a0 and b1 > b0:
            sub[a0 - i0:a1 - i0, b0 - j0:b1 - j0] = np.nan_to_num(chm[a0:a1, b0:b1])
        g = (np.clip(sub.T[::-1] / 0.40, 0, 1) * 255).astype(np.uint8)
        side = panels[0].size[0] if panels else 1200
        chm_im = Image.fromarray(g).convert("RGB").resize((side, side), Image.NEAREST); dd = ImageDraw.Draw(chm_im)
        dd.text((8, 6), f"laser {ld} canopy height (white = 0.40 m), 1 cm cells, raw laser frame", fill=(255, 230, 0), font=fnt)
        dd.text((8, 30), f"id {iid}: h95 {p['height_p95'] * 100:.0f} cm, area {p['area_cm2']:.0f} cm2, n pts {p.get('n_points')}", fill=(255, 230, 0), font=fnt)
        cc = side / 2; rr = 0.05 / c * (side / m); dd.ellipse((cc - rr, cc - rr, cc + rr, cc + rr), outline=(0, 220, 255), width=2)
        chm_im.save(f"{a.out_dir}/zoom_{label}_chm{ld}.png")
        panels.append(chm_im)
        W = sum(q.size[0] for q in panels); H = max(q.size[1] for q in panels)
        sheet = Image.new("RGB", (W, H), (0, 0, 0)); xoff = 0
        for q in panels:
            sheet.paste(q, (xoff, 0)); xoff += q.size[0]
        out = f"{a.out_dir}/zoom_{label}.jpg"
        sheet.save(out, quality=90); tr["sheet"] = out; rec["targets"][label] = tr
        print("wrote", out, sheet.size)
    json.dump(rec, open(f"{a.out_dir}/stake_zoom.json", "w"), indent=1)


if __name__ == "__main__":
    main()
