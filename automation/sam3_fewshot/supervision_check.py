"""Is the fine-tuned fruit supervision worse or just denser? On 05 chunk 1_0 keyframes of the fruit3d trees, compare the OLD (stock SAM3)
and NEW (fine-tuned) compiled fruit maps against 3D-confirmed oranges (tracks + keyframe triangulation): per supervision, fruit components
per frame, their median area, the share of components with a confirmed orange within R px (precision proxy, two references: oranges
confirmed with the stock detector and with the fine-tuned one), and the recall of confirmed oranges. Writes a 2-frame overlay sheet.
sam3/h3dgs env: supervision_check.py -> prints + /home/paperspace/data/demo_video_v2/sam3_fewshot/supervision_old_vs_new.jpg"""
import json, os, glob, numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as nd
S = "/home/paperspace/data/citrus_all/05_13D_Jackal"; N = f"{S}/experimental/h3dgs_native"; OLD, NEW = f"{N}/chunk_1_0_sam3/supervision/trees_only", f"{N}/chunk_1_0_sam3_ft/supervision/trees_only"; D = f"{S}/prod/scratch_sam3"; R = 12
ok = lambda v: v["reproj_med_px"] <= 6 and v["parallax_deg"] >= 2 and v["depth_min"] > 0.5 and v["depth_max"] < 12 and v["dist_centroid"] < 6
conf = {"stock": {}, "finetuned": {}}   # kf name -> [(u, v)]
for W in glob.glob(f"{D}/fruit3d_t*/"):
    meta = json.load(open(W + "meta.json")); kfof = {r["name"]: f"kf_{r['donor_kf']:06d}.png" for r in meta["frames"] if r["is_kf"] and r.get("kf_dt_ms", 0) == 0}   # the frame that IS the keyframe image (is_kf also flags frames up to ~300 ms from it)
    for ref, fd in (("stock", "fewshot"), ("finetuned", "fewshot_union40_ep1")):
        if not os.path.exists(f"{W}{fd}/tri_refined_kfonly.json"): continue
        tri = json.load(open(f"{W}{fd}/tri_refined_kfonly.json")); tr = json.load(open(f"{W}{fd}/tracks.json"))
        for t, v in tri.items():
            if not ok(v): continue
            for o in tr[t]:
                if o["in_tree"] and o["name"] in kfof: conf[ref].setdefault(kfof[o["name"]], []).append((o["u"], o["v"]))
frames = sorted(set(conf["stock"]) | set(conf["finetuned"])); frames = [f for f in frames if os.path.exists(f"{OLD}/{f}") and os.path.exists(f"{NEW}/{f}")]
print(f"[sup] {len(frames)} chunk-1_0 keyframes with a confirmed orange (stock-confirmed views on {len(conf['stock'])} kf, fine-tuned-confirmed on {len(conf['finetuned'])} kf)")
stats = {}
for lab, SUP in (("old (stock SAM3)", OLD), ("new (fine-tuned)", NEW)):
    ncomp = 0; areas = []; hit = {"stock": 0, "finetuned": 0}; rec = {"stock": [0, 0], "finetuned": [0, 0]}
    for f in frames:
        m = np.array(Image.open(f"{SUP}/{f}"), np.uint16); fr = (m >= 10000) & (m != 65535); lab_, n = nd.label(fr); ncomp += n
        cents = nd.center_of_mass(fr, lab_, range(1, n + 1)) if n else []; areas += [int((lab_ == i).sum()) for i in range(1, n + 1)]
        for ref in ("stock", "finetuned"):
            pts = conf[ref].get(f, [])
            hit[ref] += sum(1 for cy, cx in cents if any(np.hypot(cx - u, cy - v) <= R for u, v in pts))
            rec[ref][0] += sum(1 for u, v in pts if any(np.hypot(cx - u, cy - v) <= R for cy, cx in cents)); rec[ref][1] += len(pts)
    print(f"[sup] {lab:18s}: {ncomp} fruit components on {len(frames)} frames ({ncomp / len(frames):.1f}/frame, median area {np.median(areas) if areas else 0:.0f} px); "
          f"on a stock-confirmed orange {hit['stock'] / max(ncomp, 1):.2f}, on a fine-tuned-confirmed orange {hit['finetuned'] / max(ncomp, 1):.2f}; "
          f"recall of confirmed oranges: stock-set {rec['stock'][0]}/{rec['stock'][1]} = {rec['stock'][0] / max(rec['stock'][1], 1):.2f}, fine-tuned-set {rec['finetuned'][0]}/{rec['finetuned'][1]} = {rec['finetuned'][0] / max(rec['finetuned'][1], 1):.2f}", flush=True)
# overlay sheet: two keyframes with the most fine-tuned-confirmed oranges
pick = sorted(frames, key=lambda f: -len(conf["finetuned"].get(f, [])))[:2]; F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20); tiles = []
for f in pick:
    for lab, SUP in (("old", OLD), ("new", NEW)):
        im = np.array(Image.open(f"{D}/{f}").convert("RGB")).astype(np.float32); m = np.array(Image.open(f"{SUP}/{f}"), np.uint16); fr = (m >= 10000) & (m != 65535); tr = (m < 10000) & (m != 65535)
        im[tr] = 0.75 * im[tr] + 0.25 * np.array([0, 90, 255]); im[fr] = 0.2 * im[fr] + 0.8 * np.array([255, 0, 220]); pil = Image.fromarray(im.astype(np.uint8)); d = ImageDraw.Draw(pil)
        for u, v in conf["finetuned"].get(f, []): d.ellipse([u - 14, v - 14, u + 14, v + 14], outline=(0, 235, 255), width=2)
        for u, v in conf["stock"].get(f, []): d.ellipse([u - 10, v - 10, u + 10, v + 10], outline=(255, 255, 0), width=2)
        cap = f"{f} {lab} supervision: magenta = fruit label ({int(fr.sum()):,} px), blue tint = tree; cyan = 3D-confirmed (fine-tuned), yellow = 3D-confirmed (stock)"; d.rectangle([0, 0, d.textlength(cap, font=F) + 12, 30], fill=(0, 0, 0)); d.text((6, 4), cap, font=F, fill=(255, 255, 255)); tiles.append(pil)
sheet = Image.new("RGB", (2 * 1280 + 8, 2 * 720 + 8), (0, 0, 0))
for i, t in enumerate(tiles): sheet.paste(t, ((i % 2) * 1288, (i // 2) * 728))
out = "/home/paperspace/data/demo_video_v2/sam3_fewshot/supervision_old_vs_new.jpg"; sheet.save(out, quality=90); print("[sup] sheet ->", out, pick)
