"""CPU: project the cabbage clusters (42 global ids; 15 kept by the hierarchy's min-members 10) into frames of
IMG_7993_s0 and draw them over the photo, with the frame's SAM3 cabbage-mask count, to judge which count is real."""
import json, numpy as np
from PIL import Image, ImageDraw
D = "/home/paperspace/data/image_farm/gwakungu/2026-05-16/IMG_7993_s0_cabbage"
g = json.load(open(f"{D}/prod/bateleur/sam3_v2/global_ids.json"))
kept = {o["id"] for o in json.load(open(f"{D}/scene_graph/marker_hierarchy.json"))["objects"]}
C = {int(k): (np.array(v["world_centroid"]), v["n_members"]) for k, v in g["stats"].items()}
T = json.load(open(f"{D}/transforms.json")); fx, fy, cx, cy = T["fl_x"], T["fl_y"], T["cx"], T["cy"]
lio = {r["image_name"]: np.array(r["transform"], float) for r in json.load(open(f"{D}/lio_image_poses.json"))}
fe = json.load(open(f"{D}/prod/bateleur/sam3_v2/clip_000/frame_entries.json"))
local2name = {f["local_idx"]: (f.get("image_name") or f"image_{f.get('image_idx')}.png") for f in fe["frames"]}
nmask = {local2name.get(int(k)): len(v) for k, v in fe["frame_entries"].items()}
names = sorted(lio, key=lambda n: int(n.split("_")[1].split(".")[0]))
pick = [names[i] for i in np.linspace(3, len(names) - 4, 6).astype(int)]
tiles = []
for n in pick:
    c2w = lio[n]; w2c = np.linalg.inv(c2w)
    im = Image.open(f"{D}/images/{n}").convert("RGB"); dr = ImageDraw.Draw(im)
    vis_k = vis_d = 0
    for gid, (p, nm) in C.items():
        q = w2c[:3, :3] @ p + w2c[:3, 3]
        if q[2] <= 0.2: continue
        u, v = fx * q[0] / q[2] + cx, fy * q[1] / q[2] + cy
        if not (0 <= u < im.width and 0 <= v < im.height): continue
        k = gid in kept; vis_k += k; vis_d += (not k)
        col = (40, 220, 90) if k else (255, 150, 0); r = 22
        dr.ellipse([u - r, v - r, u + r, v + r], outline=col, width=6); dr.text((u + r + 4, v - r), f"{gid}({nm})", fill=col)
    dr.rectangle([0, 0, im.width, 70], fill=(0, 0, 0))
    dr.text((10, 10), f"{n}: SAM3 cabbage masks {nmask.get(n, 0)} | projected kept {vis_k} (green) dropped {vis_d} (orange)", fill=(255, 255, 255))
    im.thumbnail((540, 960)); tiles.append(im)
W, H = tiles[0].size; sheet = Image.new("RGB", (W * 3, H * 2))
for i, t in enumerate(tiles): sheet.paste(t, ((i % 3) * W, (i // 3) * H))
sheet.save("/tmp/appcheck/cabbage_registry_check.jpg", quality=88)
print("members per id (sorted):", sorted((nm for _, nm in C.values()), reverse=True))
print("kept ids", sorted(kept))
