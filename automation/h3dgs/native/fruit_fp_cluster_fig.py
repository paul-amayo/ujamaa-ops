#!/usr/bin/env python3
"""CPU redraw of fruit_fp_trace.py's cluster figure with per-view captions measured on the cluster itself (2026-10-04: the first
draw captioned every origin view "sit inside SAM3 fruit", but this cluster's fruit credit comes from kf_000029 and kf_000027
only; kf_000023 gives it none). Inputs: fp_trace/fruit_fp_trace_nodes.npz, the expo chunk 1_0 COLMAP poses (= the H3DGS cameras),
the cell's supervision, the kf_000025 verdict map. -> fp_trace/fruit_fp_cluster.jpg"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians/preprocess')
from read_write_model import read_images_binary, read_cameras_binary, qvec2rotmat
S = '/home/paperspace/data/citrus_all/05_13D_Jackal'; E = f'{S}/experimental/h3dgs_expo'; SP = f'{E}/camera_calibration/chunks/1_0/sparse/0'
NB = f'{S}/experimental/h3dgs_native/chunk_1_0_sam3'; SUP = f'{NB}/supervision/trees_only'; PH = f'{E}/camera_calibration/rectified/images'
D = '/home/paperspace/data/demo_video_v2/citrus_b_cut_trained'; z = np.load(f'{D}/fp_trace/fruit_fp_trace_nodes.npz')
xyz, red, fsh, views, fbv = z['xyz'], z['red'], z['fshare'], list(z['views']), z['fruit_by_view']
kd = cKDTree(xyz); mass = np.array([red[kd.query_ball_point(xyz[j], 0.3)].sum() for j in range(len(xyz))]); cl = np.array(kd.query_ball_point(xyz[mass.argmax()], 0.3))
credit = {v: float(fbv[i, cl].sum()) for i, v in enumerate(views)}; tot = sum(credit.values())
ims = {im.name: im for im in read_images_binary(f'{SP}/images.bin').values()}; fx, fy, cx, cy = list(read_cameras_binary(f'{SP}/cameras.bin').values())[0].params[:4]
v25 = np.load(f'{D}/verdicts/kf_000025.png.npz')['v']
Fb = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22); Fs = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 21)
CW, CH, Z = 200, 120, 4; tiles = []
for name in ('kf_000023.png', 'kf_000025.png', 'kf_000027.png', 'kf_000029.png'):
    im = ims[name]; R = qvec2rotmat(im.qvec); pc = (R @ xyz[cl].T + im.tvec[:, None]).T; u = fx * pc[:, 0] / pc[:, 2] + cx; vv = fy * pc[:, 1] / pc[:, 2] + cy
    img = np.array(Image.open(f'{PH}/{name}').convert('RGB')).astype(np.float32); H, W = img.shape[:2]
    s5 = np.array(Image.open(f'{SUP}/{name}'), np.uint16) == 10001; img[s5] = 0.45 * img[s5] + 0.55 * np.array([0, 235, 255]); img = img.astype(np.uint8)
    if name == 'kf_000025.png': img[v25 == 4] = (255, 40, 40)
    x0 = int(np.clip(np.median(u) - CW / 2, 0, W - CW)); y0 = int(np.clip(np.median(vv) - CH / 2, 0, H - CH))
    t = Image.fromarray(img).crop((x0, y0, x0 + CW, y0 + CH)).resize((CW * Z, CH * Z), Image.NEAREST); d = ImageDraw.Draw(t)
    for j in range(cl.size):
        X, Y = (u[j] - x0) * Z, (vv[j] - y0) * Z
        if 0 <= X < CW * Z and 0 <= Y < CH * Z: d.ellipse([X - 5, Y - 5, X + 5, Y + 5], fill=(255, 0, 220), outline=(255, 255, 255))
    c = credit.get(name, 0.0)
    cap = (f'{name[3:9]}: lit as fruit on leaves (red)' if name == 'kf_000025.png' else
           f'{name[3:9]}: on SAM3 fruit, {c / tot:.0%} of their fruit credit' if c > 0 else f'{name[3:9]}: on leaves, no fruit credit')
    d.rounded_rectangle([6, 6, 18 + d.textlength(cap, font=Fb), 38], 6, fill=(11, 11, 11)); d.text((12, 9), cap, font=Fb, fill=(255, 255, 255)); tiles.append(t)
w0, h0 = tiles[0].size; HD = 140; sheet = Image.new('RGB', (2 * w0 + 12, 2 * h0 + 12 + HD), (255, 255, 255)); dd = ImageDraw.Draw(sheet)
for k, (txt, f) in enumerate(((f'The same {cl.size} Gaussians (magenta) in four training views of tree 5, 4x zoom', Fb),
        ('cyan = SAM3 fruit masks of tree 5; red = leaf pixels that kf_000025 lights as fruit', Fs),
        (f'In 027 and 029 they fall on SAM3 oranges, so the census credits them with fruit: median {np.median(fsh[cl]):.0%} of their weight ({fsh[cl].min():.0%}-{fsh[cl].max():.0%}).', Fs),
        ('Seed B makes anything above 10 % fruit a fruit (29 of these 32). From 025 they sit on leaves beside the orange: false positives.', Fs))):
    dd.text((8, 6 + 33 * k), txt, font=f, fill=(0, 0, 0))
for k, t in enumerate(tiles): sheet.paste(t, ((k % 2) * (w0 + 12), HD + (k // 2) * (h0 + 12)))
sheet.save(f'{D}/fp_trace/fruit_fp_cluster.jpg', quality=90); print('credit by view', {k: round(v, 2) for k, v in credit.items() if v > 0}, '->', f'{D}/fp_trace/fruit_fp_cluster.jpg')
