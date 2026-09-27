"""Why did a pixel get that row? Re-render ONE demo frame through the side-cars that see it and dump, per pixel, the raw
containment score of every row word, the feature norm, the accumulated alpha and the DEPTH — beside the supervision.
(Paul, 2026-09-27: "find the worst keyframe and diagnose why this is happening".)
  HIGH_EMBEDDER_CKPT=<emb> pixi run python sidecar_row_probe.py --survey 05 --path demo_path.json --frame d_00097.png
      --models 018 ... --backdrop-dir <dir> --out <dir>"""
import argparse, json, os, re, sys
from pathlib import Path
import numpy as np, torch, cv2
from PIL import Image
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH')
from high_splat_hierarchy_accuracy import build_hyper_embedder
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras, CameraType
import lorentz as L, open_clip
dev = 'cuda'; torch.set_grad_enabled(False)
ap = argparse.ArgumentParser(); ap.add_argument('--survey', required=True); ap.add_argument('--path', required=True); ap.add_argument('--frame', required=True)
ap.add_argument('--models', nargs='+', required=True); ap.add_argument('--backdrop-dir', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--seed-tag', default='glref_bg_f1.0_r2'); ap.add_argument('--verdict-tag', default='bg_f1.0_r2')
a = ap.parse_args(); S = Path('/home/paperspace/data/citrus_all') / a.survey; P = S / 'experimental/h3dgs'; OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
pj = json.load(open(a.path)); F = next(f for f in pj['frames'] if f['name'] == a.frame); K = pj['intrinsics']; sc = pj['scale']
W, H = int(round(K['w'] * sc)), int(round(K['h'] * sc))
meta = json.load(open(P / 'export_meta.json')); R_W = np.asarray(meta.get('world_rotation_to_zup') or meta['world_rotation_lio_to_h3dgs'], np.float64)[:3, :3]; GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])
VL = f'/home/paperspace/logs/sidecar_{a.survey}_verdicts.log'
def verdict_thr(block):
    out = {}
    for line in open(VL):
        if line.startswith(f'[{block} sidecar {a.verdict_tag} '):
            m = re.search(r'"([a-z]+)": thr ([0-9.]+)', line)
            if m: out[m.group(1)] = float(m.group(2))
    return out
hyper = build_hyper_embedder(os.environ['HIGH_EMBEDDER_CKPT'], dev); curv = hyper.curv.exp()
clip_m, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k'); clip_m = clip_m.to(dev).eval()
tok = open_clip.get_tokenizer('ViT-B-16')
def word_vec(ws):
    t = clip_m.encode_text(tok(list(ws)).to(dev)).float(); return t / t.norm(dim=-1, keepdim=True)
def heats(feat, E):
    Hh, Ww, D = feat.shape; ft = torch.from_numpy(np.asarray(feat).reshape(-1, D)).to(dev); void = ft.norm(dim=-1) < 0.5; out = []
    for j in range(0, ft.shape[0], 8192):
        sl = ft[j:j + 8192]; h = L.exp_map0(sl, curv=curv); path, mask = L.get_interpolated_hyperbolic_features(h, steps=4, curv=curv, max_dist=11.1, return_mask=True)
        d = hyper.decode_features(path, project=True); d = d / d.norm(dim=-1, keepdim=True); sm = (d @ E.T).view(sl.shape[0], 4, -1).masked_fill(mask.to(dev).bool().unsqueeze(-1), -1.0)
        d0 = hyper.decode_features(h, project=True); d0 = d0 / d0.norm(dim=-1, keepdim=True); out.append(torch.maximum(sm.max(1).values, d0 @ E.T))
    hm = torch.cat(out); hm[void] = -1.0
    return hm.view(Hh, Ww, -1).cpu().numpy(), void.view(Hh, Ww).cpu().numpy()
sup = None
for bd in sorted((S / 'prod/tassili/blocks_ns/lio_row100').glob('block_[0-9][0-9][0-9]')):
    own = {Path(f['file_path']).name for f in json.load(open(bd / 'transforms.json'))['frames']}
    if F['src'] in own and (bd / 'supervision/trees_only' / F['src']).exists(): sup = bd / 'supervision/trees_only' / F['src']; break
H_ = json.load(open(S / 'prod/bateleur/scene_graph/marker_hierarchy.json')); tree_row = {t: r['id'] for r in H_['rows'] for t in r['object_ids']}
allrow = sorted({w for v in pj['rows'].values() for w in v})
res = {}
for b in a.models:
    O = S / 'experimental/h3dgs_sidecar' / f'block_{b}'; run = sorted((O / 'splat_runs_FEATFIX').glob(f'stage2_censusinit_{a.seed_tag}/high/*'))[-1]
    rwords = pj['rows'].get(b, [])
    if not rwords: continue
    dp = json.load(open(run / 'dataparser_transforms.json')); T = np.asarray(dp['transform'], np.float64); s = float(dp['scale'])
    _, pipe, _, step = eval_setup(run / 'config.yml', test_mode='inference'); model = pipe.model; model.eval(); model.step = step
    c2w_h = np.asarray(F['c2w_h']); c2w_cv = np.eye(4); c2w_cv[:3, :3] = R_W.T @ c2w_h[:3, :3]; c2w_cv[:3, 3] = R_W.T @ c2w_h[:3, 3]; g = c2w_cv @ GL2CV
    M = np.zeros((3, 4)); M[:, :3] = T[:, :3] @ g[:3, :3]; M[:, 3] = s * (T[:, :3] @ g[:3, 3] + T[:, 3])
    cam = Cameras(camera_to_worlds=torch.tensor(M, dtype=torch.float32)[None], fx=K['fl_x'] * sc, fy=K['fl_y'] * sc, cx=K['cx'] * sc, cy=K['cy'] * sc,
                  width=W, height=H, camera_type=CameraType.PERSPECTIVE).to(model.device)
    model.image_encoder.set_positives(['__probe__']); outs = model.get_outputs_for_camera(cam)
    feat = outs['high_features'].cpu().numpy(); feat = feat[0] if feat.ndim == 4 else feat
    alpha = outs['accumulation'].squeeze(-1).cpu().numpy(); depth = outs['depth'].squeeze(-1).cpu().numpy() / s     # ns units -> metres
    hm, void = heats(feat, word_vec(rwords)); vt = verdict_thr(b)
    res[b] = dict(raw=hm, void=void, alpha=alpha, depth=depth, words=rwords, thr=np.array([vt.get(w, 0.8) for w in rwords], np.float32))
    print(f'[probe] {b}: rows {rwords} thr {res[b]["thr"].tolist()}, void {100*void.mean():.1f}% of pixels, depth p50 {np.median(depth[alpha>0.5]):.1f} m', flush=True)
    del model, pipe; torch.cuda.empty_cache()
# composite exactly as the demo does (winner = highest raw score, gate = that word's own threshold)
br = np.full((H, W), -1, np.int32); brm = np.full((H, W), -9.0, np.float32); bwin = np.full((H, W), '', dtype=object); braw = np.zeros((H, W, len(allrow)), np.float32) - 9
bdep = np.zeros((H, W), np.float32)
for b, r in res.items():
    rid = r['raw'].argmax(-1); mg = np.take_along_axis(r['raw'] - r['thr'][None, None], rid[..., None], -1)[..., 0]
    m = mg > brm; br[m] = np.array([allrow.index(w) for w in r['words']], np.int32)[rid][m]; brm[m] = mg[m]; bwin[m] = b; bdep[m] = r['depth'][m]
    for i, w in enumerate(r['words']):
        j = allrow.index(w); braw[:, :, j] = np.maximum(braw[:, :, j], r['raw'][:, :, i])
g = np.full((H, W), -1, np.int32)
if sup is not None:
    sarr = np.array(Image.open(sup), np.uint16); sarr = np.array(Image.fromarray(sarr).resize((W, H), Image.NEAREST))
    for t in [int(v) for v in np.unique(sarr) if v != 65535]:
        rr = tree_row.get(t, -3); g[sarr == t] = rr if (rr is not None and 0 <= rr < len(allrow)) else -2
sk = np.array(Image.open(S / 'prod/tassili/sky_masks' / F['src']).convert('L').resize((W, H), Image.NEAREST)) > 127
drawn = (brm >= 0) & ~sk
print(f"[probe] {F['src']}: drawn {int(drawn.sum())} px; supervision labels {int((g >= 0).sum())} px", flush=True)
for r in range(len(allrow)):
    m = drawn & (br == r)
    if not m.any(): continue
    d = bdep[m]; sep = braw[:, :, r][m] - braw[:, :, 1 - r][m] if len(allrow) == 2 else np.zeros(m.sum())
    print(f"[probe]   drawn '{allrow[r]}': {int(m.sum())} px, depth p50 {np.median(d):.1f} m p90 {np.percentile(d,90):.1f} m; "
          f"score {braw[:,:,r][m].mean():.3f}, lead over the other row {sep.mean():+.3f} (25th pct {np.percentile(sep,25):+.3f}); "
          f"supervision here: {int((g[m]==r).sum())} same row, {int(((g[m]>=0)&(g[m]!=r)).sum())} other row, {int((g[m]==-1).sum())} unlabelled", flush=True)
np.savez_compressed(OUT / f"probe_{F['src']}.npz", br=br, brm=brm, braw=braw, depth=bdep, g=g, sky=sk)
img = cv2.imread(str(Path(a.backdrop_dir) / F['name'])); img = cv2.resize(img, (W, H)); img[sk] = 0
cols = [(0, 220, 255), (255, 200, 0)]
dv = img.copy()
for r in range(len(allrow)):
    m = drawn & (br == r); dv[m] = (0.45 * dv[m] + 0.55 * np.array(cols[r % 2])).astype(np.uint8)
lead = np.clip((braw[:, :, 0] - braw[:, :, 1]) * 6 + 0.5, 0, 1) if len(allrow) == 2 else np.zeros((H, W))
lv = cv2.applyColorMap((lead * 255).astype(np.uint8), cv2.COLORMAP_COOL); lv[sk] = 0
dd = np.clip(bdep / 40.0, 0, 1); dvz = cv2.applyColorMap((dd * 255).astype(np.uint8), cv2.COLORMAP_TURBO); dvz[sk | ~drawn] = 0
lead_txt = f"score lead: magenta = '{allrow[0]}' ahead, cyan = '{allrow[1]}' ahead, pale = the two within 0.08" if len(allrow) == 2 else 'score lead'
for im, t in ((dv, f"drawn ({F['src']})"), (lv, lead_txt), (dvz, 'depth of drawn pixels (blue 0 m -> red 40 m+)')):
    cv2.putText(im, t, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite(str(OUT / f"probe_{F['src']}"), np.concatenate([dv, lv, dvz], 0))
print(f"[probe] wrote {OUT / ('probe_' + F['src'])}", flush=True)
