#!/usr/bin/env python3
"""sites_pick_tangent.py (2026-11-11, letter v39 Fig. 3) — picks the robot-survey panels of Fig. 3 under the rule fixed in the lab notebook
before any image was seen: readout = the letter's encoder path (protocol_tangent_vf.jsonl); candidates = held-out TREE heads >= 3 % of the
frame whose argmax IoU lies within --tol of the survey median over those heads; every candidate frame is rendered void-free and the sharpest is
shown (variance of the Laplacian of the rendered grey image inside the head's supervision bounding box). Writes <fig dir>/sites/picks.json and
<survey>_all.npz (rgb, void, meta, one tangent-cosine map per tree word) for `sites_strip.py --keep-picks --stage 2`.
nerf_new pixi env:  sites_pick_tangent.py <fig dir> --surveys a,b,c [--tol 0.03]"""
import argparse, json, subprocess, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
from high_splat_hierarchy_accuracy import build_hyper_embedder
ap = argparse.ArgumentParser(); ap.add_argument('fig'); ap.add_argument('--surveys', required=True); ap.add_argument('--tol', type=float, default=0.03)
ap.add_argument('--render', default='/home/paperspace/logs/sec4/bin/splat_feature_render.py'); ap.add_argument('--tmp', default='/mnt/shared/voidfree_tmp/sites'); a = ap.parse_args()
FIG = Path(a.fig) / 'sites'; FIG.mkdir(parents=True, exist_ok=True); dev = 'cuda'
inv = [json.loads(l) for l in open('/home/paperspace/logs/identity_inventory_v2.jsonl')]; SUP = {c['identity']: Path(c['sup']) for c in inv}
def render(N, frames, out):
    N = Path(N); cfg = json.load(open(N / 'W.json'))['config']; sp = N.parent / 'sparse' / '0'
    if not (sp / 'images.bin').exists(): sp = Path(cfg).parents[3] / 'sparse' / '0'
    if all((Path(out) / f.replace('.png', '.npz')).exists() for f in frames): return
    cmd = ['/home/paperspace/.pixi/bin/pixi', 'run', 'python', a.render, '--config', cfg, '--features', str(N / 'features_B.bin'), '--frames', *frames, '--out-dir', str(out), '--colmap-sparse', str(sp), '--void-free', '0.5']
    r = subprocess.run(cmd, cwd='/home/paperspace/code/nerf_new', capture_output=True, text=True)
    if r.returncode: sys.exit(f'[pick] render failed for {N}: {r.stderr[-400:]}')
def lap_var(g):
    L = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]; return float(L.var())
picks = []
for sv in a.surveys.split(','):
    heads = []
    for c in [c for c in inv if c['survey'] == sv]:
        p = Path(c['identity']) / 'protocol_tangent_vf.jsonl'
        if not p.exists(): continue
        for l in open(p):
            h = json.loads(l)
            if h.get('split') == 'eval' and h.get('kind') == 'TREE' and h.get('gt_frac', 0) >= 0.03: h['identity'] = c['identity']; heads.append(h)
    med = float(np.median([h['argmax']['iou'] for h in heads])); cands = [h for h in heads if abs(h['argmax']['iou'] - med) <= a.tol]
    by = defaultdict(set)
    for h in cands: by[h['identity']].add(h['frame'])
    scored = []
    for N, frs in by.items():
        out = Path(a.tmp) / N.replace('/home/paperspace/data/', '').replace('/', '__'); render(N, sorted(frs), out)
        for h in [h for h in cands if h['identity'] == N]:
            z = np.load(out / h['frame'].replace('.png', '.npz')); rgb = z['rgb']; H, W = rgb.shape[:2]
            ids = np.array(Image.fromarray(np.array(Image.open(SUP[N] / h['frame']))).resize((W, H), Image.NEAREST)); t = ids == int(h['id'])
            ys, xs = np.where(t); g = rgb.astype(float).mean(-1) / 255; sh = lap_var(g[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
            scored.append((sh, h, out))
    sh, h, out = max(scored, key=lambda x: x[0])
    print(f"[pick] {sv}: {len(heads)} heads >= 3 %, median {med:.3f}; {len(cands)} within {a.tol} ({len(by)} cells); sharpest {h['frame']} id {h['id']} IoU {h['argmax']['iou']:.3f}, "
          f"sharpness {sh:.5f} (candidates {min(s for s, _, _ in scored):.5f}-{max(s for s, _, _ in scored):.5f})", flush=True)
    N = Path(h['identity']); z = np.load(out / h['frame'].replace('.png', '.npz')); f = torch.tensor(z['features'], dtype=torch.float32, device=dev); Hh, Ww, D = f.shape; f = f.reshape(-1, D)
    hyper = build_hyper_embedder(json.load(open(N / 'features_B.bin.json'))['embedder'], dev)
    bank = np.load(N / 'text_bank.npz', allow_pickle=True); words = list(bank['words']); meta = json.loads(str(bank['meta'])); wt = {int(k): w for k, w in meta['word_table'].items()}
    tw = [w for k, w in sorted(wt.items()) if k < 10000 and w in words]
    with torch.no_grad():
        g_ = hyper.encode_features(torch.tensor(bank['emb'], dtype=torch.float32, device=dev)[[words.index(w) for w in tw]], project=False); g_ = g_ / g_.norm(dim=-1, keepdim=True)
        S = (f / f.norm(dim=-1, keepdim=True).clamp(min=1e-6)) @ g_.T
    void = (f.norm(dim=-1) < 0.5).view(Hh, Ww).cpu().numpy()
    np.savez_compressed(FIG / f'{sv}_all.npz', rgb=z['rgb'], void=void, meta=json.dumps(meta), **{w: S[:, i].view(Hh, Ww).cpu().numpy().astype(np.float16) for i, w in enumerate(tw)})
    picks.append({'survey': sv, 'identity': h['identity'], 'frame': h['frame'], 'word': h['word'], 'id': int(h['id']), 'argmax_iou': h['argmax']['iou'], 'median': med, 'n': len(heads),
                  'gt_frac': h['gt_frac'], 'readout': 'tangent_vf', 'rule': f'within {a.tol} of the median, sharpest render (Laplacian variance in the head box)', 'sharpness': sh, 'n_candidates': len(cands)})
json.dump(picks, open(FIG / 'picks.json', 'w'), indent=1); print('[pick] wrote', FIG / 'picks.json')
