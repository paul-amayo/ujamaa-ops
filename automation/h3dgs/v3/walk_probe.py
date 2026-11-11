"""walk_probe.py (2026-11-11, Paul: "if you geodesic walk the feature to row level and decode, wouldn't that work?"). Citrus 05 cell 1_0.
Address level: each tree's g(t) walked to radius r (rescaled tangent vector = point at geodesic distance r from the origin), decoded, cosine
to the row anchors, for r = each row's own radius and fixed radii. Pixel level (void-free maps of /tmp/d36void/maps, held-out protocol frames):
row IoU under the argmax for the decoded own point, walk-to-the-query's-radius, containment and the tangent cosine."""
import json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
from high_splat_hierarchy_accuracy import build_hyper_embedder
import lorentz as L
N = Path('/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/v3/chunk_1_0/identity'); dev = 'cuda'
hyper = build_hyper_embedder(json.load(open(N / 'features_B.bin.json'))['embedder'], dev); curv = hyper.curv.exp()
bank = np.load(N / 'text_bank.npz', allow_pickle=True); words = list(bank['words']); meta = json.loads(str(bank['meta']))
raw = torch.tensor(bank['emb'], dtype=torch.float32, device=dev); anc = raw / raw.norm(dim=-1, keepdim=True)
H = json.load(open(meta['hierarchy_json'])); row_of = {int(o['id']): int(o['row_id']) for o in H['objects']}
wt = {int(k): w for k, w in meta['word_table'].items()}; rw = {int(k): w for k, w in meta['row_words'].items()}
rows = [r for r in sorted(rw) if rw[r] in words]; trees = [t for t in sorted(wt) if t < 10000 and wt[t] in words and row_of.get(t) in rows]
dec = lambda v: (lambda d: d / d.norm(dim=-1, keepdim=True))(hyper.decode_features(L.exp_map0(v, curv=curv), project=True))
with torch.no_grad():
    gT = hyper.encode_features(raw[[words.index(wt[t]) for t in trees]], project=False); gR = hyper.encode_features(raw[[words.index(rw[r]) for r in rows]], project=False)
    aR = anc[[words.index(rw[r]) for r in rows]]; rR = gR.norm(dim=-1); nT = gT / gT.norm(dim=-1, keepdim=True)
    own = torch.tensor([rows.index(row_of[t]) for t in trees], device=dev)
    print(f'row radii {[round(x, 2) for x in rR.tolist()]}; tree radius median {gT.norm(dim=-1).median():.2f}')
    def rep(name, S):
        so = S.gather(1, own[:, None])[:, 0]; Sm = S.clone(); Sm.scatter_(1, own[:, None], -9)
        print(f'  {name:34s} own row wins {(S.argmax(1) == own).float().mean():.2f}   own {so.median():.3f} vs best other {Sm.max(1).values.median():.3f}')
    S = torch.stack([dec(nT * rR[j]) @ aR[j] for j in range(len(rows))], 1); rep('walk to each row\'s own radius', S)
    for r in (1.0, 2.0, 2.78, 3.5, 4.78):
        rep(f'walk to radius {r}', dec(nT * r) @ aR.T)
    # pixel level
    SUP = Path('/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs_native/chunk_1_0_sam3/supervision/trees_only')
    ev = set(json.load(open(N / 'split_names.json'))['eval']); M = Path('/tmp/d36void/maps'); tdir = gR / rR[:, None]
    res = {k: [] for k in ('decoded own', 'walk to query radius', 'containment', 'tangent cosine')}; nf = 0
    for fr in json.load(open(N / 'frames.json'))['frames']:
        if fr not in ev or not (M / fr.replace('.png', '.npz')).exists(): continue
        nf += 1; z = np.load(M / fr.replace('.png', '.npz')); feat = torch.tensor(z['features'], dtype=torch.float32, device=dev); Hh, Ww, D = feat.shape; ft = feat.reshape(-1, D)
        void = (ft.norm(dim=-1) < 0.5).cpu().numpy(); I = np.array(Image.open(SUP / fr).resize((Ww, Hh), Image.NEAREST)).astype(np.int64).reshape(-1)
        out = {k: [] for k in res}
        for c0 in range(0, ft.shape[0], 32768):
            f = ft[c0:c0 + 32768]; n = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-6); h = L.exp_map0(f, curv=curv)
            d0 = hyper.decode_features(h, project=True); d0 = d0 / d0.norm(dim=-1, keepdim=True); out['decoded own'].append(d0 @ aR.T)
            out['walk to query radius'].append(torch.stack([dec(n * rR[j]) @ aR[j] for j in range(len(rows))], 1))
            path, mask = L.get_interpolated_hyperbolic_features(h, steps=4, curv=curv, max_dist=11.1, return_mask=True)
            d = hyper.decode_features(path, project=True); d = d / d.norm(dim=-1, keepdim=True)
            sm = (d.view(f.shape[0], 4, -1) @ aR.T).masked_fill(mask.to(dev).bool().view(f.shape[0], 4)[..., None], -1.0).max(1).values
            out['containment'].append(torch.maximum(d0 @ aR.T, sm)); out['tangent cosine'].append(n @ tdir.T)
        for k, v in out.items():
            am = np.array(rows)[torch.cat(v).argmax(1).cpu().numpy()]; am[void] = -1
            for r in rows:
                t = np.isin(I, [x for x, rr in row_of.items() if rr == r])
                if t.mean() < 0.002: continue
                p = am == r; u = (p | t).sum(); res[k].append(float((p & t).sum() / u) if u else 0.0)
    print(f'pixel level, {nf} held-out frames, row IoU median (n):')
    for k, v in res.items(): print(f'  {k:22s} {np.median(v):.3f} ({len(v)})')
