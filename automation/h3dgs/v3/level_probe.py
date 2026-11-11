import json, sys
"""level_probe.py (2026-11-11, Paul: "how does the tangent cosine explain levels; why did row decoding get 0"). On citrus 05 cell 1_0,
compares each tree address with the row addresses three ways: tangent direction cos(g(tree), g(row)), anchor cos(t_tree, t_row), and the
decoded own point cos(h(exp(g(tree))), t_row). nerf_new pixi env, no arguments."""
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
from high_splat_hierarchy_accuracy import build_hyper_embedder
import lorentz as L
N = Path('/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/v3/chunk_1_0/identity'); dev = 'cuda'
hyper = build_hyper_embedder(json.load(open(N / 'features_B.bin.json'))['embedder'], dev); curv = hyper.curv.exp()
bank = np.load(N / 'text_bank.npz', allow_pickle=True); words = list(bank['words']); meta = json.loads(str(bank['meta']))
raw = torch.tensor(bank['emb'], dtype=torch.float32, device=dev); anc = raw / raw.norm(dim=-1, keepdim=True)
H = json.load(open(meta['hierarchy_json'])); row_of = {int(o['id']): int(o['row_id']) for o in H['objects']}
wt = {int(k): w for k, w in meta['word_table'].items()}; rw = {int(k): w for k, w in meta['row_words'].items()}
trees = [t for t in sorted(wt) if t < 10000 and wt[t] in words and row_of.get(t) in rw and rw[row_of[t]] in words]
rows = [r for r in sorted(rw) if rw[r] in words]
with torch.no_grad():
    gT = hyper.encode_features(raw[[words.index(wt[t]) for t in trees]], project=False)
    gR = hyper.encode_features(raw[[words.index(rw[r]) for r in rows]], project=False)
    aT = anc[[words.index(wt[t]) for t in trees]]; aR = anc[[words.index(rw[r]) for r in rows]]
    dT = hyper.decode_features(L.exp_map0(gT, curv=curv), project=True); dT = dT / dT.norm(dim=-1, keepdim=True)
own = torch.tensor([rows.index(row_of[t]) for t in trees], device=dev)
def rep(name, S):
    am = S.argmax(1); acc = (am == own).float().mean().item()
    so = S.gather(1, own[:, None])[:, 0]; Sm = S.clone(); Sm.scatter_(1, own[:, None], -9); oth = Sm.max(1).values
    cnt = torch.bincount(am, minlength=len(rows)).tolist()
    print(f'{name:38s} own row wins {acc:5.2f}   own-row score {so.median():.3f} vs best other {oth.median():.3f}   argmax row histogram {cnt}')
print(f'{len(trees)} trees, {len(rows)} rows')
print(f'tangent radius |g(t)|: rows median {gR.norm(dim=-1).median():.2f}  trees median {gT.norm(dim=-1).median():.2f}')
nT = gT / gT.norm(dim=-1, keepdim=True); nR = gR / gR.norm(dim=-1, keepdim=True)
rep('tangent direction cos(g(tree), g(row))', nT @ nR.T)
rep('anchor cos(t_tree, t_row)', aT @ aR.T)
rep('decoded own cos(h(g(tree)), t_row)', dT @ aR.T)
