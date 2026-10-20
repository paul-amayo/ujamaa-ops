#!/usr/bin/env python3
"""super_eval.py (2026-10-20, CVPR super hierarchy, the decisive test) — the composed query "trunk of tree k" on the held-out frames of a cell.
Super field (parts/maps_super, the super embedder): tree-level argmax by containment over the cell's tree addresses (as argmax_eval.py),
then, inside the pixels of tree k, the part level read at the pixel's own point: (a) NODE path, argmax by cosine among tree k's part nodes
(image anchors) and the tree's own address, the pixel is "trunk of tree k" when that argmax is k's trunk node; (b) WORD path, LERF relevancy
of an open word ("trunk", and the unprompted "bark", "stem") >= 0.5 against the canonical negatives. Reference: the union of the SAM 3
"tree trunk" masks of tree k on that frame (parts/verdict). Comparators on the same heads and reference: HiGH as published (maps_v3 + its
embedder; tree argmax from its own field, "trunk" by the word path, no part nodes exist) and the LangSplat-style field (arms/maps_lite,
"trunk" relevancy >= 0.5 over the whole frame, no identity). Heads: every (held-out frame, tree) with a trunk mask; the main number is the
median IoU over heads whose trunk covers >= --min-frac of the frame; instance recall (any predicted pixel inside the trunk) beside it.
nerf_new pixi env:  super_eval.py --identity <N> --parts <P> --emb-super <ckpt> [--out <json>] [--dump <dir>] [--min-frac 0.002]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, open_clip
from PIL import Image
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH')
from high_splat_hierarchy_accuracy import build_hyper_embedder
import lorentz as L
ap = argparse.ArgumentParser(); ap.add_argument('--identity', required=True); ap.add_argument('--parts', required=True); ap.add_argument('--emb-super', required=True); ap.add_argument('--out', default=''); ap.add_argument('--dump', default='')
ap.add_argument('--min-frac', type=float, default=0.002); ap.add_argument('--words', default='trunk,bark,stem'); ap.add_argument('--rel', type=float, default=0.5); ap.add_argument('--max-dist', type=float, default=11.1); ap.add_argument('--maps-dir', default='maps_super'); a = ap.parse_args()
N, P = Path(a.identity), Path(a.parts); dev = 'cuda'; torch.set_grad_enabled(False); VOID, MAXD, STEPS, FRUIT, PART = 0.5, a.max_dist, 4, 10000, 20000
bank = np.load(N / 'text_bank.npz', allow_pickle=True); meta = json.loads(str(bank['meta'])); wt = {int(k): w for k, w in meta['word_table'].items()}; words = [str(w) for w in bank['words']]; sup = Path(meta['manifest']).parent
E = torch.tensor(bank['emb'], dtype=torch.float32, device=dev); E = E / E.norm(dim=-1, keepdim=True); tree_ids = sorted(k for k in wt if k < FRUIT and wt[k] in words); T_cols = torch.tensor([words.index(wt[k]) for k in tree_ids], device=dev)
nz = np.load(P / 'nodes.npz'); node_id = [int(i) for i in nz['ids']]; node_tree = [int(t) for t in nz['tree']]; node_type = [int(t) for t in nz['type']]; NA = torch.tensor(np.asarray(nz['emb'], np.float32), device=dev); NA = NA / NA.norm(dim=-1, keepdim=True)
clip, _, _ = open_clip.create_model_and_transforms('ViT-B-16', 'laion2b_s34b_b88k', device=dev); tok = open_clip.get_tokenizer('ViT-B-16')
enc = lambda ws: torch.nn.functional.normalize(clip.encode_text(tok(ws).to(dev)).float(), dim=-1)
WORDS = [w.strip() for w in a.words.split(',')]; EW = enc(WORDS); ENEG = enc(['object', 'things', 'stuff', 'texture'])
rel = lambda cp, cn: torch.sigmoid(10.0 * (cp.unsqueeze(-1) - cn.unsqueeze(-2))).min(-1).values   # [..., w] relevancy, LERF
nrm = lambda x: x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
hyp_s = build_hyper_embedder(a.emb_super, dev); hyp_v = build_hyper_embedder(json.load(open(N / 'features_B.bin.json'))['embedder'], dev)
lb = np.load(N / 'arms/lite_targets/basis.npz'); LC = torch.tensor(lb['components'], dtype=torch.float32, device=dev); LM = torch.tensor(lb['mean'], dtype=torch.float32, device=dev)
def decode(hyper, ft):
    """own-point decoded unit features [P, 512], waypoint decoded [P, STEPS, 512] (masked rows = 0), void [P]"""
    curv = hyper.curv.exp(); void = ft.norm(dim=-1) < VOID; own = torch.zeros(ft.shape[0], 512, device=dev); path = torch.zeros(ft.shape[0], STEPS, 512, device=dev)
    for j in range(0, ft.shape[0], 8192):
        sl = ft[j:j + 8192]; h = L.exp_map0(sl, curv=curv); own[j:j + 8192] = nrm(hyper.decode_features(h, project=True))
        pth, msk = L.get_interpolated_hyperbolic_features(h, steps=STEPS, curv=curv, max_dist=MAXD, return_mask=True)
        path[j:j + 8192] = nrm(hyper.decode_features(pth, project=True)).view(sl.shape[0], STEPS, -1).masked_fill(msk.to(dev).bool().unsqueeze(-1), 0.0)
    return own, path, void
def tree_argmax(own, path, void):
    S = torch.maximum((path @ E[T_cols].T).max(1).values, own @ E[T_cols].T); S[void] = -1.0; return torch.tensor(tree_ids, device=dev)[S.argmax(1)], S
ev = set(json.load(open(N / 'split_names.json'))['eval']); frames = [f for f in json.load(open(N / 'frames.json'))['frames'] if f in ev]
def prf(pred, truth):
    i = (pred & truth).sum().item(); u = (pred | truth).sum().item(); return (i / u if u else 0.0, i / max(pred.sum().item(), 1), i / max(truth.sum().item(), 1), bool(i > 0))
heads = []; dump = Path(a.dump) if a.dump else None
if dump: dump.mkdir(parents=True, exist_ok=True)
for f in frames:
    stem = f.replace('.png', ''); zs = np.load(P / a.maps_dir / f'{stem}.npz'); fs = torch.tensor(zs['features'], dtype=torch.float32, device=dev); H, W = fs.shape[:2]; fs = fs.reshape(-1, fs.shape[-1]); Px = H * W
    pj = P / 'verdict' / f'{stem}.json'
    if not pj.exists(): continue
    recs = json.load(open(pj)); zp = np.load(P / 'verdict' / f'{stem}.npz'); Hf, Wf = zp['shape']; M = np.unpackbits(zp['masks'], axis=1)[:, :Hf * Wf].reshape(-1, Hf, Wf).astype(bool)
    trunk_of = {}
    for i, r in enumerate(recs):
        if r['prompt'] == 'tree trunk': trunk_of.setdefault(r['tree'], np.zeros((Hf, Wf), bool)); trunk_of[r['tree']] |= M[i]
    if not trunk_of: continue
    ids = np.array(Image.fromarray(np.array(Image.open(sup / f))).resize((W, H), Image.NEAREST))
    own_s, path_s, void_s = decode(hyp_s, fs); lab_s, _ = tree_argmax(own_s, path_s, void_s); live_s = ~void_s
    rel_s = rel(own_s @ EW.T, own_s @ ENEG.T); rel_s[void_s] = 0
    zv = np.load(N / 'maps_v3' / f'{stem}.npz'); fv = torch.tensor(zv['features'], dtype=torch.float32, device=dev).reshape(-1, 32); own_v, path_v, void_v = decode(hyp_v, fv); lab_v, _ = tree_argmax(own_v, path_v, void_v); rel_v = rel(own_v @ EW.T, own_v @ ENEG.T); rel_v[void_v] = 0
    zl = np.load(N / 'arms/maps_lite' / f'{stem}.npz'); fl = torch.tensor(zl['features'], dtype=torch.float32, device=dev).reshape(-1, 32); void_l = fl.norm(dim=-1) < VOID; own_l = nrm(fl @ LC + LM); rel_l = rel(own_l @ EW.T, own_l @ ENEG.T); rel_l[void_l] = 0
    part_pred = torch.full((Px,), -1, dtype=torch.long, device=dev)   # node id of the part-level argmax within the pixel's tree, -2 = the tree itself
    for k in set(lab_s[live_s].tolist()):
        sel = (lab_s == k) & live_s; cand = [j for j in range(len(node_id)) if node_tree[j] == k]
        if not cand: part_pred[sel] = -2; continue
        C = torch.cat([NA[cand], E[words.index(wt[k])][None]], 0); sc = own_s[sel] @ C.T; am = sc.argmax(1); ids_c = torch.tensor([node_id[j] for j in cand] + [-2], device=dev); part_pred[sel] = ids_c[am]
    for k, tm in trunk_of.items():
        truth = torch.from_numpy(np.array(Image.fromarray(tm).resize((W, H), Image.NEAREST))).to(dev).reshape(-1); frac = truth.sum().item() / Px
        trunk_node = next((node_id[j] for j in range(len(node_id)) if node_tree[j] == k and node_type[j] == 0), None)
        rec = {'frame': f, 'tree': int(k), 'trunk_px': int(truth.sum().item()), 'trunk_frac': frac, 'tree_frac': float((ids == k).mean()), 'has_trunk_node': trunk_node is not None, 'n_part_nodes': sum(1 for j in range(len(node_id)) if node_tree[j] == k)}
        in_k = (lab_s == k) & live_s; rec['super_node'] = prf(in_k & (part_pred == trunk_node), truth) if trunk_node is not None else (0.0, 0.0, 0.0, False)
        tt = truth.bool(); rec['trunk_px_live'] = float(live_s[tt].float().mean().item()); rec['trunk_px_tree_k'] = float(in_k[tt].float().mean().item())
        pk = part_pred[tt & in_k]; rec['trunk_px_part_argmax'] = {'trunk': float((pk == trunk_node).float().mean().item()) if trunk_node is not None and len(pk) else 0.0, 'tree': float((pk == -2).float().mean().item()) if len(pk) else 0.0, 'other_part': float(((pk != -2) & (pk != (trunk_node if trunk_node is not None else -9))).float().mean().item()) if len(pk) else 0.0}
        for wi, w in enumerate(WORDS): rec[f'super_word_{w}'] = prf(in_k & (rel_s[:, wi] >= a.rel), truth); rec[f'v3_word_{w}'] = prf((lab_v == k) & ~void_v & (rel_v[:, wi] >= a.rel), truth); rec[f'lite_word_{w}'] = prf(~void_l & (rel_l[:, wi] >= a.rel), truth)
        rec['tree_iou_super'] = prf(in_k, torch.from_numpy(ids == k).to(dev).reshape(-1))[0]; rec['tree_iou_v3'] = prf((lab_v == k) & ~void_v, torch.from_numpy(ids == k).to(dev).reshape(-1))[0]
        heads.append(rec); print(f"[super] {f} tree {k}: trunk {100 * frac:.2f} % of frame | trunk px live {rec['trunk_px_live']:.2f}, in tree k {rec['trunk_px_tree_k']:.2f}, part argmax trunk/tree/other {rec['trunk_px_part_argmax']['trunk']:.2f}/{rec['trunk_px_part_argmax']['tree']:.2f}/{rec['trunk_px_part_argmax']['other_part']:.2f} | node {rec['super_node'][0]:.3f} (P {rec['super_node'][1]:.2f} R {rec['super_node'][2]:.2f}) | word trunk super {rec['super_word_trunk'][0]:.3f} v3 {rec['v3_word_trunk'][0]:.3f} lite {rec['lite_word_trunk'][0]:.3f} | tree IoU super {rec['tree_iou_super']:.3f} v3 {rec['tree_iou_v3']:.3f}" + ('' if trunk_node is not None else ' | NO TRUNK NODE'), flush=True)
    if dump: np.savez_compressed(dump / f'{stem}.npz', tree_super=lab_s.view(H, W).cpu().numpy().astype(np.int32), live_super=live_s.view(H, W).cpu().numpy(), part_super=part_pred.view(H, W).cpu().numpy().astype(np.int32), rel_super=rel_s.view(H, W, -1).cpu().numpy(), tree_v3=lab_v.view(H, W).cpu().numpy().astype(np.int32), live_v3=(~void_v).view(H, W).cpu().numpy(), rel_v3=rel_v.view(H, W, -1).cpu().numpy(), rel_lite=rel_l.view(H, W, -1).cpu().numpy(), live_lite=(~void_l).view(H, W).cpu().numpy(), words=np.array(WORDS), rgb=zs['rgb'] if 'rgb' in zs.files else np.zeros(0))
med = lambda v: float(np.median(v)) if len(v) else float('nan'); keys = ['super_node'] + [f'{p}_word_{w}' for p in ('super', 'v3', 'lite') for w in WORDS]
out = {'frames': len(frames), 'heads': len(heads), 'min_frac': a.min_frac, 'rel': a.rel, 'per_head': heads, 'summary': {}}
for name, hs in (('all', heads), ('main', [h for h in heads if h['trunk_frac'] >= a.min_frac])):
    out['summary'][name] = {'n': len(hs), 'no_trunk_node': sum(1 for h in hs if not h['has_trunk_node'])}
    for k in keys: out['summary'][name][k] = {'iou_med': med([h[k][0] for h in hs]), 'prec_med': med([h[k][1] for h in hs]), 'rec_med': med([h[k][2] for h in hs]), 'inst_recall': float(np.mean([h[k][3] for h in hs])) if hs else float('nan')}
    out['summary'][name]['tree_iou_super_med'] = med([h['tree_iou_super'] for h in hs]); out['summary'][name]['tree_iou_v3_med'] = med([h['tree_iou_v3'] for h in hs])
    print(f"[super] {name} heads n={len(hs)} (no trunk node: {out['summary'][name]['no_trunk_node']}): " + '; '.join(f"{k} IoU {out['summary'][name][k]['iou_med']:.3f} P {out['summary'][name][k]['prec_med']:.2f} R {out['summary'][name][k]['rec_med']:.2f} inst-recall {out['summary'][name][k]['inst_recall']:.2f}" for k in keys) + f"; tree IoU on these heads super {out['summary'][name]['tree_iou_super_med']:.3f} v3 {out['summary'][name]['tree_iou_v3_med']:.3f}")
if a.out: json.dump(out, open(a.out, 'w'), indent=1); print('[super] wrote', a.out)
