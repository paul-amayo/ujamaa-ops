#!/usr/bin/env python3
"""argmax_eval.py — the paper's E1 protocol on the rendered feature maps of one identity cell (2026-10-13).
containment_eval.py scores every head with the best of 57 thresholds: an oracle per head, the upper bound. This tool keeps that
oracle as a check and adds, from ONE decode per frame, the protocols the paper reports:
  argmax    per pixel, the word with the highest score among the cell's words of that level (trees among tree words, rows among row
            words, fruit among fruit + tree words); presence is only the norm check (||f|| < 0.5 = no identity)
  curves    IoU of every head at the 57 thresholds, with and without the argmax rule -> protocol_table.py fits ONE threshold per
            level on a cell's census frames and applies it to the held-out frames (and across surveys)
  nesting   fraction of non-void pixels whose row-level argmax is the row of their tree-level argmax (cross-level consistency);
            for ROW heads also 'argmax_union' = the row as the union of its member trees' argmax pixels (scene graph, no row words)
Spaces (--space, the ablation arms of the paper): hyper = containment (exp_map0 -> 4-step walk, extrapolated steps masked, own point,
cosine to CLIP text, max; containment_eval's decode, void pixels -1); hyper_own = the pixel's own point only (arm D, what the ancestors
add); euclid = a Euclidean embedder's field decoded at its own point (arm B); flat = three PCA-32 blocks tree|row|fruit decoded by the
inverse PCA (arm C, --basis). Inputs resolve from the cell itself: text_bank.npz meta (supervision manifest -> id maps; hierarchy json),
features_B.bin.json (the embedder the seed was built against — embedder-era pairing) unless overridden. Writes
<identity>/protocol_<tag>.jsonl (one line per head + one per frame) and prints containment_eval-style lines.
nerf_new pixi env:  argmax_eval.py --identity <ws>/identity [--tag v3] [--maps maps_v3/%s.npz] [--space auto|hyper|hyper_own|euclid|flat]
                        [--basis <flat basis npz>] [--emb ckpt] [--sup dir] [--hier json] [--out jsonl] [--check-oracle]"""
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH')
from high_splat_hierarchy_accuracy import build_hyper_embedder
from word_utils import get_word_for_id
import lorentz as L
import open_clip
FRUIT_ID_BASE, UNLAB, VOID_NORM, MAX_DIST, STEPS = 10000, 65535, 0.5, 11.1, 4
THR = np.concatenate([np.linspace(0.4, 0.95, 45), np.linspace(0.955, 0.999, 12)])   # containment_eval.best_mask's sweep
ap = argparse.ArgumentParser(); ap.add_argument('--identity', required=True); ap.add_argument('--tag', default='v3'); ap.add_argument('--maps', default='maps_v3/%s.npz')
ap.add_argument('--space', default='auto', choices=['auto', 'hyper', 'hyper_own', 'euclid', 'flat', 'langsplat']); ap.add_argument('--basis', default='', help='--space flat: <features>.basis.npz from build_flat_seed.py')
ap.add_argument('--ae-ckpt', default='', help='--space langsplat: the cell autoencoder (3 -> 512) LangSplat trained'); ap.add_argument('--ls-level', type=int, default=1, help='--space langsplat: which of the 3 rendered levels to score')
ap.add_argument('--category', default='', help="category protocol (E3): 'word=GROUP,...' with GROUP in ALL (every labelled px), TREES (tree ids), FRUIT (fruit ids), e.g. tree=ALL,orange=FRUIT — scored with LERF relevancy against the canonical negatives, same for every space")
ap.add_argument('--fruit-rule', default='mixed', choices=['mixed', 'level'], help="FRUIT heads: 'mixed' = argmax over fruit + tree words (a leaf-dominated blend decodes to the tree); 'level' = the same rule as every other level — argmax among fruit words only, presence where the pixel's tangent norm sits at fruit depth (>= the midpoint between the tree and fruit target norms: depth == radius)")
ap.add_argument('--frames-json', default='', help='verdict frames to score (default <identity>/frames.json; E4 uses frames_fruit.json)')
ap.add_argument('--void-norm', type=float, default=None, help="presence check ||f|| < this = no identity (default 0.5, containment_eval's constant for hyperbolic tangents)")
ap.add_argument('--void-ref', nargs=2, default=None, metavar=('ARM_BIN', 'HYPER_BIN'), help='scale the 0.5 check to the arm: 0.5 x median nonzero seed norm of ARM_BIN / that of HYPER_BIN (fields whose targets are not tangent-sized)')
ap.add_argument('--out', default=''); ap.add_argument('--sup', default=''); ap.add_argument('--hier', default=''); ap.add_argument('--emb', default='')
ap.add_argument('--dump-pred', default='', help='write <dir>/<frame>.npz per frame: tree / fruit_level / fruit_mixed argmax ids per pixel (-1 = none) + live, for qualitative figures')
ap.add_argument('--check-oracle', action='store_true', help='compare the oracle sweep with the [<tag> frame] lines already in verdicts.log')
ap.add_argument('--clip-model', default='ViT-B-16'); ap.add_argument('--clip-pretrained', default='laion2b_s34b_b88k')
a = ap.parse_args(); N = Path(a.identity); dev = 'cuda'; torch.set_grad_enabled(False); t0 = time.time()
if a.void_ref is not None:
    def _med(b):
        j = json.load(open(b + '.json')); F_ = np.fromfile(b, dtype=np.float32).reshape(j['n'], j['dim']); nz_ = np.linalg.norm(F_, axis=1); return float(np.median(nz_[nz_ > 0]))
    VOID_NORM = 0.5 * _med(a.void_ref[0]) / _med(a.void_ref[1]); print(f'[void] relative presence check: ||f|| < {VOID_NORM:.3f} (0.5 x {_med(a.void_ref[0]):.2f} / {_med(a.void_ref[1]):.2f})', flush=True)
elif a.void_norm is not None: VOID_NORM = a.void_norm; print(f'[void] presence check ||f|| < {VOID_NORM:.3f}', flush=True)
bank = np.load(N / 'text_bank.npz', allow_pickle=True); meta = json.loads(str(bank['meta']))
SUP = Path(a.sup) if a.sup else Path(meta['manifest']).parent; HJ = Path(a.hier) if a.hier else Path(meta['hierarchy_json'])
EMB = a.emb or json.load(open(N / 'features_B.bin.json'))['embedder']; OUT = Path(a.out) if a.out else N / f'protocol_{a.tag}.jsonl'
frames = json.load(open(a.frames_json or N / 'frames.json'))['frames']; ev = set(json.load(open(N / 'split_names.json'))['eval'])
Hj = json.load(open(HJ)); row_of = {o['id']: o['row_id'] for o in Hj['objects']}; tree_of_fruit = {f['id']: f['tree_id'] for f in Hj.get('fruits', [])}
# --- vocabulary of the cell: the ids the bank carries (= the ids painted in this cell's supervision) + their rows; LIVE words ---
wt = {int(k): v for k, v in meta['word_table'].items()}; rw = {int(k): v for k, v in meta.get('row_words', {}).items()}
tree_ids = sorted(k for k in wt if k < FRUIT_ID_BASE); fruit_ids = sorted(k for k in wt if k >= FRUIT_ID_BASE); row_ids = sorted(rw)
ids = tree_ids + fruit_ids + row_ids; kinds = ['TREE'] * len(tree_ids) + ['FRUIT'] * len(fruit_ids) + ['ROW'] * len(row_ids)
WORD = lambda k, i: get_word_for_id(i - FRUIT_ID_BASE if k == 'FRUIT' else i, {'TREE': 'mask', 'FRUIT': 'fruit', 'ROW': 'row'}[k])
words = [WORD(k, i) for k, i in zip(kinds, ids)]
stale = [(k, i, w, (rw if k == 'ROW' else wt).get(i)) for k, i, w in zip(kinds, ids, words) if (rw if k == 'ROW' else wt).get(i) != w]   # per kind: tree and row ids share numbers
if stale: print(f'[vocab] WARNING {len(stale)} bank words differ from the live word table (kind, id, live, bank): {stale[:5]}', flush=True)
clip, _, _ = open_clip.create_model_and_transforms(a.clip_model, pretrained=a.clip_pretrained, precision='fp16'); clip.eval(); clip = clip.to(dev)
tok = open_clip.get_tokenizer(a.clip_model)
hyper = build_hyper_embedder(EMB, dev) if a.space not in ('flat', 'langsplat') else None; curv = hyper.curv.exp() if hyper is not None else None
EUCLID = bool(getattr(hyper.config, 'euclidean', False)) if hyper is not None else False
SPACE = a.space if a.space != 'auto' else ('euclid' if EUCLID else 'hyper')
if SPACE == 'euclid' and not EUCLID: sys.exit('[space] --space euclid needs a Euclidean embedder (config euclidean=True)')
if SPACE in ('hyper', 'hyper_own') and EUCLID: sys.exit('[space] this embedder is Euclidean; use --space euclid')
if SPACE == 'flat':
    if not a.basis: sys.exit('[space] --space flat needs --basis')
    _B = np.load(a.basis); comp = torch.from_numpy(_B['components']).float().to(dev); bmean = torch.from_numpy(_B['mean']).float().to(dev); K = comp.shape[0]
if SPACE == 'langsplat':   # LangSplat's own decode + relevancy: AE 3 -> 512, LERF softmax against the canonical negatives, min over them
    if not a.ae_ckpt: sys.exit('[space] --space langsplat needs --ae-ckpt')
    sys.path.insert(0, '/home/paperspace/code/LangSplat/autoencoder'); from model import Autoencoder
    AE = Autoencoder([256, 128, 64, 32, 3], [16, 32, 64, 128, 256, 256, 512]).to(dev); AE.load_state_dict(torch.load(a.ae_ckpt, map_location=dev)); AE.eval()
    LS_NEG = ('object', 'things', 'stuff', 'texture'); VOID_NORM = 1e-3   # LangSplat has no 'no identity' notion: only uncovered pixels are void
def encode(ws):
    e = clip.encode_text(tok(ws).to(dev)).float(); return e / e.norm(dim=-1, keepdim=True)
E = encode(words); col = {(k, i): j for j, (k, i) in enumerate(zip(kinds, ids))}
R_MID = None
if a.fruit_rule == 'level' and hyper is not None and not EUCLID and fruit_ids:   # fruit depth from the embedder's own targets (RAW CLIP -> encode -> tangent norm)
    with torch.no_grad():
        _ce = clip.encode_text(tok([WORD('TREE', t) for t in tree_ids] + [WORD('FRUIT', f) for f in fruit_ids]).to(dev)).float()
        _nt = torch.tensor([0] * len(tree_ids) + [4] * len(fruit_ids), device=dev); _tl = L.log_map0(hyper.encode_features(_ce, project=True, node_types=_nt), curv=curv).norm(dim=-1)
    R_MID = 0.5 * (_tl[:len(tree_ids)].median().item() + _tl[len(tree_ids):].median().item()); print(f'[fruit-rule] level: tree targets median norm {_tl[:len(tree_ids)].median().item():.2f}, fruit {_tl[len(tree_ids):].median().item():.2f} -> fruit depth >= {R_MID:.2f}', flush=True)
CAT = [(kv.split('=')[0], kv.split('=')[1] if '=' in kv else 'ALL') for kv in a.category.split(',') if kv]
if CAT:   # category protocol: the vocabulary is the category words; LERF relevancy (min over the canonical negatives) replaces the cosine
    LS_NEG = ('object', 'things', 'stuff', 'texture'); E_CAT = encode([w for w, _ in CAT]); E_NEG = encode(list(LS_NEG))
    def rel_from_cos(cp, cn): return torch.sigmoid(10.0 * (cp.unsqueeze(-1) - cn.unsqueeze(-2))).min(-1).values   # cp [..., c], cn [..., k] -> [..., c]
if SPACE == 'langsplat': E_NEG = encode(list(LS_NEG))
def relevancy(dec):   # LangSplat/eval get_relevancy, vectorised over the vocabulary: min over negatives of softmax(10 * [pos, neg])[pos]
    pos = dec @ E.T; neg = dec @ E_NEG.T
    return torch.sigmoid(10.0 * (pos.unsqueeze(-1) - neg.unsqueeze(1))).min(-1).values
print(f'[vocab] {len(tree_ids)} trees, {len(fruit_ids)} fruit, {len(row_ids)} rows | space {SPACE} | sup {SUP} | hier {HJ.name} | emb {Path(EMB).parent.parent.name if hyper is not None else "-"}', flush=True)
def column(kind, i):   # a head whose word the bank lacks (row of a tree outside the bank, ...) gets encoded on the fly
    global E
    if (kind, i) not in col:
        w = WORD(kind, i); E = torch.cat([E, encode([w])]); col[(kind, i)] = E.shape[0] - 1; ids.append(i); kinds.append(kind); words.append(w)
        {'TREE': tree_ids, 'FRUIT': fruit_ids, 'ROW': row_ids}[kind].append(i); print(f'[vocab] added {kind} {i} "{w}" (not in the bank)', flush=True)
    return col[(kind, i)]
def prf(pred, gt):
    inter = (pred & gt).sum().item(); u = (pred | gt).sum().item(); p = pred.sum().item(); g = gt.sum().item()
    return inter / max(u, 1), inter / max(p, 1), inter / max(g, 1)
THR_T = torch.tensor(THR, device=dev, dtype=torch.float32)
def sweep(heat, gt, extra=None):   # IoU / precision / recall at EVERY threshold from two histograms (== the 57 separate `heat >= thr` masks, 114x fewer passes)
    sel = heat if extra is None else heat.masked_fill(~extra, -1.0)
    idx = torch.searchsorted(THR_T, sel.float(), right=True)                     # value v counts for thresholds THR[j] with THR[j] <= v, i.e. j < idx
    hist_all = torch.bincount(idx, minlength=len(THR) + 1).float(); hist_gt = torch.bincount(idx[gt], minlength=len(THR) + 1).float()
    P_ = hist_all.sum() - hist_all.cumsum(0)[:len(THR)]; TP = hist_gt.sum() - hist_gt.cumsum(0)[:len(THR)]; G = float(gt.sum().item())
    iou = TP / (P_ + G - TP).clamp(min=1); return iou.cpu().numpy(), (TP / P_.clamp(min=1)).cpu().numpy(), (TP / max(G, 1)).cpu().numpy()
def curve(heat, gt, extra=None): return sweep(heat, gt, extra)[0].tolist()
nrm = lambda x: x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
lines = []; parity = []
for fr in frames:
    stem = fr.replace('.png', ''); z = np.load(N / (a.maps % stem)); feat = torch.from_numpy(z['features'].astype(np.float32)).to(dev); H, W, D = feat.shape
    ft = feat.reshape(-1, D); P = ft.shape[0]; void = (ft[:, 3 * (a.ls_level - 1):3 * a.ls_level] if SPACE == 'langsplat' else ft).norm(dim=-1) < VOID_NORM
    gt16 = np.array(Image.fromarray(np.array(Image.open(SUP / fr), np.uint16)).resize((W, H), Image.NEAREST)); g = torch.from_numpy(gt16.astype(np.int64)).to(dev).reshape(-1)
    uid = sorted(int(u) for u in np.unique(gt16) if 0 <= u < UNLAB); trees = [u for u in uid if u < FRUIT_ID_BASE]; fruits = [u for u in uid if u >= FRUIT_ID_BASE]
    heads = []   # (kind, id, gt mask) exactly as containment_eval builds its cases
    for t in trees:
        tf = [f for f in fruits if tree_of_fruit.get(f - FRUIT_ID_BASE) == t]; gt_t = g == t
        for f in tf: gt_t = gt_t | (g == f)
        heads.append(('TREE', t, gt_t)); heads += [('FRUIT', f, g == f) for f in tf]
    for r in sorted({row_of[t] for t in trees if t in row_of}):
        gt_r = torch.zeros_like(g, dtype=torch.bool)
        for t in trees:
            if row_of.get(t) == r:
                gt_r |= g == t
                for f in fruits:
                    if tree_of_fruit.get(f - FRUIT_ID_BASE) == t: gt_r |= g == f
        heads.append(('ROW', r, gt_r))
    if CAT:   # category heads replace the instance heads; S holds relevancies in (0, 1)
        grp = {'ALL': g < UNLAB, 'TREES': (g < FRUIT_ID_BASE), 'FRUIT': (g >= FRUIT_ID_BASE) & (g < UNLAB)}
        heads = [('CAT', ci, grp[gname]) for ci, (w, gname) in enumerate(CAT) if grp[gname].sum() > 0]
        S = torch.zeros(P, len(CAT), device=dev)
        for j in range(0, P, 8192):
            sl = ft[j:j + 8192]
            if SPACE == 'flat' and D == K: dec = nrm(sl @ comp + bmean)[:, None]
            elif SPACE == 'flat': dec = nrm(sl.view(-1, 3, K) @ comp + bmean)                       # [n, 3, 512]: a category may live in any block -> max over blocks
            elif SPACE == 'euclid': dec = nrm(hyper.decode_features(sl, project=True))[:, None]
            elif SPACE == 'langsplat': dec = nrm(AE.decode(sl[:, 3 * (a.ls_level - 1):3 * a.ls_level]))[:, None]
            else:
                h = L.exp_map0(sl, curv=curv); own = nrm(hyper.decode_features(h, project=True))[:, None]
                if SPACE == 'hyper_own': dec = own
                else:
                    path, mask = L.get_interpolated_hyperbolic_features(h, steps=STEPS, curv=curv, max_dist=MAX_DIST, return_mask=True)
                    dp = nrm(hyper.decode_features(path, project=True)).view(sl.shape[0], STEPS, -1).masked_fill(mask.to(dev).bool().unsqueeze(-1), 0.0); dec = torch.cat([dp, own], 1)
            S[j:j + 8192] = rel_from_cos(dec @ E_CAT.T, dec @ E_NEG.T).max(1).values
        S[void] = -1.0; live = ~void; lab_c = S.argmax(1); split = 'eval' if fr in ev else 'train'
        lines.append({'tag': a.tag, 'frame': fr, 'split': split, 'kind': 'FRAME', 'space': SPACE, 'void_frac': void.float().mean().item(), 'nesting': float('nan'), 'heads': len(heads), 'hw': [H, W]})
        for kind, ci, gt in heads:
            heat = S[:, ci]; cv, cp, cr = sweep(heat, gt); b = int(np.argmax(cv)); o_iou, o_prec, o_rec = (float(cv[b]), float(cp[b]), float(cr[b])) if cv[b] > 0 else (0.0, 0.0, 0.0); o_thr = float(THR[b]) if cv[b] > 0 else float('nan')
            pred = live & (lab_c == ci); a_iou, a_prec, a_rec = prf(pred, gt); half = prf(live & (heat >= 0.5), gt)
            lines.append({'tag': a.tag, 'frame': fr, 'split': split, 'kind': 'CAT', 'id': ci, 'word': CAT[ci][0], 'group': CAT[ci][1], 'gt_px': int(gt.sum().item()), 'gt_frac': gt.sum().item() / P, 'gt_void_frac': 1 - (gt & live).sum().item() / max(gt.sum().item(), 1),
                          'oracle': {'thr': o_thr, 'iou': o_iou, 'prec': o_prec, 'rec': o_rec}, 'argmax': {'iou': a_iou, 'prec': a_prec, 'rec': a_rec}, 'rel05': dict(zip(('iou', 'prec', 'rec'), half)), 'curve_thr': [round(x, 4) for x in cv.tolist()], 'curve_argmax_thr': [round(x, 4) for x in curve(heat, gt, pred)]})
            print(f'[{a.tag} {fr}] CAT "{CAT[ci][0]}"={CAT[ci][1]}: oracle thr {o_thr:.2f} IoU {o_iou:.3f} | rel>=0.5 IoU {half[0]:.3f} | argmax IoU {a_iou:.3f} prec {a_prec:.3f} rec {a_rec:.3f} | gt {100 * gt.sum().item() / P:.1f}%', flush=True)
        print(f'[{a.tag} {fr}] void {100 * void.float().mean().item():.1f}% | {len(heads)} category heads | {time.time() - t0:.0f}s', flush=True); continue
    for k, i, _ in heads: column(k, i)
    V = E.shape[0]; S = torch.full((P, V), -1.0, device=dev)
    T_cols = torch.tensor([col[('TREE', t)] for t in tree_ids], device=dev, dtype=torch.long); R_cols = torch.tensor([col[('ROW', r)] for r in row_ids], device=dev, dtype=torch.long)
    F_cols = torch.tensor([col[('FRUIT', f)] for f in fruit_ids], device=dev, dtype=torch.long); FT_cols = torch.cat([F_cols, T_cols])
    for j in range(0, P, 8192):
        sl = ft[j:j + 8192]
        if SPACE == 'flat' and D == K:   # ONE flat block (LangSplat-lite): the same decoded vector answers every level's words
            S[j:j + 8192] = nrm(sl @ comp + bmean) @ E.T; continue
        if SPACE == 'flat':   # three flat blocks, each decoded by the inverse PCA to CLIP space; a block below the norm check carries nothing
            bl = sl.view(-1, 3, K); dec = nrm(bl @ comp + bmean)
            S[j:j + 8192, T_cols] = dec[:, 0] @ E[T_cols].T
            if len(row_ids): S[j:j + 8192, R_cols] = dec[:, 1] @ E[R_cols].T
            if len(fruit_ids): S[j:j + 8192, F_cols] = (dec[:, 2] @ E[F_cols].T).masked_fill((bl[:, 2].norm(dim=-1) < VOID_NORM).unsqueeze(-1), -1.0)
            continue
        if SPACE == 'euclid': S[j:j + 8192] = nrm(hyper.decode_features(sl, project=True)) @ E.T; continue
        if SPACE == 'langsplat': S[j:j + 8192] = relevancy(AE.decode(sl[:, 3 * (a.ls_level - 1):3 * a.ls_level])); continue
        h = L.exp_map0(sl, curv=curv); own = nrm(hyper.decode_features(h, project=True)) @ E.T
        if SPACE == 'hyper_own': S[j:j + 8192] = own; continue
        path, mask = L.get_interpolated_hyperbolic_features(h, steps=STEPS, curv=curv, max_dist=MAX_DIST, return_mask=True)
        sm = (nrm(hyper.decode_features(path, project=True)) @ E.T).view(sl.shape[0], STEPS, V).masked_fill(mask.to(dev).bool().unsqueeze(-1), -1.0)
        S[j:j + 8192] = torch.maximum(sm.max(1).values, own)
    S[void] = -1.0; live = ~void
    lab_t = S[:, T_cols].argmax(1); lab_r = S[:, R_cols].argmax(1) if len(row_ids) else None; lab_ft = S[:, FT_cols].argmax(1)
    if R_MID is not None and len(fruit_ids): lab_f_only = S[:, F_cols].argmax(1); deep = ft.norm(dim=-1) >= R_MID
    row_true = torch.tensor([row_of.get(t, -1) for t in tree_ids], device=dev)[lab_t]   # the row of each pixel's tree-level argmax (scene graph)
    nest = float('nan')
    if lab_r is not None:
        row_hat = torch.tensor(row_ids, device=dev)[lab_r]; ok = live & (row_true >= 0); nest = ((row_true == row_hat) & ok).sum().item() / max(ok.sum().item(), 1)
    split = 'eval' if fr in ev else 'train'
    if a.dump_pred:   # per-pixel decisions of the level rules, for figures (tree argmax; fruit by the level rule and by the mixed rule)
        Path(a.dump_pred).mkdir(parents=True, exist_ok=True); none = torch.full((P,), -1, dtype=torch.int64, device=dev)
        tree_px = torch.where(live, torch.tensor(tree_ids, device=dev)[lab_t], none)
        fids = torch.tensor(fruit_ids if fruit_ids else [-1], device=dev)
        f_lvl = torch.where(live & deep, fids[lab_f_only], none) if (R_MID is not None and len(fruit_ids)) else none
        f_mix = torch.where(live & (lab_ft < len(fruit_ids)), fids[lab_ft.clamp(max=max(len(fruit_ids) - 1, 0))], none)
        np.savez_compressed(Path(a.dump_pred) / f'{stem}.npz', tree=tree_px.view(H, W).cpu().numpy().astype(np.int32), fruit_level=f_lvl.view(H, W).cpu().numpy().astype(np.int32),
                            fruit_mixed=f_mix.view(H, W).cpu().numpy().astype(np.int32), live=live.view(H, W).cpu().numpy())
    lines.append({'tag': a.tag, 'frame': fr, 'split': split, 'kind': 'FRAME', 'space': SPACE, 'void_frac': void.float().mean().item(), 'nesting': nest, 'heads': len(heads), 'hw': [H, W]})
    for kind, i, gt in heads:
        c = col[(kind, i)]; heat = S[:, c]
        cv, cp, cr = sweep(heat, gt); b = int(np.argmax(cv)); o_iou, o_prec, o_rec = (float(cv[b]), float(cp[b]), float(cr[b])) if cv[b] > 0 else (0.0, 0.0, 0.0); o_thr = float(THR[b]) if cv[b] > 0 else float('nan'); cv = cv.tolist()
        if kind == 'TREE': pred = live & (lab_t == tree_ids.index(i))
        elif kind == 'ROW': pred = live & (lab_r == row_ids.index(i)) if lab_r is not None else torch.zeros_like(live)
        elif R_MID is not None: pred = live & deep & (lab_f_only == fruit_ids.index(i))   # fruit scored like every level: among fruit words, presence by depth
        else: pred = live & (lab_ft == fruit_ids.index(i))
        a_iou, a_prec, a_rec = prf(pred, gt); gl = gt & live
        rec = {'tag': a.tag, 'frame': fr, 'split': split, 'kind': kind, 'id': i, 'word': words[c], 'gt_px': int(gt.sum().item()), 'gt_frac': gt.sum().item() / P,
               'gt_void_frac': 1 - gl.sum().item() / max(gt.sum().item(), 1), 'oracle': {'thr': o_thr, 'iou': o_iou, 'prec': o_prec, 'rec': o_rec},
               'argmax': {'iou': a_iou, 'prec': a_prec, 'rec': a_rec}, 'curve_thr': [round(x, 4) for x in cv], 'curve_argmax_thr': [round(x, 4) for x in curve(heat, gt, pred)]}
        if kind == 'ROW': rec['argmax_union'] = dict(zip(('iou', 'prec', 'rec'), prf(live & (row_true == i), gt)))   # row = union of its member trees' argmax pixels (scene graph, no row words)
        if kind == 'TREE':
            rec['sibling_confusion'] = ((lab_t != tree_ids.index(i)) & gl).sum().item() / max(gl.sum().item(), 1)   # GT px of this tree that another tree won
            if lab_r is not None and row_of.get(i) in row_ids: rec['nest_pred'] = ((lab_r == row_ids.index(row_of[i])) & pred).sum().item() / max(pred.sum().item(), 1)
        lines.append(rec); parity.append((fr, kind, i, words[c], o_thr, o_iou, o_prec, o_rec))
        print(f'[{a.tag} {fr}] {kind + (" of" if kind == "FRUIT" else ""):10s} {i:<5d} "{words[c]}": oracle thr {o_thr:.2f} IoU {o_iou:.3f} | argmax IoU {a_iou:.3f} prec {a_prec:.3f} rec {a_rec:.3f} | gt {100 * rec["gt_frac"]:.2f}% of frame', flush=True)
    print(f'[{a.tag} {fr}] void {100 * void.float().mean().item():.1f}% | nesting {nest:.3f} | {len(heads)} heads | {time.time() - t0:.0f}s', flush=True)
with open(OUT, 'w') as f:
    for l in lines: f.write(json.dumps(l) + '\n')
hd = [l for l in lines if l['kind'] != 'FRAME']
def med(k, s, key): v = [l[key]['iou'] for l in hd if l['kind'] == k and l['split'] == s]; return f'{np.median(v):.3f} (n={len(v)})' if v else '—'
if CAT: print(f'[protocol] {N} [{SPACE}] category: ' + '; '.join(f'"{w}" {s}: oracle {np.median([l["oracle"]["iou"] for l in hd if l["word"] == w and l["split"] == s]):.3f} rel>=0.5 {np.median([l["rel05"]["iou"] for l in hd if l["word"] == w and l["split"] == s]):.3f} argmax {np.median([l["argmax"]["iou"] for l in hd if l["word"] == w and l["split"] == s]):.3f} (n={sum(1 for l in hd if l["word"] == w and l["split"] == s)})' for w, _ in CAT for s in ('train', 'eval') if any(l['word'] == w and l['split'] == s for l in hd)) + f' | {time.time() - t0:.0f}s -> {OUT}', flush=True)
print(f'[protocol] {N} [{SPACE}]: ' + '; '.join(f'{k} {s}: oracle {med(k, s, "oracle")} argmax {med(k, s, "argmax")}' for k in ('TREE', 'ROW', 'FRUIT') for s in ('train', 'eval') if any(l['kind'] == k and l['split'] == s for l in hd)) + f' | nesting mean {np.nanmean([l["nesting"] for l in lines if l["kind"] == "FRAME"]):.3f} | {time.time() - t0:.0f}s -> {OUT}', flush=True)
if a.check_oracle and (N / 'verdicts.log').exists():
    pat = re.compile(r'^\[(\S+) (\S+)\] (TREE|ROW|FRUIT of) +(\d+) +"([^"]*)": thr (\S+) IoU (\S+) prec (\S+) rec (\S+)'); ref = {}
    for line in open(N / 'verdicts.log', errors='replace'):
        m = pat.match(line)
        if m and m.group(1) == a.tag: ref[(m.group(2), m.group(3).replace(' of', ''), m.group(5))] = (float(m.group(7)), m.group(6))   # keyed by WORD: 'FRUIT of <tree>' carries the tree id
    d = [(abs(o_iou - ref[(fr, k, w)][0]), fr, k, i, o_iou, ref[(fr, k, w)][0]) for fr, k, i, w, o_thr, o_iou, _, _ in parity if (fr, k, w) in ref]
    worst = max(d) if d else None
    print(f'[check] oracle vs verdicts.log: {len(d)} of {len(parity)} heads matched, max |dIoU| {worst[0]:.4f}' + (f' ({worst[1]} {worst[2]} {worst[3]}: {worst[4]:.3f} vs {worst[5]:.3f})' if worst else ''), flush=True)
