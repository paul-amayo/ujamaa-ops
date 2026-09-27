#!/usr/bin/env python3
"""Build a census-init stage2 checkpoint: majority-label feature init.

Takes the interaction-census W matrix (gaussian_interaction_census.py), builds
dataloader-EXACT targets (RAW un-normalized CLIP -> encode_features with
node_types, fruit=4), assigns each gaussian its weight-majority label's target
tangent vector (total W > 1.0; rest stay zero), and writes the features into a
copy of the given stage2-init checkpoint. Recipe of record since 2026-08-07
(zero-init retired: "deflation" was unfinished growth).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'interfaces' / 'rerun' / 'HiGH'))
from high_splat_hierarchy_accuracy import build_hyper_embedder   # noqa: E402
from word_utils import get_word_for_id                           # noqa: E402
import lorentz as L                                              # noqa: E402
import open_clip                                                 # noqa: E402

FRUIT_ID_BASE = 10000   # must match scripts/compile_supervision.py:FRUIT_ID_BASE

ap = argparse.ArgumentParser()
ap.add_argument('--w-npz', required=True)
ap.add_argument('--embedder', required=True)
ap.add_argument('--src-ckpt', required=True, type=Path)
ap.add_argument('--dst-dir', required=True, type=Path)
ap.add_argument('--tree-floor', type=float, default=1.0,
                help="assignment floor on a gaussian's total census weight for tree/row-argmax gaussians "
                     "(recipe of record 1.0). Denser geometries (the H3DGS side-car: 4.5x the gaussians of a "
                     "block, median per-gaussian weight 100x smaller) leave most voted gaussians at zero under "
                     "1.0 and the blended features shrink toward the origin (2026-09-27).")
ap.add_argument('--bg-competes', action='store_true',
                help="with a census that carries the unlabelled row (CENSUS_WITH_BG=1): withhold identity from a "
                     "gaussian whose alpha mass into unlabelled pixels exceeds its best label's mass (it mostly "
                     "serves ground/sky and would bleed a shrunk tree feature there, which decodes as the row). "
                     "Off by default (recipe of record unchanged).")
ap.add_argument('--bg-ratio', type=float, default=1.0,
                help="with --bg-competes: withhold identity only when the void mass is at least this multiple of the "
                     "best label's mass (1.0 = void >= label). Silhouette gaussians of a thin canopy against sky have "
                     "void ~ label and were being withheld at 1.0 (block 013 'refused': 12-19 %% of its pixels with no "
                     "identity), while ground/sky floaters sit at void >> label.")
ap.add_argument('--fruit-floor', type=float, default=0.01,
                help="assignment floor for FRUIT-argmax gaussians. The global "
                     "floor (tot > 1.0) is calibrated on trees, whose gradient "
                     "mass comes from ~100k painted px; a 100-300 px fruit "
                     "spreads tot medians of 0.03-0.04 over its members, so "
                     "the same floor dropped 78-80%% of fruit 10000/10004's "
                     "gaussians on 05 b000 (2026-08-26) — they rendered weak, "
                     "fell under the ||f||<0.5 norm gate, and the fruit was "
                     "invisible to queries. Trees keep the 1.0 floor.")
ap.add_argument('--fruit-share-assign', type=float, default=0.0,
                help="direct seeding (Paul, 2026-08-26): a gaussian whose FRUIT "
                     "share of its supervised diet exceeds this gets its majority "
                     "FRUIT entity's target, even where a tree wins the argmax. "
                     "The argmax hands a fruit's own pixels to the tree (fruit is "
                     "never the majority of a shared carrier's diet — measured "
                     "0-10%% fruit blend at fruit px on 05 b000), so majority "
                     "assignment structurally cannot seed fruit; share assignment "
                     "can. Lower = more fruit recall, more tree-px halo. 0 = off.")
ap.add_argument('--fruit-share-min', type=float, default=0.0,
                help="margin rule (2026-08-07, 'lets try 2'): a gaussian gets "
                     "a FRUIT target only if that fruit's weight share of its "
                     "total supervision exceeds this; mixed halo gaussians "
                     "fall back to their best NON-fruit label. 0 = plain "
                     "majority (the original rule).")
args = ap.parse_args()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_grad_enabled(False)
z = np.load(args.w_npz)
W, labels = z['W'], [int(u) for u in z['labels']]
m, _, _ = open_clip.create_model_and_transforms('ViT-B-16', 'laion2b_s34b_b88k', device=dev)
tok = open_clip.get_tokenizer('ViT-B-16')
hyper = build_hyper_embedder(args.embedder, dev)
curv = hyper.curv.exp()
words = [(get_word_for_id(u - FRUIT_ID_BASE, 'fruit') if u >= FRUIT_ID_BASE else get_word_for_id(u, 'mask'))
         for u in labels]
nt = torch.tensor([4 if u >= FRUIT_ID_BASE else 0 for u in labels], device=dev)
ce = m.encode_text(tok(words).to(dev)).float()          # RAW — never unit-normalize
hf = hyper.encode_features(ce, project=True, node_types=nt)
tl = L.log_map0(hf, curv=curv).cpu().numpy()
print('target norms:', dict(zip(labels, np.linalg.norm(tl, axis=1).round(2))))

bg_mass = None
if 65535 in labels:   # census run with CENSUS_WITH_BG=1: row 65535 = each gaussian's alpha mass into UNLABELLED pixels
    bg_i = labels.index(65535); bg_mass = W[bg_i]
    keep_rows = [i for i in range(len(labels)) if i != bg_i]
    W = W[keep_rows]; labels = [labels[i] for i in keep_rows]; tl = tl[keep_rows]
    print(f'background row present: {100.0 * np.mean(bg_mass > 0):.1f}% of gaussians touch unlabelled pixels')
tot = W.sum(0)
maj = W.argmax(0)
fruit_rows_all = np.array([u >= FRUIT_ID_BASE for u in labels])
is_fruit_maj = fruit_rows_all[maj]
floor = np.where(is_fruit_maj, args.fruit_floor, args.tree_floor)
assign = tot > floor
if args.bg_competes and bg_mass is not None:
    _withheld = assign & (bg_mass >= args.bg_ratio * W.max(0))
    assign &= ~_withheld
    print(f'background competes: {int(_withheld.sum())} gaussians above the floor withheld (void mass >= {args.bg_ratio:g} x best label mass)')
n_fr = int((assign & is_fruit_maj).sum())
print(f'floors: tree {args.tree_floor}, fruit {args.fruit_floor} -> {n_fr} fruit-argmax gaussians assigned')
if args.fruit_share_min > 0:
    fruit_rows = fruit_rows_all
    share = W[maj, np.arange(W.shape[1])] / np.maximum(tot, 1e-9)
    demote = assign & is_fruit_maj & (share < args.fruit_share_min)
    if fruit_rows.all():
        demote[:] = False
    else:
        Wnf = W.copy()
        Wnf[fruit_rows] = -1.0
        maj = np.where(demote, Wnf.argmax(0), maj)
    kept = int((assign & fruit_rows[maj]).sum())
    print(f'margin rule: {int(demote.sum())} mixed halo gaussians demoted to tree; '
          f'{kept} keep a fruit target (share >= {args.fruit_share_min})')
if args.fruit_share_assign > 0:
    Wf = W.copy()
    Wf[~fruit_rows_all] = 0.0
    fshare = np.zeros_like(tot)
    sup_ = tot > 0
    fshare[sup_] = Wf.sum(0)[sup_] / tot[sup_]
    fmaj = np.where(Wf.sum(0) > 0, Wf.argmax(0), maj)
    promote = sup_ & (fshare > args.fruit_share_assign)
    maj = np.where(promote, fmaj, maj)
    assign = assign | promote
    print(f'share-assign: {int(promote.sum())} gaussians seeded with their fruit '
          f'target (fruit share > {args.fruit_share_assign})')
feats = np.zeros((W.shape[1], tl.shape[1]), np.float32)
feats[assign] = tl[maj[assign]]
print(f'assigned {assign.sum()}/{W.shape[1]} gaussians ({100*assign.sum()/W.shape[1]:.1f}%)')

ck = torch.load(args.src_ckpt, map_location='cpu')
key = [k for k in ck['pipeline'] if k.endswith('gauss_params.high_features')][0]
assert ck['pipeline'][key].shape == feats.shape, (ck['pipeline'][key].shape, feats.shape)
ck['pipeline'][key] = torch.from_numpy(feats)
args.dst_dir.mkdir(parents=True, exist_ok=True)
out = args.dst_dir / args.src_ckpt.name
torch.save(ck, out)
print(f'census-init checkpoint -> {out}')
