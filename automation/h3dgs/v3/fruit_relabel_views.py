#!/usr/bin/env python3
"""fruit_relabel_views.py — the mask-views fruit rule (Paul 2026-10-14: "surely we'd just use 24 and 25"): a gaussian is FRUIT when, over the
views in which it touches a fruit mask (census W2 from splat_census.py --fruit-views-npz), at least --min-share of its LABELLED votes are fruit
(unlabelled px never vote; tree / unlabelled votes from views where the fruit is not visible never count). Starts from a base assignment
(build_census_init's <features>.labels.npy: trees by the ordinary census), promotes qualifying gaussians to their majority fruit word and demotes
base fruit labels that fail the rule to their best non-fruit label in W (or none). Writes <out>.labels.npy for reseed_from_labels.py.
system python3:  fruit_relabel_views.py --base <features.bin.labels.npy> --w-npz <W.npz> --w2-npz <W2.npz> --out <labels.npy> [--min-share 0.5] [--min-votes 1.0]"""
import argparse, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--base', required=True); ap.add_argument('--w-npz', required=True); ap.add_argument('--w2-npz', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--min-share', type=float, default=0.5); ap.add_argument('--min-votes', type=float, default=1.0); a = ap.parse_args()
FRUIT, UNLAB = 10000, 65535
base = np.load(a.base).astype(np.int32); Z = np.load(a.w_npz); W, lab = Z['W'], Z['labels']; Z2 = np.load(a.w2_npz); W2 = Z2['W']; assert (Z2['labels'] == lab).all() and W2.shape == W.shape and base.shape[0] == W.shape[1]
fr = np.array([(FRUIT <= l < UNLAB) for l in lab]); labd = np.array([l != UNLAB for l in lab]); nonf = labd & ~fr
f2 = W2[fr].sum(0); l2 = W2[labd].sum(0); share = np.where(l2 > 0, f2 / np.maximum(l2, 1e-9), 0.0)
fruit_ok = (l2 >= a.min_votes) & (share >= a.min_share)
best_fruit = lab[fr][W2[fr].argmax(0)]                                   # the fruit word with most mask-view votes
best_nonf = lab[nonf][W[nonf].argmax(0)]; has_nonf = W[nonf].max(0) > 0  # fallback for demoted base-fruit labels
new = base.copy(); base_fruit = (base >= FRUIT) & (base < UNLAB)
promote = fruit_ok; demote = base_fruit & ~fruit_ok
new[promote] = best_fruit[promote]; new[demote] = np.where(has_nonf[demote], best_nonf[demote], -1)
np.save(a.out, new.astype(np.int32))
print(f'[relabel-views] base fruit {int(base_fruit.sum())} -> fruit {int(((new >= FRUIT) & (new < UNLAB)).sum())} (promoted {int((promote & ~base_fruit).sum())}, kept {int((promote & base_fruit).sum())}, demoted {int(demote.sum())}); '
      f'judged (mask-view labelled votes >= {a.min_votes}): {int((l2 >= a.min_votes).sum())}, share >= {a.min_share}: {int(fruit_ok.sum())} -> {a.out}', flush=True)
