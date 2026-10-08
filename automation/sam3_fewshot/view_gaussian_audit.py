"""One training view, gaussian level (Paul 2026-10-07: "where do the false positives or negatives come from; from a gaussian supervision
perspective this should be almost binary"). Inputs: the cell's global census W (seed rules of seed B reproduced: argmax with bg at ratio 2,
floors tree 1.0 / fruit 0.01, fruit-share promotion > 0.1) and a single-view census W_view of the same view. Reports, for the view's
fruit pixels, which gaussians render them (fruit-seeded vs tree-seeded = the FN side) and, for the fruit-seeded gaussians visible in the
view, where their weight lands (fruit pixels vs leaf/other = the FP side), plus the fruit-share distribution of the gaussians involved.
h3dgs env: view_gaussian_audit.py <cell dir> <view dir>"""
import json, sys, numpy as np
cell, view = sys.argv[1], sys.argv[2]; z = np.load(f"{cell}/W.npz"); W, L = z["W"], [int(x) for x in z["labels"]]; zv = np.load(f"{view}/W_view.npz"); Wv, Lv = zv["W"], [int(x) for x in zv["labels"]]
isf = np.array([10000 <= l < 65535 for l in L]); isbg = np.array([l == 65535 for l in L]); Wa = W.copy(); Wa[isbg] *= 2.0   # bg competes at ratio 2
maj = Wa.argmax(0); tot = W[~isbg].sum(0); fmaj = isf[maj]; floor = np.where(fmaj, 0.01, 1.0); assigned = tot >= floor
fshare = np.where(tot > 0, W[isf].sum(0) / np.maximum(tot, 1e-9), 0.0); fruit_seeded = assigned & (fmaj | (fshare > 0.1)); tree_seeded = assigned & ~fruit_seeded
fv = np.array([10000 <= l < 65535 for l in Lv]); tv = np.array([l < 10000 for l in Lv]); wf_v = Wv[fv].sum(0); wt_v = Wv[tv].sum(0); wb_v = Wv[[l == 65535 for l in Lv]].sum(0)
seen = (wf_v + wt_v + wb_v) > 0; print(f"[view] {view.split('/')[-2]}: {int(seen.sum()):,} gaussians render into the view; seed (global): fruit-seeded {int(fruit_seeded.sum()):,}, of which visible here {int((fruit_seeded & seen).sum()):,}")
# FN side: who renders the fruit pixels of this view
F = wf_v.sum(); on_f = wf_v[fruit_seeded].sum(); on_t = wf_v[tree_seeded].sum(); on_u = wf_v[~assigned].sum()
print(f"[FN side] fruit-pixel weight in this view: {on_f / F:.2f} from fruit-seeded gaussians, {on_t / F:.2f} from tree-seeded, {on_u / F:.2f} from unassigned (no identity) -> only the first part can light")
g = (wf_v > 0); print(f"          gaussians touching fruit pixels here: {int(g.sum()):,}; their GLOBAL fruit share: <0.1 {int((g & (fshare < 0.1)).sum()):,}, 0.1-0.5 {int((g & (fshare >= 0.1) & (fshare < 0.5)).sum()):,}, 0.5-0.9 {int((g & (fshare >= 0.5) & (fshare < 0.9)).sum()):,}, >=0.9 {int((g & (fshare >= 0.9)).sum()):,}")
sh_v = wf_v[g] / (wf_v[g] + wt_v[g] + wb_v[g]); print(f"          their fruit share IN THIS VIEW alone: median {np.median(sh_v):.2f}, >=0.9 {int((sh_v >= 0.9).sum()):,}, <0.5 {int((sh_v < 0.5).sum()):,}  (a gaussian that is mostly leaf even in one view is bigger than the orange)")
# FP side: where the fruit-seeded gaussians' weight lands in this view
fs = fruit_seeded & seen; Wfs = wf_v[fs].sum(); Wts = wt_v[fs].sum(); Wbs = wb_v[fs].sum()
print(f"[FP side] fruit-seeded gaussians visible here: their alpha weight lands {Wfs / (Wfs + Wts + Wbs):.2f} on fruit pixels, {Wts / (Wfs + Wts + Wbs):.2f} on tree pixels, {Wbs / (Wfs + Wts + Wbs):.2f} on unlabelled -> the leaf share is lit as fruit")
print(f"          fruit-seeded gaussians here by global share: 0.1-0.5 {int((fs & (fshare < 0.5)).sum()):,} (promoted by the share rule), 0.5-0.9 {int((fs & (fshare >= 0.5) & (fshare < 0.9)).sum()):,}, >=0.9 {int((fs & (fshare >= 0.9)).sum()):,}; weight on tree pixels from the 0.1-0.5 ones {wt_v[fs & (fshare < 0.5)].sum() / max(Wts, 1e-9):.2f}")
