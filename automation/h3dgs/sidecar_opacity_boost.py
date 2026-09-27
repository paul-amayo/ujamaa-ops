"""Identity-proxy opacity boost for a side-car seed checkpoint (2026-09-27). The H3DGS side-car reproduces see-through
canopies (accumulated alpha p10 0.81 on a tree mask vs 0.996 for the block model), so on 12-19 % of a canopy's pixels the
alpha blend is won by the tree BEHIND (sidecar_feature_diag.py: 'indicated' pixels argmax 'reigns' 18.8 %, 'mosul' ->
'secretariat' 11.9 %; own-argmax pixels have full-length vectors). The side-car is an identity field, not a radiance
field (RGB comes from H3DGS itself), so raise the opacity of every gaussian that CARRIES an identity (nonzero census
feature) to at least --min-opacity: the nearest labelled surface then owns the pixel. Void gaussians are left as they are.
  python sidecar_opacity_boost.py <seed ckpt in> <ckpt out> [--min-opacity 0.95]"""
import argparse, torch
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--min-opacity", type=float, default=0.95); a = ap.parse_args()
ck = torch.load(a.src, map_location="cpu", weights_only=False); P = ck["pipeline"]
kf = [k for k in P if k.endswith("gauss_params.high_features")][0]; ko = kf.replace("high_features", "opacities")
carry = P[kf].abs().sum(1) > 0; op = torch.sigmoid(P[ko].reshape(-1)); boosted = carry & (op < a.min_opacity)
op2 = torch.where(boosted, torch.full_like(op, a.min_opacity), op); P[ko] = torch.log(op2 / (1 - op2)).reshape(P[ko].shape)
torch.save(ck, a.dst)
print(f"[boost] {int(carry.sum())} identity-carrying gaussians of {carry.numel()}; {int(boosted.sum())} raised to opacity {a.min_opacity} (their median was {op[boosted].median().item() if boosted.any() else float('nan'):.3f}) -> {a.dst}", flush=True)
