"""Per-block colour -> 32-d HiGH target tables for the H3DGS feature stage, built exactly as HighDataloader.create():
palette colour -> id -> vocab word (MASK_WORDS / FRUIT_WORDS, node type leaf/fruit) -> CLIP text embedding
(open_clip ViT-B-16 laion2b) -> HyperEmbedder.encode_features(project=True, node_types) -> log_map0 (survey embedder
from HIGH_EMBEDDER_CKPT). Writes <block>/semantic_v2_B/high_targets.npz {colors uint8 [n,3], targets float32 [n,32],
words}. Run in the nerf_new pixi env:
  HIGH_EMBEDDER_CKPT=<survey embedder ckpt> python export_high_targets.py <blocks_ns cfg dir>"""
import os, sys, json, glob, numpy as np, torch
from pathlib import Path
from high.data.vocab import MASK_WORDS, FRUIT_WORDS
from high.encoders.clip_encoder import ClipNetwork, ClipNetworkConfig
from high.encoders.hyperbolic_ae import load_hyper_embedder
import high.encoders.lorentz as L
NODE_TYPE_LEAF, NODE_TYPE_FRUIT = 0, 4
cfg_dir = Path(sys.argv[1]); ckpt = os.environ["HIGH_EMBEDDER_CKPT"]
clip = ClipNetwork(ClipNetworkConfig()); he = load_hyper_embedder(ckpt, "cuda"); he.eval(); curv = he.curv.exp()
print(f"[targets] embedder {ckpt} curv {float(curv):.3f}; MASK_WORDS {len(MASK_WORDS)}, FRUIT_WORDS {len(FRUIT_WORDS)}", flush=True)
done = 0
for b in sorted(cfg_dir.glob("block_[0-9][0-9][0-9]")):
    pal_p = b / "semantic_v2_B" / "palette.json"
    if not pal_p.exists(): continue
    pal = json.load(open(pal_p)); man_p = b / "semantic_v2_B" / "supervision_manifest.json"
    fruit_base = int(json.load(open(man_p)).get("fruit_id_base", 200)) if man_p.exists() else 200
    colors, words, ntypes = [], [], []
    for k, idx in pal.items():
        r, g, bb = [int(x) for x in k.split(",")]
        if (r, g, bb) == (0, 0, 0): continue
        idx = int(idx)
        if idx >= fruit_base:
            fi = idx - fruit_base; w = FRUIT_WORDS[fi] if fi < len(FRUIT_WORDS) else f"{FRUIT_WORDS[fi % len(FRUIT_WORDS)]} {fi // len(FRUIT_WORDS)}"; nt = NODE_TYPE_FRUIT
        elif idx < len(MASK_WORDS): w, nt = MASK_WORDS[idx], NODE_TYPE_LEAF
        else: w, nt = "Background", NODE_TYPE_LEAF
        colors.append((r, g, bb)); words.append(w); ntypes.append(nt)
    with torch.no_grad():
        ce = clip.encode_text(words)
        hyper = he.encode_features(ce.float(), project=True, node_types=torch.tensor(ntypes, dtype=torch.long, device=ce.device))
        lat = L.log_map0(hyper, curv=curv).float().cpu().numpy()
    np.savez(b / "semantic_v2_B" / "high_targets.npz", colors=np.array(colors, np.uint8), targets=lat.astype(np.float32), words=np.array(words))
    done += 1
    if done <= 2 or done % 10 == 0: print(f"[targets] {b.name}: {len(colors)} colours, target norms p50 {np.median(np.linalg.norm(lat, axis=1)):.2f}, sample words {words[:3]}", flush=True)
print(f"[targets] wrote high_targets.npz for {done} blocks", flush=True)
