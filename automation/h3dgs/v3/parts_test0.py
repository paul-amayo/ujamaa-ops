#!/usr/bin/env python3
"""parts_test0.py (2026-10-20, CVPR super hierarchy, test 0) — are the parts of neighbouring trees separable in CLIP image space, and
does a part word find its part inside a tree? On the frames that parts_masks.py processed: the two-level mask-crop CLIP embedding of
every supervised tree mask and of every part mask (the HiGH MaskCropEncoder, ViT-B-16 laion2b, the model of the text bank), then
  (1) cosines: tree vs tree in the frame; same-prompt parts across trees; different-prompt parts within a tree; same-prompt within a tree
  (2) nearest neighbour of each part among the frame's other parts: same tree? same prompt? (chance levels reported)
  (3) text -> part retrieval inside each tree: for a query word, the top-scoring part of the tree by cosine; accuracy against the prompt
      that produced the parts, for the prompted words and for unprompted synonyms (bark, stem, wood, foliage, canopy)
  (4) for the trunk word: cosine to the trunk part against cosine to the whole tree crop (is the part more trunk-like than its tree?)
Writes <parts dir>/test0.json and prints the summary.  nerf_new pixi env:  parts_test0.py <identity dir> <images dir> <parts dir>"""
import json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH')
from clip_image_embedder import MaskCropEncoder, MaskCropEncoderConfig
N, IM, PD = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); LEVELS = int(sys.argv[4]) if len(sys.argv) > 4 else 2; dev = 'cuda'; torch.set_grad_enabled(False)
meta = json.loads(str(np.load(N / 'text_bank.npz', allow_pickle=True)['meta'])); sup = Path(meta['manifest']).parent
enc = MaskCropEncoder(MaskCropEncoderConfig(), device=dev)
WORDS = {'tree trunk': 'tree trunk', 'trunk': 'tree trunk', 'bark': 'tree trunk', 'stem': 'tree trunk', 'wood': 'tree trunk', 'branch': 'branch', 'branches': 'branch', 'leaves': 'leaves', 'foliage': 'leaves', 'canopy': 'leaves'}
T = enc.get_text_embeddings(list(WORDS)); T = torch.nn.functional.normalize(T.float(), dim=-1).cpu().numpy()
def embed(img, masks):
    crops = torch.stack([enc._get_multi_level_crops(torch.from_numpy(m.astype(np.uint8)), img, num_levels=LEVELS, pad=True) for m in masks])
    out = []
    for j in range(0, len(crops), 16): out.append(enc.model.encode_image(crops[j:j + 16].flatten(0, 1).to(dev), normalize=False).float().view(-1, LEVELS, 512).mean(1).cpu())
    e = torch.cat(out).numpy(); return e / np.linalg.norm(e, axis=1, keepdims=True)
S = {'tree_tree': [], 'same_prompt_cross_tree': {}, 'diff_prompt_same_tree': [], 'same_prompt_same_tree': [], 'nn_same_tree': [], 'nn_same_tree_chance': [], 'nn_same_prompt': [], 'nn_same_prompt_chance': [],
     'retrieval': {w: [] for w in WORDS}, 'trunk_part_minus_tree': [], 'frames': 0, 'parts': 0, 'trees': 0}
for jf in sorted(PD.glob('kf_*.json')):
    recs = json.load(open(jf)); f = jf.name.replace('.json', '.png')
    if not recs: continue
    z = np.load(PD / jf.name.replace('.json', '.npz')); H, W = z['shape']; M = np.unpackbits(z['masks'], axis=1)[:, :H * W].reshape(-1, H, W).astype(bool)
    img = torch.from_numpy(np.array(Image.open(IM / f).convert('RGB'))).permute(2, 0, 1); ids = np.array(Image.open(sup / f))
    trees = sorted({r['tree'] for r in recs}); Et = embed(img, [ids == t for t in trees]); Ep = embed(img, M); S['frames'] += 1; S['parts'] += len(recs); S['trees'] += len(trees)
    tp = np.array([r['tree'] for r in recs]); pp = np.array([r['prompt'] for r in recs]); C = Ep @ Ep.T; np.fill_diagonal(C, -2)
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)): S['tree_tree'].append(float(Et[i] @ Et[j]))
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            c = float(Ep[i] @ Ep[j])
            if tp[i] != tp[j] and pp[i] == pp[j]: S['same_prompt_cross_tree'].setdefault(pp[i], []).append(c)
            elif tp[i] == tp[j] and pp[i] != pp[j]: S['diff_prompt_same_tree'].append(c)
            elif tp[i] == tp[j] and pp[i] == pp[j]: S['same_prompt_same_tree'].append(c)
        if len(recs) > 1:
            k = int(C[i].argmax()); S['nn_same_tree'].append(float(tp[k] == tp[i])); S['nn_same_tree_chance'].append(float(((tp == tp[i]).sum() - 1) / (len(recs) - 1)))
            S['nn_same_prompt'].append(float(pp[k] == pp[i])); S['nn_same_prompt_chance'].append(float(((pp == pp[i]).sum() - 1) / (len(recs) - 1)))
    for ti, t in enumerate(trees):
        sel = np.where(tp == t)[0]; present = set(pp[sel])
        if len(present) < 2: continue
        sc = Ep[sel] @ T.T   # [parts of the tree, words]
        for wi, w in enumerate(WORDS):
            if WORDS[w] not in present: continue
            S['retrieval'][w].append(float(pp[sel][int(sc[:, wi].argmax())] == WORDS[w]))
        tr = [i for i in sel if pp[i] == 'tree trunk']
        if tr: S['trunk_part_minus_tree'].append(float(max(Ep[i] @ T[list(WORDS).index('trunk')] for i in tr) - Et[ti] @ T[list(WORDS).index('trunk')]))
med = lambda v: round(float(np.median(v)), 3) if len(v) else None; mean = lambda v: round(float(np.mean(v)), 3) if len(v) else None
out = {'frames': S['frames'], 'trees': S['trees'], 'parts': S['parts'], 'cos_tree_tree_median': med(S['tree_tree']), 'cos_tree_tree_range': [round(float(np.min(S['tree_tree'])), 3), round(float(np.max(S['tree_tree'])), 3)] if S['tree_tree'] else None,
       'cos_same_prompt_cross_tree_median': {p: med(v) for p, v in S['same_prompt_cross_tree'].items()}, 'cos_diff_prompt_same_tree_median': med(S['diff_prompt_same_tree']), 'cos_same_prompt_same_tree_median': med(S['same_prompt_same_tree']),
       'nn_same_tree': mean(S['nn_same_tree']), 'nn_same_tree_chance': mean(S['nn_same_tree_chance']), 'nn_same_prompt': mean(S['nn_same_prompt']), 'nn_same_prompt_chance': mean(S['nn_same_prompt_chance']),
       'text_to_part_top1_in_tree': {w: {'acc': mean(v), 'n': len(v)} for w, v in S['retrieval'].items()}, 'trunk_word_part_minus_tree_median': med(S['trunk_part_minus_tree'])}
out['crop_levels'] = LEVELS; json.dump(out, open(PD / f'test0_levels{LEVELS}.json', 'w'), indent=1)
print(f"[test0] {out['frames']} frames, {out['trees']} tree crops, {out['parts']} parts")
print(f"[test0] cosines: tree-tree {out['cos_tree_tree_median']} (range {out['cos_tree_tree_range']}); same prompt across trees {out['cos_same_prompt_cross_tree_median']}; different prompts within a tree {out['cos_diff_prompt_same_tree_median']}; same prompt within a tree {out['cos_same_prompt_same_tree_median']}")
print(f"[test0] nearest part: same tree {out['nn_same_tree']} (chance {out['nn_same_tree_chance']}); same prompt {out['nn_same_prompt']} (chance {out['nn_same_prompt_chance']})")
print('[test0] text -> part top-1 inside a tree: ' + ', '.join(f"{w} {d['acc']} (n={d['n']})" for w, d in out['text_to_part_top1_in_tree'].items()))
print(f"[test0] 'trunk' word: cosine to the trunk part minus cosine to the whole tree, median {out['trunk_word_part_minus_tree_median']}")
