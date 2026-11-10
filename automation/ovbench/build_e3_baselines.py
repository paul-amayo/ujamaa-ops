#!/usr/bin/env python3
"""build_e3_baselines.py — gather ovbench per-cell summaries into the paper's provenance file numbers/e3_baselines.json (2026-11-10).
For every method x cell x query: median IoU at relevancy >= 0.5 and at the oracle threshold (57-value grid), heads, training time and
steps, deviations from the published recipe; HiGH's medians on the same frames alongside; methods that did not run with the reason.
  build_e3_baselines.py --root /home/paperspace/logs/ovbench --out e3_baselines.json"""
import argparse, json, glob, os, re
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('--root', default='/home/paperspace/logs/ovbench'); ap.add_argument('--out', required=True)
a = ap.parse_args(); R = Path(a.root)
LERF_DEV = [
    'Source: kerrj/lerf at db08d57 (the last upstream commit, 2024-03-13), extracted with git archive from the history of the box clone '
    '(RSojen/lerf-scene-graph, a fork with sky loss, modified negatives and fused bounding-box embeddings, which is NOT used).',
    'BACKEND: tiny-cuda-nn is not installed on the box and installs only from GitHub (blocked). LERF\'s two hash grids and its CLIP / DINO '
    'heads run on nerfstudio\'s torch HashEncoding / MLP of the same shape (levels, resolutions, 2^19 tables, 8 features, 256-wide 4- and '
    '1-hidden-layer heads) — the fallback nerfstudio itself uses for nerfacto. Differences: biases in the torch heads (CutlassMLP has none), '
    'hash init +-1e-3 (tcnn +-1e-4), torch grid-vertex alignment, about 2x slower. LERF\'s nerfacto half runs on the same fallback.',
    'DINO ViT-S/8 instantiated through timm 0.6.7 (vit_small_patch8_224_dino = the official dino_deitsmall8_pretrain.pth from '
    'dl.fbaipublicfiles.com) behind dino\'s VisionTransformer interface, because torch.hub fetches facebookresearch/dino from GitHub; '
    'wrapper == timm forward to 0.0 at 224 x 224 (dino_timm.check()).',
    'Type annotations of five config fields made tyro-compatible (defaults unchanged); LERFPipeline\'s datamanager.to(device) guarded '
    '(nerfstudio 1.1 DataManager has no .to). nerfstudio is nerf_new\'s 1.1.5 (LERF db08d57 targets 1.0).',
    'Training split: the HiGH splatfacto run\'s own train_filenames (= identity split_names train); held-out verdict frames are in neither '
    'training set. Dataparser settings identical to the HiGH splatfacto run (orientation up, centre poses, auto scale).',
    'Memory-only settings: --vis tensorboard (published default: viewer), eval chunk 8192 rays (default 32768), render chunk 4096; LERF picks '
    'each query\'s best scale over the whole image, so chunking does not change the maps.',
    'Rendered at HiGH\'s 640 x 360 (render_scale 0.5) through the run\'s dataparser transform; held-out cameras from the chunk COLMAP model '
    '(self-checked against a workspace camera).',
    'Relevancy: LERF\'s own (30 scales 0-1.5, best scale per query per image, min over the canonical negatives object/things/stuff/texture). '
    'No void: every pixel is scored (HiGH marks void pixels).',
    'Query list per cell = "tree", "orange", "the tree on the right", "tree <id>" for the cell\'s 38-ish scene-graph trees and "row <id>" for their rows '
    '(includes "row -1", the scene graph\'s unassigned row); every query is rendered in one pass, LERF picks each one\'s scale separately.']
CELL_DEV = {
    '05_13D_Jackal_1_0': ['Trained in two segments of one run: steps 0-25000 (8,867 s wall, including the CLIP/DINO cache load and nerfstudio\'s '
                          'periodic full-resolution eval images), then stopped at nerfstudio\'s step-25000 all-image evaluation (71 val images at '
                          '1280 x 720 through the torch backend, eval only) and resumed from the step-24000 checkpoint (model, optimisers, schedulers) '
                          'to 30000 with that evaluation disabled (2,086 s). Steps 24000-25000 were therefore trained twice from the same state; '
                          'the final weights are step 29999.',
                          'The CLIP pyramid / DINO caches were built once (attempt 1, ~6 min for 294 views) and reused by the final run.',
                          'Four earlier attempts failed or were stopped (tyro annotations; nerfstudio 1.1 datamanager.to; tiny-cuda-nn missing; '
                          'memory (eval chunk 32768 grew the trainer to 27 GB on the shared card); root disk filled at step 2030 by this session\'s '
                          'own checkpoint copy) — logs in attempt_logs/.'],
    '_default': ['Eval-only schedule: --steps-per-eval-image and --steps-per-eval-all-images set beyond the run (no periodic full-resolution '
                 'eval renders); the trained weights do not depend on them.']}
SKIPPED = {
    'openhype': {'ran': False, 'reason': 'OpenHype extracts its mask hierarchy only with Semantic-SAM (openhype_preprocess/img_seg_model.py supports '
                 'model_type "semantic_sam" alone). Semantic-SAM, its detectron2-xyz dependency and the SwinL checkpoint swinl_only_sam_many2many.pth '
                 'are not on the box and come only from GitHub (UX-Decoder/Semantic-SAM, MaureenZOU/detectron2-xyz, a GitHub release asset), which the '
                 'box may not fetch; its NeRF stage also needs tiny-cuda-nn (GitHub only, absent). No OpenHype preprocessing, AE or NeRF output exists '
                 'on the box. Not run; nothing substituted.'},
    'langsplat': {'ran': False, 'reason': 'Not run in this session. Its preprocessing (SAM ViT-H automatic masks, 32 points/side + 1 crop layer, '
                  'CLIP per mask at three levels) was measured on 05 1_0 in earlier sessions at 4,539 s and 2,482 s for the first image '
                  '(lab notebook 2026-10-12/13), CPU-bound at ~9 cores; at those measured rates the 294 training views of 05 1_0 would need 203-371 h of preprocessing '
                  'before any training. The time went to LERF.'}}
out = {'generated': __import__('datetime').datetime.utcnow().isoformat(timespec='seconds') + 'Z', 'protocol': {
    'frames': '<identity>/frames.json (census/training + 10 held-out); headline = held-out', 'min_frac': 0.002, 'oracle_grid': 'protocol_table.THR (57 values 0.4-0.999)',
    'standard_threshold': 'relevancy >= 0.5', 'queries': {'tree': 'category, every labelled pixel (E3 "ALL")', 'orange': 'category, fruit pixels (ids >= 10000)',
    'tree <id>': 'identity address, scene-graph tree id; GT = tree + its fruit', 'row <id>': 'identity address, marker_hierarchy row_id; GT = member trees',
    'the tree on the right': 'spatial phrase; GT = the TREE head (>= 0.2 %) whose supervised centroid is furthest right'},
    'argmax05': 'TREE/ROW only: pixel goes to the address with the highest relevancy among the cell\'s addresses of that level AND relevancy >= 0.5 (open-vocabulary analogue of HiGH argmax)',
    'high': 'HiGH medians from <identity>/protocol_v3.jsonl (argmax / oracle) and protocol_cat_v3.jsonl (rel05 / oracle / argmax), same frames and filter; v3vf when present'},
    'methods': {}}
for m in ('lerf',):
    cells = {}
    for sj in sorted(glob.glob(str(R / m / '*' / 'summary.json'))):
        W = Path(sj).parent; s = json.load(open(sj)); tt = json.load(open(W / 'train_time.json')) if (W / 'train_time.json').exists() else {}
        steps = None; cfg = W / 'outputs' / f'ovbench_{W.name}' / m / 'run' / 'config.yml'
        if cfg.exists():
            mm = re.search(r'^max_num_iterations: (\d+)', open(cfg).read(), re.M); steps = int(mm.group(1)) if mm else None
        ck = sorted(glob.glob(str(cfg.parent / 'nerfstudio_models' / '*.ckpt')))
        cells[W.name] = {'train_time': tt, 'cell_deviations': CELL_DEV.get(W.name, CELL_DEV['_default']), 'max_num_iterations': steps, 'final_checkpoint': os.path.basename(ck[-1]) if ck else None,
                         'train_views': None, 'frames_scored': s['frames'], 'gt_check': s['gt_check'], 'method_medians': s['method_medians'], 'high_medians': s['high_medians']}
        q = W / 'queries.json'
        if q.exists(): cells[W.name]['n_queries'] = len(json.load(open(q))['queries'])
        tl = W / 'train.log'
        if tl.exists():
            mm = re.search(r'Caching all (\d+) images', open(tl, errors='replace').read()); cells[W.name]['train_views'] = int(mm.group(1)) if mm else None
    out['methods'][m] = {'ran': bool(cells), 'variant': 'lerf (default config: ViT-B-16 laion2b_s34b_b88k, hashgrids (19,19)/(12,12)/((16,128),(128,512)), 24 LERF samples, 30k steps)',
                         'deviations': LERF_DEV, 'cells': cells}
out['methods'].update(SKIPPED)
json.dump(out, open(a.out, 'w'), indent=1); print(f'[e3_baselines] {a.out}: ' + ', '.join(f'{k} {len(v.get("cells", {}))} cells' for k, v in out['methods'].items()))
