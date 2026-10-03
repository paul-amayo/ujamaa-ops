#!/usr/bin/env python3
"""Text bank for native H3DGS identity serving (UJAMAA, 2026-10-03): the CLIP text embedding of every word a query can
resolve to — the supervision manifest's word_table (trees, fruit) and the hierarchy's row words — computed exactly as
high/encoders/clip_encoder.py does (stock open_clip ViT-B-16 laion2b_s34b_b88k, fp16, raw words), because the h3dgs env
that serves the hierarchy has no open_clip. Extra free-text words with --extra.
nerf_new env:  pixi run python build_text_bank.py --manifest <supervision/*/manifest.json> --hierarchy-json <json>
                  --out <cell dir>/text_bank.npz [--extra orange tree row]"""
import argparse, json, sys
import numpy as np, torch, open_clip
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH')
from word_utils import get_word_for_id
ap = argparse.ArgumentParser()
ap.add_argument('--manifest', required=True); ap.add_argument('--hierarchy-json', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--extra', nargs='*', default=[])
a = ap.parse_args()
wt = {str(k): v for k, v in json.load(open(a.manifest)).get('word_table', {}).items()}
H = json.load(open(a.hierarchy_json))
rows = sorted({int(o['row_id']) for o in H['objects'] if o.get('row_id') is not None and int(o['row_id']) >= 0})
row_words = {str(r): get_word_for_id(r, 'row') for r in rows}
words = sorted(set(wt.values()) | set(row_words.values()) | set(a.extra))
m, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k', precision='fp16')
m.eval(); m = m.to('cuda'); tok = open_clip.get_tokenizer('ViT-B-16')
with torch.no_grad():
    E = torch.cat([m.encode_text(tok(words[i:i + 64]).to('cuda')).float() for i in range(0, len(words), 64)]).cpu().numpy()
meta = {'word_table': wt, 'row_words': row_words, 'manifest': a.manifest, 'hierarchy_json': a.hierarchy_json,
        'clip': 'ViT-B-16/laion2b_s34b_b88k fp16 (high/encoders/clip_encoder.py)'}
np.savez(a.out, words=np.array(words), emb=E.astype(np.float32), meta=json.dumps(meta))
print(f'[text-bank] {len(words)} words ({len(wt)} word-table ids, {len(row_words)} rows, {len(a.extra)} extra) -> {a.out}')
