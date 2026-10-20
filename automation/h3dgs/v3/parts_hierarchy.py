#!/usr/bin/env python3
"""parts_hierarchy.py (2026-10-20, CVPR super hierarchy) — the survey's scene graph plus the part-type nodes of one cell, for the embedder:
copies marker_hierarchy.json and adds "parts": [{id, tree_id, type}] from parts/nodes.npz and "part_anchors": the npz path.
system python3:  parts_hierarchy.py <marker_hierarchy.json> <nodes.npz> <out json>"""
import json, sys
import numpy as np
H = json.load(open(sys.argv[1])); z = np.load(sys.argv[2]); objs = {int(o['id']) for o in H['objects']}
parts = [{'id': int(i), 'tree_id': int(t), 'type': int(ty)} for i, t, ty in zip(z['ids'], z['tree'], z['type']) if int(t) in objs]
H['parts'] = parts; H['part_anchors'] = sys.argv[2]; H.setdefault('_provenance', {}); H['_provenance']['parts'] = f'{len(parts)} part-type nodes (trunk 0 / branch 1 / leaves 2) under the trees of one cell, anchors = mean mask-crop CLIP image embeddings (parts_nodes.py)'
json.dump(H, open(sys.argv[3], 'w'), indent=1); print(f"[hier] {len(H['objects'])} objects, {len(H['rows'])} rows, {len(H.get('fruits', []))} fruits, {len(parts)} parts -> {sys.argv[3]}")
