"""trunk_fruitmask.py (2026-10-20, super hierarchy LoD route) — trunk ∪ fruit as the LoD small-object class for HIGH_FRUIT_MASK_DIR: per training image a uint8 PNG, 0 unlabelled, 1 labelled non-target
(tree ids), 2 target (trunk node px 20000+4k+0 from parts/v4/supervision, real fruit ids 10000..19999). Resized NEAREST to the image size."""
import json, sys, numpy as np
from pathlib import Path
from PIL import Image
W = Path('/home/paperspace/data/citrus_all/05_13D_Jackal'); SUP = W / 'experimental/v3/chunk_1_0/parts/v4/supervision'; IM = W / 'experimental/splatfacto_1_0/images'; OUT = W / 'experimental/v3/chunk_1_0/parts/v4/fruitmask_trunk'; OUT.mkdir(exist_ok=True)
n = 0; px = {'trunk': 0, 'fruit': 0, 'labelled': 0, 'total': 0}; frames_with_trunk = 0; resized = 0
for p in sorted(SUP.glob('kf_*.png')):
    ids = np.array(Image.open(p)).astype(np.int64); trunk = (ids >= 20000) & (ids < 65535) & ((ids - 20000) % 4 == 0); fruit = (ids >= 10000) & (ids < 20000); lab = ids != 65535
    m = np.zeros(ids.shape, np.uint8); m[lab] = 1; m[trunk | fruit] = 2
    im = Image.open(IM / p.name); 
    if im.size != (m.shape[1], m.shape[0]): m = np.array(Image.fromarray(m).resize(im.size, Image.NEAREST)); resized += 1
    Image.fromarray(m).save(OUT / p.name); n += 1; px['trunk'] += int(trunk.sum()); px['fruit'] += int(fruit.sum()); px['labelled'] += int(lab.sum()); px['total'] += ids.size; frames_with_trunk += int(trunk.any())
json.dump({'rule': 'HIGH_FRUIT_MASK_DIR for the trunk LoD run (10-20): 2 = trunk node px (parts/v4/supervision ids 20000+4k) or real fruit ids (10000..19999), 1 = other labelled px, 0 = unlabelled', 'source': str(SUP), 'frames': n, 'frames_with_trunk': frames_with_trunk, 'px': px, 'resized_frames': resized}, open(OUT / 'manifest.json', 'w'), indent=1)
print(f"[fruitmask] {n} frames ({frames_with_trunk} with a trunk; {resized} resized) -> {OUT}; trunk px {px['trunk']} ({100*px['trunk']/px['total']:.3f} % of all px), fruit px {px['fruit']}, labelled {100*px['labelled']/px['total']:.1f} %")
