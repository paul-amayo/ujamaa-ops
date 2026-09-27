"""Per-gaussian supervision-interaction census through the RASTERIZER.

For every supervision label u and frame f: render the 32-d feature map with
grad enabled and take d(sum over u's pixels of channel-0)/d(high_features).
Because rasterized features are linear in per-gaussian features, that gradient
IS each gaussian's total alpha-blend weight into u's pixels in f. Accumulate
over all frames -> W[u] in R^N. No projection geometry, no markers — the
renderer itself reports who serves which supervision.

Then: for tree 73 (and 84, 60), which of its gaussians also serve OTHER
labels (collision), which labels, at what weight share, and where they sit.
Saves W to npz for the feature-init experiment.
"""
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
from high_splat_hierarchy_accuracy import build_hyper_embedder  # noqa: F401,E402
from splat_world_bridge import load_bridge, internal_to_world   # noqa: E402
from nerfstudio.utils.eval_utils import eval_setup              # noqa: E402

UNLAB = 65535
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--run-glob', required=True,
                help="glob for the run's config.yml (any run on the target geometry)")
ap.add_argument('--supervision-dir', required=True, type=Path)
ap.add_argument('--out-npz', required=True)
args = ap.parse_args()
SUP = args.supervision_dir
OUT = args.out_npz

CFG = sorted(glob.glob(args.run_glob))[-1]
_, pipe, _, _step = eval_setup(Path(CFG))
# step=0 collapses SH to degree 0 (fix 4e95db73) -> dark RGB
pipe.model.step = _step
model = pipe.model
model.eval()
model.image_encoder.set_positives(['__census__'])
hf = model.gauss_params['high_features']
hf.requires_grad_(True)
N = hf.shape[0]
ds = pipe.datamanager.train_dataset
names = [Path(f).name for f in ds.image_filenames]

W = defaultdict(lambda: np.zeros(N, np.float32))
torch.set_grad_enabled(True)
done = 0
for idx, name in enumerate(names):
    sf = SUP / name
    if not sf.exists():
        continue
    a0 = np.array(Image.open(sf), np.uint16)
    cam = ds.cameras[idx:idx + 1].to(model.device)
    cam.rescale_output_resolution(0.5)
    outs = model.get_outputs(cam)
    feat = outs['high_features']
    Hh, Ww = feat.shape[:2]
    a = np.array(Image.fromarray(a0).resize((Ww, Hh), Image.NEAREST))
    # 0 <= u: id 0 is a real tree (new censuses assign gid 0 to each survey's
    # first tree; only 65535 is void). The old `0 < u` gave every survey's
    # tree-0 gaussians NO identity at seed time — same one-character disease
    # as containment_eval's grader, found via kf_000007's unscored 192k-px
    # canopy (asus IoU 0.026 with its GT plainly in frame, 2026-08-19).
    uids = [int(u) for u in np.unique(a) if 0 <= u != UNLAB]
    # CENSUS_WITH_BG=1 (2026-09-27, H3DGS side-car): also census the UNLABELLED pixels as row 65535 — a gaussian's
    # alpha mass into void pixels — so the init can withhold identity from gaussians that mostly serve ground/sky
    # (they otherwise bleed a shrunk tree feature onto unlabelled pixels, which decodes as the row). Off by default.
    if os.environ.get('CENSUS_WITH_BG') == '1' and (a == UNLAB).any():
        uids.append(UNLAB)
    for k, u in enumerate(uids):
        m = torch.from_numpy(a == u).to(feat.device)
        loss = feat[..., 0][m].sum()
        g, = torch.autograd.grad(loss, hf, retain_graph=(k < len(uids) - 1))
        W[u] += g[:, 0].detach().cpu().numpy()
    del outs, feat
    done += 1
    if done % 25 == 0:
        print(f'  {done} frames censused')

labels = sorted(W)
Wm = np.stack([W[u] for u in labels])            # [L, N]
np.savez_compressed(OUT, W=Wm, labels=np.array(labels))
print(f'saved W {Wm.shape} -> {OUT}')

print('census complete — analyse/init from the npz')
