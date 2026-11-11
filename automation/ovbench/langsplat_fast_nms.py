#!/usr/bin/env python3
"""langsplat_fast_nms.py (2026-11-11, UJAMAA ovbench) — writes <LangSplat>/preprocess_fast.py: the released preprocess.py with only mask_nms's
pairwise Python loop replaced by the same computation on the GPU. In October the released preprocessing took 2,482-4,539 s per 1280x720
image at 891 % CPU and 0 % GPU: after SAM, the masks are compared pair by pair on the CPU (two full-image logical ops per pair, per level).
Here the intersections are one exact float32 matrix product of the binary masks (TF32 off; pixel counts < 2^24), and the unions, ratios and
the inner-overlap rule are the released expressions on the same (i <= j) pairs, so the kept masks are identical (checked by
langsplat_nms_check.py against the released function). Nothing else in the file changes.
usage: python3 langsplat_fast_nms.py /home/paperspace/code/LangSplat"""
import sys
from pathlib import Path
L = Path(sys.argv[1]); s = (L / 'preprocess.py').read_text()
a = s.index('def mask_nms('); b = s.index('def masks_update(')
FAST = '''def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """released mask_nms with its pairwise loop vectorised on the GPU (UJAMAA ovbench 2026-11-11); same pairs, same expressions"""
    scores, idx = scores.sort(0, descending=True)
    masks_ord = masks[idx.view(-1), :]
    n = masks_ord.shape[0]
    torch.backends.cuda.matmul.allow_tf32 = False
    M = masks_ord.reshape(n, -1).to("cuda", torch.float32)
    inter = M @ M.T
    area = M.sum(1)
    union = area[:, None] + area[None, :] - inter
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device="cuda"))
    iou_matrix = torch.where(upper, inter / union, torch.zeros_like(inter))
    ri = inter / area[:, None]
    rj = inter / area[None, :]
    val = 1 - rj * ri
    c1 = upper & (ri < 0.5) & (rj >= 0.85)
    c2 = upper & (ri >= 0.85) & (rj < 0.5)
    inner_iou_matrix = torch.zeros_like(inter)
    inner_iou_matrix[c1] = val[c1]
    inner_iou_matrix.T[c2] = val[c2]
    iou_matrix = iou_matrix.cpu(); inner_iou_matrix = inner_iou_matrix.cpu()
'''
rest = s[a:b]; tail = rest[rest.index('    iou_matrix.triu_(diagonal=1)'):]
(L / 'preprocess_fast.py').write_text(s[:a] + FAST + tail + s[b:]); print('wrote', L / 'preprocess_fast.py')
