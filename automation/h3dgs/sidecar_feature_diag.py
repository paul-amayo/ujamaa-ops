"""Where does a seed's rendered identity go wrong on one frame? For every supervision label on the frame: the pixels'
nearest census target (cosine in the tangent space, over the labels of the census) and how long the rendered features
are relative to that label's target — separates "label confusion" (argmax = another tree) from "shrunk features"
(right direction, short vector -> decodes at the row level) and "no identity" (below containment_eval's 0.5 gate).
  HIGH_EMBEDDER_CKPT=<emb> pixi run python sidecar_feature_diag.py --config <seed config.yml> --w-npz <census npz>
      --embedder <emb> --supervision-dir <ids> --frame kf_XXXXXX.png"""
import argparse, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH')
from high_splat_hierarchy_accuracy import render_frame, build_hyper_embedder
from word_utils import get_word_for_id
from nerfstudio.utils.eval_utils import eval_setup
import lorentz as L, open_clip
FRUIT_ID_BASE = 10000; UNLAB = 65535; dev = 'cuda'; torch.set_grad_enabled(False)
ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--w-npz', required=True); ap.add_argument('--embedder', required=True)
ap.add_argument('--supervision-dir', required=True); ap.add_argument('--frame', required=True); a = ap.parse_args()
_, pipe, _, step = eval_setup(Path(a.config)); model = pipe.model; model.eval(); model.step = step; ds = pipe.datamanager.train_dataset
names = [Path(f).name for f in ds.image_filenames]; i = names.index(a.frame); cam = ds.cameras[i:i + 1].to(model.device); cam.rescale_output_resolution(0.5)
rgb, alpha, feat = render_frame(model, cam, model.config.lang_field_dim); H, W_, D = feat.shape; f = torch.from_numpy(np.asarray(feat).reshape(-1, D)).to(dev); fn = f.norm(dim=-1)
z = np.load(a.w_npz); labels = [int(u) for u in z['labels'] if int(u) != UNLAB]
m, _, _ = open_clip.create_model_and_transforms('ViT-B-16', 'laion2b_s34b_b88k', device=dev); tok = open_clip.get_tokenizer('ViT-B-16')
hyper = build_hyper_embedder(a.embedder, dev); curv = hyper.curv.exp()
words = [(get_word_for_id(u - FRUIT_ID_BASE, 'fruit') if u >= FRUIT_ID_BASE else get_word_for_id(u, 'mask')) for u in labels]
nt = torch.tensor([4 if u >= FRUIT_ID_BASE else 0 for u in labels], device=dev)
tl = L.log_map0(hyper.encode_features(m.encode_text(tok(words).to(dev)).float(), project=True, node_types=nt), curv=curv)   # (L, D) targets
tn = tl.norm(dim=-1); cos = (f @ tl.T) / (fn[:, None] * tn[None, :] + 1e-9); am = cos.argmax(1)
sup = np.array(Image.fromarray(np.array(Image.open(Path(a.supervision_dir) / a.frame), np.uint16)).resize((W_, H), Image.NEAREST)).reshape(-1)
print(f'[diag] {a.frame}: {len(labels)} census labels {dict(zip(labels, words))}; target norms {dict(zip(labels, tn.cpu().numpy().round(2)))}')
for u in [int(v) for v in np.unique(sup) if v != UNLAB]:
    if u not in labels: continue
    mk = torch.from_numpy(sup == u).to(dev); n = int(mk.sum()); ui = labels.index(u); gate = (fn[mk] < 0.5)
    hit = (am[mk] == ui) & ~gate; conf = {}
    for j, v in enumerate(labels):
        c = int(((am[mk] == j) & ~gate).sum())
        if c: conf[words[j]] = round(100.0 * c / n, 1)
    r_hit = (fn[mk][hit] / tn[ui]).cpu().numpy(); r_all = (fn[mk][~gate] / tn[ui]).cpu().numpy()
    print(f'[diag] label {u} "{words[ui]}" ({n} px): argmax = own {100.0 * hit.float().mean():.1f}%, no identity {100.0 * gate.float().mean():.1f}%, others {dict(sorted(conf.items(), key=lambda kv: -kv[1]))}; '
          f'|f|/|t| own-argmax median {np.median(r_hit) if len(r_hit) else float("nan"):.2f} p10 {np.percentile(r_hit, 10) if len(r_hit) else float("nan"):.2f}; own-cos median {cos[mk][:, ui][hit].median().item() if hit.any() else float("nan"):.3f}')
mk = torch.from_numpy(sup == UNLAB).to(dev); gate = fn[mk] < 0.5; amu = am[mk][~gate]
print(f'[diag] unlabelled ({int(mk.sum())} px): no identity {100.0 * gate.float().mean():.1f}%; identity-carrying pixels argmax ' + str({words[j]: round(100.0 * int((amu == j).sum()) / max(1, int((~gate).sum())), 1) for j in range(len(labels)) if int((amu == j).sum())}) + f'; their |f|/|t| median {(fn[mk][~gate] / tn[am[mk][~gate]]).median().item() if (~gate).any() else float("nan"):.2f}')
