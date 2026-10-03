"""Fruit mask anatomy at the verdict threshold: GT blobs, boundary-tolerant precision/recall, orange-level hits.
Same heat as containment_eval.containment_heat (own point + 4 ancestor steps, norm gate 0.5), CLIP fp16 as clip_encoder.py."""
import sys, json, numpy as np, torch, open_clip
from PIL import Image
from scipy import ndimage
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
import lorentz as L
from high_splat_hierarchy_accuracy import build_hyper_embedder
from word_utils import get_word_for_id
S = '/home/paperspace/data/citrus_all/05_13D_Jackal'; N = f'{S}/experimental/h3dgs_native/chunk_1_0_expo_fruit'
SUP = f'{S}/experimental/h3dgs_sidecar_chunks/chunk_1_0_expo/supervision/trees_fruit_v3'
hyper = build_hyper_embedder(f'{S}/prod/bateleur/embedder/05_13D_v1g/ckpts/model_best.pth', 'cuda'); curv = hyper.curv.exp()
m, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k', precision='fp16'); m = m.eval().cuda(); tok = open_clip.get_tokenizer('ViT-B-16')
def heat(feat, word):
    e = m.encode_text(tok(word).cuda()).float(); e = e / e.norm(dim=-1, keepdim=True)
    ft = torch.from_numpy(feat.reshape(-1, feat.shape[-1])).float().cuda(); out = []
    with torch.no_grad():
        for j in range(0, ft.shape[0], 4096):
            h = L.exp_map0(ft[j:j + 4096], curv=curv); path, mask = L.get_interpolated_hyperbolic_features(h, steps=4, curv=curv, max_dist=11.1, return_mask=True)
            d = hyper.decode_features(path, project=True); d = d / d.norm(dim=-1, keepdim=True); sm = (d @ e.T).view(-1, 4).masked_fill(mask.cuda().bool(), -1.0)
            d0 = hyper.decode_features(h, project=True); d0 = d0 / d0.norm(dim=-1, keepdim=True); out.append(torch.maximum(sm.max(1).values, (d0 @ e.T).squeeze(-1)).cpu().numpy())
    hm = np.concatenate(out).reshape(feat.shape[:2]); hm[np.linalg.norm(feat, axis=-1) < 0.5] = -1.0; return hm
for frame, runs in (('kf_000025.png', {'native': 0.91, 'sidecar': 0.79}), ('kf_003285.png', {'native': 0.84, 'sidecar': 0.68})):
    a = np.array(Image.open(f'{SUP}/{frame}'), np.uint16)
    fid = next(int(u) for u in np.unique(a) if 10000 <= u < 65535 and get_word_for_id(int(u) - 10000, 'fruit') == 'wildly')
    for tag, thr in runs.items():
        z = np.load(f'{N}/maps_{"native" if tag == "native" else "sidecar"}/{frame[:-4]}_{"tau3" if tag == "native" else "sidecar"}.npz'); f = z['features']
        gt = np.array(Image.fromarray(a).resize((f.shape[1], f.shape[0]), Image.NEAREST)) == fid
        p = heat(f, 'wildly') >= thr
        inter = (p & gt).sum(); iou = inter / max((p | gt).sum(), 1)
        lab, n = ndimage.label(gt, structure=np.ones((3, 3))); sizes = ndimage.sum(gt, lab, range(1, n + 1))
        tol = {}
        for k in (1, 2):
            gk = ndimage.binary_dilation(gt, iterations=k); pk = ndimage.binary_dilation(p, iterations=k)
            tol[k] = ((p & gk).sum() / max(p.sum(), 1), (gt & pk).sum() / max(gt.sum(), 1))
        hit = sum(1 for i in range(1, n + 1) if (ndimage.binary_dilation(lab == i, iterations=1) & p).any())
        plab, pn = ndimage.label(p, structure=np.ones((3, 3))); g1 = ndimage.binary_dilation(gt, iterations=1)
        false_blobs = sum(1 for i in range(1, pn + 1) if not ((plab == i) & g1).any())
        lab_full = np.array(Image.fromarray(a).resize((f.shape[1], f.shape[0]), Image.NEAREST)); off = p & ~ndimage.binary_dilation(gt, iterations=1)
        tree_of = {fo['id']: fo['tree_id'] for fo in json.load(open(f'{S}/prod/bateleur/scene_graph/marker_hierarchy.json')).get('fruits', [])}
        t5 = tree_of.get(fid - 10000); v = lab_full[off]
        cls = {'other fruit': int(((v >= 10000) & (v < 65535)).sum()), f'canopy of the same tree ({t5})': int((v == t5).sum()),
               'other trees': int(((v < 10000) & (v != t5)).sum()), 'unlabelled': int((v == 65535).sum())}
        print(f'    {frame} {tag} lit px > 1 px from the fruit: {int(off.sum())} -> ' + ', '.join(f'{k} {c}' for k, c in cls.items()))
        print(f'{frame} {tag:8s} thr {thr}: GT {int(gt.sum())} px in {n} blobs (median {np.median(sizes):.0f} px, ~{2*np.sqrt(np.median(sizes)/np.pi):.0f} px across; largest {int(sizes.max())}) | lit {int(p.sum())} px | IoU {iou:.3f} | '
              f'within 1 px: prec {tol[1][0]:.2f} rec {tol[1][1]:.2f} | within 2 px: prec {tol[2][0]:.2f} rec {tol[2][1]:.2f} | GT blobs lit {hit}/{n} | lit blobs off-fruit {false_blobs}/{pn}')
