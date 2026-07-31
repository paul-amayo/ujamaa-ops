"""Render GT|prediction strips for the G2 sky-variant comparison.
Usage (nerf_new pixi, cwd aru_sil_core/src, transforms must be GL-swapped):
  python g2_render.py <out_dir> <frame_idx...> -- <name=config.yml> ...
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
ARU = Path("/home/paperspace/code/aru_sil_core")
sys.path.insert(0, str(ARU / "src/scripts"))
from high_splat_hierarchy_accuracy import render_frame
from nerfstudio.utils.eval_utils import eval_setup

args = sys.argv[1:]
sep = args.index("--")
out = Path(args[0]); out.mkdir(parents=True, exist_ok=True)
frames = [int(x) for x in args[1:sep]]
variants = [a.split("=", 1) for a in args[sep + 1:]]

for name, cfg in variants:
    _, pipe, _, step = eval_setup(Path(cfg))
    model = pipe.model; model.eval()
    tds = pipe.datamanager.train_dataset
    for fi in frames:
        cam = tds.cameras[fi:fi + 1].to(model.device)
        rgb, alpha, _ = render_frame(model, cam, model.config.lang_field_dim)
        rgb = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
        rgb = rgb.astype(np.float32)
        if rgb.max() > 1.5:  # render_frame returns 0-255 floats
            rgb = rgb / 255.0
        pred = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        gt = np.asarray(Image.open(tds.image_filenames[fi]).convert("RGB").resize(
            (pred.shape[1], pred.shape[0])))
        strip = np.concatenate([gt, pred], axis=1)
        Image.fromarray(strip).resize((strip.shape[1] // 2, strip.shape[0] // 2)).save(
            out / f"{name}_f{fi:03d}.jpg", quality=82)
        print(f"[g2] {name} frame {fi} saved", flush=True)
    del pipe, model
    import torch; torch.cuda.empty_cache()
print("G2 RENDERS DONE")
