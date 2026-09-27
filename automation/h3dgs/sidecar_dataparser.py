"""nerfstudio's dataparser transform for a block dir, computed exactly the way ns-train's nerfstudio-data parser does
(orientation 'up', center 'poses', auto-scale), written as <out>/dataparser_transforms.json — so the H3DGS side-car
converter can place gaussians in the frame a later bootstrap on the SAME transforms.json will use, without needing an
earlier run of that block (04's block runs were made through the OpenCV-era poses, so their frames are not reusable).
  pixi run --manifest-path /home/paperspace/code/nerf_new/pixi.toml python sidecar_dataparser.py <block dir> <out dir>"""
import json, sys
from pathlib import Path
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
BD, OUT = Path(sys.argv[1]), Path(sys.argv[2])
cfg = NerfstudioDataParserConfig(data=BD, orientation_method="up", center_method="poses", auto_scale_poses=True, scale_factor=1.0, scene_scale=1.0, eval_mode="interval", eval_interval=10)
outs = cfg.setup()._generate_dataparser_outputs(split="train")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "dataparser_transforms.json").write_text(json.dumps({"transform": outs.dataparser_transform.tolist(), "scale": float(outs.dataparser_scale)}, indent=1))
print(f"[dataparser] {BD.name}: scale {float(outs.dataparser_scale):.4f}, {len(outs.image_filenames)} train images -> {OUT / 'dataparser_transforms.json'}", flush=True)
