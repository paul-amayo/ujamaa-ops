#!/bin/bash
# setup_lerf_src.sh — ovbench (2026-11-10): published LERF source for the benchmark, outside every git checkout.
# The pixi env's editable `lerf` is the RSojen/lerf-scene-graph fork (sky loss, modified negatives, fused bounding-box embeddings,
# plus uncommitted edits) — not LERF as published. Its git history contains kerrj/lerf up to db08d57 (2024-03-13, "fix camera
# optimization bug", the last upstream commit), so `git archive db08d57` reproduces the authors' release without any network.
# One change, forced by the box: dino_extractor.create_model gets DINO ViT-S/8 from dino_timm.DinoViTS8 (same official weights via
# timm, see dino_timm.py) instead of torch.hub.load('facebookresearch/dino:main', ...), which fetches GitHub. PYTHONPATH=$DST makes
# this copy shadow the fork for ns-train's `lerf` entry point (checked: lerf.__file__).
# Second change, type annotations only: upstream config fields carry annotations that do not match their defaults
# (patch_tile_size_range: Tuple[int, int] = (0.05, 0.5); hashgrid_layers: Tuple[int] = (12, 12); negatives: Tuple[str] = 4 words, ...);
# tyro in nerf_new refuses them when building the ns-train CLI. They become variadic / float tuples with the SAME default values —
# no behaviour change (the same edit the fork made; the fork's other edits are not taken).
# Third change, API compatibility: LERFPipeline calls self.datamanager.to(device); DataManager stopped being an nn.Module after
# nerfstudio 1.0 (nerf_new is 1.1.5) and has no .to. The datamanager is built with device=device and holds no parameters, so the call
# is guarded with hasattr (the fork comments it out) — no behaviour change.
# Fourth change, BACKEND (a real deviation, reported): LERFField builds its two hash grids and its CLIP / DINO heads with
# tiny-cuda-nn (tcnn.Encoding HashGrid, tcnn.Network CutlassMLP). tiny-cuda-nn is not installed in any env on the box and installs
# only from GitHub (blocked). They become nerfstudio's own torch implementations of the same modules — HashEncoding(implementation=
# 'torch') with the same levels / resolutions / 2^19 tables / 8 features per level, MLP(implementation='torch') with the same widths and
# depths (tcnn n_hidden_layers h -> nerfstudio num_layers h + 1). nerfstudio uses exactly this fallback for nerfacto when tcnn is
# missing (LERF's nerfacto half already runs on it here). Differences: torch Linear layers carry biases (CutlassMLP has none), the
# hash-table init is nerfstudio's (uniform +-1e-3 vs tcnn's +-1e-4), torch grid-vertex alignment, fp32 tables under autocast, speed.
set -euo pipefail
DST=/home/paperspace/logs/ovbench/src/lerf_kerrj_db08d57; BIN=/home/paperspace/logs/ovbench/bin
rm -rf $DST; mkdir -p $DST
git -C /home/paperspace/code/lerf-scene-graph archive db08d57 | tar -x -C $DST
cp $BIN/dino_timm.py $DST/lerf/data/utils/dino_timm.py
python3 - $DST/lerf/data/utils/dino_extractor.py <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
old = "        if 'dino' in model_type:\n            model = torch.hub.load('facebookresearch/dino:main', model_type)\n"
new = ("        if 'dino' in model_type:\n            # ovbench: torch.hub fetches GitHub (blocked on the A100); same network + official weights through timm\n"
       "            assert model_type == 'dino_vits8', model_type\n            from lerf.data.utils.dino_timm import DinoViTS8\n            model = DinoViTS8()\n")
assert s.count(old) == 1, 'patch anchor not found'; open(p, 'w').write(s.replace(old, new)); print('[setup] dino_extractor.create_model patched')
EOF
python3 - $DST <<'EOF'
import sys
R = {'lerf/data/lerf_datamanager.py': [('patch_tile_size_range: Tuple[int, int] = (0.05, 0.5)', 'patch_tile_size_range: Tuple[float, float] = (0.05, 0.5)')],
     'lerf/lerf.py': [('hashgrid_layers: Tuple[int] = ', 'hashgrid_layers: Tuple[int, ...] = '), ('hashgrid_resolutions: Tuple[Tuple[int]] = ', 'hashgrid_resolutions: Tuple[Tuple[int, ...], ...] = '),
                      ('hashgrid_sizes: Tuple[int] = ', 'hashgrid_sizes: Tuple[int, ...] = ')],
     'lerf/lerf_pipeline.py': [('        self.datamanager.to(device)', "        if hasattr(self.datamanager, 'to'): self.datamanager.to(device)  # ovbench: DataManager has no .to in nerfstudio 1.1")],
     'lerf/encoders/openclip_encoder.py': [('negatives: Tuple[str] = ', 'negatives: Tuple[str, ...] = ')], 'lerf/encoders/clip_encoder.py': [('negatives: Tuple[str] = ', 'negatives: Tuple[str, ...] = ')]}
for f, subs in R.items():
    p = sys.argv[1] + '/' + f; s = open(p).read()
    for old, new in subs:
        assert s.count(old) == 1, (f, old); s = s.replace(old, new)
    open(p, 'w').write(s)
print('[setup] annotations + datamanager.to guard patched')
p = sys.argv[1] + '/lerf/lerf_field.py'; s = open(p).read()
subs = [('''        tot_out_dims = sum([e.n_output_dims for e in self.clip_encs])''', '''        tot_out_dims = sum([e.get_out_dim() for e in self.clip_encs])  # ovbench: torch backend'''),
        ('''        self.clip_net = tcnn.Network(
            n_input_dims=tot_out_dims + 1,
            n_output_dims=clip_n_dims,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": 4,
            },
        )''', '''        # ovbench: tcnn CutlassMLP (256 x 4 hidden, ReLU) -> nerfstudio torch MLP of the same shape (tiny-cuda-nn unavailable)
        self.clip_net = MLP(in_dim=tot_out_dims + 1, num_layers=5, layer_width=256, out_dim=clip_n_dims, activation=nn.ReLU(), out_activation=None, implementation="torch")
        self.clip_net.n_input_dims = tot_out_dims + 1'''),
        ('''        self.dino_net = tcnn.Network(
            n_input_dims=tot_out_dims,
            n_output_dims=384,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": 1,
            },
        )''', '''        self.dino_net = MLP(in_dim=tot_out_dims, num_layers=2, layer_width=256, out_dim=384, activation=nn.ReLU(), out_activation=None, implementation="torch")'''),
        ('''        enc = tcnn.Encoding(
            n_input_dims=indim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": levels,
                "n_features_per_level": 8,
                "log2_hashmap_size": hash_size,
                "base_resolution": start_res,
                "per_level_scale": growth,
            },
        )''', '''        # ovbench: tcnn HashGrid -> nerfstudio torch HashEncoding, same levels / resolutions (same growth) / table size / features
        enc = HashEncoding(num_levels=levels, min_res=start_res, max_res=end_res, log2_hashmap_size=hash_size, features_per_level=8, implementation="torch")'''),
        ('''from nerfstudio.fields.base_field import Field''', '''from nerfstudio.fields.base_field import Field
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.field_components.mlp import MLP''')]
s = s.replace('\r\n', '\n')
for old, new in subs:
    assert s.count(old) == 1, old[:60]; s = s.replace(old, new)
open(p, 'w').write(s); print('[setup] lerf_field: tcnn -> nerfstudio torch backend')
EOF
: > $DST/OVBENCH_PATCH.diff
for f in lerf/data/utils/dino_extractor.py lerf/data/lerf_datamanager.py lerf/lerf.py lerf/lerf_pipeline.py lerf/lerf_field.py lerf/encoders/openclip_encoder.py lerf/encoders/clip_encoder.py; do
  diff -u --strip-trailing-cr <(git -C /home/paperspace/code/lerf-scene-graph show db08d57:$f) $DST/$f >> $DST/OVBENCH_PATCH.diff || true
done
echo "[setup] $DST ready; patch recorded in $DST/OVBENCH_PATCH.diff"
