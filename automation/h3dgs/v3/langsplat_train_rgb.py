#!/usr/bin/env python
"""langsplat_train_rgb.py — LangSplat's own trainer in RGB mode (paper E3, 2026-10-13). LangSplat's README trains the RGB 3DGS with the
graphdeco repo and starts the three language-feature trainings from that checkpoint; its train.py can do the RGB stage itself once
include_feature is False, which the CLI cannot set (a store_true flag defaulting to True). Same args as LangSplat/train.py, minus the
feature-level suffix: -s <dataset> -m <model dir> [--iterations 30000] -> <model dir>/chkpnt<iterations>.pth. langsplat env."""
import os, sys
sys.path.insert(0, '/home/paperspace/code/LangSplat'); os.chdir('/home/paperspace/code/LangSplat')
from argparse import ArgumentParser
import torch
from arguments import ModelParams, OptimizationParams, PipelineParams
from utils.general_utils import safe_state
from gaussian_renderer import network_gui
import train as T
parser = ArgumentParser(); lp = ModelParams(parser); op = OptimizationParams(parser); pp = PipelineParams(parser)
parser.add_argument('--port', type=int, default=56000 + os.getpid() % 1000); parser.add_argument('--quiet', action='store_true')
args = parser.parse_args(sys.argv[1:]); dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
opt.include_feature = False; dataset.feature_level = -1
print(f'[langsplat-rgb] {dataset.source_path} -> {dataset.model_path}, {opt.iterations} iterations', flush=True)
safe_state(args.quiet); network_gui.init('127.0.0.1', args.port); torch.autograd.set_detect_anomaly(False)
T.training(dataset, opt, pipe, [], [opt.iterations], [opt.iterations], None, -1)
print('[langsplat-rgb] done', flush=True)
