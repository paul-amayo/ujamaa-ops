#!/bin/bash
# after the super chain: identity guard (argmax_eval on the super field) + the decisive composed-query test
set -uo pipefail
W=/home/paperspace/data/citrus_all/05_13D_Jackal; WS=$W/experimental/v3/chunk_1_0; N=$WS/identity; P=$WS/parts; NS=/home/paperspace/code/nerf_new; V3=/home/paperspace/code/automation/h3dgs/v3
EMB=$W/experimental/identity/embedder/05_13D_super_v1/ckpts/model_best.pth; S=/tmp/claude-1000/-home-paperspace-code/a2c976c5-e79c-4d4c-bb12-07b7819bb9d8/scratchpad
(cd $NS && pixi run python $V3/argmax_eval.py --identity $N --tag super --space hyper --emb $EMB --maps $P/maps_super/%s.npz --out $P/protocol_super.jsonl) 2>&1 | grep -aE '^\[protocol\]|Error|Traceback' | cut -c1-300
(cd $NS && pixi run python $V3/super_eval.py --identity $N --parts $P --emb-super $EMB --out $P/super_eval.json --dump $P/super_dump) 2>&1 | grep -aE '^\[super\]|Error|Traceback' | cut -c1-420
echo SCORE-DONE
