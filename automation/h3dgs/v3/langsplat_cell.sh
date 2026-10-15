#!/bin/bash
# langsplat_cell.sh — the LangSplat baseline on one v3 chunk (paper E3, 2026-10-13), time-boxed: SAM+CLIP preprocessing of the dataset that
# langsplat_prep.py built (training split only), LangSplat's autoencoder, its RGB 3DGS (langsplat_train_rgb.py), the three language-feature
# levels, renders of the cell's verdict frames (langsplat_render_named.py) and the E1 scorer per level (argmax_eval.py --space langsplat).
#   WS=<chunk ws> N=<identity dir> CASE=<name> [ITER=30000] [SPARSE=<chunk sparse/0>] langsplat_cell.sh
set -uo pipefail; : ${WS:?} ${N:?} ${CASE:?}; ITER=${ITER:-30000}; SPARSE=${SPARSE:-$WS/sparse/0}
LS=/home/paperspace/code/LangSplat; V3=/home/paperspace/code/automation/h3dgs/v3; NS=/home/paperspace/code/nerf_new; DS=$WS/langsplat
source /home/paperspace/miniconda3/etc/profile.d/conda.sh; conda activate langsplat
L=/home/paperspace/logs/langsplat_$CASE.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $CASE: $*" | tee -a $L; }; t0=$(date +%s); step(){ say "$1 $(( ($(date +%s)-t0)/60 )) min"; }
FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))"); NI=$(ls $DS/images | wc -l)
say "=== $DS ($NI images) | verdict frames $(echo $FR | wc -w) | iterations $ITER"
[ $(ls $DS/language_features/*_f.npy 2>/dev/null | wc -l) -ge $NI ] || (cd $LS && nice -n 10 python preprocess.py --dataset_path $DS > $L.pre 2>&1); step "preprocess: $(ls $DS/language_features/*_f.npy 2>/dev/null | wc -l) feature files"
[ -e $DS/ae_best_ckpt.pth ] || (cd $LS/autoencoder && nice -n 10 python train.py --dataset_path $DS --dataset_name $CASE --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 > $L.ae 2>&1 \
   && python test.py --dataset_path $DS --dataset_name $CASE >> $L.ae 2>&1 && cp ckpt/$CASE/best_ckpt.pth $DS/ae_best_ckpt.pth); step "autoencoder: $(ls $DS/language_features_dim3 2>/dev/null | wc -l) dim-3 files"
[ -e $DS/output/rgb/chkpnt$ITER.pth ] || (cd $LS && nice -n 10 python $V3/langsplat_train_rgb.py -s $DS -m $DS/output/rgb --iterations $ITER --quiet > $L.rgb 2>&1); step "rgb 3dgs: $(ls $DS/output/rgb/chkpnt$ITER.pth 2>/dev/null)"
for lv in 1 2 3; do [ -e $DS/output/ls_$lv/chkpnt30000.pth ] || (cd $LS && nice -n 10 python train.py -s $DS -m $DS/output/ls --start_checkpoint $DS/output/rgb/chkpnt$ITER.pth --feature_level $lv --quiet > $L.lv$lv 2>&1); step "level $lv: $(ls $DS/output/ls_$lv/chkpnt30000.pth 2>/dev/null)"; done
[ -e $N/langsplat/maps/$(echo $FR | cut -d' ' -f1 | sed 's/.png/.npz/') ] || python $V3/langsplat_render_named.py --rgb-ckpt $DS/output/rgb/chkpnt$ITER.pth --feature-ckpts $DS/output/ls_1/chkpnt30000.pth $DS/output/ls_2/chkpnt30000.pth $DS/output/ls_3/chkpnt30000.pth --sparse $WS/sparse/0 $SPARSE --frames $FR --out-dir $N/langsplat/maps 2>&1 | grep -aE "^\[ls-render\]|Error|Traceback" | tee -a $L
for lv in 1 2 3; do [ -s $N/protocol_langsplat_L$lv.jsonl ] || (cd $NS && pixi run python $V3/argmax_eval.py --identity $N --tag langsplat_L$lv --space langsplat --ls-level $lv --ae-ckpt $DS/ae_best_ckpt.pth --maps langsplat/maps/%s.npz 2>&1) | grep -aE "^\[(protocol|vocab)\]|Error|Traceback" | tee -a $L; done
say "=== done in $(( ($(date +%s)-t0)/60 )) min"
