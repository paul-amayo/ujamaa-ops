#!/bin/bash
# Regression harness (roadmap Phase 1): fixed fixtures + expected values so
# pipeline/code changes are testable. Two tiers:
#
#   Tier A (CPU, ~1 min)  A1 K-domain consistency: every survey's kf json count
#                            must equal its kf_images count (the silent-skew
#                            class caught on 02, latent on 03 — see Q4)
#                         A2 sankofa ledger_v2 datum invariants (272 matched
#                            13B pairs, 0.000 m pair medians in BOTH frames)
#                         A3 embedder round-trip: v3vocab1k decodes its own
#                            word targets at cos >= 0.999 (CPU)
#   Tier B (GPU, ~15 min) scene-baseline probe: block_001 (04_13D) trained
#                         2000 iters on the scene.json recipe -> train PSNR +
#                         gauss count vs expected +/- tolerance
#
# Usage:
#   bash automation/regression_harness.sh              # A then B
#   bash automation/regression_harness.sh --cpu-only   # A only
#   bash automation/regression_harness.sh --calibrate  # (re)record expected.json
#
# Expected values live in automation/regression_expected.json (committed).
# Every run appends a line to /home/paperspace/logs/regression_harness.log.
set -u
CODE=/home/paperspace/code
EXPECTED=$CODE/automation/regression_expected.json
NS_PIXI=$CODE/nerf_new/pixi.toml
BLOCK=/home/paperspace/data/citrus_all/04_13D_Jackal/blocks_ns/lio_row6F/block_001_L095_sky
LOG=/home/paperspace/logs/regression_harness.log
CPU_ONLY=0; CALIBRATE=0
for a in "$@"; do case "$a" in
    --cpu-only) CPU_ONLY=1 ;;
    --calibrate) CALIBRATE=1 ;;
    *) echo "unknown flag $a"; exit 2 ;;
esac; done

FAIL=0
note() { echo "$1"; echo "$(date -u +%FT%TZ) $1" >> "$LOG"; }

# ---------- A1: K-domain consistency ----------
echo "== A1 K-domain consistency =="
python3 - << 'EOF' || FAIL=1
import json, sys
from pathlib import Path
bad = 0
for root in sorted(Path('/home/paperspace/data/citrus_all').glob('*_Jackal')):
    kf_json = root / 'lio_image_poses_kf20cm.json'
    kf_dir = root / 'kf_images'
    if not (kf_json.exists() and kf_dir.is_dir()):
        continue
    nj = len(json.load(open(kf_json)))
    np_ = len(list(kf_dir.glob('kf_*.png')))
    ok = nj == np_
    print(f"  {root.name}: json {nj} vs kf_images {np_} {'OK' if ok else 'MISMATCH'}")
    bad += (not ok)
# known-legacy skew is allowed to persist until Q4 re-verification, but it
# must be VISIBLE — fail only if a previously-consistent survey regresses.
LEGACY = {'01_13B_Jackal', '03_13B_Jackal', '04_13D_Jackal', '05_13D_Jackal'}
hard = [str(r) for r in []]
for root in sorted(Path('/home/paperspace/data/citrus_all').glob('*_Jackal')):
    kf_json, kf_dir = root / 'lio_image_poses_kf20cm.json', root / 'kf_images'
    if kf_json.exists() and kf_dir.is_dir():
        if len(json.load(open(kf_json))) != len(list(kf_dir.glob('kf_*.png'))) \
                and root.name not in LEGACY:
            hard.append(root.name)
if hard:
    print(f"  FAIL: non-legacy K-domain mismatch: {hard}")
    sys.exit(1)
print(f"  ({bad} legacy skews visible — Q4 owns them)")
EOF
[ $? -ne 0 ] && FAIL=1

# ---------- A2: ledger_v2 datum invariants ----------
echo "== A2 ledger_v2 datum invariants =="
python3 - << 'EOF' || FAIL=1
import json, math, statistics, sys
d = json.load(open('/home/paperspace/data/citrus_all/sankofa_substrate/ledger_v2.json'))
flat = []
for v in (d.values() if isinstance(d, dict) else d):
    flat.extend(v if isinstance(v, list) else [v])
b13 = [r for r in flat if r.get('farm') == '13B' and r.get('lat')]
c1 = {r['canonical_id']: r for r in b13 if r['epoch'] == '2023-07-16'}
c2 = {r['canonical_id']: r for r in b13 if r['epoch'] == '2023-07-18'}
both = sorted(set(c1) & set(c2))
mlat = statistics.median(abs(c2[c]['lat'] - c1[c]['lat']) for c in both) * 111320
men = statistics.median(abs(c2[c]['en_13b_frame'][1] - c1[c]['en_13b_frame'][1]) for c in both)
ok = len(both) >= 272 and mlat < 0.9 and men < 0.9
print(f"  pairs {len(both)} (>=272), |dlat| median {mlat:.2f} m, |den_north| median {men:.2f} m -> {'OK' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
EOF
[ $? -ne 0 ] && FAIL=1

# ---------- A3: embedder round-trip ----------
echo "== A3 embedder round-trip (v3vocab1k, CPU) =="
pixi run --manifest-path "$NS_PIXI" python - << 'EOF' || FAIL=1
import sys
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH')
import torch
from high_splat_hierarchy_accuracy import build_hyper_embedder
from word_utils import get_word_for_id
import open_clip
dev = 'cpu'
hyper = build_hyper_embedder('/home/paperspace/data/high/nerf/04_13D_v3vocab1k/ckpts/model_best.pth', dev)
oc, _, _ = open_clip.create_model_and_transforms('ViT-B-16', 'laion2b_s34b_b88k', device=dev)
tok = open_clip.get_tokenizer('ViT-B-16')
ok = True
for wid, lvl, nt in ((6, 'row', 0), (84, 'mask', 0), (17, 'fruit', 4)):
    w = get_word_for_id(wid, lvl if lvl != 'mask' else 'mask')
    ce = oc.encode_text(tok([w])).float()
    hf = hyper.encode_features(ce, project=True, node_types=torch.tensor([nt]))
    de = hyper.decode_features(hf, project=True)
    cos = torch.nn.functional.cosine_similarity(ce, de).item()
    print(f"  {lvl}:{w!r} decode cos {cos:.4f}")
    ok &= cos >= 0.999
sys.exit(0 if ok else 1)
EOF
[ $? -ne 0 ] && FAIL=1

# ---------- Tier B: GPU scene-baseline probe ----------
if [ "$CPU_ONLY" = "1" ]; then
    note "regression tier A only: $([ $FAIL = 0 ] && echo PASS || echo FAIL)"
    exit $FAIL
fi
echo "== B scene-baseline probe (block_001, 2000 iters) =="
PROBE_DIR=$BLOCK/splat_runs_REGRESSION
EXP=probe_$(date +%Y%m%d_%H%M)
ROOT=/home/paperspace/data/citrus_all/04_13D_Jackal
eval "$(python3 - "$ROOT" << 'EOF'
import json, shlex, sys
sc = json.load(open(f"{sys.argv[1]}/scene.json"))
b = sc["baseline"]
for k, v in b.get("env", {}).items():
    print(f'export {k}={shlex.quote(str(v))}')
if sc.get("supervision_dir"):
    print(f'export TREE_WEIGHT_DIR={shlex.quote(sys.argv[1] + "/" + sc["supervision_dir"])}')
flags = {k: v for k, v in b["flags"].items() if k != "max-num-iterations"}
print('FLAGS=' + shlex.quote(" ".join(f"--{k} {v}" for k, v in flags.items())))
EOF
)"
EMPTY=/home/paperspace/logs/empty_semantic; mkdir -p $EMPTY
cd $CODE/nerf_new
echo "n" | MAX_JOBS=4 pixi run ns-train high \
    --data "$BLOCK" --output-dir "$PROBE_DIR" --experiment-name "$EXP" \
    --pipeline.datamanager.semantic-dir $EMPTY \
    $FLAGS \
    --max-num-iterations 2000 \
    --vis tensorboard nerfstudio-data \
    --eval-mode interval --eval-interval 10 \
    > /home/paperspace/logs/regression_probe.log 2>&1 || {
        note "regression B: TRAIN FAILED (see regression_probe.log)"; exit 1; }

RESULT=$(pixi run --manifest-path "$NS_PIXI" python - "$PROBE_DIR/$EXP" << 'EOF'
import glob, math, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
evs = sorted(glob.glob(sys.argv[1] + '/high/*/events.out.tfevents.*'))
ea = EventAccumulator(evs[-1]); ea.Reload()
tags = ea.Tags()['scalars']
def last(*cands):
    # first candidate that matches a tag (train-side tags preferred by order)
    for sub in cands:
        t = [x for x in tags if sub.lower() in x.lower()]
        if t:
            return ea.Scalars(t[0])[-1].value
    return float('nan')
# doctrine: headline = train FG PSNR (never full-image once sky masks exist)
pv = last('Train Metrics Dict/psnr_fg', 'psnr_fg')
gv = last('Train Metrics Dict/gaussian_count', 'gaussian_count', 'num_points')
if math.isnan(pv) or math.isnan(gv):
    print(f"TAG-MISS: scalars present: {sorted(tags)}", file=sys.stderr)
    sys.exit(3)
print(f"{pv:.2f} {gv:.0f}")
EOF
)
[ $? -ne 0 ] && { note "regression B: metric extraction failed (TAG-MISS)"; exit 1; }
PSNR=$(echo "$RESULT" | awk '{print $1}')
GAUSS=$(echo "$RESULT" | awk '{print $2}')
echo "  probe: train FG-PSNR@2k=$PSNR gauss=$GAUSS"

if [ "$CALIBRATE" = "1" ] || [ ! -f "$EXPECTED" ]; then
    python3 -c "
import json, math, sys
p, g = float('$PSNR'), float('$GAUSS')
if math.isnan(p) or math.isnan(g):
    sys.exit('refusing to calibrate on nan')
json.dump({'block': '$BLOCK', 'iters': 2000, 'metric': 'Train Metrics Dict/psnr_fg',
           'train_psnr_fg': p, 'gauss': g,
           'tol_psnr_db': 0.5, 'tol_gauss_frac': 0.10}, open('$EXPECTED','w'), indent=1)" \
        || { note "regression CALIBRATION REFUSED (nan)"; exit 1; }
    note "regression CALIBRATED: FG-PSNR@2k=$PSNR gauss=$GAUSS -> $EXPECTED"
    exit $FAIL
fi
python3 - "$PSNR" "$GAUSS" << 'EOF' || FAIL=1
import json, math, sys
psnr, gauss = float(sys.argv[1]), float(sys.argv[2])
e = json.load(open('/home/paperspace/code/automation/regression_expected.json'))
if math.isnan(psnr) or math.isnan(gauss):
    print("  nan metrics — FAIL"); sys.exit(1)
dp = abs(psnr - e['train_psnr_fg']); dg = abs(gauss - e['gauss']) / max(e['gauss'], 1)
ok = dp <= e['tol_psnr_db'] and dg <= e['tol_gauss_frac']
print(f"  vs expected: dFG-PSNR {dp:.2f} dB (tol {e['tol_psnr_db']}), dgauss {100*dg:.1f}% (tol {100*e['tol_gauss_frac']:.0f}%) -> {'OK' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
EOF
[ $? -ne 0 ] && FAIL=1

note "regression harness: $([ $FAIL = 0 ] && echo PASS || echo FAIL) (A + B)"
exit $FAIL
