#!/bin/bash
# Night queue 2026-08-13 (Paul: "a night queue that takes both cpu and gpu,
# autonomous"). Two tracks; every stage guarded + resumable; nothing
# destructive; no rm, no git push. Markers: *-DONE / *-FAIL per stage,
# NIGHT-GPU-DONE / NIGHT-CPU-DONE at track ends. Morning consolidation
# (notebook entry) happens in-session from these logs.
#
# EXCLUDED on purpose (not autonomous-safe): klapmuts re-block (mask coverage
# unverified), Dec pose-domain wiring (ZED-odom frame vs L2C question),
# supervision palette fix + 05 stage2 (task #8 needs design), destructive
# recluster (shares its detections dir), the held 01/03 legacy queue, and
# the ~380G deletion list (Paul's rm).
set -u
LOGS=/home/paperspace/logs
SRC=/home/paperspace/code/aru_sil_core/src/scripts
NS_PIXI=/home/paperspace/code/nerf_new/pixi.toml
AB_LOG=$LOGS/ab_perpass_block000.log
mark() { echo "[$(date +%H:%M:%S)] $1"; }

# ============================ CPU TRACK ====================================
(
  exec > "$LOGS/night_cpu_20260813.log" 2>&1
  mark "CPU track start"

  # C1. Q4 phase-0: READ-ONLY pose-delta audit — quantify the skew between
  # the mono-K poses and the stale kf jsons the legacy registries were
  # clustered under (ts-matched). Evidence report for the recluster.
  cd /home/paperspace/code/nerf_new
  pixi run python - << 'EOF' > "$LOGS/q4_pose_delta_report.txt" 2>&1
import json
import sys
import numpy as np
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
import kf_domain
for s in ('03_13B_Jackal', '04_13D_Jackal', '05_13D_Jackal'):
    root = f'/home/paperspace/data/citrus_all/{s}'
    try:
        d = kf_domain.load(root)
        stale = json.load(open(f'{root}/lio_image_poses_kf20cm.json'))
    except Exception as e:
        print(f'{s}: SKIP ({e})')
        continue
    by_ts = {int(e['timestamp_ns']): np.asarray(e['transform'])[:3, 3] for e in stale}
    deltas, worst = [], []
    for k, t in enumerate(d.ts):
        p = by_ts.get(int(t))
        if p is None or np.isnan(d.poses[k]).any():
            continue
        dd = float(np.linalg.norm(d.poses[k][:3, 3] - p))
        deltas.append(dd)
        worst.append((dd, k))
    a = np.array(deltas)
    worst.sort(reverse=True)
    print(f'{s}: {len(a)}/{len(d.ts)} K ts-matched to the stale json | '
          f'delta median {np.median(a)*1000:.1f} mm  p95 {np.percentile(a,95)*1000:.1f} mm  '
          f'max {a.max()*1000:.1f} mm | unmatched K (skew set): {len(d.ts)-len(a)}')
    print(f'  worst-10 K: {[(k, round(v,3)) for v, k in worst[:10]]}')
EOF
  grep -q "delta median" "$LOGS/q4_pose_delta_report.txt" \
      && mark "C1-DONE q4 pose-delta audit" || mark "C1-FAIL q4 audit"

  # C2. Dec kf20cm cut + PNG extraction (CPU/IO only — pose-domain wiring
  # deliberately NOT touched; TRANSFORM_MONO is only used for spacing).
  DEC=/home/paperspace/data/klapmuts/dec_2025_a300/monolithics
  if [ ! -f "$DEC/image_left_kf20cm.monolithic" ]; then
    /home/paperspace/code/aru_sil_core/build/bin/dump_keyframe_image_monolithic \
      --IMAGE_MONO "$DEC/image_left.monolithic" \
      --TRANSFORM_MONO "$DEC/zed_transform.monolithic" \
      --MIN_DIST 0.20 \
      --OUT_MONO "$DEC/image_left_kf20cm.monolithic" \
      > "$LOGS/dec_kf_cut.log" 2>&1 \
      && mark "C2a-DONE dec kf mono cut" || mark "C2a-FAIL dec kf cut"
  else
    mark "C2a-SKIP dec kf mono exists"
  fi
  if [ -f "$DEC/image_left_kf20cm.monolithic" ]; then
    pixi run --manifest-path "$NS_PIXI" python "$SRC/extract_kf_pngs.py" \
      --data-dir /home/paperspace/data/klapmuts/dec_2025_a300 \
      --mono "$DEC/image_left_kf20cm.monolithic" \
      > "$LOGS/dec_kf_extract.log" 2>&1 \
      && mark "C2b-DONE dec kf_images ($(ls /home/paperspace/data/klapmuts/dec_2025_a300/kf_images 2>/dev/null | wc -l) PNGs)" \
      || mark "C2b-FAIL dec extract"
  fi

  # C3. Housekeeping: sweep TSV into the notebook tree, registry regen,
  # klapmuts kf_domain cache prebuild (descriptor-safe standalone process).
  cp /tmp/claude-1000/-home-paperspace-code/c47d8606-c5fe-4343-ae48-8faa25cdc994/scratchpad/psnr_sweep_all.tsv \
     /home/paperspace/code/lab_notebook/psnr_sweep_20260813.tsv 2>/dev/null \
     && mark "C3a-DONE sweep tsv -> lab_notebook" || mark "C3a-SKIP sweep tsv"
  python3 /home/paperspace/code/automation/build_dataset_registry.py \
     > "$LOGS/datasets_regen.log" 2>&1 \
     && mark "C3b-DONE DATASETS.md regen" || mark "C3b-FAIL registry regen"
  pixi run python "$SRC/kf_domain.py" --data-dir /home/paperspace/data/klapmuts \
     > "$LOGS/klap_kfdomain.log" 2>&1 \
     && mark "C3c-DONE klap kf_domain cache" || mark "C3c-FAIL klap cache"

  mark "NIGHT-CPU-DONE"
) &
CPU_PID=$!

# ============================ GPU TRACK ====================================
exec > "$LOGS/night_gpu_20260813.log" 2>&1
mark "GPU track start (waiting for the A/B chain)"

# G0. Chain behind the per-pass A/B (arm B + verdict own the GPU now).
for _ in $(seq 1 120); do
    grep -q "A/B COMPLETE\|AB-FAIL" "$AB_LOG" 2>/dev/null && break
    sleep 120
done
mark "G0 gate passed: $(grep -oE 'A/B COMPLETE|AB-FAIL [a-zA-Z ]*' "$AB_LOG" | tail -1)"

# G1. Full regression harness (A + B) — tonight's changes get the fixture.
bash /home/paperspace/code/automation/regression_harness.sh \
    > "$LOGS/night_harness.log" 2>&1 \
    && mark "G1-DONE harness $(grep -oE 'PASS|FAIL' "$LOGS/night_harness.log" | tail -1)" \
    || mark "G1-FAIL harness (see night_harness.log)"

# G2. 05 stage-1 fleet slice: row blocks 001..008 through the two-stage
# pipeline (stage2 skips loudly per block until supervision compile lands —
# task #8). ~65 min/block.
bash "$SRC/run_unified_pipeline.sh" 05_13D_Jackal 1 8 \
    > "$LOGS/05_fleet_night_20260813.log" 2>&1 \
    && mark "G2-DONE 05 blocks 001-008" || mark "G2-FAIL 05 fleet (see log)"

# G3. Quick consistency re-check + end marker.
bash /home/paperspace/code/automation/regression_harness.sh --cpu-only \
    > "$LOGS/night_harness_end.log" 2>&1 \
    && mark "G3-DONE tier A recheck" || mark "G3-FAIL tier A recheck"

wait "$CPU_PID" 2>/dev/null
mark "NIGHT-GPU-DONE"
