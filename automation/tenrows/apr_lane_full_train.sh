#!/bin/bash
# Paul 2026-10-06 "A": April lane 2 full, resampled to December's 9.5 cm per frame, then E2 (every 2nd frame, every 10th held out)
# -> project h3dgs_e2s. Waits for apr_lane_full_chain.sh (SfM + warp v2 + heading check), trains only if the placement is at
# December's level (pre-stated: heading deviation max <= 3.43 and p99 <= 2.93 deg, warp residual p90 <= 0.082 m - December lane 2's
# own values), then place (warp + resampled prep) -> train -> hierarchy -> post-opt -> eval.
LD=/home/paperspace/data/klapmuts/apr_2026_zed/experimental/lane2_apr_full; L=/home/paperspace/logs/apr_lane2_full_train.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
say "waiting for the chain"
until grep -q 'chain done' /home/paperspace/logs/apr_lane2_full_chain.log 2>/dev/null; do
  pgrep -f '^bash /home/paperspace/logs/apr_lane_full_chain[.]sh' > /dev/null || { sleep 5; grep -q 'chain done' /home/paperspace/logs/apr_lane2_full_chain.log || { say "CHAIN ENDED WITHOUT 'chain done' - not training"; exit 1; }; }
  sleep 30
done
V=$(python3 - <<'PY'
import re
s = open('/home/paperspace/logs/apr_lane2_full_chain.log').read()
h = re.findall(r'\[heading-check\].*?p99 ([\d.]+) max ([\d.]+) deg', s); w = re.findall(r'\[lane-warp\].*?\(p90 [\d.]+ -> ([\d.]+)\)', s)
if not h or not w: print('FAIL no heading-check or lane-warp line'); raise SystemExit
p99, mx = map(float, h[-1]); p90 = float(w[-1])
print(('PASS' if (mx <= 3.43 and p99 <= 2.93 and p90 <= 0.082) else 'FAIL') + f' heading p99 {p99} max {mx} deg, warp p90 {p90} m (December: p99 2.93, max 3.43, p90 0.082)')
PY
)
say "placement check: $V"
case "$V" in PASS*) ;; *) say "not training"; exit 1 ;; esac
say "training project h3dgs_e2s (resample 9.5 cm, every 2nd frame)"
cd /home/paperspace/logs && LANE_PROJ=h3dgs_e2s PREP_EXTRA="--resample-cm 9.5" bash /home/paperspace/logs/apr_lane_run_v2.sh $LD place train hierarchy post eval 2>&1 | tee -a $L
say "=== done"
