#!/bin/bash
# gen2 M0 pilot (2026-08-21): rebuild every rotation survey's hierarchy with
# the CENSUS row solver (build_marker_hierarchy --row-solver census) into
# <root>/experimental/gen2_m0/<name>/ — NO prod writes — then diff against
# prod with automation/gen2_m0_diff.py (table + figs + report.json).
# CPU only, niced (GLOMAP bursts in the GPU slots own the cores). Re-runnable:
# every rebuild overwrites its experimental/ copy.
# Usage: gen2_m0_pilot.sh [name ...]    names: 05 04 01 03 02 apr apr_mv10
#   variants (experiments, same outputs tree): 02_dir985 01_dir985 03_dir985
#   (citrus dir gate 0.985), apr_band06 apr_mv10_band06 (0.6 m band)
set -u
export PATH=/home/paperspace/.local/bin:/home/paperspace/.pixi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
SRC=/home/paperspace/code/aru_sil_core/src/scripts
AUTO=/home/paperspace/code/automation
CITRUS=/home/paperspace/data/citrus_all
KLAP=/home/paperspace/data/klapmuts
cd /home/paperspace/code/nerf_new || exit 1

build() {   # name root markers_tag census_tag [extra builder args]
    local name=$1 root=$2 mtag=$3 ctag=$4; shift 4
    local dir=$root/experimental/gen2_m0/$name out
    out=$dir/marker_hierarchy.json
    mkdir -p "$dir"
    if nice -n 15 pixi run python "$SRC/build_marker_hierarchy.py" \
        --semantic-monolithic "$root/prod/monos/filtered_semantic_v2.monolithic" \
        --marker-monolithic "$root/prod/bateleur/scene_graph/markers_${mtag}.monolithic" \
        --census-json "$root/prod/bateleur/sam3_${ctag}/global_ids.json" \
        --data-dir "$root" \
        --dominant-direction-xz "1,0" --out "$out" "$@" > "$dir/build.log" 2>&1; then
        echo "BUILT $name -> $out"
    else
        echo "FAILED $name (see $dir/build.log)"; tail -5 "$dir/build.log"
    fi
}
want() { [ $# -eq 0 ] && return 0; local n; for n in "$@"; do [ "$n" = "$NAME" ] && return 0; done; return 1; }

for spec in "05:05_13D_Jackal:$CITRUS:v2:v2" "04:04_13D_Jackal:$CITRUS:v2:v2" \
            "01:01_13B_Jackal:$CITRUS:v2:v2" "03:03_13B_Jackal:$CITRUS:v2:v2" \
            "02:02_13B_Jackal:$CITRUS:v2:v2" \
            "apr:apr_2026_zed:$KLAP:v2:v2" "apr_mv10:apr_2026_zed:$KLAP:mv10:mv10" \
            "02_dir985:02_13B_Jackal:$CITRUS:v2:v2:--dominant-direction-threshold 0.985" \
            "02_dir95:02_13B_Jackal:$CITRUS:v2:v2:--dominant-direction-threshold 0.95" \
            "01_dir985:01_13B_Jackal:$CITRUS:v2:v2:--dominant-direction-threshold 0.985" \
            "03_dir985:03_13B_Jackal:$CITRUS:v2:v2:--dominant-direction-threshold 0.985" \
            "apr_band06:apr_2026_zed:$KLAP:v2:v2:--row-band-m 0.6" \
            "apr_mv10_band06:apr_2026_zed:$KLAP:mv10:mv10:--row-band-m 0.6"; do
    IFS=: read -r NAME SURVEY BASE MTAG CTAG EXTRA <<< "$spec"
    want "$@" || continue
    # shellcheck disable=SC2086  # EXTRA is a deliberate word-split flag list
    build "$NAME" "$BASE/$SURVEY" "$MTAG" "$CTAG" ${EXTRA:-}
done
nice -n 15 python3 "$AUTO/gen2_m0_diff.py" "$@"
