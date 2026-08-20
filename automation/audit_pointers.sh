#!/bin/bash
# Ground-truth pointer audit: derive each survey's true resume point from
# stage2 ARTIFACT COUNTS, not from log-visible failures. Exists because
# 2026-08-20's wall rewinds were first set from the log and undercounted
# (02/03 real wall b005, set b011 — the artifacts told the truth).
S=/home/paperspace/logs/week_prod_20260814_state
for spec in "01_13B_Jackal:/home/paperspace/data/citrus_all" \
            "02_13B_Jackal:/home/paperspace/data/citrus_all" \
            "03_13B_Jackal:/home/paperspace/data/citrus_all" \
            "04_13D_Jackal:/home/paperspace/data/citrus_all" \
            "05_13D_Jackal:/home/paperspace/data/citrus_all" \
            "apr_2026_zed:/home/paperspace/data/klapmuts"; do
    n=${spec%%:*}; root=${spec##*:}
    cfg=$root/$n/prod/tassili/blocks_ns/lio_row100
    # first block index with NO stage2 ckpt = true resume point
    resume=""
    for d in $(ls -d "$cfg"/block_* 2>/dev/null | sort); do
        b=${d##*block_}
        if ! ls "$d"/splat_runs_FEATFIX/stage2_censusinit_*/high/*/nerfstudio_models/*.ckpt \
            >/dev/null 2>&1; then resume=$((10#$b)); break; fi
    done
    ptr=$(cat "$S/$n.next" 2>/dev/null)
    match=OK; [ "$ptr" != "$resume" ] && match="MISMATCH (pointer $ptr, artifacts say $resume)"
    echo "$n: first-unbuilt=b$resume pointer=$ptr $match"
done
