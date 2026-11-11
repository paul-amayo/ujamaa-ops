#!/bin/bash
# assemble_results.sh (A100 session): RESULTS.md = the prose (figures/RESULTS_prose.md) + the tables generated from the records.
B=/mnt/shared/azalai_rr; T=/mnt/shared/overnight_20261112/AZ1_row_to_relevancy; cd /home/paperspace/code/nerf_new
OMP_NUM_THREADS=1 /home/paperspace/.pixi/bin/pixi run python $B/bin/azalai_rr_tables.py $B/figures/tables.md > /dev/null 2>&1
OMP_NUM_THREADS=1 /home/paperspace/.pixi/bin/pixi run python $B/bin/azalai_rr_figure.py $T/results/rr_summary_figure.png > /dev/null 2>&1
{ cat $B/figures/RESULTS_prose.md; echo; echo '## Tables (generated from the records by `code/azalai_rr_tables.py`)'; echo; cat $B/figures/tables.md; } > $T/results/RESULTS.md
grep -c '' $T/results/RESULTS.md
