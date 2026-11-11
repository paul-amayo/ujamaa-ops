#!/bin/bash
# mi_stops_123.sh (A100 session 11-11): pos01, pos02, pos03 of 230613 in sequence through mi_stop_a100.sh; writes logs/STOPS_123_DONE
for s in 1 2 3; do bash /mnt/shared/must-c_bulk/work/identity_feas/bin/mi_stop_a100.sh 230613_pos0$s pos0$s; done
date -u > /mnt/shared/must-c_bulk/work/identity_feas/logs/STOPS_123_DONE
