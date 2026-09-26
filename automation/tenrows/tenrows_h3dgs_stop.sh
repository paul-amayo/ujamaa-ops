#!/bin/bash
# tenrows_h3dgs_stop.sh <h3dgs project dir> — stop the h3dgs_survey.sh run working in that project (driver + colmap /
# glomap / python children), patterns anchored on the project path (nothing else is matched); kept in a script file so
# no interactive shell carries the pattern text (pkill self-match trap).
P=${1:?project dir}
pkill -f "^bash /home/paperspace/logs/h3dgs_survey.sh .* $P$" && echo "driver stopped"; pkill -f "^bash /home/paperspace/logs/h3dgs_survey.sh .* $P " && echo "driver stopped"
sleep 1
for pat in "colmap [a-z_]+ --database_path $P" "colmap [a-z_]+ .*--input_path $P" "glomap mapper --database_path $P" "python [^ ]*(make_chunk|prepare_chunk|make_chunks_depth_scale|copy_file_to_chunks|h3dgs_lidar_init|h3dgs_export|h3dgs_depth_05|h3dgs_eval_compact)\.py .*$P" "train_(single|coarse|post)\.py .*$P" "GaussianHierarchy(Creator|Merger) .*$P"; do
  pkill -f "$pat" && echo "killed: $pat"
done
sleep 2; pgrep -af "$P" | grep -v "tenrows_h3dgs_stop" | cut -c1-160 || echo "nothing left on $P"
