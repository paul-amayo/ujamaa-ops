#!/bin/bash
# Stop the ten_rows_ins11 survey run (its bash driver and any colmap/glomap/python child working in the ins11 project).
# Patterns are anchored on the ins11 project path so the rgb_zed survey and the stereo chain are never matched;
# kept in a script file so no interactive shell carries the pattern text (pkill self-match trap).
P=/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/h3dgs_ins11
pkill -f "^bash /home/paperspace/logs/h3dgs_survey.sh /home/paperspace/data/klapmuts/dec_2025_ten_rows $P" && echo "driver stopped"
sleep 1
for pat in "colmap [a-z_]+ --database_path $P" "colmap [a-z_]+ .*--input_path $P" "glomap mapper --database_path $P" "python [^ ]*(make_chunk|prepare_chunk|make_chunks_depth_scale|copy_file_to_chunks|h3dgs_lidar_init|h3dgs_export|h3dgs_depth_05)\.py .*$P" "train_(single|coarse|post)\.py .*$P"; do
  pkill -f "$pat" && echo "killed: $pat"
done
sleep 1; pgrep -af "$P" | grep -v "tenrows_ins11_stop" | cut -c1-160 || echo "nothing left on $P"
