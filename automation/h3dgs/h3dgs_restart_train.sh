#!/bin/bash
# Stop the running H3DGS training chain (trainer script, full_train.py, train_single/train_post),
# quarantine the partial chunk output, and relaunch the trainer (which keeps the finished scaffold).
OUT=/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs/output
for pat in "h3dgs_train_05.sh" "full_train.py" "train_single.py" "train_post.py"; do
  for p in $(pgrep -f "$pat"); do kill "$p" 2>/dev/null && echo "killed $p ($pat)"; done
done
sleep 4
echo "still running: $(pgrep -f 'full_train.py|train_single.py' | wc -l)"
for c in $(ls $OUT/trained_chunks 2>/dev/null); do
  [ -e "$OUT/trained_chunks/$c/hierarchy.hier_opt" ] && continue
  mv "$OUT/trained_chunks/$c" "$OUT/trained_chunks_partial_${c}_$(date +%H%M)" && echo "quarantined partial chunk $c"
done
nohup /home/paperspace/logs/h3dgs_train_05.sh > /home/paperspace/logs/h3dgs_train_05.out 2>&1 &
sleep 3
echo "trainer relaunched: $(pgrep -f h3dgs_train_05.sh | wc -l) process(es)"
