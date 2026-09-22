#!/bin/bash
# Keep the H3DGS hierarchy backend (:8006) up for Tassili whenever the GPU can take it: down during any
# post-optimisation / merge / held-out render of the 05 runner (those OOM next to it), up during training
# phases and idle time. Re-merges the finished chunks into merged_partial.hier when a new one completes.
PROJ=/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs
OUT=$PROJ/output; CH=$PROJ/camera_calibration/chunks; REPO=/home/paperspace/code/hierarchical-3d-gaussians
L=/home/paperspace/logs/hier_backend_scheduler.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
say "scheduler up"
unresp=0
while true; do
  busy=$(pgrep -fc "train_post.py|GaussianHierarchyMerger|render_hierarchy.py|h3dgs_chunk_eval|h3dgs_seam_eval")
  # "up" = a backend PROCESS exists. A healthz timeout alone never triggers a restart (it would race a
  # second instance onto the port); only a process that stays unresponsive for >90 s is force-restarted.
  up=0; pgrep -f "hier_render_service.py" > /dev/null && up=1
  if [ "$up" = 1 ]; then
    if curl -sf -m10 http://127.0.0.1:8006/healthz > /dev/null 2>&1; then unresp=0; else unresp=$((unresp+1)); fi
    if [ "$unresp" -ge 5 ]; then say "backend unresponsive for >90 s — force restart"; /home/paperspace/logs/stop_hier_service.sh >> $L 2>&1; up=0; unresp=0; fi
  fi
  if [ "$busy" -gt 0 ]; then
    [ "$up" = 1 ] && { say "GPU-heavy H3DGS step running — stopping backend"; /home/paperspace/logs/stop_hier_service.sh >> $L 2>&1; }
    sleep 20; continue
  fi
  done_chunks=$(for c in $(ls $CH 2>/dev/null); do [ -e $OUT/trained_chunks/$c/hierarchy.hier_opt ] && echo -n "$c "; done | sed 's/ $//')
  if [ -n "$done_chunks" ] && [ "$(cat $OUT/merged_partial.chunks 2>/dev/null)" != "$done_chunks" ]; then
    [ "$up" = 1 ] && { /home/paperspace/logs/stop_hier_service.sh >> $L 2>&1; up=0; }
    say "merging finished chunks: $done_chunks"
    (cd $REPO && submodules/gaussianhierarchy/build/GaussianHierarchyMerger $OUT/trained_chunks 0 $CH $OUT/merged_partial.hier $done_chunks >> $L 2>&1) \
      && echo "$done_chunks" > $OUT/merged_partial.chunks && say "merged ($(du -h $OUT/merged_partial.hier | cut -f1))" || say "MERGE FAILED"
  fi
  if [ "$up" = 0 ] && [ -e $OUT/merged_partial.hier ]; then
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -ge 16000 ]; then
      say "starting backend (free ${free} MiB) with chunks: $(cat $OUT/merged_partial.chunks)"
      /home/paperspace/logs/start_hier_service.sh $OUT/merged_partial.hier $(cat $OUT/merged_partial.chunks) >> $L 2>&1
    fi
  fi
  sleep 20
done
