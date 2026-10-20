#!/bin/bash
# Move cold data from the A100 root disk to the Paperspace shared drive (Paul, 20 Oct: "move all").
# Per directory: copy to /mnt/shared/data/<same relative path>, verify file count + apparent bytes match,
# then delete the local copy and leave a symlink at the old path. Any mismatch or rsync error -> local copy kept.
set -u
SRC=/home/paperspace/data; DST=/mnt/shared/data; L=/home/paperspace/logs/disk_move_20261020.log
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a $L; }
ITEMS="klapmuts/ten_row_10_am citrus_all/03_13B_Jackal/prod/tassili klapmuts/rosbag2_2026_02_02-08_03_05 klapmuts/apr_2026_zed/prod/tassili
citrus_all/02_13B_Jackal/prod/tassili public/fruitnerf/FruitNeRF_Dataset klapmuts/dec_2025_ten_rows/prod/tassili klapmuts/apr_2026_zed/experimental/h3dgs
citrus_all/04_13D_Jackal/experimental/h3dgs gendia/mango/images_orig gendia/sukuma/images_orig high citrus_all/01_13B_Jackal/prod/scratch_sam3
citrus_all/03_13B_Jackal/prod/scratch_sam3 gendia/pawpaw/images_orig image_farm/kendu_bay/2026-05-14/IMG_7967"
mountpoint -q /mnt/shared || ls /mnt/shared >/dev/null; mountpoint -q /mnt/shared || { say "ABORT: /mnt/shared not mounted"; exit 1; }
say "=== start; root free $(df -h / | awk 'NR==2{print $4}')"
for rel in $ITEMS; do
  s="$SRC/$rel"; d="$DST/$rel"
  [ -L "$s" ] && { say "skip $rel (already a symlink)"; continue; }
  [ -d "$s" ] || { say "skip $rel (missing)"; continue; }
  if fuser -sm "$s" 2>/dev/null && [ -n "$(lsof +D "$s" 2>/dev/null | head -2 | tail -1)" ]; then say "skip $rel (files open)"; continue; fi
  nl=$(find "$s" -type l | wc -l)
  mkdir -p "$d"; t0=$(date +%s)
  rsync -rlt --no-perms --no-owner --no-group "$s/" "$d/" 2>>$L; rc=$?
  ns=$(find "$s" -type f | wc -l); nd=$(find "$d" -type f | wc -l)
  bs=$(du -sb --apparent-size "$s" | cut -f1); bd=$(du -sb --apparent-size "$d" | cut -f1)
  if [ $rc -eq 0 ] && [ "$ns" = "$nd" ] && [ "$bs" = "$bd" ]; then
    rm -rf "$s" && ln -s "$d" "$s" && say "moved $rel: $ns files, $((bs/1073741824)) GB, symlinks inside $nl, $(( $(date +%s)-t0 )) s"
  else
    say "KEPT LOCAL $rel: rsync rc=$rc files $ns/$nd bytes $bs/$bd (copy left on the share for inspection)"
  fi
done
say "=== done; root free $(df -h / | awk 'NR==2{print $4}'); share free $(df -h /mnt/shared | awk 'NR==2{print $4}')"; touch /home/paperspace/logs/disk_move_20261020.DONE
