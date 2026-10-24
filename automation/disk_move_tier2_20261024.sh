#!/bin/bash
# Tier-2 move (Paul, 24 Oct: "move tier 2 to shared2"): six re-ingestable monolithics -> /mnt/shared2/ujamaa_work/data/<same path>,
# byte-compared with cmp, then the local file is replaced by a symlink (readers keep their paths). Any mismatch -> local kept.
set -u
SRC=/home/paperspace/data; DST=/mnt/shared2/ujamaa_work/data; L=/home/paperspace/logs/disk_move_tier2_20261024.log
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a $L; }
K=klapmuts
ITEMS="$K/apr_2026_1130/prod/monos/monolithics/image_right.monolithic $K/apr_2026_1400/prod/monos/monolithics/image_right.monolithic
$K/apr_2026_1517/prod/monos/monolithics/image_right.monolithic $K/apr_2026_d2_1230/prod/monos/monolithics/image_right.monolithic
$K/apr_2026_d2_1552/prod/monos/monolithics/image_right.monolithic $K/feb_2026_1003/prod/monos/monolithics/image_left.monolithic"
ls /mnt/shared2 >/dev/null; mountpoint -q /mnt/shared2 || { say "ABORT: /mnt/shared2 not mounted"; exit 1; }
say "=== start; root free $(df -h / | awk 'NR==2{print $4}'); shared2 free $(df -h /mnt/shared2 | awk 'NR==2{print $4}')"
for rel in $ITEMS; do
  s="$SRC/$rel"; d="$DST/$rel"
  [ -L "$s" ] && { say "skip $rel (already a symlink)"; continue; }
  [ -f "$s" ] || { say "skip $rel (missing)"; continue; }
  [ -n "$(lsof "$s" 2>/dev/null | tail -n +2)" ] && { say "skip $rel (open)"; continue; }
  mkdir -p "$(dirname "$d")"; t0=$(date +%s)
  cp --preserve=timestamps "$s" "$d" 2>>$L
  if [ "$(stat -c %s "$s")" = "$(stat -c %s "$d")" ] && cmp -s "$s" "$d"; then
    rm "$s" && ln -s "$d" "$s" && say "moved $rel: $(numfmt --to=iec $(stat -c %s "$d")), byte-identical, $(( $(date +%s)-t0 )) s"
  else say "KEPT LOCAL $rel: copy differs"; fi
done
say "=== done; root free $(df -h / | awk 'NR==2{print $4}'); shared2 free $(df -h /mnt/shared2 | awk 'NR==2{print $4}')"; touch /home/paperspace/logs/disk_move_tier2_20261024.DONE
