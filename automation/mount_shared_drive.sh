#!/bin/bash
# mount_shared_drive.sh (2026-10-21) — mount a Paperspace shared drive (SMB) the way /mnt/shared is mounted: root-only credentials file
# /etc/smb-<name>.cred, fstab entry (vers=3.0, uid/gid 1000, mfsymlinks, _netdev, nofail, systemd automount, 30 s timeout). It test-mounts
# first and touches fstab only after the test succeeds (fstab is backed up). The password is prompted without echo, or read from SMB_PASS.
#   bash mount_shared_drive.sh '<network path: \\10.x.x.x\sdXXXX or //10.x.x.x/sdXXXX>' [mount point, default /mnt/shared2]
set -euo pipefail
SRC=${1:?network path from the Paperspace console}; MP=${2:-/mnt/shared2}
SRC=$(printf '%s' "$SRC" | tr '\\' '/'); case $SRC in //*) ;; /*) SRC="/$SRC";; *) SRC="//$SRC";; esac
HOST=$(printf '%s' "$SRC" | cut -d/ -f3); NAME=$(basename "$MP"); CRED=/etc/smb-$NAME.cred
grep -qs " $MP " /proc/mounts && { echo "[mount] $MP is already mounted"; exit 1; }
grep -qsE "^[^#]*[[:space:]]$MP[[:space:]]" /etc/fstab && { echo "[mount] $MP already has an fstab entry"; exit 1; }
timeout 5 bash -c "echo > /dev/tcp/$HOST/445" 2>/dev/null || { echo "[mount] $HOST:445 is not reachable from this machine (private network $(ip -4 -o addr show eth0 | awk '{print $4}'))"; exit 1; }
U=${SMB_USER:-}; PW=${SMB_PASS:-}
[ -n "$U" ] || read -rp "username for $SRC: " U
[ -n "$PW" ] || { read -rsp "password: " PW; echo; }
sudo install -m 600 -o root -g root /dev/null "$CRED"; printf 'username=%s\npassword=%s\n' "$U" "$PW" | sudo tee "$CRED" > /dev/null; unset PW SMB_PASS
sudo mkdir -p "$MP"
if ! sudo mount -t cifs "$SRC" "$MP" -o credentials=$CRED,uid=1000,gid=1000,vers=3.0,mfsymlinks; then echo "[mount] test mount failed; fstab untouched, credentials removed"; sudo rm -f "$CRED"; exit 1; fi
sudo umount "$MP"; B=/etc/fstab.bak.$(date +%Y%m%d-%H%M%S); sudo cp /etc/fstab "$B"
echo "$SRC $MP cifs credentials=$CRED,uid=1000,gid=1000,vers=3.0,mfsymlinks,_netdev,nofail,x-systemd.automount,x-systemd.mount-timeout=30 0 0" | sudo tee -a /etc/fstab > /dev/null
sudo systemctl daemon-reload; sudo systemctl start "$(systemd-escape -p --suffix=automount "$MP")"
ls "$MP" > /dev/null; echo "[mount] $SRC on $MP (fstab backup $B)"; df -h "$MP" | tail -1
