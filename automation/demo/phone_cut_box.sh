#!/bin/bash
# phone_cut_box.sh — one 720p phone-friendly file from the demo segments that live on the box (UJAMAA, 2026-10-03; Paul:
# "send the ujamaa demo for phone viewing"). v3's order (assemble_v2.py) WITHOUT the globe zooms and title cards (those
# live on the laptop): Citrus B 0_0 reel, fruit clip (variant a), Citrus A 3_1 reel, Klapmuts lane walk (first 35 s),
# Gwakungu cabbage reel, Kendu Bay ground-crop walk. Portrait clips pillarboxed; H.264 CRF 26, faststart, silent.
set -euo pipefail
N='scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,fps=30,format=yuv420p,setsar=1'
ffmpeg -y -v error -i /home/paperspace/logs/demo_chunks/05_0_0_720/demo.mp4 -i /home/paperspace/logs/demo_chunks/05_1_0_fruit_720/demo_a.mp4 -i /home/paperspace/logs/demo_chunks/01_3_1_720/demo.mp4 -t 35 -i /home/paperspace/data/demo_video_v2/klapmuts-dec25.mp4 -i /home/paperspace/logs/demo_chunks/gwakungu_7993/demo.mp4 -i /home/paperspace/data/demo_video_v2/kendu-0514-plants.mp4 \
  -filter_complex "[0:v]$N[a];[1:v]$N[b];[2:v]$N[c];[3:v]$N[d];[4:v]$N[e];[5:v]$N[f];[a][b][c][d][e][f]concat=n=6:v=1:a=0[v]" \
  -map "[v]" -c:v libx264 -preset medium -crf 26 -profile:v high -level 4.0 -movflags +faststart -an /home/paperspace/data/demo_video_v2/ujamaa_demo_phone_20261003.mp4
