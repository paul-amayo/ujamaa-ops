#!/usr/bin/env python3
"""First combined cut of the AIC FS 2026 demo video (Paul, 2 Oct: "make the first combined cut").

Inputs (all already rendered — nothing here touches the A100):
  reels/<cell>_720/demo.mp4     peer's "ask the orchard" reels, 1280x720, 8 fps content in a 30 fps container
  <farm>.mp4                    1 Oct live-stream walks (Klapmuts 24 fps landscape; Kendu Bay / Gwakungu 12 fps portrait)
Cards are drawn with PIL (this ffmpeg has no drawtext). Everything is normalised to 1280x720 / 30 fps / yuv420p
with a short fade on each segment, then concatenated. Output: ujamaa_demo_v1.mp4 beside the inputs.

Defaults taken for v1 (Paul to overrule): Citrus B = chunk 1_0 (the orange segment); Klapmuts slowed 2x and cut
at 90 % (the exit is blown out); phone clips slowed 2x and pillarboxed; English-only cards (taglines wait for
native checks); the globe transitions are v2.
"""
import subprocess, sys, os, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(os.environ.get("DEMO_ROOT", "/Users/paulamayo/data/for_a100/demo_video_20261001"))
OUT = ROOT / "ujamaa_demo_v1.mp4"
WORK = ROOT / "_v1_work"; WORK.mkdir(exist_ok=True)
W, H, FPS = 1280, 720, 30
FONT_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_R = "/System/Library/Fonts/Supplemental/Arial.ttf"
BG, FG, ACCENT, DIM = (11, 15, 10), (240, 236, 226), (233, 150, 55), (150, 150, 140)

def card(name, title, lines, secs, accent_line=None):
    """Static title card -> mp4 of `secs` seconds."""
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    ft = ImageFont.truetype(FONT_B, 72); fl = ImageFont.truetype(FONT_R, 34); fa = ImageFont.truetype(FONT_B, 34)
    y = 230
    d.text((90, y), title, font=ft, fill=FG); y += 110
    for ln in lines:
        d.text((92, y), ln, font=fl, fill=DIM); y += 50
    if accent_line:
        y += 20; d.text((92, y), accent_line, font=fa, fill=ACCENT)
    d.text((90, H - 70), "UJAMAA  ·  Pillar 3  ·  AIC FS 2026", font=ImageFont.truetype(FONT_R, 22), fill=DIM)
    png = WORK / f"{name}.png"; im.save(png)
    mp4 = WORK / f"{name}.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-framerate", str(FPS), "-t", str(secs), "-i", str(png),
         "-vf", f"fade=t=in:st=0:d=0.5,fade=t=out:st={secs-0.5}:d=0.5,format=yuv420p",
         "-c:v", "libx264", "-crf", "20", "-r", str(FPS), str(mp4)])
    return mp4

def clip(name, src, secs=None, slow=1.0, portrait=False, start=0.0):
    """Normalise a rendered clip: optional slow-down, trim, pillarbox, fade."""
    dur = secs if secs else probe(src) * slow
    vf = []
    if slow != 1.0: vf.append(f"setpts={slow}*PTS")
    if portrait: vf.append(f"scale=-2:{H},pad={W}:{H}:(ow-iw)/2:0:color=0x0b0f0a")
    else: vf.append(f"scale={W}:{H}")
    vf += [f"fade=t=in:st=0:d=0.5", f"fade=t=out:st={dur-0.6:.2f}:d=0.6", "format=yuv420p"]
    mp4 = WORK / f"{name}.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", str(src)]
    if secs: cmd += ["-t", str(secs)]
    cmd += ["-vf", ",".join(vf), "-r", str(FPS), "-an", "-c:v", "libx264", "-crf", "20", str(mp4)]
    run(cmd); return mp4

def probe(p):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", str(p)]).decode().strip())

def run(cmd):
    r = subprocess.run(cmd);
    if r.returncode: sys.exit(f"failed: {' '.join(cmd)}")

reels = ROOT / "reels"
segs = [
    card("c0", "UJAMAA", ["Ask the orchard: digital twins of farms that answer questions",
                          "in the farmer's own language — rendered, counted, found again."], 6,
         accent_line="Western Cape  ·  Kenya  ·  a citrus orchard"),
    card("c1", "Citrus farm B", ["100 trees · 3 surveys · 154 oranges confirmed by hand",
                                 "One pass down the row, every tree known by row and position."], 4,
         accent_line="Ni mti gani wenye machungwa mengi zaidi? Nionyeshe."),
    clip("r1", reels / "05_1_0_720" / "demo.mp4"),
    card("c2", "Citrus farm A", ["290 trees · surveyed July 2023 · 28 dB reconstruction",
                                 "Rows, trees, and the trees at the end of a row — by containment."], 4,
         accent_line="Show me the trees at the end of row 22."),
    clip("r2", reels / "01_3_1_720" / "demo.mp4"),
    card("c3", "Klapmuts, Western Cape", ["912 berry bags in December · 911 in April",
                                          "825 found again four months later, bag by bag."], 4,
         accent_line="How many bushes did we find again?"),
    clip("r3", ROOT / "klapmuts-dec25.mp4", secs=round(probe(ROOT / "klapmuts-dec25.mp4") * 0.9 * 2, 1), slow=2.0),
    card("c4", "Gwakungu, Kenya", ["Phone survey, 16 May 2026 · 29 cabbages counted along the path",
                                   "No robot, no lidar: a walk with a phone."], 4,
         accent_line="How many cabbages are there?"),
    clip("r4", ROOT / "gwakungu-cabbage.mp4", slow=2.0, portrait=True),
    card("c5", "Kendu Bay, Kenya", ["Phone survey, 14 May 2026 · a ground crop in 61 frames",
                                    "The same pipeline, from a citrus orchard to a smallholding."], 4),
    clip("r5", ROOT / "kendu-0514-plants.mp4", slow=2.0, portrait=True),
    card("c6", "Ask us · Offer", ["ASK: farms and phone surveys to twin; partners for the Kenyan sites",
                                  "OFFER: the pipeline, trained models, and answers in your language"], 8,
         accent_line="Paul Amayo · UCT · paul.amayo@uct.ac.za"),
]
lst = WORK / "concat.txt"
lst.write_text("".join(f"file '{p}'\n" for p in segs))
run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(OUT)])
print(OUT, f"{probe(OUT):.1f} s", json.dumps([f"{p.stem}:{probe(p):.1f}" for p in segs]))
