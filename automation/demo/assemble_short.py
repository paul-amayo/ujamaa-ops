#!/usr/bin/env python3
"""90-second cut-down of the demo video for the gallery loop (Paul, 5 Oct: "we are running late"): the same
sources as assemble_v2.py, question segments only, globe zooms trimmed to their last 4 s. 1920x1080 / 30 fps / silent.
Output: ujamaa_demo_short.mp4 (DEMO_OUT overrides). CPU only.
"""
import subprocess, sys, os, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(os.environ.get("DEMO_ROOT", "/Users/paulamayo/data/for_a100/demo_video_20261001"))
OUT = ROOT / os.environ.get("DEMO_OUT", "ujamaa_demo_short.mp4")
WORK = ROOT / "_short_work"; WORK.mkdir(exist_ok=True)
W, H, FPS = 1920, 1080, 30
FONT_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_R = "/System/Library/Fonts/Supplemental/Arial.ttf"
BG, FG, ACCENT, DIM = (11, 15, 10), (240, 236, 226), (233, 150, 55), (150, 150, 140)
FOOT = "UJAMAA  ·  Pillar 3  ·  AIC FS 2026"

def run(cmd):
    if subprocess.run(cmd).returncode: sys.exit("failed: " + " ".join(map(str, cmd)))

def probe(p):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", str(p)]).decode().strip())

def draw_card(im, title, lines, accent_line, y=330):
    d = ImageDraw.Draw(im)
    d.text((135, y), title, font=ImageFont.truetype(FONT_B, 104), fill=FG); y += 160
    for ln in lines:
        d.text((138, y), ln, font=ImageFont.truetype(FONT_R, 48), fill=DIM); y += 72
    if accent_line:
        d.text((138, y + 30), accent_line, font=ImageFont.truetype(FONT_B, 48), fill=ACCENT)
    d.text((135, H - 100), FOOT, font=ImageFont.truetype(FONT_R, 30), fill=DIM)

def card(name, title, lines, secs, accent_line=None):
    im = Image.new("RGB", (W, H), BG); draw_card(im, title, lines, accent_line)
    png = WORK / f"{name}.png"; im.save(png); mp4 = WORK / f"{name}.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-framerate", str(FPS), "-t", str(secs), "-i", png,
         "-vf", f"fade=t=in:st=0:d=0.4,fade=t=out:st={secs-0.4}:d=0.4,format=yuv420p",
         "-c:v", "libx264", "-crf", "18", "-r", str(FPS), mp4])
    return mp4

def titled_globe(name, src, secs, title, lines, accent_line):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); draw_card(im, title, lines, accent_line, y=300)
    png = WORK / f"{name}_ovl.png"; im.save(png); mp4 = WORK / f"{name}.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-t", str(secs), "-i", src, "-i", png, "-filter_complex",
         f"[0:v]crop={W-420}:{H}:0:0,pad={W}:{H}:420:0:color=0x0b0f0a[g];[g][1:v]overlay=0:0,"
         f"fade=t=in:st=0:d=0.6,fade=t=out:st={secs-0.5:.2f}:d=0.5,format=yuv420p",
         "-c:v", "libx264", "-crf", "18", "-r", str(FPS), "-an", mp4])
    return mp4

def clip(name, src, start=0.0, secs=None, portrait=False, fade_in=0.4, fade_out=0.5):
    dur = secs if secs else probe(src) - start
    vf = [f"scale=-2:{H}:flags=lanczos,pad={W}:{H}:(ow-iw)/2:0:color=0x0b0f0a" if portrait
          else f"scale={W}:{H}:flags=lanczos",
          f"fade=t=in:st=0:d={fade_in}", f"fade=t=out:st={dur-fade_out:.2f}:d={fade_out}", "format=yuv420p"]
    mp4 = WORK / f"{name}.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", src]
    if secs: cmd += ["-t", str(secs)]
    cmd += ["-vf", ",".join(vf), "-r", str(FPS), "-an", "-c:v", "libx264", "-crf", "18", mp4]
    run(cmd); return mp4

G, R, V = ROOT / "globe", ROOT / "reels", ROOT / "v2_walks"
zoom = lambda n, f: clip(n, G / f, start=2.0, secs=4.0, fade_in=0.2)   # last 4 s of a 6 s zoom
segs = [
    titled_globe("t0", G / "earth_idle.mp4", 5, "UJAMAA",
                 ["Ask the orchard.", "Farms that answer questions", "in the farmer's own language."],
                 "Western Cape  ·  Kenya  ·  a citrus orchard"),
    zoom("g_cit", "citrus.mp4"),
    clip("r_00", R / "05_0_0_720" / "demo.mp4", start=3.8, secs=12.8),      # show me this row → which tree is this?
    clip("fruit", R / "05_1_0_fruit_720" / "demo_a.mp4", secs=12.0),        # which tree has the most oranges?
    clip("r_31", R / "01_3_1_720" / "demo.mp4", start=32.5, secs=11.4),     # by row → trees at the end of row 22
    zoom("g_kl", "klapmuts.mp4"),
    clip("w_kl", V / "klapmuts-dec25.mp4", start=5.0, secs=10.0),
    zoom("g_ge", "gendia.mp4"),
    clip("r_cab", R / "gwakungu_7993" / "demo.mp4", start=3.6, secs=7.2, portrait=True),  # which cabbage → how many
    zoom("g_kb", "kendu_bay.mp4"),
    clip("w_kb", V / "kendu-0514-plants.mp4", secs=5.0, portrait=True),
    card("c6", "Ask us · Offer", ["ASK: farms and phone surveys to twin; partners for the Kenyan sites",
                                  "OFFER: the pipeline, trained models, and answers in your language"], 6,
         accent_line="Paul Amayo · UCT · paul.amayo@uct.ac.za"),
]
lst = WORK / "concat.txt"; lst.write_text("".join(f"file '{p}'\n" for p in segs))
run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", OUT])
print(OUT, f"{probe(OUT):.1f} s", json.dumps([f"{p.stem}:{probe(p):.1f}" for p in segs]))
