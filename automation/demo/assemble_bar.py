#!/usr/bin/env python3
"""Final-cut assembler under Paul's per-frame bar (5 Oct): only videos whose frames carry >= 25 dB (cabbages 23 by his
allowance) and a measured IoU caption go in. Sources: the box's bar videos (box_weekend/) + the laptop globe clips.
1920x1080 / 30 fps / silent. Optional farms are included when their bar walk exists (klapmuts_bar/, kendu_bar/).
"""
import subprocess, sys, os, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(os.environ.get("DEMO_ROOT", "/Users/paulamayo/data/for_a100/demo_video_20261001"))
OUT = ROOT / os.environ.get("DEMO_OUT", "ujamaa_demo_bar.mp4")
WORK = ROOT / "_bar_work"; WORK.mkdir(exist_ok=True)
W, H, FPS = 1920, 1080, 30
FONT_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"; FONT_R = "/System/Library/Fonts/Supplemental/Arial.ttf"
BG, FG, ACCENT, DIM = (11, 15, 10), (240, 236, 226), (233, 150, 55), (150, 150, 140)
FOOT = "UJAMAA  ·  Pillar 3  ·  AIC FS 2026"

def run(cmd):
    if subprocess.run(cmd).returncode: sys.exit("failed: " + " ".join(map(str, cmd)))
def probe(p):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)]).decode().strip())
def draw_card(im, title, lines, accent_line, y=330):
    d = ImageDraw.Draw(im)
    d.text((135, y), title, font=ImageFont.truetype(FONT_B, 104), fill=FG); y += 160
    for ln in lines: d.text((138, y), ln, font=ImageFont.truetype(FONT_R, 48), fill=DIM); y += 72
    if accent_line: d.text((138, y + 30), accent_line, font=ImageFont.truetype(FONT_B, 48), fill=ACCENT)
    d.text((135, H - 100), FOOT, font=ImageFont.truetype(FONT_R, 30), fill=DIM)
def card(name, title, lines, secs, accent_line=None):
    im = Image.new("RGB", (W, H), BG); draw_card(im, title, lines, accent_line)
    png = WORK / f"{name}.png"; im.save(png); mp4 = WORK / f"{name}.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-framerate", str(FPS), "-t", str(secs), "-i", png,
         "-vf", f"fade=t=in:st=0:d=0.4,fade=t=out:st={secs-0.4}:d=0.4,format=yuv420p", "-c:v", "libx264", "-crf", "18", "-r", str(FPS), mp4])
    return mp4
def titled_globe(name, src, secs, title, lines, accent_line):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); draw_card(im, title, lines, accent_line, y=300)
    png = WORK / f"{name}_ovl.png"; im.save(png); mp4 = WORK / f"{name}.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-t", str(secs), "-i", src, "-i", png, "-filter_complex",
         f"[0:v]crop={W-420}:{H}:0:0,pad={W}:{H}:420:0:color=0x0b0f0a[g];[g][1:v]overlay=0:0,fade=t=in:st=0:d=0.6,fade=t=out:st={secs-0.5:.2f}:d=0.5,format=yuv420p",
         "-c:v", "libx264", "-crf", "18", "-r", str(FPS), "-an", mp4])
    return mp4
def clip(name, src, start=0.0, secs=None, portrait=False, fade_in=0.4, fade_out=0.5, slow=1.0):
    dur = (secs if secs else probe(src) - start) * slow
    vf = ([f"setpts={slow}*PTS"] if slow != 1.0 else []) + [f"scale=-2:{H}:flags=lanczos,pad={W}:{H}:(ow-iw)/2:0:color=0x0b0f0a" if portrait else f"scale={W}:{H}:flags=lanczos",
          f"fade=t=in:st=0:d={fade_in}", f"fade=t=out:st={dur-fade_out:.2f}:d={fade_out}", "format=yuv420p"]
    mp4 = WORK / f"{name}.mp4"; cmd = ["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", src]
    if secs: cmd += ["-t", str(secs)]
    cmd += ["-vf", ",".join(vf), "-r", str(FPS), "-an", "-c:v", "libx264", "-crf", "18", mp4]; run(cmd); return mp4

G, B = ROOT / "globe", ROOT / "box_weekend"
def src(n):
    """The box video without dB / training-view captions once the peer delivers it (Paul, 5 Oct); else the captioned one."""
    nc = B / n / f"{n}_nocap.mp4"
    return nc if nc.exists() else B / n / f"{n}.mp4"
zoom = lambda n, f: clip(n, G / f, start=1.5, secs=4.5, fade_in=0.2)
segs = [
    titled_globe("t0", G / "earth_idle.mp4", 5, "UJAMAA", ["Ask the orchard.", "Farms that answer questions", "in the farmer's own language."],
                 "Western Cape  ·  Kenya  ·  a citrus orchard"),
    card("c1", "Citrus farm A", ["290 trees · 36 rows · surveyed July 2023", "Identity by containment, IoU on screen"], 3,
         accent_line="Show me this row · which tree is this? · group them by row"),
    clip("ca", src("citrus_a_bar_v4")),
    card("c2", "Citrus farm B", ["100 trees · 3 surveys · 154 oranges confirmed by hand"], 3,
         accent_line="Ni mti gani wenye machungwa mengi zaidi?  ·  Which tree has the most oranges?"),
    clip("cb", src("citrus_b_cut"), slow=2.5),   # Paul: too fast to read
]
if (B / "klapmuts_bar" / "klapmuts_bar.mp4").exists():
    segs += [zoom("g_kl", "klapmuts.mp4"),
             card("c3", "Klapmuts, Western Cape", ["912 berry bags in December · 911 in April · 825 found again"], 3, accent_line="How many bushes did we find again?"),
             clip("kl", src("klapmuts_bar"))]
segs += [zoom("g_gw", "gendia.mp4"),   # the 'gendia' stop is Gwakungu, Nyahururu (relabelled 5 Oct)
         card("c4", "Gwakungu, Nyahururu", ["Phone survey, 16 May 2026 · cabbages counted along the path"], 3,
              accent_line="Which cabbage is this?  ·  How many cabbages are there?"),
         clip("cab", src("cabbage_cut"), portrait=True)]
if (B / "kendu_bar" / "kendu_bar.mp4").exists():
    segs += [zoom("g_kb", "kendu_bay.mp4"),
             card("c5", "Gendia, Kendu Bay", ["Phone survey, 14 May 2026 · a ground crop in 61 frames"], 3),
             clip("kb", src("kendu_bar"), portrait=True)]
segs += [card("c6", "Ask us · Offer", ["ASK: farms and phone surveys to twin; partners for the Kenyan sites", "OFFER: the pipeline, trained models, and answers in your language"], 6,
              accent_line="Paul Amayo · UCT · paul.amayo@uct.ac.za")]
lst = WORK / "concat.txt"; lst.write_text("".join(f"file '{p}'\n" for p in segs))
run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", OUT])
print(OUT, f"{probe(OUT):.1f} s", json.dumps([f"{p.stem}:{probe(p):.1f}" for p in segs]))
