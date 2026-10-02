#!/usr/bin/env python3
"""Second cut of the AIC FS 2026 demo video (Paul's v1 review, 2 Oct): globe → farm framing, Citrus B driven from
chunk 0_0 (1_0's turn poses were the grey frames) with the fruit story as its own clip, the three non-citrus walks
re-shot at 8 fps, the Gwakungu reel with registry + queries. 1920x1080 / 30 fps / silent. Nothing here touches the A100.

Inputs under DEMO_ROOT (default /Users/paulamayo/data/for_a100/demo_video_20261001):
  globe/*.mp4                       laptop-rendered globe clips (1080p, 6 s each)
  reels/05_0_0_720, 01_3_1_720      peer's "ask the orchard" reels (1280x720, 8 fps in 30)
  reels/05_1_0_fruit_720/demo_a.mp4 peer's fruit clip (falls back to the 1_0 reel's fruit segment if absent)
  reels/gwakungu_7993/demo.mp4      cabbage reel (540x960 portrait, 5 fps)
  v2_walks/{klapmuts-dec25,kendu-0514-plants}.mp4   8 fps live-stream walks (v2 shots)
Cards are PIL (no drawtext in this ffmpeg). Output: ujamaa_demo_v2.mp4.
"""
import subprocess, sys, os, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(os.environ.get("DEMO_ROOT", "/Users/paulamayo/data/for_a100/demo_video_20261001"))
OUT = ROOT / "ujamaa_demo_v2.mp4"
WORK = ROOT / "_v2_work"; WORK.mkdir(exist_ok=True)
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
         "-vf", f"fade=t=in:st=0:d=0.5,fade=t=out:st={secs-0.5}:d=0.5,format=yuv420p",
         "-c:v", "libx264", "-crf", "18", "-r", str(FPS), mp4])
    return mp4

def titled_globe(name, src, title, lines, accent_line):
    """The idle globe with the title composited on the left (transparent PNG overlay, fades with the clip)."""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); draw_card(im, title, lines, accent_line, y=300)
    png = WORK / f"{name}_ovl.png"; im.save(png); mp4 = WORK / f"{name}.mp4"; dur = probe(src)
    run(["ffmpeg", "-v", "error", "-y", "-i", src, "-i", png, "-filter_complex",
         f"[0:v][1:v]overlay=0:0,fade=t=in:st=0:d=0.8,fade=t=out:st={dur-0.6:.2f}:d=0.6,format=yuv420p",
         "-c:v", "libx264", "-crf", "18", "-r", str(FPS), "-an", mp4])
    return mp4

def clip(name, src, secs=None, slow=1.0, portrait=False, start=0.0, fade=True):
    dur = (secs if secs else (probe(src) - start)) * slow
    vf = []
    if slow != 1.0: vf.append(f"setpts={slow}*PTS")
    if portrait: vf.append(f"scale=-2:{H}:flags=lanczos,pad={W}:{H}:(ow-iw)/2:0:color=0x0b0f0a")
    else: vf.append(f"scale={W}:{H}:flags=lanczos")
    if fade: vf += ["fade=t=in:st=0:d=0.5", f"fade=t=out:st={dur-0.6:.2f}:d=0.6"]
    vf.append("format=yuv420p")
    mp4 = WORK / f"{name}.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", src]
    if secs: cmd += ["-t", str(secs)]
    cmd += ["-vf", ",".join(vf), "-r", str(FPS), "-an", "-c:v", "libx264", "-crf", "18", mp4]
    run(cmd); return mp4

G, R, V = ROOT / "globe", ROOT / "reels", ROOT / "v2_walks"
fruit_src = R / "05_1_0_fruit_720" / "demo_a.mp4"
if fruit_src.exists(): fruit = clip("fruit", fruit_src)
else:
    print("fruit clip not yet delivered — stand-in: the 1_0 reel's fruit segment", file=sys.stderr)
    fruit = clip("fruit", R / "05_1_0_720" / "demo.mp4", start=1.4, secs=5.4)

segs = [
    titled_globe("t0", G / "earth_idle.mp4", "UJAMAA",
                 ["Ask the orchard: farms that answer questions", "in the farmer's own language."],
                 "Western Cape  ·  Kenya  ·  a citrus orchard"),
    # citrus (continent only)
    clip("g_cit", G / "citrus.mp4"),
    card("c1", "Citrus farm B", ["100 trees · 3 surveys · 154 oranges confirmed by hand"], 3,
         accent_line="Ni mti gani wenye machungwa mengi zaidi?  ·  Which tree has the most oranges?"),
    clip("r_00", R / "05_0_0_720" / "demo.mp4"),
    fruit,
    card("c2", "Citrus farm A", ["290 trees · surveyed July 2023 · 28 dB reconstruction"], 3,
         accent_line="Show me the trees at the end of row 22."),
    clip("r_31", R / "01_3_1_720" / "demo.mp4"),
    clip("g_cit_out", G / "citrus_out.mp4"),
    # Klapmuts
    clip("g_kl", G / "klapmuts.mp4"),
    card("c3", "Klapmuts, Western Cape", ["912 berry bags in December · 911 in April · 825 found again"], 3,
         accent_line="How many bushes did we find again?"),
    clip("w_kl", V / "klapmuts-dec25.mp4", secs=35),
    clip("g_kl_out", G / "klapmuts_out.mp4"),
    # Gwakungu (Gendia)
    clip("g_ge", G / "gendia.mp4"),
    card("c4", "Gwakungu, Kenya", ["Phone survey, 16 May 2026 · 29 cabbages in two rows, counted along the path"], 3,
         accent_line="How many cabbages are there?"),
    clip("r_cab", R / "gwakungu_7993" / "demo.mp4", portrait=True),
    clip("g_ge_out", G / "gendia_out.mp4"),
    # Kendu Bay
    clip("g_kb", G / "kendu_bay.mp4"),
    card("c5", "Kendu Bay, Kenya", ["Phone survey, 14 May 2026 · a ground crop in 61 frames"], 3),
    clip("w_kb", V / "kendu-0514-plants.mp4", portrait=True),
    clip("g_kb_out", G / "kendu_bay_out.mp4"),
    card("c6", "Ask us · Offer", ["ASK: farms and phone surveys to twin; partners for the Kenyan sites",
                                  "OFFER: the pipeline, trained models, and answers in your language"], 8,
         accent_line="Paul Amayo · UCT · paul.amayo@uct.ac.za"),
]
lst = WORK / "concat.txt"; lst.write_text("".join(f"file '{p}'\n" for p in segs))
run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", OUT])
print(OUT, f"{probe(OUT):.1f} s", json.dumps([f"{p.stem}:{probe(p):.1f}" for p in segs]))
