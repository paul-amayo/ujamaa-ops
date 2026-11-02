#!/usr/bin/env python3
"""Build the UJAMAA Google.org film from a structured script (final_script.json: vo[] + shots[]).

  python3 compose_film.py <final_script.json> <out.mp4> [--voice Daniel] [--real-vo <paul_recorded.wav>]

Picture: each shot = one asset cut [in_s, out_s] to exactly its slot, 1920x1080 30 fps; landscape sources are scaled and
centre-cropped, vertical sources sit on a blurred, darkened fill of themselves. Optional on_screen_text per shot is a small
lower-right card (fades). Sound: 'paul_2023_audio' lines are cut from the 2023 talk at talk_in_s/talk_out_s; 'paul_2026_recorded'
lines are a SCRATCH track synthesised with macOS `say` (rate fitted to the slot) until Paul records them — pass --real-vo with his
recording (one file, lines in script order, each starting at its t_start) to use that instead. No music. Loudness-normalised.
Writes <out>.report.json: per VO line the slot length vs the synthesised length, and any line that had to be sped up."""
import json, subprocess, sys, re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

a = sys.argv[1:]; SCRIPT = Path(a[0]); OUT = Path(a[1])
VOICE = a[a.index("--voice") + 1] if "--voice" in a else "Daniel"
REAL = Path(a[a.index("--real-vo") + 1]) if "--real-vo" in a else None
S = json.load(open(SCRIPT)); G = Path("/Users/paulamayo/data/for_a100/google_org_2026-11"); W = OUT.parent / ("_work_" + OUT.stem); W.mkdir(parents=True, exist_ok=True)
TALK = G / "02_voices/_source/ebe_talk_2023_amayo.mp4"
FPS, SW, SH = 30, 1920, 1080
FB, FR = "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"

def run(cmd): subprocess.run(cmd, check=True)
def dur(p): return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)]))
def dims(p):
    s = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", str(p)]))["streams"][0]
    return s["width"], s["height"]
def resolve(asset):
    p = Path(asset); cands = [p, G / asset, G / "01_field" / p.name, G / "03_3d_renders" / p.name, G / "04_app" / p.name, G / "05_globe" / p.name, G / "02_voices" / p.name, G / "06_finished_video" / p.name]
    for c in cands:
        if c.exists(): return c
    raise SystemExit(f"asset not found: {asset}")

# ---------------------------------------------------------------- picture
def card_png(text, name):
    im = Image.new("RGBA", (SW, SH), (0, 0, 0, 0)); d = ImageDraw.Draw(im); f = ImageFont.truetype(FR, 34)
    lines = text.split("\n"); w = max(d.textlength(l, font=f) for l in lines) + 56; h = 26 + 48 * len(lines)
    x, y = SW - 100 - int(w), SH - 110 - int(h)
    d.rectangle([x, y, x + w, y + h], fill=(11, 15, 10, 196)); d.rectangle([x, y, x + 7, y + h], fill=(233, 150, 55, 255))
    for i, l in enumerate(lines): d.text((x + 28, y + 14 + 48 * i), l, font=f, fill=(240, 236, 226))
    p = W / f"{name}.png"; im.save(p); return p

pieces = []
for k, sh in enumerate(sorted(S["shots"], key=lambda s: s["t_start"])):
    src = resolve(sh["asset"]); slot = round(sh["t_end"] - sh["t_start"], 3); w, h = dims(src); out = W / f"shot{k:02d}.mp4"
    avail = dur(src) - sh["in_s"]; take = min(slot, avail)
    if w >= h: vf = f"scale={SW}:{SH}:force_original_aspect_ratio=increase,crop={SW}:{SH}"
    else: vf = None
    base = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(sh["in_s"]), "-i", str(src), "-t", str(take)]
    if vf: fc = f"[0:v]{vf},fps={FPS}[v]"
    else: fc = (f"[0:v]split[a][b];[a]scale={SW}:{SH}:force_original_aspect_ratio=increase,crop={SW}:{SH},boxblur=40:2,eq=brightness=-0.12[bg];"
                f"[b]scale=-2:{SH}[fg];[bg][fg]overlay=(W-w)/2:0,fps={FPS}[v]")
    if take < slot - 0.05:   # asset too short for the slot: hold its last frame
        fc += f";[v]tpad=stop_mode=clone:stop_duration={slot - take:.3f}[v]"
    txt = (sh.get("on_screen_text") or "").strip()
    if txt:
        png = card_png(txt, f"card{k:02d}"); base += ["-loop", "1", "-i", str(png)]
        fc += f";[1:v]format=rgba,fade=t=in:st=0.3:d=0.4:alpha=1,fade=t=out:st={max(0.4, slot - 0.8):.2f}:d=0.4:alpha=1[c];[v][c]overlay=0:0:shortest=1[v]"
    run(base + ["-filter_complex", fc + ";[v]format=yuv420p[vo]", "-map", "[vo]", "-t", str(slot), "-an", "-c:v", "libx264", "-crf", "17", "-preset", "medium", str(out)])
    pieces.append(out)
lst = W / "video.txt"; lst.write_text("".join(f"file '{p}'\n" for p in pieces)); video = W / "video.mp4"
run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(video)])
TOTAL = dur(video)

# ---------------------------------------------------------------- sound
report = []; tracks = []   # (path, start_s)
if REAL:
    tracks.append((REAL, 0.0))
for i, line in enumerate(S["vo"]):
    slot = line["t_end"] - line["t_start"]; src = line["source"]
    if src == "silence" or not line.get("text", "").strip(): continue
    if src == "paul_2023_audio":
        d = line["talk_out_s"] - line["talk_in_s"]; p = W / f"vo{i:02d}_2023.wav"
        run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(line["talk_in_s"]), "-i", str(TALK), "-t", str(d), "-vn",
             "-af", f"afade=t=in:d=0.12,afade=t=out:st={max(0, d - 0.25):.2f}:d=0.25", "-ar", "48000", "-ac", "2", str(p)])
        tracks.append((p, line["t_start"])); report.append({"line": i, "source": src, "slot_s": round(slot, 2), "audio_s": round(d, 2)}); continue
    if REAL: continue   # recorded narration already covers these
    words = len(re.findall(r"[\w'’-]+", line["text"])); rate = 165; p = None
    for attempt in range(4):   # fit the scratch read to the slot
        aiff = W / f"vo{i:02d}_say.aiff"; p = W / f"vo{i:02d}_say.wav"
        run(["say", "-v", VOICE, "-r", str(rate), "-o", str(aiff), line["text"]])
        run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(aiff), "-ar", "48000", "-ac", "2", "-af", "afade=t=in:d=0.05", str(p)])
        d = dur(p)
        if d <= slot - 0.15 or attempt == 3: break
        rate = int(rate * min(1.35, d / max(slot - 0.3, 0.5)) + 2)
    tracks.append((p, line["t_start"])); report.append({"line": i, "source": "scratch_say", "words": words, "slot_s": round(slot, 2), "audio_s": round(d, 2), "rate_wpm": rate, "fits": d <= slot})
# mix: silent bed + every track delayed to its start
inputs = ["-f", "lavfi", "-t", str(TOTAL), "-i", "anullsrc=r=48000:cl=stereo"]; fc = []
for j, (p, st) in enumerate(tracks):
    inputs += ["-i", str(p)]; fc.append(f"[{j + 1}:a]adelay={int(st * 1000)}|{int(st * 1000)}[t{j}]")
mix = "[0:a]" + "".join(f"[t{j}]" for j in range(len(tracks))) + f"amix=inputs={len(tracks) + 1}:normalize=0:dropout_transition=0,loudnorm=I=-16:TP=-1.5:LRA=11[a]"
audio = W / "audio.wav"
run(["ffmpeg", "-nostdin", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(fc + [mix]), "-map", "[a]", "-t", str(TOTAL), str(audio)])
run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video), "-i", str(audio), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(OUT)])
json.dump({"out": str(OUT), "duration_s": round(TOTAL, 2), "shots": len(pieces), "voice": VOICE if not REAL else str(REAL), "lines": report}, open(str(OUT) + ".report.json", "w"), indent=1)
tight = [r for r in report if r.get("fits") is False]
print(f"{OUT} {TOTAL:.1f} s, {len(pieces)} shots, {len(report)} voice lines; {len(tight)} scratch lines longer than their slot" + (": " + ", ".join(str(r['line']) for r in tight) if tight else ""))
