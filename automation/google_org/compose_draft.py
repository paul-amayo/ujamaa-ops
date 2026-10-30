#!/usr/bin/env python3
"""Draft composition for the Google.org reel (Paul, 30 Oct 2026: "show me how a full video could look like composed").
A ~100 s sketch from the google_org_2026-11 assets, to show the shape, not a final edit:
  1. voice 1 (collaborative approach) on camera, then over the globe tour
  2. voice 2 (lowering the barrier) on camera, then over phone field footage (face-blurred clips only)
  3. voice 3 (robot-equipped phones) on camera, then over the 3D renders (oranges, cabbages, chillies)
  4. the product: a question asked in Kiswahili in the live app, answered, the tree lit (natural silence)
  5. end card: the credit line
Vertical clips sit on a blurred, enlarged copy of themselves. One optional lower third on the first voice shot.
No music. Output: google_org_2026-11/_draft/ujamaa_google_org_draft.mp4 (1920x1080, 30 fps, AAC)."""
import subprocess, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

G = Path("/Users/paulamayo/data/for_a100/google_org_2026-11"); OUT = G / "_draft"; W = OUT / "_work"; W.mkdir(parents=True, exist_ok=True)
FPS, SW, SH = 30, 1920, 1080
FB, FR = "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"

def run(cmd): subprocess.run(cmd, check=True)
def dur(p): return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)]))
def dims(p):
    s = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", str(p)]))["streams"][0]
    return s["width"], s["height"]

def video(src, start, secs, name):
    """a video-only 1920x1080 30 fps piece; vertical sources over a blurred fill of themselves"""
    w, h = dims(src); out = W / f"{name}.mp4"
    if w >= h:
        vf = f"scale={SW}:{SH}:force_original_aspect_ratio=increase,crop={SW}:{SH},fps={FPS},format=yuv420p"
        args = ["-vf", vf]
    else:
        fc = (f"[0:v]split[a][b];[a]scale={SW}:{SH}:force_original_aspect_ratio=increase,crop={SW}:{SH},boxblur=40:2,eq=brightness=-0.12[bg];"
              f"[b]scale=-2:{SH}[fg];[bg][fg]overlay=(W-w)/2:0,fps={FPS},format=yuv420p")
        args = ["-filter_complex", fc]
    run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(start), "-i", str(src), "-t", str(secs), *args, "-an",
         "-c:v", "libx264", "-crf", "18", "-preset", "medium", str(out)])
    return out

def concat_video(parts, name):
    lst = W / f"{name}.txt"; lst.write_text("".join(f"file '{p}'\n" for p in parts)); out = W / f"{name}.mp4"
    run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)]); return out

def lower_third(name, l1, l2):
    im = Image.new("RGBA", (SW, SH), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    f1, f2 = ImageFont.truetype(FB, 46), ImageFont.truetype(FR, 32)
    wbox = max(d.textlength(l1, font=f1), d.textlength(l2, font=f2)) + 60; x, y = SW - 110 - int(wbox), SH - 210   # right side: the EBE Talks logo sits bottom-left
    d.rectangle([x - 30, y - 22, x - 30 + wbox, y + 108], fill=(11, 15, 10, 200)); d.rectangle([x - 30, y - 22, x - 22, y + 108], fill=(233, 150, 55, 255))
    d.text((x, y - 6), l1, font=f1, fill=(240, 236, 226)); d.text((x, y + 54), l2, font=f2, fill=(200, 196, 186))
    p = W / f"{name}.png"; im.save(p); return p

def voice_segment(name, voice, on_camera, broll, lt=None):
    """voice on camera for `on_camera` s, then b-roll for the rest; the voice's audio runs throughout"""
    total = dur(voice); cam = video(voice, 0, on_camera, f"{name}_cam")
    if lt:
        ov = W / f"{name}_cam_lt.mp4"
        run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(cam), "-loop", "1", "-i", str(lt), "-filter_complex",
             f"[1:v]format=rgba,fade=t=in:st=0.6:d=0.4:alpha=1,fade=t=out:st={on_camera - 1.0}:d=0.4:alpha=1[l];[0:v][l]overlay=0:0:shortest=1,format=yuv420p",
             "-c:v", "libx264", "-crf", "18", str(ov)]); cam = ov
    rest = total - on_camera; parts, left = [cam], rest
    for i, (src, st, d) in enumerate(broll):
        d = min(d, left) if i < len(broll) - 1 else left
        parts.append(video(src, st, d, f"{name}_b{i}")); left -= d
    v = concat_video(parts, f"{name}_v"); out = W / f"{name}.mp4"
    run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(v), "-i", str(voice), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
         "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-shortest", str(out)])
    return out

def silent_segment(name, pieces):
    v = concat_video([video(s, st, d, f"{name}_{i}") for i, (s, st, d) in enumerate(pieces)], f"{name}_v"); out = W / f"{name}.mp4"
    run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(v), "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)]); return out

def end_card(name, secs=4.0):
    im = Image.new("RGB", (SW, SH), (11, 15, 10)); d = ImageDraw.Draw(im)
    t1, t2 = "UCT African Robotics Unit · UJAMAA", "Robotics opened the path. The phone walks it."
    f1, f2 = ImageFont.truetype(FB, 64), ImageFont.truetype(FR, 38)
    d.text(((SW - d.textlength(t1, font=f1)) / 2, SH / 2 - 70), t1, font=f1, fill=(240, 236, 226))
    d.text(((SW - d.textlength(t2, font=f2)) / 2, SH / 2 + 20), t2, font=f2, fill=(233, 150, 55))
    p = W / f"{name}.png"; im.save(p); out = W / f"{name}.mp4"
    run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-loop", "1", "-framerate", str(FPS), "-t", str(secs), "-i", str(p), "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
         "-vf", f"fade=t=in:st=0:d=0.6,fade=t=out:st={secs - 0.6}:d=0.6,format=yuv420p", "-c:v", "libx264", "-crf", "18", "-c:a", "aac", "-shortest", str(out)])
    return out

V, R, F, A, GL = G / "02_voices", G / "03_3d_renders", G / "01_field", G / "04_app", G / "05_globe"
segs = [
    voice_segment("s1", V / "01_amayo_collaborative_approach.mp4", 7.0, [(GL / "tour.mp4", 0, 15.0)],
                  lt=lower_third("lt1", "Paul Amayo", "UCT African Robotics Unit · UJAMAA")),
    voice_segment("s2", V / "02_amayo_lowering_the_barrier.mp4", 6.0,
                  [(F / "06_gwakungu_open_field.mp4", 1.0, 6.5), (F / "08_gwakungu_cabbages.mp4", 2.0, 7.0)]),
    voice_segment("s3", V / "03_amayo_robot_equipped_phones.mp4", 5.0,
                  [(R / "02_citrus_b_oranges_lane.mp4", 0.5, 7.0), (R / "01_citrus_a_orchard_query.mp4", 30.0, 4.5),
                   (R / "05_gwakungu_cabbages_query.mp4", 2.0, 3.5), (R / "06_gwakungu_red_chillies.mp4", 3.0, 6.0)]),
    silent_segment("s4", [(A / "02_citrus_b_oranges.mp4", 18.0, 7.0), (A / "02_citrus_b_oranges.mp4", 45.0, 12.0)]),
    end_card("s5"),
]
lst = OUT / "_work" / "final.txt"; lst.write_text("".join(f"file '{p}'\n" for p in segs))
out = OUT / "ujamaa_google_org_draft.mp4"
run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c:v", "libx264", "-crf", "18", "-preset", "medium",
     "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)])
print(out, f"{dur(out):.1f} s", [f"{p.stem}:{dur(p):.1f}" for p in segs])
