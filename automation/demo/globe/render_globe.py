#!/usr/bin/env python3
"""Render globe.html headless into 1920x1080 / 30 fps mp4 clips for the demo video.

Renderer: Google Chrome (headless=new) driven over the DevTools protocol with a tiny
pure-stdlib websocket client (no playwright / node on this Mac). globe.html is served
from a local http.server (file:// would block the WebGL texture). Every frame is
window.renderFrame(i) -> PNG data URL, so capture is exact and wall-clock free.

    python3 render_globe.py                      # all clips -> OUT_DIR
    python3 render_globe.py --clips klapmuts klapmuts_out --frames 180
    python3 render_globe.py --chrome /path/to/Chrome --keep-frames

Outputs (OUT_DIR): <clip>.mp4 per clip, checks/<clip>_f{090,179}.png, sheet.png, sheet.md.
"""
import argparse
import base64
import http.server
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from functools import partial

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = '/Users/paulamayo/data/for_a100/demo_video_20261001/globe'
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

# clip name -> (stop, mode)
CLIPS = {
    'earth_idle':    ('earth_idle', 'in'),
    'klapmuts':      ('klapmuts', 'in'),
    'klapmuts_out':  ('klapmuts', 'out'),
    'kendu_bay':     ('kendu_bay', 'in'),
    'kendu_bay_out': ('kendu_bay', 'out'),
    'gendia':        ('gendia', 'in'),
    'gendia_out':    ('gendia', 'out'),
    'citrus':        ('citrus', 'in'),
    'citrus_out':    ('citrus', 'out'),
}


# ----------------------------------------------------------------------------- websocket
class WS:
    """Minimal RFC6455 client: text frames, masking, 64-bit lengths, continuation, ping."""

    def __init__(self, url):
        assert url.startswith('ws://')
        hostport, _, path = url[5:].partition('/')
        host, _, port = hostport.partition(':')
        self.sock = socket.create_connection((host, int(port or 80)))
        self.sock.settimeout(120)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (f'GET /{path} HTTP/1.1\r\nHost: {hostport}\r\nUpgrade: websocket\r\n'
               f'Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n')
        self.sock.sendall(req.encode())
        buf = b''
        while b'\r\n\r\n' not in buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise RuntimeError('websocket handshake failed')
            buf += chunk
        if b' 101 ' not in buf.split(b'\r\n', 1)[0]:
            raise RuntimeError('websocket upgrade refused: ' + buf.decode(errors='replace')[:200])
        self.rest = buf.split(b'\r\n\r\n', 1)[1]

    def _recv_exact(self, n):
        out = self.rest[:n]
        self.rest = self.rest[n:]
        while len(out) < n:
            chunk = self.sock.recv(min(1 << 20, n - len(out)))
            if not chunk:
                raise RuntimeError('websocket closed')
            out += chunk
        return out

    def send_text(self, text):
        data = text.encode()
        mask = os.urandom(4)
        hdr = bytes([0x81])
        n = len(data)
        if n < 126:
            hdr += bytes([0x80 | n])
        elif n < 65536:
            hdr += bytes([0x80 | 126]) + struct.pack('>H', n)
        else:
            hdr += bytes([0x80 | 127]) + struct.pack('>Q', n)
        masked = bytes(b ^ mask[i & 3] for i, b in enumerate(data))
        self.sock.sendall(hdr + mask + masked)

    def recv_text(self):
        msg = b''
        while True:
            b0, b1 = self._recv_exact(2)
            fin, op = b0 & 0x80, b0 & 0x0F
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack('>H', self._recv_exact(2))[0]
            elif n == 127:
                n = struct.unpack('>Q', self._recv_exact(8))[0]
            if b1 & 0x80:
                mask = self._recv_exact(4)
                payload = bytes(b ^ mask[i & 3] for i, b in enumerate(self._recv_exact(n)))
            else:
                payload = self._recv_exact(n)
            if op == 0x9:                      # ping -> pong
                self.sock.sendall(bytes([0x8A, 0x80]) + os.urandom(4))
                continue
            if op == 0x8:
                raise RuntimeError('websocket closed by peer')
            if op in (0x1, 0x2, 0x0):
                msg += payload
                if fin:
                    return msg.decode()

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class CDP:
    def __init__(self, ws_url):
        self.ws = WS(ws_url)
        self.n = 0

    def call(self, method, **params):
        self.n += 1
        self.ws.send_text(json.dumps({'id': self.n, 'method': method, 'params': params}))
        while True:
            m = json.loads(self.ws.recv_text())
            if m.get('id') == self.n:
                if 'error' in m:
                    raise RuntimeError(f'{method}: {m["error"]}')
                return m.get('result', {})

    def eval(self, expr):
        r = self.call('Runtime.evaluate', expression=expr, returnByValue=True, awaitPromise=True)
        if r.get('exceptionDetails'):
            raise RuntimeError(f'page error: {r["exceptionDetails"].get("text")} {r.get("result")}')
        return r.get('result', {}).get('value')


# ----------------------------------------------------------------------------- chrome
def free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]
    s.close()
    return p


def start_http(root):
    port = free_port()
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a, **k):
            pass
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', port), partial(Quiet, directory=root))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, port


def start_chrome(chrome, url, profile, extra):
    port = free_port()
    cmd = [chrome, '--headless=new', f'--remote-debugging-port={port}', f'--user-data-dir={profile}',
           '--no-first-run', '--no-default-browser-check', '--hide-scrollbars', '--window-size=1920,1080',
           '--disable-extensions', '--mute-audio', '--ignore-gpu-blocklist', *extra, url]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < 60:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list', timeout=2) as r:
                targets = json.load(r)
            for t in targets:
                if t.get('type') == 'page' and 'globe.html' in t.get('url', ''):
                    return proc, t['webSocketDebuggerUrl']
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError('chrome exited early')
        time.sleep(0.25)
    proc.kill()
    raise RuntimeError('chrome devtools endpoint did not appear')


# ----------------------------------------------------------------------------- encode / sheet
def encode(frames_dir, fps, out_mp4):
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', os.path.join(frames_dir, '%04d.png'),
           '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out_mp4]
    subprocess.run(cmd, check=True)


def duration(mp4):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', mp4],
                       capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


def make_sheet(out_dir, clips, nframes, fps, cols=(0, 60, 120, -1)):
    from PIL import Image, ImageDraw, ImageFont
    checks = os.path.join(out_dir, 'checks')
    tw, th, pad, cap = 464, 261, 8, 26
    sheet = Image.new('RGB', (pad + len(cols) * (tw + pad), pad + len(clips) * (th + cap + pad)), (11, 15, 10))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 18)
    except Exception:
        font = ImageFont.load_default()
    lines = ['# globe clips', '', f'{nframes} frames @ {fps} fps, 1920x1080, libx264 crf 18 yuv420p', '',
             '| clip | stop | mode | duration s |', '|---|---|---|---|']
    for r, clip in enumerate(clips):
        y = pad + r * (th + cap + pad)
        mp4 = os.path.join(out_dir, clip + '.mp4')
        dur = duration(mp4) if os.path.exists(mp4) else 0
        stop, mode = CLIPS[clip]
        lines.append(f'| {clip}.mp4 | {stop} | {mode} | {dur:.2f} |')
        draw.text((pad, y + 4), f'{clip}.mp4  ({stop}, {mode}, {dur:.2f} s)', fill=(220, 224, 216), font=font)
        for c, fi in enumerate(cols):
            idx = nframes - 1 if fi < 0 else fi
            p = os.path.join(checks, f'{clip}_f{idx:03d}.png')
            if os.path.exists(p):
                im = Image.open(p).convert('RGB').resize((tw, th), Image.LANCZOS)
                sheet.paste(im, (pad + c * (tw + pad), y + cap))
                draw.text((pad + c * (tw + pad) + 6, y + cap + th - 24), f'f{idx:03d}', fill=(233, 150, 55), font=font)
    sheet.save(os.path.join(out_dir, 'sheet.png'))
    with open(os.path.join(out_dir, 'sheet.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=OUT_DIR)
    ap.add_argument('--clips', nargs='*', default=list(CLIPS), choices=list(CLIPS))
    ap.add_argument('--frames', type=int, default=180)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--chrome', default=CHROME)
    ap.add_argument('--keep-frames', action='store_true', help='keep PNG frames under <out>/frames/<clip>/')
    ap.add_argument('--check-frames', default='0,60,90,120,179', help='frame indices copied to <out>/checks/')
    ap.add_argument('--chrome-flag', action='append', default=[], help='extra Chrome flag (repeatable)')
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, 'checks'), exist_ok=True)
    check_idx = {min(int(x), args.frames - 1) for x in args.check_frames.split(',') if x.strip()}
    srv, http_port = start_http(HERE)
    profile = tempfile.mkdtemp(prefix='globe_chrome_')
    url = f'http://127.0.0.1:{http_port}/globe.html?stop=earth_idle&frame=0'
    t0 = time.time()
    proc, ws_url = start_chrome(args.chrome, url, profile, args.chrome_flag)
    cdp = CDP(ws_url)
    while not cdp.eval('!!window.__ready'):
        err = cdp.eval('window.__error || ""')
        if err:
            raise RuntimeError('page error: ' + err)
        if time.time() - t0 > 180:
            raise RuntimeError('page never became ready')
        time.sleep(0.25)
    info = cdp.eval('JSON.stringify(window.clipInfo())')
    print(f'chrome ready in {time.time() - t0:.1f}s: {info}', flush=True)

    try:
        for clip in args.clips:
            stop, mode = CLIPS[clip]
            cdp.eval(f'window.setClip({json.dumps(stop)}, {json.dumps(mode)}); true')
            frames_dir = os.path.join(args.out, 'frames', clip) if args.keep_frames else tempfile.mkdtemp(prefix=f'globe_{clip}_')
            os.makedirs(frames_dir, exist_ok=True)
            t1 = time.time()
            for i in range(args.frames):
                data_url = cdp.eval(f'window.renderFrame({i})')
                png = base64.b64decode(data_url.split(',', 1)[1])
                path = os.path.join(frames_dir, f'{i:04d}.png')
                with open(path, 'wb') as f:
                    f.write(png)
                if i in check_idx:
                    shutil.copyfile(path, os.path.join(args.out, 'checks', f'{clip}_f{i:03d}.png'))
            out_mp4 = os.path.join(args.out, clip + '.mp4')
            encode(frames_dir, args.fps, out_mp4)
            print(f'{clip}: {args.frames} frames in {time.time() - t1:.1f}s -> {out_mp4} ({duration(out_mp4):.2f}s)', flush=True)
            if not args.keep_frames:
                shutil.rmtree(frames_dir, ignore_errors=True)
    finally:
        proc.kill()
        srv.shutdown()
        shutil.rmtree(profile, ignore_errors=True)

    make_sheet(args.out, args.clips, args.frames, args.fps)
    print('sheet:', os.path.join(args.out, 'sheet.png'))


if __name__ == '__main__':
    main()
