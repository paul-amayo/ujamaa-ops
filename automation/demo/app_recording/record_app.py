#!/usr/bin/env python3
"""Screen-record the UJAMAA live app (http://localhost:8011/app/) for the Google.org video.

Headless Chrome (--headless=new, 1920x1080) driven over the DevTools protocol with a
stdlib websocket client (pattern from ../globe/render_globe.py). Frames come from
Page.startScreencast (timestamped) topped up by Page.captureScreenshot when the
screencast goes quiet, so a static page still yields frames. Encoding is real time:
each frame is held until the next frame's timestamp, then resampled to 30 fps CFR,
libx264 crf 18 yuv420p. No overlays, no music, no captions.

Needs the A100 tunnel (8011 app/API, 8024 render stream):
    ssh -N -f -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \\
        -L 8011:localhost:8011 -L 8024:localhost:8024 paperspace@184.105.3.25

    python3 record_app.py                          # all shots
    python3 record_app.py --shots picker cabbage   # some shots
    python3 record_app.py --explore '#/sankofa?survey=klapmuts-dec25'   # screenshot + DOM text

The stage is ONE shared renderer: shots that need 3D POST /api/stage and wait for
"ready". The survey staged at start is re-staged at the end.
"""
import argparse
import base64
import json
import os
import queue
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
import urllib.request

APP = 'http://localhost:8011'
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
OUT_DIR = '/Users/paulamayo/data/for_a100/google_org_2026-11/04_app'
W, H, FPS = 1920, 1080, 30


# ----------------------------------------------------------------------------- websocket
class WS:
    """Minimal RFC6455 client: text frames, masking, 64-bit lengths, continuation, ping."""

    def __init__(self, url):
        hostport, _, path = url[5:].partition('/')
        host, _, port = hostport.partition(':')
        self.sock = socket.create_connection((host, int(port or 80)))
        self.sock.settimeout(None)
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
            raise RuntimeError('websocket upgrade refused')
        self.rest = buf.split(b'\r\n\r\n', 1)[1]
        self.wlock = threading.Lock()

    def _recv_exact(self, n):
        out = self.rest[:n]
        self.rest = self.rest[n:]
        parts = [out]
        got = len(out)
        while got < n:
            chunk = self.sock.recv(min(1 << 20, n - got))
            if not chunk:
                raise RuntimeError('websocket closed')
            parts.append(chunk)
            got += len(chunk)
        return b''.join(parts)

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
        m4 = mask * (n // 4 + 1)
        masked = bytes(a ^ b for a, b in zip(data, m4))
        with self.wlock:
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
            payload = self._recv_exact(n)
            if op == 0x9:
                with self.wlock:
                    self.sock.sendall(bytes([0x8A, 0x80]) + os.urandom(4))
                continue
            if op == 0x8:
                raise RuntimeError('websocket closed by peer')
            if op in (0x1, 0x2, 0x0):
                msg += payload
                if fin:
                    return msg.decode()


class CDP:
    """CDP session with a reader thread: responses by id, events to handlers."""

    def __init__(self, ws_url):
        self.ws = WS(ws_url)
        self.n = 0
        self.lock = threading.Lock()
        self.pending = {}
        self.handlers = {}
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while True:
            try:
                m = json.loads(self.ws.recv_text())
            except Exception:
                for q in self.pending.values():
                    q.put({'error': 'closed'})
                return
            if 'id' in m:
                q = self.pending.pop(m['id'], None)
                if q:
                    q.put(m)
            else:
                h = self.handlers.get(m.get('method'))
                if h:
                    try:
                        h(m.get('params', {}))
                    except Exception as e:
                        print('handler error', e)

    def call(self, method, timeout=120, **params):
        with self.lock:
            self.n += 1
            i = self.n
        q = queue.Queue()
        self.pending[i] = q
        self.ws.send_text(json.dumps({'id': i, 'method': method, 'params': params}))
        m = q.get(timeout=timeout)
        if 'error' in m:
            raise RuntimeError(f'{method}: {m["error"]}')
        return m.get('result', {})

    def send_nowait(self, method, **params):
        with self.lock:
            self.n += 1
            i = self.n
        self.ws.send_text(json.dumps({'id': i, 'method': method, 'params': params}))

    def eval(self, expr):
        r = self.call('Runtime.evaluate', expression=expr, returnByValue=True, awaitPromise=True)
        if r.get('exceptionDetails'):
            raise RuntimeError(f'page error: {r["exceptionDetails"]}')
        return r.get('result', {}).get('value')


def free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]
    s.close()
    return p


# ----------------------------------------------------------------------------- browser
class Browser:
    def __init__(self, chrome=CHROME):
        self.profile = tempfile.mkdtemp(prefix='ujamaa_rec_')
        port = free_port()
        cmd = [chrome, '--headless=new', f'--remote-debugging-port={port}', f'--user-data-dir={self.profile}',
               '--no-first-run', '--no-default-browser-check', '--hide-scrollbars', f'--window-size={W},{H}',
               '--disable-extensions', '--mute-audio', '--ignore-gpu-blocklist', '--enable-gpu',
               '--disable-background-timer-throttling', '--disable-renderer-backgrounding',
               '--disable-backgrounding-occluded-windows', '--force-device-scale-factor=1', 'about:blank']
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.time()
        ws = None
        while time.time() - t0 < 60 and not ws:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list', timeout=2) as r:
                    for t in json.load(r):
                        if t.get('type') == 'page':
                            ws = t['webSocketDebuggerUrl']
            except Exception:
                time.sleep(0.25)
        if not ws:
            raise RuntimeError('chrome devtools endpoint did not appear')
        self.cdp = CDP(ws)
        c = self.cdp
        c.call('Page.enable')
        c.call('Runtime.enable')
        c.call('Emulation.setDeviceMetricsOverride', width=W, height=H, deviceScaleFactor=1, mobile=False)
        c.call('Emulation.setFocusEmulationEnabled', enabled=True)
        c.call('Page.bringToFront')
        self.rec = None

    def close(self):
        try:
            self.proc.terminate()
            self.proc.wait(10)
        except Exception:
            self.proc.kill()
        shutil.rmtree(self.profile, ignore_errors=True)

    # -- navigation / queries
    def goto(self, url, settle=2.0):
        self.cdp.call('Page.navigate', url=url)
        self.wait_js('document.readyState === "complete"', 60)
        time.sleep(settle)

    def reload(self, settle=2.0):
        self.cdp.call('Page.reload', ignoreCache=True)
        time.sleep(0.5)
        self.wait_js('document.readyState === "complete"', 60)
        time.sleep(settle)

    def js(self, expr):
        return self.cdp.eval(expr)

    def wait_js(self, expr, timeout=60, poll=0.25, what=None):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                v = self.js(expr)
                if v:
                    return v
            except RuntimeError:
                pass
            time.sleep(poll)
        raise TimeoutError(f'timed out waiting for {what or expr}')

    def rect(self, js_el):
        """Centre of the element returned by js_el (an expression), scrolled into view."""
        r = self.js(f'''(() => {{ const e = {js_el}; if (!e) return null;
            e.scrollIntoView({{block: 'nearest', inline: 'nearest'}});
            const b = e.getBoundingClientRect(); return [b.x + b.width / 2, b.y + b.height / 2]; }})()''')
        if not r:
            raise RuntimeError(f'element not found: {js_el}')
        return r

    def move_mouse(self, x, y, steps=12, dur=0.4):
        x0, y0 = getattr(self, '_mouse', (W / 2, H / 2))
        for i in range(1, steps + 1):
            t = i / steps
            t = t * t * (3 - 2 * t)
            self.cdp.call('Input.dispatchMouseEvent', type='mouseMoved', x=x0 + (x - x0) * t, y=y0 + (y - y0) * t)
            time.sleep(dur / steps)
        self._mouse = (x, y)

    def click(self, js_el, move=True):
        x, y = self.rect(js_el)
        if move:
            self.move_mouse(x, y)
        for t in ('mousePressed', 'mouseReleased'):
            self.cdp.call('Input.dispatchMouseEvent', type=t, x=x, y=y, button='left', clickCount=1)
        self._mouse = (x, y)

    def type(self, text, cps=14):
        for ch in text:
            self.cdp.call('Input.insertText', text=ch)
            time.sleep(1.0 / cps)

    def wheel(self, x, y, dy):
        self.cdp.call('Input.dispatchMouseEvent', type='mouseWheel', x=x, y=y, deltaX=0, deltaY=dy)

    def screenshot(self, path):
        d = self.cdp.call('Page.captureScreenshot', format='png')['data']
        with open(path, 'wb') as f:
            f.write(base64.b64decode(d))

    # -- recording
    def start_recording(self, frames_dir):
        os.makedirs(frames_dir, exist_ok=True)
        rec = {'dir': frames_dir, 'frames': [], 'last': time.time(), 'on': True, 'lock': threading.Lock()}
        self.rec = rec

        def save(data, ts, src):
            with rec['lock']:
                i = len(rec['frames'])
                p = os.path.join(frames_dir, f'{i:06d}.jpg')
                rec['frames'].append((ts, p, src))
                rec['last'] = time.time()
            with open(p, 'wb') as f:
                f.write(base64.b64decode(data))

        def on_frame(params):
            self.cdp.send_nowait('Page.screencastFrameAck', sessionId=params['sessionId'])
            if rec['on']:
                save(params['data'], time.time(), 'sc')

        self.cdp.handlers['Page.screencastFrame'] = on_frame
        self.cdp.call('Page.startScreencast', format='jpeg', quality=95, maxWidth=W, maxHeight=H, everyNthFrame=1)

        def topup():   # static page -> no screencast frames; poll a screenshot when quiet
            while rec['on']:
                time.sleep(0.05)
                if time.time() - rec['last'] > 0.1:
                    try:
                        d = self.cdp.call('Page.captureScreenshot', format='jpeg', quality=95, timeout=10)['data']
                        if rec['on']:
                            save(d, time.time(), 'shot')
                    except Exception:
                        pass
        rec['thread'] = threading.Thread(target=topup, daemon=True)
        rec['thread'].start()
        rec['t0'] = time.time()
        return rec

    def stop_recording(self):
        rec = self.rec
        rec['on'] = False
        rec['t1'] = time.time()
        rec['thread'].join(5)
        try:
            self.cdp.call('Page.stopScreencast')
        except Exception:
            pass
        self.cdp.handlers.pop('Page.screencastFrame', None)
        self.rec = None
        return rec


def encode(rec, out_mp4, fps=FPS, max_s=60.0):
    """Hold each frame until the next one's wall-clock time; resample to CFR fps.
    Longer than max_s -> the head is trimmed so the clip ends on the held result."""
    frames = sorted(rec['frames'])
    frames = [f for f in frames if os.path.getsize(f[1]) > 0]
    lst = os.path.join(rec['dir'], 'list.ffconcat')
    with open(lst, 'w') as f:
        f.write('ffconcat version 1.0\n')
        for k, (ts, p, _) in enumerate(frames):
            nxt = frames[k + 1][0] if k + 1 < len(frames) else rec['t1']
            f.write(f"file '{p}'\nduration {max(nxt - ts, 0.001):.4f}\n")
        f.write(f"file '{frames[-1][1]}'\n")
    total = rec['t1'] - frames[0][0]
    trim = ['-ss', f'{total - max_s:.3f}'] if max_s and total > max_s else []
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, *trim,
                    '-vf', f'fps={fps},scale={W}:{H}:flags=lanczos:in_range=full:out_range=tv,format=yuv420p', '-c:v', 'libx264', '-crf', '18',
                    '-pix_fmt', 'yuv420p', '-color_range', 'tv',
                    '-preset', 'slow', '-r', str(fps), '-movflags', '+faststart', out_mp4], check=True)
    sc = sum(1 for f in frames if f[2] == 'sc')
    span = frames[-1][0] - frames[0][0] if len(frames) > 1 else 0
    return {'frames': len(frames), 'screencast': sc, 'mean_fps': len(frames) / max(span, 1e-6)}


# ----------------------------------------------------------------------------- stage
def api(path, body=None, timeout=30):
    req = urllib.request.Request(APP + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={'Content-Type': 'application/json'}, method='POST' if body is not None else 'GET')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def stage(survey, timeout=600):
    st = api('/api/stage')
    if st.get('survey') != survey or st.get('state') != 'ready':
        api('/api/stage', {'survey': survey})
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = api('/api/stage')
        if st.get('survey') == survey and st.get('state') == 'ready':
            print(f'  stage {survey} ready ({time.time() - t0:.0f} s)')
            return st
        time.sleep(2)
    raise TimeoutError(f'stage {survey} not ready: {st}')


# ----------------------------------------------------------------------------- page helpers
def by_text(sel, text):
    return f'[...document.querySelectorAll({json.dumps(sel)})].find(e => e.offsetParent !== null && e.textContent.trim().startsWith({json.dumps(text)}))'


CHAT_IN = 'document.querySelector(\'input[placeholder^="Ask the twin"]\')'
SEND = by_text('button.send', 'Send')
TWIN_COUNT = 'document.querySelectorAll(".adx-m.twin .adx-meta").length'


def hud_fps(b):
    return b.js('''(() => { const e = [...document.querySelectorAll('.tsl-ctl')].find(x => /Live/.test(x.textContent));
                    return e ? e.textContent : null; })()''')


def wait_live(b, timeout=180):
    b.wait_js('''[...document.querySelectorAll('.tsl-ctl')].some(x => /Live · [1-9]/.test(x.textContent))''',
              timeout, what='HUD "Live · N fps"')
    print('  ', hud_fps(b))


def ask(b, text, timeout=150, pre=0.6):
    n0 = b.js(TWIN_COUNT)
    b.click(CHAT_IN)
    time.sleep(pre)
    b.type(text)
    time.sleep(0.8)
    b.click(SEND)
    t0 = time.time()
    b.wait_js(f'{TWIN_COUNT} > {n0}', timeout, poll=0.5, what='Adinkra answer')
    ans = b.js('''(() => { const m = [...document.querySelectorAll('.adx-m.twin')].pop(); return m ? m.innerText : null; })()''')
    print(f'   answer after {time.time() - t0:.1f} s: {ans!r}')
    return ans


def smooth_scroll(b, el_js, total, secs, x=None, y=None):
    """Scroll the element (or window when el_js is None) by total px over secs, eased."""
    n = int(secs * 30)
    target = el_js or 'document.scrollingElement'
    start = b.js(f'{target}.scrollTop')
    for i in range(1, n + 1):
        t = i / n
        e = t * t * (3 - 2 * t)
        b.js(f'{target}.scrollTop = {start + total * e:.1f}')
        time.sleep(secs / n)


def scroll_container(b):
    """The element that actually scrolls on the picker page."""
    return b.js('''(() => { const c = [document.scrollingElement, ...document.querySelectorAll('*')]
        .filter(e => e && e.scrollHeight > e.clientHeight + 40 && ['auto','scroll'].includes(getComputedStyle(e).overflowY) || e === document.scrollingElement)
        .sort((a, b) => (b.scrollHeight - b.clientHeight) - (a.scrollHeight - a.clientHeight))[0];
        if (!c) return null; c.setAttribute('data-rec-scroll', '1'); return [c.tagName, c.className, c.scrollHeight, c.clientHeight]; })()''')


# ----------------------------------------------------------------------------- shots
def shot_picker(b, out):
    b.goto(f'{APP}/app/#/', settle=1)
    b.reload(settle=4)
    b.wait_js('document.querySelectorAll(".app-farm").length > 0', 60, what='farm cards')
    info = scroll_container(b)
    print('   scroll container', info)
    el = 'document.querySelector("[data-rec-scroll]")'
    maxs = b.js(f'{el}.scrollHeight - {el}.clientHeight')
    b.js(f'{el}.scrollTop = 0')
    rec = b.start_recording(out['frames'])
    time.sleep(4)
    # step through the farm cards left to right (local selection only; the shared stage is untouched)
    for farm in ('Citrus farm A', 'Citrus farm B', 'Klapmuts', 'Kendu Bay', 'Gwakungu'):
        try:
            b.click(f'''[...document.querySelectorAll('.app-farm')].find(e => e.textContent.trim().startsWith({json.dumps(farm)}))?.querySelector('.app-farm-sv')''')
        except Exception as e:
            print('   card', farm, e)
        time.sleep(3.5)
    smooth_scroll(b, el, maxs, 6)
    time.sleep(3)
    smooth_scroll(b, el, -maxs, 4)
    time.sleep(2)
    return b.stop_recording()


def open_tassili(b, survey):
    stage(survey)
    b.goto(f'{APP}/app/#/tassili?survey={survey}', settle=1)
    b.reload(settle=3)
    wait_live(b)
    time.sleep(4)


def press_play(b):
    b.click('''[...document.querySelectorAll('span.b')].find(e => e.offsetParent && /▶|❚❚/.test(e.textContent))''')


args_pause_s = 3.2   # seconds after the show_fruit answer before pausing the walk


def shot_citrus_b(b, out):
    open_tassili(b, 'citrus-b-05')
    rec = b.start_recording(out['frames'])
    time.sleep(2)
    press_play(b)
    time.sleep(8)
    out['answer'] = ask(b, 'Ni mti gani wenye machungwa mengi zaidi? Nionyeshe.')
    if 'SHOW_FRUIT' not in (out['answer'] or '').upper():
        b.stop_recording()
        raise RuntimeError(f"Adinkra did not run show_fruit: {out['answer']!r}")
    # show_fruit walks in from ~4 m before tree 5's best view (walk pace ~1.3 m/s); pause there and hold
    time.sleep(args_pause_s)
    print('   at pause:', b.js('''[...document.querySelectorAll('.app-view:not([hidden]) *')]
        .map(e => e.childElementCount ? '' : e.textContent).find(t => /^near /i.test(t)) || null'''))
    press_play(b)   # the play/pause toggle -> pause
    time.sleep(8.5)
    out['hud'] = hud_fps(b)
    return b.stop_recording()


def shot_cabbage(b, out):
    open_tassili(b, 'gwakungu-cabbage')
    rec = b.start_recording(out['frames'])
    time.sleep(3)
    a1 = ask(b, 'How many cabbages are there?')
    time.sleep(4)
    a2 = ask(b, 'Which cabbage is in front of me? Show me.')
    time.sleep(10)
    out['answer'] = [a1, a2]
    out['hud'] = hud_fps(b)
    return b.stop_recording()


def open_page(b, route, survey, marker):
    stage(survey)
    b.goto(f'{APP}/app/#/{route}?survey={survey}', settle=1)
    b.reload(settle=4)
    b.wait_js(marker, 60, what=marker)
    time.sleep(3)


def seg(label):
    return f'''[...document.querySelectorAll('.ln-seg span')].find(e => e.offsetParent && e.textContent.trim() === {json.dumps(label)})'''


def search(b, text):
    b.click('document.querySelector(".ln-side input.app-find")')
    time.sleep(0.4)
    b.type(text, cps=10)
    time.sleep(0.5)
    b.cdp.call('Input.dispatchKeyEvent', type='keyDown', key='Enter', code='Enter', windowsVirtualKeyCode=13, text='\r')
    b.cdp.call('Input.dispatchKeyEvent', type='keyUp', key='Enter', code='Enter', windowsVirtualKeyCode=13)


def shot_bateleur(b, out):
    open_page(b, 'bateleur', 'klapmuts-dec25', 'document.querySelectorAll(".btx svg circle").length > 100')
    rec = b.start_recording(out['frames'])
    time.sleep(4)
    b.click(seg('Row'))                      # colour the registry by row
    time.sleep(5)
    b.click('''[...document.querySelectorAll('.rowlist > div')].find(e => e.textContent.trim().startsWith('Row 7'))''')
    time.sleep(6)                            # first bush of row 7 selected -> plant record
    search(b, 'bush 400')
    time.sleep(7)
    out['answer'] = b.js('document.querySelector(".ln-right").innerText')
    return b.stop_recording()


def shot_sankofa(b, out, survey='klapmuts-dec25', q='Hoeveel bosse het ons weer gevind?'):
    open_page(b, 'sankofa', survey, 'document.body.innerText.includes("SANKOFA") || document.body.innerText.includes("Sankofa")')
    rec = b.start_recording(out['frames'])
    time.sleep(5)
    out['answer'] = ask(b, q)
    time.sleep(9)
    return b.stop_recording()


def shot_sankofa_citrus_a(b, out):
    """Extra: Sankofa with real history (Klapmuts history is empty on the live API)."""
    return shot_sankofa(b, out, 'citrus-a-01', 'Watter bome het die meeste verander?')


SHOTS = {
    'picker': ('01_picker', shot_picker),
    'citrus_b': ('02_citrus_b_oranges', shot_citrus_b),
    'cabbage': ('03_cabbage_query', shot_cabbage),
    'bateleur': ('04_bateleur_map', shot_bateleur),
    'sankofa': ('05_sankofa_history', shot_sankofa),
    'sankofa_citrus_a': ('05b_sankofa_history_citrus_a', shot_sankofa_citrus_a),
}


def check_frames(mp4, out_png_prefix, n=4):
    d = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', mp4],
                             capture_output=True, text=True).stdout.strip() or 0)
    pngs = []
    for k in range(n):
        t = d * (k + 0.5) / n if k < n - 1 else max(d - 0.5, 0)
        p = f'{out_png_prefix}_{k}_{t:05.1f}s.png'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{t:.2f}', '-i', mp4, '-frames:v', '1', p], check=True)
        pngs.append(p)
    return d, pngs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=OUT_DIR)
    ap.add_argument('--shots', nargs='*', default=list(SHOTS), choices=list(SHOTS))
    ap.add_argument('--chrome', default=CHROME)
    ap.add_argument('--keep-frames', action='store_true')
    ap.add_argument('--restore', default=None, help='survey to stage at the end (default: whatever was staged at start)')
    ap.add_argument('--explore', default=None, help='hash route to screenshot + dump text, e.g. "#/tassili?survey=x"')
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, 'checks'), exist_ok=True)
    found = api('/api/stage').get('survey')
    restore = args.restore or found
    print('stage at start:', found)

    b = Browser(args.chrome)
    try:
        if args.explore:
            b.goto(f'{APP}/app/{args.explore}', settle=1)
            b.reload(settle=8)
            p = os.path.join(args.out, 'checks', 'explore.png')
            b.screenshot(p)
            print(b.js('document.body.innerText')[:4000])
            print('screenshot', p)
            return
        report = []
        for key in args.shots:
            name, fn = SHOTS[key]
            print(f'== {name}')
            out = {'frames': os.path.join(args.out, 'frames', name)}
            shutil.rmtree(out['frames'], ignore_errors=True)
            try:
                rec = fn(b, out)
            except Exception as e:
                if b.rec:
                    b.stop_recording()
                print(f'   FAILED: {e}')
                b.screenshot(os.path.join(args.out, 'checks', f'{name}_fail.png'))
                report.append({'shot': name, 'error': str(e)})
                continue
            mp4 = os.path.join(args.out, name + '.mp4')
            stats = encode(rec, mp4)
            d, pngs = check_frames(mp4, os.path.join(args.out, 'checks', name))
            print(f'   {mp4}: {d:.1f} s, {stats}')
            report.append({'shot': name, 'mp4': mp4, 'duration_s': round(d, 2), 'answer': out.get('answer'),
                           'hud': out.get('hud'), **stats, 'checks': pngs})
            if not args.keep_frames:
                shutil.rmtree(out['frames'], ignore_errors=True)
        with open(os.path.join(args.out, 'record_report.json'), 'a') as f:
            f.write(json.dumps({'when': time.strftime('%Y-%m-%d %H:%M:%S'), 'shots': report}, ensure_ascii=False) + '\n')
    finally:
        b.close()
        if restore and api('/api/stage').get('survey') != restore:
            print('restoring stage', restore)
            stage(restore)


if __name__ == '__main__':
    main()
