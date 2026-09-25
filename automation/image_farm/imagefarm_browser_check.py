"""Real-browser check of the Tassili viewer on the :8004 checkpoint backend: load /tassili/ with the stream_url override,
wait for frames, screenshot as loaded, press Play, screenshot again, report the stream/backend state and console errors."""
from playwright.sync_api import sync_playwright
import json, time, sys
OUT = "/home/paperspace/logs"; TAG = sys.argv[1] if len(sys.argv) > 1 else "imagefarm"; console = []
with sync_playwright() as pw:
    b = pw.chromium.launch(args=["--use-gl=angle", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1440, "height": 810})
    pg.on("console", lambda m: console.append(f"[{m.type}] {m.text[:160]}"))
    pg.on("pageerror", lambda e: console.append(f"[pageerror] {str(e)[:160]}"))
    pg.goto("http://127.0.0.1:8001/tassili/?stream_url=ws://127.0.0.1:8004/ws", timeout=30000)
    time.sleep(25)
    st = pg.evaluate("""() => {
        const s = window.state; if (!s) return {err:'no state'};
        const e = s.engine, st = s.stream, stats = s.stats || {};
        return {backend: stats.backend || (st && st.url) || null, frames: stats.frames || null, fps: stats.fps || null,
                totalLength: e ? +e.totalLength.toFixed(1) : null, playing: e ? e.playing : null,
                cam_pos: s.camera ? s.camera.position.toArray().map(x=>+x.toFixed(2)) : null};
    }""")
    print("[browser] as-loaded:", json.dumps(st)); pg.screenshot(path=f"{OUT}/{TAG}_0_loaded.png")
    pg.evaluate("""() => { const b=[...document.querySelectorAll('button')].find(x=>/play|▶/i.test(x.textContent)); if(b)b.click(); }""")
    time.sleep(12); pg.screenshot(path=f"{OUT}/{TAG}_1_playing.png")
    st2 = pg.evaluate("""() => { const s = window.state; const stats = s.stats || {}; return {playing: s.engine && s.engine.playing, frames: stats.frames || null, fps: stats.fps || null, backend: stats.backend || null}; }""")
    print("[browser] playing:", json.dumps(st2))
    b.close()
errs = [c for c in console if 'error' in c.lower() or 'pageerror' in c]
print("[browser] console errors:", errs[:6] if errs else "none")
