"""Count how many distinct recorded keyframes the walkthrough replays during a Play, through the real browser.
  python imagefarm_replay_check.py "<url query string>" [seconds]"""
from playwright.sync_api import sync_playwright
import json, time, sys
Q = sys.argv[1] if len(sys.argv) > 1 else "stream_url=ws://127.0.0.1:8004/ws"; SECS = float(sys.argv[2]) if len(sys.argv) > 2 else 25
with sync_playwright() as pw:
    b = pw.chromium.launch(args=["--use-gl=angle", "--enable-unsafe-swiftshader"]); pg = b.new_page(viewport={"width": 1440, "height": 810})
    pg.goto(f"http://127.0.0.1:8001/tassili/?{Q}", timeout=30000); time.sleep(20)
    info = pg.evaluate("() => { const e = window.state.engine; return {frames_loaded: e.frames ? e.frames.length : null, totalLength: +e.totalLength.toFixed(2), speedScale: e.speedScale, pace: e.paceKey}; }")
    print("[replay] loaded:", json.dumps(info))
    pg.evaluate("""() => { const b=[...document.querySelectorAll('button')].find(x=>/play|▶/i.test(x.textContent)); if(b)b.click(); }""")
    seen = []; t0 = time.time()
    while time.time() - t0 < SECS:
        r = pg.evaluate("() => { const e = window.state.engine; const f = e.recordedFrameAt ? e.recordedFrameAt(e.s) : null; return {s: +e.s.toFixed(2), playing: e.playing, name: f ? f.image_name : null}; }")
        if r["name"] and (not seen or seen[-1] != r["name"]): seen.append(r["name"])
        if not r["playing"] and time.time() - t0 > 3: break
        time.sleep(0.25)
    print(f"[replay] {len(seen)} distinct keyframes replayed in {time.time()-t0:.0f}s (s reached {r['s']} of {info['totalLength']}, playing={r['playing']}); first {seen[:3]} last {seen[-3:]}")
    b.close()
