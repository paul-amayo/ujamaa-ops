"""Real-browser check: open Tassili with the hierarchy backend selected (?stream_url=ws://127.0.0.1:8006/ws),
seek the walkthrough to a keyframe inside a finished chunk, capture the streamed frame the browser shows and
the stream stats, then press play for a few seconds and capture again."""
import base64, json, sys, time, urllib.request
from playwright.sync_api import sync_playwright
target = sys.argv[1] if len(sys.argv) > 1 else "kf_001411.png"
url = "http://127.0.0.1:8001/tassili/?stream_url=ws://127.0.0.1:8006/ws"
traj = json.load(urllib.request.urlopen("http://127.0.0.1:8001/scene/trajectory?stride=1"))["frames"]
k = next(i for i, f in enumerate(traj) if f["image_name"] == target)
grab = """()=>{const im=[...document.images].find(i=>i.src.startsWith('blob:'));if(!im||!im.naturalWidth)return null;const c=document.createElement('canvas');c.width=im.naturalWidth;c.height=im.naturalHeight;c.getContext('2d').drawImage(im,0,0);return {w:im.naturalWidth,h:im.naturalHeight,url:window.state.stream.url,stats:window.state.stream.stats,block:window.state.stream.replayBlock,d:c.toDataURL('image/jpeg',0.85).split(',')[1]};}"""
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1280, "height": 720})
    logs = []; pg.on("console", lambda m: logs.append(m.text))
    pg.goto(url); pg.wait_for_function("()=>window.state && window.state.engine && window.state.stream && window.state.stream.ws", timeout=60000)
    time.sleep(2)
    s = pg.evaluate(f"()=>{{const e=window.state.engine; return (e.frameS && e.frameS[{k}]!=null) ? e.frameS[{k}] : e.totalLength*{k}/{len(traj)};}}")
    pg.evaluate(f"()=>{{const st=window.state; st.engine.pause(); st.engine.seekS({s}); const f=st.engine.recordedFrameAt({s}); st.stream.replayC2W=f.matrix; st.stream.replayBlock=f.block; st.stream.replayPrefetch=null;}}")
    time.sleep(4)
    r = pg.evaluate(grab)
    if r:
        open("/home/paperspace/logs/hier_browser_rest.jpg", "wb").write(base64.b64decode(r["d"]))
        print(f"[browser] at rest near {target}: stream url {r['url']}, frame {r['w']}x{r['h']}, stats {json.dumps(r['stats'])}, replayBlock {r['block']}")
    else:
        print("[browser] no streamed frame at rest")
    s_before = pg.evaluate("()=>window.state.engine.s")
    pg.evaluate("()=>{window.state.engine.setPace && window.state.engine.setPace('walk'); window.state.engine.play();}")
    frames = 0; t0 = time.time(); last = None
    while time.time() - t0 < 8:
        r2 = pg.evaluate(grab)
        if r2 and r2["d"] != last: frames += 1; last = r2["d"]
        time.sleep(0.1)
    if r2: open("/home/paperspace/logs/hier_browser_play.jpg", "wb").write(base64.b64decode(r2["d"]))
    s_after = pg.evaluate("()=>window.state.engine.s")
    print(f"[browser] playing 8 s: {frames} distinct frames, engine s {s_before:.1f} -> {s_after:.1f} m, replayBlock {r2['block'] if r2 else None}, stats {json.dumps(r2['stats']) if r2 else None}")
    pg.screenshot(path="/home/paperspace/logs/hier_browser_page.png")
    errs = [l for l in logs if "error" in l.lower()][:5]; print("[browser] console errors:", errs)
    b.close()
