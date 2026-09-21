"""Verify the owner-hint path end to end in a real browser: (1) trajectory frames carry
`block`; (2) the stream's outgoing pose message includes it; (3) mid-motion frames are
clean (the service renders the owning checkpoint). Saves rest + mid-motion frames."""
from playwright.sync_api import sync_playwright
import time, base64, json
errs = []
with sync_playwright() as pw:
    b = pw.chromium.launch(args=["--use-gl=angle", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1280, "height": 720})
    pg.on("pageerror", lambda e: errs.append(str(e)[:140]))
    pg.goto("http://127.0.0.1:8001/tassili/", timeout=30000); time.sleep(22)
    st = pg.evaluate("""() => { const e = window.state.engine, s = window.state.stream;
        const rf = e.recordedFrameAt(e.s);
        return { frames_with_block: e.frames.filter(f => f.block != null).length, n: e.frames.length,
                 rest_block: rf && rf.block, rest_name: rf && rf.image_name,
                 stream_block: s.replayBlock, stream_has_pose: Array.isArray(s.replayC2W) }; }""")
    print("[own] rest:", json.dumps(st))
    def grab(name):
        d = pg.evaluate("""() => { const im=[...document.images].find(i=>i.src.startsWith('blob:')); if(!im) return null;
            const c=document.createElement('canvas'); c.width=im.naturalWidth; c.height=im.naturalHeight;
            c.getContext('2d').drawImage(im,0,0); return c.toDataURL('image/jpeg',0.92).split(',')[1]; }""")
        if d and len(d) > 20000: open(f"/home/paperspace/logs/{name}.jpg", "wb").write(base64.b64decode(d)); return True
        return False
    grab("own_rest")
    pg.evaluate("() => { window.state.engine.setPace('walk'); window.state.engine.play(); }")
    got = []
    for i in range(12):
        time.sleep(1.0)
        blk = pg.evaluate("() => window.state.stream.replayBlock")
        if grab(f"own_motion_{i}"): got.append((i, blk))
    b.close()
print("[own] mid-motion frames captured (i, block):", got)
print("[own] errors:", errs[:4] if errs else "none")
