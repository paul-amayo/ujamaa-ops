"""Real-browser check of a nerfstudio ns-viewer (viser) page: load it, wait, screenshot, nudge the camera with a drag,
screenshot again, report console errors and whether the background render image is present/non-black.
  python nsviewer_browser_check.py [port] [tag]"""
from playwright.sync_api import sync_playwright
import json, sys, time
PORT = sys.argv[1] if len(sys.argv) > 1 else "7007"; TAG = sys.argv[2] if len(sys.argv) > 2 else "nsviewer"; OUT = "/home/paperspace/logs"; console = []
with sync_playwright() as pw:
    b = pw.chromium.launch(args=["--use-gl=angle", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1440, "height": 810})
    pg.on("console", lambda m: console.append(f"[{m.type}] {m.text[:200]}"))
    pg.on("pageerror", lambda e: console.append(f"[pageerror] {str(e)[:200]}"))
    pg.goto(f"http://127.0.0.1:{PORT}", timeout=30000)
    time.sleep(20); pg.screenshot(path=f"{OUT}/{TAG}_0_loaded.png")
    info = pg.evaluate("""() => { const c=[...document.querySelectorAll('canvas')].map(x=>({w:x.width,h:x.height,cls:x.className.slice(0,40)}));
        const imgs=[...document.querySelectorAll('img')].map(x=>({w:x.naturalWidth,h:x.naturalHeight,src:(x.src||'').slice(0,40)}));
        return {title: document.title, canvases: c, imgs: imgs, text: document.body.innerText.slice(0,300)}; }""")
    print("[browser] loaded:", json.dumps(info)[:900])
    pg.mouse.move(700, 400); pg.mouse.down(); pg.mouse.move(760, 420, steps=8); pg.mouse.up()
    time.sleep(8); pg.screenshot(path=f"{OUT}/{TAG}_1_dragged.png")
    b.close()
errs = [c for c in console if 'error' in c.lower() or 'pageerror' in c or 'warn' in c.lower()]
print("[browser] console (errors/warnings):", errs[:8] if errs else "none"); print("[browser] console total", len(console))
