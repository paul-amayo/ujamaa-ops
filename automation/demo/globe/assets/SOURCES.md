# Earth textures for the globe sequence

Both are NASA Blue Marble images (NASA Earth Observatory / Visible Earth), public domain, downloaded 2026-10-01:

- `earth_5400.jpg` — "Blue Marble: Next Generation, topography and bathymetry, August 2004", 5400×2700
  https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73776/world.topo.bathy.200408.3x5400x2700.jpg
- `earth_2048.jpg` — "Land, shallow water, topography", 2048×1024 (fallback for low-memory clients)
  https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57752/land_shallow_topo_2048.jpg

Use: the globe-to-farm transitions in the demo video and, later, the landing page's "choose a farm". The citrus
farms zoom to the continent only (Paul, 2026-10-01): no farm location is shown for them.

## three.js

- `three.min.js` — three.js r160 (0.160.0), MIT licence, downloaded 2026-10-02 from
  https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js
  (sha256 170c6789f43217c96b3170f4b42fafe135de7f7cd48497a4218f9757ee1d49fa). Last release that still ships the
  UMD `three.min.js` bundle (it logs a deprecation warning); `globe.html` loads it with a plain `<script>` tag.

## Render recipe

`globe.html` is driven frame-by-frame (`?stop=&mode=&frame=` or `window.renderFrame(i)`, never wall-clock);
`render_globe.py` serves this directory over a local http.server, drives headless Google Chrome via the DevTools
protocol (stdlib-only websocket client), and encodes 1920×1080 / 30 fps / libx264 crf 18 clips with ffmpeg.
