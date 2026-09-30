"""POST /mask on the identity side-car at walk poses near a tree; report how much fires per pose."""
import json, sys, urllib.request, struct
import numpy as np, cv2
SURVEY, TREE = sys.argv[1], int(sys.argv[2])
get = lambda u: json.loads(urllib.request.urlopen(u, timeout=30).read())
pl = next(o for o in get(f"http://127.0.0.1:8011/api/registry?survey={SURVEY}")["objects"] if o["n"] == TREE); t = np.array(pl["xyz"])
fr = get("http://127.0.0.1:8031/scene/trajectory")["frames"]
M = [np.array(f["matrix"], float).reshape(4, 4) for f in fr]; P = np.array([m[:3, 3] for m in M])
i = int(np.argmin(np.linalg.norm(P - t, axis=1)))
best = None
for off in (0, 1, 2, 3, 4, 6, 8):
    j = max(0, i - off); c2w = M[j]; fwd = -c2w[:3, 2]; v = t - c2w[:3, 3]; d = float(np.linalg.norm(v))
    ahead = float(np.dot(v / max(d, 1e-6), fwd / np.linalg.norm(fwd)))
    body = json.dumps({"c2w": c2w.flatten().tolist(), "w": 1280, "h": 720, "fovy": 1.194, "quality": 90, "query": {"tree": TREE}}).encode()
    req = urllib.request.Request("http://127.0.0.1:8025/mask", data=body, headers={"Content-Type": "application/json"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    seq, W, H, nb, _, rms, tms = struct.unpack("<IHHHHff", raw[:20])
    a = cv2.imdecode(np.frombuffer(raw[20:], np.uint8), cv2.IMREAD_GRAYSCALE)
    fired = int((a > 80).sum()); print(f"off {off}: pose {j} dist {d:.1f} m, cos(ahead) {ahead:+.2f}, blocks {nb}, render {rms:.0f} ms, mask max {int(a.max())}, fired px {fired}")
    if best is None or fired > best[0]: best = (fired, j, a)
cv2.imwrite(f"/tmp/appcheck/mask_{SURVEY}_{TREE}.png", best[2]); print("best pose", best[1], "fired", best[0])
