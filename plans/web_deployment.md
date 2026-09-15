# UJAMAA demo → dedicated website: deployment roadmap

Status: DRAFT for Paul, 2026-09-15. Goal: the demo stack currently reached by
SSH tunnels to the A100 becomes a real site — a domain, TLS, auth, and no
laptop-side tunnels — without buying hardware before we have to.

## 0. What is actually running (measured 2026-09-15 on the A100)

| port | service | role | GPU? |
|---|---|---|---|
| 8001 | `splat_viewer/server.py` (aru_sil_core) | static front-end (`static/`), scene manifests, `/scene/hierarchy` | no |
| 8002 | query server | hierarchy/agronomy queries | no |
| 8003 | Adinkra panel server | panel agents → ollama | via ollama |
| 8004 | `render_service.py` (nerf_new pixi, **13 workers**) | server-side splat rendering + queries (moved here 2026-08-30; the download-era `/splats/*` endpoints were removed) | **yes** |
| 11434 | ollama (loopback) | Gemma-4-12B Q4_K_M for Adinkra | **yes** (~9 GB) |

Assets: prod `.splat` files are small — ~30 MB/block, **0.1 GB total**, with
`.br` (brotli) companions already built. The 752 GB under `citrus_all/` is
training data, not demo payload; the site never touches it.

**Finding to fix regardless of the migration: 8001–8004 bind `0.0.0.0` on a
box with a public IP.** Whether they are truly world-reachable depends on the
Paperspace firewall — verify, and either way the target state is: all
services loopback-only, reached through an authenticated tunnel.

Two facts shape the plan:
1. The demo is **GPU-tethered twice** (render service + ollama). A dedicated
   *website* is therefore mostly a routing/auth problem, not a hosting one —
   the GPU stays wherever the GPU is.
2. The GPU is **contended**: G0 fine-tune until ~Sep 20, fleet runs after,
   and the evening fleet policy stops ollama. A public demo and an
   experiment box have opposite uptime cultures; the roadmap's later phases
   are about separating them.

## P0 — ujamaa.ai in front of the current box (≈1 day, no new spend)

No code changes; the servers do not move.

1. **Domain: already owned — `ujamaa.ai`, registered at GoDaddy.** Keep the
   registration there; move only the DNS to a free Cloudflare zone (add the
   site in Cloudflare → change the two nameservers in GoDaddy's dashboard,
   ~30 min + propagation). Cloudflare Tunnel and Access need the zone on
   Cloudflare's DNS; GoDaddy stays the registrar of record. (A
   `*.uct.ac.za` alias via ICTS remains a nice-to-have, not a blocker.)
2. **Cloudflare Tunnel** (`cloudflared` systemd service on the A100, free):
   outbound-only connection, so all 800x ports get **rebound to
   127.0.0.1** and the box exposes nothing. Routes:
   `demo.ujamaa.ai` → 8001, `query.` → 8002, `adinkra.` → 8003,
   `render.` → 8004 (or one hostname + path routing if the front-end's
   URLs are made relative).
3. **Cloudflare Access** (free ≤50 users): email-OTP allow-list (us + WWW +
   collaborators). The demo is not for the open internet yet — Gemma terms,
   survey imagery governance, and an unhardened LLM endpoint all say
   invite-only first.
4. TLS, caching of the static front-end and `.splat.br`, and basic
   analytics come with the proxy for free.

Deliverable: `https://demo.ujamaa.ai` works from any browser with no SSH.
Risk unchanged: it is still the experiment box behind it.

## P1 — make the stack a service, not a session (≈2–3 days, no spend)

1. **systemd units** for all four servers + ollama (the render service and
   viewer currently die with the shell that started them). Restart=always,
   health-check endpoints, journald logs.
2. **Reconcile the evening fleet policy**: it currently stops ollama —
   either exempt the demo ollama or accept published "demo hours" on the
   status page. Decide explicitly; today it silently breaks Adinkra at
   night.
3. **Config out of code**: one `demo.env` (DATA_ROOT, survey id, block
   whitelist, ports) so the stack is re-creatable on any box — this is the
   real prerequisite for P2.
4. **Demo asset bundle**: the ~1 GB the site actually needs (splats,
   hierarchy bin/json, RGB crops the viewer serves) rsync'd into
   `/home/paperspace/demo_assets/` with a manifest — separable from the
   752 GB survey tree, backed up to the Mac.
5. Uptime monitoring (UptimeRobot free tier on the four hostnames) wired
   into the ops dashboard's GPU panel.

## P2 — take the demo off the experiment GPU (decision gate, ~$40–80/mo)

Options, cheapest first — measure before choosing (one afternoon:
`nvidia-smi` deltas per render worker, tokens/s for Gemma Q4 on smaller
cards):

- **A. Client-side rendering comeback**: the `.splat` files are 30 MB/block
  and `gaussian-splats-3d` rendered them in-browser before 08-30. If the
  walkthrough at demo scale (3–10 blocks) is acceptable client-side, the
  GPU renderer drops out of the demo entirely and only ollama needs a GPU.
  Cost: front-end work, zero hardware. This is the highest-leverage option
  and we owned it three weeks ago — the question is only why it was
  abandoned (scale? quality?) — answer it with a measurement, not memory.
- **B. Small dedicated GPU box** (Paperspace A4000 16 GB, ~$0.76/h — ~$550/mo
  24/7, or ~$80/mo at demo-hours-only with scheduled start/stop): runs
  ollama Q4 (9 GB) + render service if A fails. The `demo.env` bundle from
  P1 makes this a one-day move; the tunnel makes it invisible to users
  (repoint the route).
- **C. Stay on the A100** with cgroup/MPS partitioning: free but the demo
  competes with training forever; acceptable only until the first external
  demo with a deadline.

Recommendation: A for rendering + B-at-demo-hours for ollama, C as the
interim default. Revisit after the G0 full gate (~Sep 20) — if G1 goes
ahead, the A100 is booked solid and C stops being viable.

## P3 — hardening before any non-invited audience

- Rate-limit / queue the Adinkra endpoint (one ollama = one concurrent
  generation; a public form field pointed at it is a free DoS).
- **Gemma licence re-read** before opening beyond research collaborators
  (the memo's §7 note applies to serving, not just fine-tuning); same pass
  for survey-imagery/data governance (farm identifiability).
- Cloudflare Access stays even when "public": switch allow-list → any
  Google/email login, keeping abuse traceable.
- Status page + demo-hours honesty if ollama is scheduled.

## Sequence and cost summary

| phase | wall time | recurring cost | unblocks |
|---|---|---|---|
| P0 tunnel + DNS + auth | ~1 day | domain already owned | WWW & collaborators browse the demo today |
| P1 systemd + env + bundle | 2–3 days | 0 | reproducibility; P2 |
| P2 GPU separation | 0.5 day measure + 1 day move | $0–80/mo (option-dependent) | demo uptime decoupled from research |
| P3 hardening | 1–2 days | 0 | audience beyond the allow-list |

P0 can start now (it doesn't touch the GPU or the G0 run). P2's decision
waits for the render-cost measurement and the G0 gate.
