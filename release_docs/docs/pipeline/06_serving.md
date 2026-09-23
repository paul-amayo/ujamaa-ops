# 6 · Serving the twin

The quickstart's `demo_up.sh` is this page automated against sample data;
here is what it starts, pointed at *your* survey, and how to run it as a
service rather than a session.

## The processes

| port | service | env | GPU |
|---|---|---|---|
| 8001 | walkthrough viewer (static front-end + scene manifests + `/scene/hierarchy`) | system python | no |
| 8002 | twin query API | system python | no |
| 8003 | Adinkra — panel agents → the language model | system python | via ollama |
| 8004 | render service (streams the splat blocks) | **pixi** | yes |
| 11434 | ollama (loopback) | — | yes, ~9 GB |

```bash
./scripts/demo_up.sh data/<survey_id>     # same entry point as the quickstart
```

What that sets, if you run pieces by hand (the configuration contract):

| variable | consumed by | meaning |
|---|---|---|
| `ADINKRA_HIERARCHY` | 8003 | the survey's `marker_hierarchy.json` |
| `ADINKRA_SCORES` | 8003 | reconstruction-quality record (3D-view panel) |
| `ADINKRA_LEDGER` | 8003 | the site ledger (change panel) |
| `RENDER_BLOCKS_ROOT` | 8004 | the survey's blocks directory |
| `RENDER_RUN_GLOB` | 8004 | per-block glob to the trained run's `config.yml` |
| `RENDER_CKPT_POSES` | 8004 | pose convention of those checkpoints (`opengl`) |
| `RENDER_VRAM_BUDGET` / `RENDER_PRELOAD` | 8004 | how many blocks stay resident (~0.5–1 GB each; loads take ~20 s) |
| `HIGH_LLM_URL` / `HIGH_LLM_MODEL_ALIAS` | 8003 | ollama endpoint + model tag |

## Language interface — what to expect operationally

- First query after a cold start loads the model (up to minutes);
  afterwards it stays resident (~9 GB).
- **One generation at a time**: ollama serialises; concurrent users
  queue. Answer latency is roughly 10–30 s by language on a data-centre
  GPU (measured medians 11 s English … 32 s isiXhosa) — design the
  interaction as ask-and-return, and read
  [docs/EVALUATION.md](../EVALUATION.md) before treating answers as
  decisions.
- Adding a language is a documented one-file change (see
  [extending](../ARCHITECTURE.md)); our own added languages shipped
  machine-drafted and *unvalidated* — label yours honestly too.

## Running it as a service (recommended beyond your first afternoon)

- **systemd units** for all five processes (templates in
  `deploy/systemd/`): `Restart=always`, ordered after ollama, journald
  logs. The failure mode they prevent: servers started from a shell die
  with the shell — silently, at your first closed laptop.
- **Bind loopback, expose through a tunnel** — the shipped units bind
  `127.0.0.1`; put an authenticated reverse proxy or tunnel in front for
  remote users. Never expose 8003 raw: an open text endpoint into a GPU
  is a free denial-of-service.
- **GPU sharing with training**: the render service holds VRAM per
  resident block and ollama holds ~9 GB. On one GPU, stop serving while
  training (`demo_down.sh`) or cap `RENDER_VRAM_BUDGET` and accept slower
  loads.

## Verify

`./scripts/check_serving.sh` — hits each port's health endpoint, asks
Adinkra one English registry question, and confirms the answer cites your
plant count. If 8003 answers with connection errors, ollama is down —
it is a separate service, and the panels do not start it for you.
