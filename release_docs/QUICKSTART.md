# Quickstart — the demo stack on a real orchard, in an afternoon

You'll download a real (permitted, anonymised) citrus-orchard survey we
captured in South Africa — already processed into a digital twin — and
stand up the full UJAMAA stack on it: the 3D walkthrough, the farm-state
panel, change-over-time, and the multilingual language interface. No
robot, no pipeline runs, one GPU.

Prerequisite: [INSTALL.md](INSTALL.md) completed, `./scripts/check_install.sh`
green.

## 1. Get the sample survey (~1–2 GB)

```bash
./scripts/get_sample_data.sh
```

Downloads and unpacks into `data/sample_orchard/`: the trained splat
reconstruction, the plant registry (107 plants, 8 rows), the change ledger
across two survey dates, and the reconstruction-quality record. See
[docs/SAMPLE_DATA.md](docs/SAMPLE_DATA.md) for exactly what's inside and
the terms it is shared under.

## 2. Start the stack

```bash
./scripts/demo_up.sh data/sample_orchard
```

Starts four services (each with its own log under `logs/`):

| port | service |
|---|---|
| 8001 | walkthrough viewer (open this one) |
| 8002 | twin query API |
| 8003 | Adinkra — the language interface |
| 8004 | GPU render service |

Wait for `ALL SERVICES UP` (first start loads the language model into GPU
memory — up to a minute or two).

## 3. Walk the orchard

Open **http://localhost:8001** in your browser.

- Drag to look, scroll to zoom, WASD (or arrow keys) to walk the rows.
- The reconstruction is real: gaps, glare and all. What you see is what
  one survey pass captured.

## 4. Ask the farm something

In the panel beside the viewer, pick a language and ask — or paste one of
these from our evaluation set:

- English: `How many plants are there in this field?` → the registry
  answer, 107.
- English: `Take me to row 4.` → the camera moves there.
- isiXhosa: `Zingaphi izityalo kule ntsimi?` → the same 107, answered in
  isiXhosa.
- English: `Do the trees need watering today?` → an honest "the farm
  state doesn't contain that" — the system is built to decline rather
  than guess. This is deliberate; see
  [docs/EVALUATION.md](docs/EVALUATION.md) for what honesty currently
  does and doesn't hold across languages.

Answers take roughly 10–30 seconds depending on language — the model runs
locally on your GPU; nothing leaves your machine.

## 5. Change over time

Switch to the change panel and ask: `Which trees have changed the most
since the last survey?` — answered from the per-tree ledger (two survey
epochs in the sample; vegetation-index change per tree).

## 6. Shut down

```bash
./scripts/demo_down.sh
```

## Where next

- **Your own orchard**: the pipeline how-tos, starting with the
  [capture guide](docs/pipeline/00_capture.md) — what to record and how,
  before any software runs.
- **How it works / extending it**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- Something broke: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md), then
  an issue with your `logs/`.
