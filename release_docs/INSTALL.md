# Installing UJAMAA

Target: fresh Ubuntu 22.04 with one NVIDIA GPU → serving stack running in
under an hour (most of it downloads). Every command is copy-pasteable; if a
step fails, that's a bug in this document — please open an issue.

UJAMAA deliberately uses **two Python environments** (this is the one piece
of honest complexity — the 3D stack and the robotics ingest stack have
incompatible dependency worlds):

| environment | used for | managed by |
|---|---|---|
| `envs/twin` (pixi) | splat training, rendering, the 3D viewer's render service | [pixi](https://pixi.sh) — pinned, reproducible |
| system `python3` | survey ingest (rosbags), the panel/language servers | apt + one `pip install -r` |

## 0. Prerequisites

- Ubuntu 22.04 (other distros likely work; only 22.04 is tested)
- An NVIDIA GPU with a driver ≥ 550 (`nvidia-smi` works). CUDA toolkit is
  NOT required system-wide; the pixi environment carries its own.
- ~15 GB free disk for software + model, plus the data you bring.

```bash
sudo apt update && sudo apt install -y git curl python3 python3-pip python3-venv
```

## 1. Clone

```bash
git clone https://github.com/<org>/ujamaa.git
cd ujamaa
```
<!-- CONSOLIDATION-CONTRACT: final org/name at repo creation; docs assume
a single clone with pipeline/, core/, adinkra/, envs/, docs/ inside. -->

## 2. The twin environment (pixi)

```bash
curl -fsSL https://pixi.sh/install.sh | bash   # or your preferred method
cd envs/twin && pixi install && cd ../..
```

This pulls the pinned nerfstudio/CUDA/pytorch stack (~6 GB first time).
Smoke test:

```bash
cd envs/twin && pixi run python -c "import torch; print(torch.cuda.get_device_name(0))" && cd ../..
```

## 3. The serving environment (system python)

```bash
python3 -m pip install -r requirements-serving.txt
```

(FastAPI/uvicorn and the panel servers' small dependency set. Survey
ingest additionally needs `numpy scipy` — included here.)

## 4. The language model (ollama + Gemma)

Adinkra serves Google's Gemma model locally via [ollama](https://ollama.com);
you download the model yourself under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms).

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M   # ~7.3 GB
```

Smoke test (first response may take a minute while the model loads):

```bash
curl -s http://localhost:11434/api/version
```

The 4-bit `Q4_K_M` build is what our published evaluation ran; a
higher-precision build will behave differently (usually better, slower —
and needs more VRAM).

## 5. Verify

```bash
./scripts/check_install.sh
```

Checks: GPU visible from the pixi env, serving deps importable, ollama up,
model present. All green → continue to [QUICKSTART.md](QUICKSTART.md),
which downloads a real sample survey and starts the full stack.

## Known sharp edges

- **Driver too old**: the pinned torch needs driver ≥ 550. Symptom:
  `CUDA error: no kernel image`. Fix: upgrade the driver, nothing else.
- **ollama stopped**: the servers do not start it for you (systemd unit
  recommended — see [docs/serving.md](docs/serving.md)). Symptom: Adinkra
  answers with connection errors.
- **Two environments, one shell**: pipeline commands are prefixed
  `pixi run` (or run inside `pixi shell`); server commands use plain
  `python3`. Each doc says which; mixing them is the #1 support question
  we designed against.
