# Migration plan — GPU work from the Paperspace VM to UCT HPC

Status: v2, 2026-08-14 — rewritten against the **confirmed** UCT HPC spec
(ucthpc.uct.ac.za). Goal: all GPU work runs on UCT HPC before the mid-Dec
2026 demo; the VM stops being the bottleneck and the single point of failure.

## 0. Both ends, measured

### The VM we are leaving

| | current |
|---|---|
| GPU | 1× Tesla V100 SXM2 32GB (sm_70), driver 580.159 — **NVML already broken**, `nvidia-smi` dead, reboot refused |
| CUDA stack | CUDA 11.8, torch 2.2, tiny-cuda-nn + gsplat built from source |
| env | pixi env in `nerf_new/` (**16 G**) |
| C++ stack | `aru_sil_core`: ROS 2 (ament/rclcpp/cv_bridge), GTSAM, PCL, Pangolin, OpenCV, protobuf, pybind11, libnabo — CPU only |
| data | 1.17 TB → after today's split: **603 G prod / 572 G experimental (deletable)** |
| GPU debt to the prod bar | ≈ **190–200 GPU-hours** (05: 35 blocks, 01: 35, 03: ~66, 04: 10, klap-apr: ~20, embedders ~10 h, SAM3) |

### The cluster we are going to (from the docs, 2026-08-14)

| fact | value | consequence for us |
|---|---|---|
| scheduler | **Slurm**; login `hpc.uct.ac.za` / `hex.uct.ac.za` | job arrays as planned |
| GPU partitions | **`a100`** — A100-40 for groups `a100free` and **`aru`**; A100-80 for others. **`l40s`** — L40S, group `l40sfree` open. `gpumk` — P100, CompSci-private | **ARU already appears on the a100 partition** — confirm your account is on it |
| GPU request syntax | `--partition=a100 --account=<group> --gres=gpu:ampere:1`; interactive via `sintx` | templates in §5 use this verbatim |
| GPU-per-job policy | *"do not reserve more than 1 GPU card per job unless … your code is capable of running on more than one"* | our per-block chain is single-GPU by construction — compliant, no change |
| walltime | `l40s` 48 h (96 cores / 8 GPUs per user); `ada` CPU 250 h; a100 not published | our jobs are 1–4 h — comfortable |
| **/home quota** | **10 GB** | **the 16 G pixi env cannot live in /home** |
| **/scratch quota** | **50 GB default**, 1 TB by 12-month agreement, 5 TB via eResearch; **not backed up**; *"not intended for long term storage"* | 50 G is far too small — the 1 TB agreement is a P0 blocker |
| installs | *"Do NOT run installs on the head node, installs must be run as jobs"* (`sintx`) | build the env inside an interactive job |
| python | Miniconda modules (`module load python/miniconda3-py3.12`) | we bring our own pixi env instead |
| transfers | **Globus** endpoint + Nextcloud portal documented | Globus Connect Personal on the VM beats rsync |
| containers | Singularity referenced in the docs, not documented as a supported service | **confirm** — it decides §2 |

**The headline finding: A100-40 is sm_80, and CUDA 11.8 supports it natively.**
The CUDA-12 / H100 risk from v1 of this plan is gone — the current stack ships
as-is. (L40S is sm_89, also supported by 11.8, if we ever use `l40sfree`.)

**The second headline: A100-40 has 40 GB vs the V100's 32 GB, and roughly
1.5–2.5× the throughput on this kind of workload.** Combined with running
4–8 blocks concurrently instead of 1, the ~200 GPU-hour debt goes from ~8 days
of perfect V100 duty to well under a day of wall-clock. That ratio is the
whole case for the move.

## 1. What moves, what stays

| workload | GPU? | destination | why |
|---|---|---|---|
| `ns-train high` stage1 + stage2 census-init (recipe of record) | yes | **HPC** | the debt; embarrassingly parallel per block |
| Embedder training (`train_hyperembedder_graph`) | yes | **HPC** | 1–2 h/survey |
| SAM3 passes (tree video-mode, fruit image-mode) | yes | **HPC** | ~40 GPU-min/survey |
| GLOMAP per-pass pose refine | yes | **HPC** | inside the per-block chain |
| Verdict batteries + Tier-B probe | yes | **HPC** | must run where training runs, else the gate is meaningless |
| Bag/mcap → monolithics, kf cuts, block partition, K-domain | no | **stays on VM** | ROS 2 + GTSAM + PCL + Pangolin; porting buys nothing |
| Ledger / association / compare (sankofa) | no | stays on VM | CPU, minutes |
| Viewer :8001, adinkra :8002 | no | **stays on VM** | long-running services; login nodes are not for daemons |

The seam is clean because the pipeline already breaks at the block boundary:
**inputs are small, outputs are big and derivable.**

## 2. Path portability — decide this first

Every artifact hard-codes `/home/paperspace/data/...`: `transforms.json` frame
paths, splat-run `config.yml` self-paths, tool defaults. Today's
`apr_2026_zed` move proved how loudly that breaks. Three options, best first:

**(a) Ask HPC support for one symlink** — `/home/paperspace` →
`/scratch/$USER/ujamaa`. A single admin `ln -s`, no security surface, and
**zero code change**: every existing absolute path resolves on both machines,
and configs written on HPC stay valid when pulled back. Cheapest by an order
of magnitude — ask for this in the P0 email.

**(b) Singularity bind-mount** (if containers are supported):
```bash
singularity exec --nv \
  --bind /scratch/$USER/ujamaa/data:/home/paperspace/data \
  --bind /scratch/$USER/ujamaa/code:/home/paperspace/code \
  ujamaa.sif bash automation/hpc_block_chain.sh 05_13D_Jackal 12
```
Same zero-code-change property, no admin favour needed.

**(c) Relative-path refactor** — a `UJAMAA_DATA_ROOT` env var plus
`transforms.json` written relative to its own directory (nerfstudio supports
this natively). This is the durable, portable-forever answer and is arguably
overdue hygiene, but it is ~2–3 days of work plus a regression pass. Do it
only if (a) and (b) both fail — or later, deliberately, when the demo is not
in the way.

## 3. Environment: pixi env on /scratch, built inside a job

/home's 10 GB rules out the env there; installs are forbidden on the head
node. So:

```bash
sintx --partition=a100 --account=aru --ntasks=8 --gres=gpu:ampere:1   # interactive
export PIXI_HOME=/scratch/$USER/ujamaa/.pixi
cd /scratch/$USER/ujamaa/code/nerf_new && pixi install                 # ~16 G
export TORCH_CUDA_ARCH_LIST=8.0 TCNN_CUDA_ARCHITECTURES=80             # A100 = sm_80
pixi run python -c "import torch; print(torch.cuda.get_device_name(0))"
```

- The pixi env **ships its own CUDA 11.8 toolchain including `nvcc`**, so
  tiny-cuda-nn and gsplat rebuild for sm_80 without depending on cluster CUDA
  modules. Rebuild them once, in the interactive job, on the target arch.
- gsplat JIT-compiles into `~/.cache/torch_extensions` — **redirect
  `TORCH_EXTENSIONS_DIR` to /scratch** or it will blow the 10 GB /home quota.
  Same for `HF_HOME` (SAM3 weights, `facebook/sam3`) and `PIP_CACHE_DIR`.
- Do **not** use `pip install --user`: `$HOME/.local` is on the 10 GB quota.
- The docs' own install instructions download from the internet inside jobs,
  which implies compute nodes have outbound access — confirm in P1; if they
  do not, pre-stage the env and HF cache and set `HF_HUB_OFFLINE=1`.
- COLMAP + GLOMAP: ship the binaries we already built, or rebuild in the same
  interactive job.

## 4. Data: push inputs, pull checkpoints, truth stays on the VM

Measured on 05 (44 blocks):

| payload | size | direction |
|---|---|---|
| kf_images | 5.2 G | ↑ once per survey |
| sky + fg masks | 36 M | ↑ |
| supervision + transforms + `init_lidar.ply` | < 1 G | ↑ |
| **per-survey training input** | **≈ 6 G** | ↑ |
| stage2 final ckpt | 1.2 G/block | ↓ |
| exported .ply/.splat + tb scalars | ~0.2 G/block | ↓ |
| stage1 / init / census intermediates | ~3 G/block | **delete on scratch after the verdict** |

Whole fleet ≈ **40 G up**, ~1.4 G down per finished block. Monolithics never
leave the VM. This is why the scratch ask is 1 TB and not 5 TB.

Rules:
- **Request the 1 TB /scratch agreement in P0** — 50 G does not hold even one
  survey's fleet (44 blocks × ~8 G raw). With per-block cleanup after the
  verdict lands, the steady-state footprint is ~2.5 G/block ≈ 500 G fleet-wide.
- **/scratch is not backed up and not long-term** (their words). `prod/` on the
  VM stays authoritative; scratch is a cache. Nothing that exists only on
  scratch is allowed to be prod.
- Transfer via **Globus** (documented endpoint) with Globus Connect Personal
  on the VM; `rsync -a --info=progress2` over SSH as the fallback.
- **Delete `experimental/` (572 G) before sizing anything** — it is safe by
  construction, and it makes the storage story ~600 G instead of 1.2 TB.

## 5. Job model: the block is the unit of parallelism

The two-stage chain is already per-block and idempotent (`.palette_v2`
marker, `transforms_lio.json` swap marker, resume-from-ckpt), so it maps
straight onto a Slurm array — one GPU each, as their policy requires:

```bash
#!/bin/sh
#SBATCH --account=aru
#SBATCH --partition=a100
#SBATCH --gres=gpu:ampere:1
#SBATCH --job-name="ujamaa-stage1"
#SBATCH --array=0-43%6            # 44 blocks, 6 concurrent
#SBATCH --ntasks=8
#SBATCH --time=04:00:00
#SBATCH --mail-user=paul.amayo@uct.ac.za
#SBATCH --mail-type=FAIL,END
#SBATCH --requeue

export TORCH_EXTENSIONS_DIR=/scratch/$USER/ujamaa/.torch_ext
export HF_HOME=/scratch/$USER/ujamaa/.hf
bash automation/hpc_block_chain.sh 05_13D_Jackal $SLURM_ARRAY_TASK_ID
```

- stage2 array chained with `--dependency=aftercorr:<jobid>` so each block's
  stage2 starts when *its own* stage1 lands, not after all 44.
- Verdict battery as a third short array; it writes
  `verdicts_censusinit_fw2.json`, which is exactly what `PROD.md` reads — so
  **the prod matrix stays honest with no manual step**.
- `--requeue` + resume-from-newest-ckpt: assume preemption on shared GPUs.
  `SAVE_BEST_EVAL_FG` already writes `nerfstudio_models_best/`, so a killed
  job's partial work is still useful.
- Keep the existing log markers (`REPL-DONE`, `SWEEP-DONE`) — the current
  tooling greps for them and will work unchanged.
- Throttle (`%6`) to stay inside the per-user GPU cap and be a good citizen;
  `l40sfree` is the overflow lane if `a100` is congested (add `8.9` to the
  arch list first).

## 6. Acceptance gate — why this migration is verifiable

We already have a null-calibrated fixture, so parity is testable:

1. **Tier A** (`automation/regression_harness.sh --cpu-only`) on HPC: K-domain
   counts, ledger datum invariants, embedder round-trip cos ≥ 0.999.
2. **Tier B probe** on block_001 @2k: expected **12.78 ± 1.5 dB** train FG
   (3σ from the 7-sample null, σ ≈ 0.47). HPC must land inside the band.
3. **Full-block parity**: retrain 05/block_001 end-to-end and require
   containment tree IoU ≥ 0.80 (VM recorded 0.958) and stage2 FG PSNR in band.
4. Only after 1–3 pass does HPC output get written into `prod/`.

Different GPU arch → different densification RNG → small numeric drift is
expected; the band is what separates drift from breakage.

## 7. Phasing (dual-run, never a hard cutover)

| phase | work | exit criterion |
|---|---|---|
| **P0 — access (0.5 d)** | email HPC support: §9 items — a100 group, 1 TB scratch, the `/home/paperspace` symlink, Singularity | account + quota + path decision confirmed |
| **P1 — env (1–2 d)** | `sintx` job: pixi env on /scratch, tcnn+gsplat for sm_80, COLMAP/GLOMAP, SAM3 cache | `ns-train high` runs 100 iters on a bind/symlinked block |
| **P2 — gate (1 d)** | Globus-push 05 inputs (6 G); run §6 | Tier A+B pass; block_001 parity |
| **P3 — first fleet (1–2 d)** | 05's remaining 35 blocks as an array; pull ckpts; regen `PROD.md` on the VM | 05 tassili column flips READY |
| **P4 — the rest (1–2 wk)** | 01/03 retrains under the recipe bar, 04 fill-in, klap-apr, embedders | fleet at the prod bar |
| **P5 — steady state** | VM = ingest + services + prod truth; HPC = all GPU | VM GPU idle; decide whether to keep the VM |

Dual-run through P4: the VM keeps training whatever HPC has not taken over, so
a slow onboarding never stalls the demo timeline.

## 8. Risks

| risk | mitigation |
|---|---|
| **/scratch stays at 50 GB** | P0 blocker — request 1 TB/12-month up front; without it, only single-block work is possible |
| No Singularity **and** no symlink | fall back to the relative-path refactor (§2c) — budget 2–3 days |
| /home 10 GB filled by caches | redirect `TORCH_EXTENSIONS_DIR`, `HF_HOME`, `PIP_CACHE_DIR`, `PIXI_HOME` to /scratch on day one |
| Scratch purge / no backup | scratch is a cache; ckpts pulled to VM `prod/` as each block lands |
| a100 queue congestion | `l40sfree` overflow (rebuild kernels with arch 8.9); submit arrays overnight |
| Preemption / walltime kill | 4 h ask for a ~1–2 h job; `--requeue`; best-eval ckpt |
| Compute nodes without internet | pre-stage env + HF cache; `HF_HUB_OFFLINE=1` |
| Two truths (VM vs HPC prod) | VM `prod/` authoritative; HPC writes to scratch until the gate passes |

## 9. P0 email to HPC support (`eresearch@uct.ac.za`)

Draft — confirm/adjust before sending:

1. Confirm my account is on the **`a100`** partition under the **`aru`** group
   (or `a100free`), and the exact `--account=` value to use. What is the
   walltime limit and per-user GPU cap on `a100`?
2. Request a **/scratch allocation of 1 TB (12-month agreement)** for the
   UJAMAA orchard-reconstruction dataset — ~600 G of prod data plus per-block
   training outputs; justification: gaussian-splat training over 8 field
   surveys, mid-December demo deliverable.
3. Is **Singularity/Apptainer** available on the GPU compute nodes, and is
   `--nv` permitted for user-supplied `.sif` images?
4. If not: would you create a symlink **`/home/paperspace` →
   `/scratch/<user>/ujamaa`** on the cluster? Our dataset artifacts embed that
   absolute path; one symlink removes an otherwise large refactor.
5. Do **GPU compute nodes have outbound internet** (conda/pip/HuggingFace)?
6. Preferred bulk-transfer route from an external cloud VM — **Globus**
   endpoint name, or rsync over SSH? Any inbound firewall requirement?
7. Are Slurm **job arrays** permitted on `a100`, and is there an array-size or
   concurrent-job cap beyond the GPU-per-user limit?

---

### Appendix — what does *not* move

`ujamaa/adinkra` (:8002), `serve-splat-viewer` / `serve-bateleur` (:8001),
ledger + association tooling, `build_prod_manifests.py`,
`build_dataset_registry.py`, and the whole rosbag/mcap ingest chain. Keeping
these on the VM is what makes this a ~3-week migration instead of a 3-month
one.
