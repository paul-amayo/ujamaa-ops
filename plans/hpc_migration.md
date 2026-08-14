# Migration plan — GPU work from the Paperspace VM to UCT HPC

Status: DRAFT for Paul, 2026-08-14. Goal: **all GPU work runs on UCT HPC**;
the VM stops being the bottleneck (and the single point of failure) before
the mid-Dec 2026 demo.

## 0. The measured starting point

| | current (VM) |
|---|---|
| GPU | 1× Tesla V100 SXM2 32GB (sm_70), driver 580.159 — **NVML already broken** (userspace 580.173 vs kernel), `nvidia-smi` dead, reboot refused |
| CUDA stack | CUDA 11.8, torch 2.2, tiny-cuda-nn + gsplat compiled from source |
| env | pixi env in `nerf_new/` (16 G on disk) |
| C++ stack | `aru_sil_core`: ROS 2 (ament/rclcpp/cv_bridge), GTSAM, PCL, Pangolin, OpenCV, protobuf, pybind11, libnabo — CPU only |
| data | ~1.17 TB total; after today's split: **603 G prod / 572 G experimental (deletable)** |
| GPU debt to reach the prod bar | ≈ **190–200 GPU-hours** (05: 35 blocks, 01: 35, 03: ~66, 04: 10, klap-apr: ~20, embedders ~10 h, SAM3 passes) |

At 65 min/block serial on the V100 that debt is ~8 days of perfect duty cycle.
As an 8-wide job array on HPC it is ~1 day. **That ratio is the whole case for
the move.**

## 1. What moves, what stays

| workload | GPU? | destination | why |
|---|---|---|---|
| `ns-train high` stage1 + stage2 census-init (the recipe of record) | yes | **HPC** | the debt; embarrassingly parallel per block |
| Embedder training (`train_hyperembedder_graph`) | yes | **HPC** | 1–2 h/survey, batchable |
| SAM3 passes (tree video-mode, fruit image-mode) | yes | **HPC** | ~40 GPU-min/survey |
| GLOMAP per-pass pose refine | yes (CUDA matching) | **HPC** | runs inside the per-block chain anyway |
| Verdict batteries (containment, pointing, aligned gates) + Tier-B probe | yes | **HPC** | must run where training runs, else the gate is meaningless |
| Bag/mcap → monolithics, kf cuts, block partition, K-domain | no | **stays on VM** | ROS 2 + GTSAM + PCL + Pangolin C++ stack; porting it buys nothing |
| Ledger / association / compare (sankofa) | no | stays on VM | CPU, minutes |
| Viewer :8001, adinkra :8002 | no | **stays on VM** | long-running services; HPC login nodes are not for daemons |

The split is clean because the two-stage pipeline already has a hard seam at
the block boundary: **inputs are small, outputs are big and derivable.**

## 2. The one design decision that matters: path portability

Every artifact in this workspace hard-codes `/home/paperspace/data/...` —
`transforms.json` frame paths, splat-run `config.yml` self-paths, tool
defaults. Today's `apr_2026_zed` move proved how loudly that breaks.

**Decision: do not rewrite paths. Bind-mount to the identical path inside the
container.**

```bash
apptainer exec --nv \
  --bind /scratch/$USER/ujamaa/data:/home/paperspace/data \
  --bind /scratch/$USER/ujamaa/code:/home/paperspace/code \
  ujamaa.sif  bash automation/hpc_block_chain.sh 05_13D_Jackal 12
```

Inside the container every existing absolute path resolves unchanged; configs
written on HPC stay valid when pulled back to the VM, and vice-versa. Zero
code churn, no path-rewrite regression class. This assumes Apptainer/
Singularity is available (§0 question 3) — it is the norm on SA clusters.

## 3. Environment: containerize, do not rebuild

Build **one** Apptainer image from the working pixi env rather than fighting
module stacks:

1. On the VM: `pixi list --manifest-path nerf_new/pixi.toml > env.lock` and
   capture the tcnn/gsplat build flags.
2. Definition file: CUDA base matching the HPC GPU arch (§0 q2) → micromamba/
   pixi env → `pip install -e nerfstudio` + `high` + `aru_sil_core/src/scripts`
   on the PYTHONPATH → **pre-build tiny-cuda-nn and gsplat kernels inside the
   image** with `TCNN_CUDA_ARCHITECTURES` / `TORCH_CUDA_ARCH_LIST` set for the
   target arch (compute nodes usually have no internet and often no nvcc).
3. Bake the SAM3 weights (`facebook/sam3` from the HF cache) into the image or
   a read-only bind; set `HF_HUB_OFFLINE=1`.
4. Ship COLMAP + GLOMAP binaries in the image (already built here).

**The arch trap, in advance:** kernels compiled for sm_70 will not run on the
HPC GPUs. A100 (sm_80) works with the current CUDA 11.8 pin. **H100 (sm_90)
does not** — it needs CUDA ≥12 and therefore a torch/nerfstudio/gsplat bump,
which is a week of work and its own regression risk. Answering §0 q2 first is
what stops that from becoming a surprise.

## 4. Data: push inputs, pull checkpoints, keep truth on the VM

Measured on 05 (44 blocks):

| payload | size | direction |
|---|---|---|
| kf_images | 5.2 G | ↑ once per survey |
| sky + fg masks | 36 M | ↑ |
| supervision + transforms + `init_lidar.ply` | < 1 G | ↑ |
| **per-survey training input** | **≈ 6 G** | ↑ |
| stage2 final ckpt | 1.2 G/block | ↓ |
| exported .ply/.splat + tb scalars | ~0.2 G/block | ↓ |
| stage1/init/census intermediates | ~3 G/block | **stays on scratch, derivable** |

So the whole fleet is ~**40 G up**, and ~1.4 G down per finished block — not
the 600 G a naive "move the datasets" reading implies. Monolithics never
leave the VM (ingest stays there).

Rules:
- **HPC scratch is a cache, not a home.** Assume a purge policy; nothing that
  only exists on scratch is allowed to be prod.
- The VM (and whatever UCT durable storage backs it) keeps `prod/` as the
  authoritative copy; `automation/build_prod_manifests.py` still runs there.
- Sync with `rsync -a --info=progress2` over SSH; ask about Globus/Datamover
  nodes (§0 q5) if the 40 G initial push is slow.
- Delete `experimental/` (572 G, safe by construction) **before** measuring
  any storage request — the ask to HPC is then ~600 G, not 1.2 TB.

## 5. Job model: the block is the unit of parallelism

The two-stage chain is already per-block and idempotent (`.palette_v2`
markers, `transforms_lio.json` swap marker, resume-from-ckpt). That maps
directly onto a Slurm array:

```bash
#SBATCH --job-name=ujamaa-stage1
#SBATCH --array=0-43%8          # 44 blocks, 8 concurrent
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00         # 65 min/block on V100 + headroom
```

- stage1 array → stage2 array with `--dependency=aftercorr:<jobid>` so each
  block's stage2 starts as soon as *its* stage1 lands (not after all 44).
- Verdict battery as a third, short array; it writes
  `verdicts_censusinit_fw2.json` — which is what `PROD.md` reads, so **the
  prod matrix stays honest without any manual step.**
- Per-block logs to `$SCRATCH/logs/<survey>/<block>.log`, same markers
  (`REPL-DONE`, `SWEEP-DONE`) the existing tooling greps for.
- Requeue-safe: `--requeue` + resume from the newest ckpt, since preemption on
  shared GPU partitions is normal.
- `SAVE_BEST_EVAL_FG` already writes `nerfstudio_models_best/` — keep it; it
  makes a preempted job's partial work still useful.

## 6. Acceptance gate (this is why the migration is verifiable)

We already have a null-calibrated fixture, so parity is testable rather than
vibes-based:

1. **Tier A** (`automation/regression_harness.sh --cpu-only`) inside the
   container: K-domain counts, ledger datum invariants, embedder round-trip
   cos ≥ 0.999.
2. **Tier B probe** on block_001 @2k: expected **12.78 ± 1.5 dB** train FG
   (3σ from the 7-sample null, σ≈0.47). HPC must land inside that band.
3. **Full-block parity**: retrain 05/block_001 end-to-end on HPC and require
   containment tree IoU ≥ 0.80 (recorded VM value 0.958) and stage2 FG PSNR
   within the same band.
4. Only after 1–3 pass does HPC output get written into `prod/`.

Different GPU arch → different densification RNG → small numeric drift is
expected; the band is what tells us drift from breakage.

## 7. Phasing (dual-run, never a hard cutover)

| phase | work | exit criterion |
|---|---|---|
| **P0 — recon (0.5 d)** | answer §0 questions with HPC support; request account/quota | GPU model, container policy, quota confirmed |
| **P1 — image (2–3 d)** | build + test `ujamaa.sif` on one GPU node, interactive | `ns-train high` runs 100 iters on a bind-mounted block |
| **P2 — gate (1 d)** | push 05 inputs (6 G); run §6 acceptance | Tier A+B pass, block_001 parity |
| **P3 — first fleet (2 d)** | 05's remaining 35 blocks as an array; pull ckpts; regen `PROD.md` on the VM | 05 tassili column flips READY |
| **P4 — the rest (1–2 wk)** | 01/03 retrains under the recipe bar, 04 fill-in, klap-apr, embedders | fleet at prod bar |
| **P5 — steady state** | VM = ingest + services + prod truth; HPC = all GPU | VM GPU idle; decide whether to keep it |

Dual-run through P4: the VM keeps training whatever HPC hasn't taken over, so
a slow HPC onboarding never stalls the demo timeline.

## 8. Risks

| risk | mitigation |
|---|---|
| **H100-only GPUs** → CUDA 11.8 pin invalid | answer §0 q2 in P0; budget a week for the CUDA-12 bump if so; V100/A100 partitions if both exist |
| Compute nodes have no internet | everything baked into the image; `HF_HUB_OFFLINE=1`; no `pip install` at runtime |
| Scratch purge eats a fleet mid-run | scratch = cache; ckpts pulled to VM `prod/` as each block lands |
| Walltime kill mid-train | 4 h ask for a 65-min job; `--requeue` + resume; best-eval ckpt |
| Shared-partition queue wait | submit arrays overnight; `%8` throttle to stay a good citizen |
| Absolute paths | bind-mount to identical paths (§2) |
| Two truths (VM vs HPC prod) | `prod/` on the VM is authoritative; HPC writes only to scratch until the gate passes |

## 9. Questions for UCT HPC support (P0 blockers)

1. Account + which cluster (UCT HPC / ilifu / CHPC) — scheduler is Slurm on the
   first two, PBS Pro on CHPC (translate §5 accordingly).
2. **GPU models and their CUDA/driver versions** (V100 / A100 / L40S / H100?) —
   decides whether the current CUDA 11.8 stack ships as-is.
3. Apptainer/Singularity available on compute nodes, and is `--nv` permitted?
   Can we bring our own `.sif`, or must images be built by support?
4. Storage: home vs scratch quotas, purge policy, and whether ~600 G project
   space is grantable for prod data.
5. Transfer path: plain `rsync` over SSH from an external cloud VM, or a
   datamover/Globus endpoint? Any inbound firewall for the VM's IP?
6. Max walltime + array limits on the GPU partition; is preemption in play?
7. Internet egress from compute nodes (expected: none) and from login nodes.

---

### Appendix — what does *not* need to move

`ujamaa/adinkra` (query server), `serve-splat-viewer` / `serve-bateleur`
(:8001), ledger + association tooling, `build_prod_manifests.py`,
`build_dataset_registry.py`, and the whole rosbag/mcap ingest chain. These are
CPU-and-services work; keeping them on the VM is what makes the GPU migration
a 3-week task instead of a 3-month one.
