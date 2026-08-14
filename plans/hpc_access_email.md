# P0 email to UCT HPC support — ready to send

To: eresearch@uct.ac.za
Cc: (supervisor/PI if required by their GPU-access policy — their docs ask for
    the supervisor in CC when requesting GPU partitions)
Subject: GPU partition + 1 TB scratch request — UJAMAA orchard reconstruction (ARU)

---

Dear HPC team,

I'm Paul Amayo (African Robotics Unit, UCT). I'm moving a GPU workload onto the
cluster: 3D gaussian-splat reconstruction of citrus orchards and a nursery site
(the UJAMAA digital-twin project), currently running on a rented cloud VM. The
work is ~200 GPU-hours of training across eight field surveys, with a public
demo deliverable in mid-December. Each job trains one reconstruction block on
**a single GPU** — no multi-GPU jobs — submitted as Slurm job arrays.

I'd be grateful for help with the following:

1. **GPU access.** Your GPU partition page lists `aru` among the groups on the
   `a100` partition. Could you confirm my account is on it, and tell me the
   exact `--account=` value to use? A100-40 suits us well (our CUDA 11.8 /
   PyTorch stack targets sm_80 directly). I'm happy to use `l40sfree` for
   overflow if `a100` is busy.

2. **Scratch allocation: 1 TB, 12-month agreement.** The 50 GB default won't
   hold a single survey (one survey is ~6 GB of input imagery plus ~2.5 GB of
   trained model per block, ~44 blocks). Steady-state footprint across the
   project is ~500 GB. I understand /scratch isn't backed up and isn't
   long-term storage — the authoritative copy of the data stays off-cluster,
   and I'll prune intermediates as each block completes.

3. **Containers.** Is Singularity/Apptainer available on the GPU compute
   nodes, and is `--nv` permitted for a user-supplied `.sif` image?

4. **If not, one small favour instead:** would you be willing to create a
   symlink `/home/paperspace` → `/scratch/<my-username>/ujamaa` on the
   cluster? Our dataset files embed that absolute path (it's the layout on the
   VM the data was built on), so a single symlink saves an otherwise
   substantial refactor. Either option 3 or 4 solves it — whichever is easier
   on your side.

5. **Do GPU compute nodes have outbound internet access** (conda/pip/
   HuggingFace)? Your docs say installs must be run as jobs rather than on the
   head node, so I assume yes, but I'd like to confirm before planning around
   it — if not, I'll pre-stage everything.

6. **Bulk transfer.** What's the preferred route for an initial ~40 GB from an
   external cloud VM — your Globus endpoint (and its name), or rsync over SSH?
   Any inbound firewall rules I should request?

7. **Job arrays.** Are Slurm arrays permitted on `a100`, and is there a cap on
   array size or concurrent tasks beyond the per-user GPU limit? Also, what is
   the walltime limit on `a100`? (I saw 48 h documented for `l40s`; my jobs run
   1–4 h each.)

Happy to provide any further detail on the workload, and glad to acknowledge
UCT eResearch/HPC in the resulting publications and demo.

Many thanks,

Paul Amayo
African Robotics Unit, University of Cape Town
paul.amayo@uct.ac.za

---

## Notes before sending (not part of the email)

- Items **1 and 2 are the P0 blockers** — everything else can be worked around.
  If you want to shorten it, send 1+2+3/4 now and ask 5–7 later.
- Item 4 is the cheap fix for the absolute-path problem
  (`plans/hpc_migration.md` §2). If they decline both 3 and 4, the fallback is
  the relative-path refactor — 2–3 days plus a regression pass.
- Their GPU page states GPU access is granted by email with the supervisor in
  CC. As PI you may be the supervisor; if the policy still wants a second
  name, CC the ARU head.
- Worth attaching nothing — a link to the project or a one-line description on
  request is usually enough.
