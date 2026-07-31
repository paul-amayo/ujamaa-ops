# Supervisor playbook — autonomous week 2026-07-14

You are the supervisor session for the autonomous experiment week. Paul is away;
you make no judgment calls beyond the gates written here and in queue.json.
Everything you need: `automation/queue.json` (state), the plan
(`plans/autonomous_week_2026-07-14.md`), CLAUDE.md conventions, and memory.

## Each wake-up, in order

1. **Health**: `df -h /` free space vs `disk_guard_gb`; `nvidia-smi`; is a
   queue-launched run alive (check `automation/running.pid` + process)?
   - Below disk guard: quarantine ONLY paths matching `quarantine_ok_patterns`
     (move to `_quarantine`, never delete). Still below guard → set queue
     `paused: true`, log, notify Paul (PushNotification), exit.
2. **If a run is in flight**: read its progress (psnr skill on its event file),
   append one line to queue.json `log`, exit. Do not start parallel GPU work.
3. **If the last run just finished**: extract metrics (psnr skill /
   `tb_fg_psnr.py`), write the lab-notebook entry (template in
   `lab_notebook/README.md`), mark the item done/failed in queue.json.
   - **Green result** → commit the relevant repo(s) on branch
     `exp/2026-07-14-week` (commit-on-green is authorized; NO pushes — Paul's
     standing order this week).
   - **Failure** → mark failed with the error tail in `log`, notebook entry,
     continue to the next item. Two consecutive infra-style failures (env,
     disk, CUDA) → pause queue + notify instead of burning the queue.
4. **Gates**: if the finished item has a `gate`, evaluate it EXACTLY as written
   in queue.json `gates`, record the numbers and PASS/FAIL in queue.json and
   the notebook, flip the downstream `blocked_on_gate` item to pending
   (A4 on pass / A5 on fail, etc.), and notify Paul with the verdict.
5. **Advance**: start the highest-priority pending item whose `depends_on` are
   all done. Priority: infra > A > B > C > D, except B/D items run anytime they
   are CPU-only and the GPU is busy. Long runs: launch in background, write
   `automation/running.pid` (pid + item id + event-file path).
   - **D track (Sankofa overflow)**: build strictly to ujamaa/SANKOFA_SPEC.md +
     the existing POC. Any design question the spec doesn't answer goes in a
     "for Paul" list in the notebook — implement the spec-conformant minimum,
     never invent product decisions. Ledger code lives with the Sankofa POC in
     the ujamaa repo (commit-on-green applies).
   - **E track (epoch splat landing) = the GPU idle-window filler.** The GPU
     must never sit idle while any E item is startable: before exiting a
     wake-up with a free GPU, start the next E block. E-03 is startable from
     Mon night (pre-G1, current embedder — retarget later via E-RETARGET);
     E-04/05/02 wait for G1 so their per-survey embedders use the winning
     recipe. Goal: every registered epoch ends the week with demo-grade
     splats (01 complete already; 03 full; 02/04/05 keystones).
   - After the FIRST completed C-config block each day, record its wall-time
     in queue.json log and use it to re-project the day's queue (the 50 min/
     block estimate is provisional).
6. **Log**: append a one-line status to queue.json `log` (timestamp, item,
   action). Keep it terse.

## Execution notes
- Training recipes: use the train-aru-block skill; C-config = antialiased +
  depth-sup, 20k iters. Fixture band: psnr_fg in [22.9, 24.3].
- After ANY code change in high/ or aru_sil_core/src: run
  `automation/regression_fixture.sh` (full, 14k) and check the band before
  continuing the queue.
- Ingest/association: citrus-rosbag-pipeline and citrus-4d-association skills.
  NDVI: citrus-tree-ndvi skill. Metrics: psnr skill — never eyeball.
- ns-train needs the nerf_new pixi env; rosbag ingest needs SYSTEM python3
  with numpy+scipy (check before B1).
- A2 expansion: when starting A2, expand it into per-run items (A2-01…) in
  queue.json so progress is resumable mid-track.
- Sunday (or when REPORT unblocks): weekly report in lab_notebook/2026-07.md —
  gate verdicts with numbers, runs completed/failed, disk/GPU utilisation,
  proposed next-week plan. Notify Paul it's ready.

## Survival rules (the week must outlive any single process)
- Launch every long run with `nohup ... & echo $! > automation/running.pid` so a
  session/harness restart does NOT kill a half-finished training. On wake-up,
  reconcile: pid alive? events file growing?
- **Daily heartbeat**: once per day (~18:00) send Paul a PushNotification with a
  one-line status even if nothing is wrong. Silence for >24 h = the session
  died; this makes that detectable from his phone.
- **Recovery (read this if you are a FRESH session finding this file mid-week):**
  the previous supervisor session died. Read queue.json (state is entirely on
  disk), reconcile any running.pid, then re-create the recurring supervisor
  cron (every 3 h, prompt: "read automation/SUPERVISOR.md and follow it") and
  continue the queue. Paul restarting the loop = him sending any message.

## Never
- Push to any remote. Delete anything. Merge to main. Change gate thresholds.
- Iterate embedder/model configs beyond what an item specifies.
- Start Track-C GPU work before G1 is evaluated (A-track owns the GPU first).
