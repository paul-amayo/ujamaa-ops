# Run of show: AIC FS 2026 gallery walk, live demo (Tuesday 6 Oct, morning)

Status: v2 for Paul, 2026-09-30. Supersedes v1, which assumed the GPU was shared with training.
The GPU is now free, so this version adds a **work and training plan** that
gives every demo farm the data each agent needs. The demo accompanies the Pillar 3 poster
(v13); its three stories are **plants individuated in 3D, the same plant tracked over time,
and questions answered in the farmer's language**.

## 1. Quality bar and what passes it (Paul, 30 Sep)

**H3DGS is the demo standard. Nothing under 25 dB is shown. Training aims for the
high 20s to early 30s on training views.** Identity (lighting a plant, a row or fruit)
on H3DGS comes from side-cars: HiGH features on the H3DGS leaves, the 09-27 recipe with
the void row, `--bg-competes --bg-ratio 2`, and no opacity boost.

Measured numbers, notebook of record. "in-cell" means held-out views inside the chunk.

| farm | H3DGS model | measured | passes 25 dB? | demo 3D (1 Oct) |
|---|---|---|---|---|
| **Citrus farm A** (01) | improved recipe, chunk 3_1 (+ 7_2, 2_0, 6_2, 2_2 built) | 3_1 in-cell 28.23 / 27.56 sky-masked | **yes** | **chunk 3_1**, side-car: trees lit (tree 164 IoU 0.97, rows 0.85–0.96) |
| **Citrus farm B** (05) | improved recipe, chunks 0_0 and **1_0** | 1_0 in-cell 26.28 / 26.55 sky-masked; 0_0 27.54 / 28.07 | **yes** | **chunk 1_0** (trees 5 and 3), side-car: trees lit (tree 5 IoU 0.73); **fruit not lit** (containment IoU 0.03 on the fruit-densified side-car — the finding, no cuts) |
| Citrus farm B (04) | original recipe, 6 chunks | not at the bar | — | Sankofa only |
| **Klapmuts** Dec lane 2 | h3dgs_e2 (half views) | training views 29.72, held-out 19.82 | **yes** (training views) | served, no side-car |
| Klapmuts Apr | 4 chunks | ~17–18 | **no** | none; April via its data (ledger v5 registry) |
| **Kendu Bay** ground crop (IMG_7975_s0) | lane-style H3DGS, 8 M | training views 26.06 / 26.78, held-out 18.74 | **yes** (training views) | **served** (kendu-0514-plants), no side-car yet |
| Gwakungu chillies (IMG_7990_s1) | lane-style H3DGS, 8 M | training 21.92, held-out 15.19 | **no** | none |
| Gwakungu cabbages (IMG_7993_s0) | 8 M: training views 23.43 / 23.61, held-out 18.16 | **below 25, shown by Paul's call (1 Oct)** | **served** in portrait; registry 29 heads in two rows; side-car pending (compiled supervision needed) |

The per-block models (the Citrus A/B glref fleet, the phone segments) are **not** shown once their H3DGS replacement lands.

Unchanged from v2: Citrus A ledger (01/02/03); Citrus B oranges (154 confirmed on 20 trees; 04: 108 on 17) and the 04↔05 pairing; Klapmuts ledger v5 (825 Dec↔Apr pairs); Gwakungu cabbages (SAM3: 956 masks / 72 frames → 42 ids).

## 2. Work plan: what I will train and build

GPU budget: A100 40 GB. The renderer (4 GB) and Gemma (8 GB) stay resident, leaving
about 28 GB for jobs, run **one at a time** in a queue. The peer session's
H3DGS 01 re-chunk (about 36 h) **stays parked until after Tuesday**, and I only read its
Klapmuts ledger files, never change them.

### Training (GPU): the queue, 30 Sep ~13:50 SAST (one job at a time, `~/logs/demo_gpu_queue4_20260930.sh`)

| # | job | who | est. | status |
|---|---|---|---|---|
| 1 | Gwakungu cabbages, H3DGS 4 M | me | — | **done: 22.78 training / 18.32 held-out, fails the bar** |
| 2 | Citrus B side-car on expo 0_0 (identity) | peer | — | **done** (tree 90 IoU 0.93; lights through the stream) |
| 3 | Citrus A side-car on 01 chunk 3_1 | peer | — | **done** (tree 164 IoU 0.97) |
| 4 | Citrus B chunk 1_0: improved recipe, in-cell score, side-car, fruit densify — **no fitted cuts** | peer | — | **done 00:02 box**: 26.28 in-cell; trees lit; fruit containment 0.03 (not shown) |
| 5 | Gwakungu chillies, H3DGS IMG_7990_s1, **8 M** | me | — | **done: 21.92 training, fails** |
| 6 | Kendu Bay ground plants, H3DGS IMG_7975_s0, **8 M** | me | — | **done: 26.06 training, passes** → served |
| 7 | Gwakungu cabbages again at **8 M** (own project `h3dgs_8m`) | me | — | **done: 23.43 training; shown (Paul)**; embedder done |
| 8 | cabbage embedder (queue v8), then the cabbage side-car on the 8 M model if it clears 25 dB; Klapmuts Dec berries (SAM3); side-cars for any other Kenyan segment ≥ 25 dB | me | ~3 h | after 7 |

Why 1_0: the peer found that 05 chunk 0_0 holds 12 trees, of which only tree 90 (8 oranges) and tree 92 (2) carry fruit; tree 5 (61) and tree 3 (14) are in cell 1_0.
Tassili guard (ujamaa 27563f3): a plant more than 6 m from the served walk is "outside the 3D section": the answer is kept and points to the map, with no walk into empty space.
**Lighting rule (Paul, 30 Sep):** trees and fruit light through the containment field's own split; no per-object thresholds fitted to SAM masks ("not real relevancy"). If raw containment misfires, it is reported, not masked.

1_0 relaunched ~21:45 SAST as its own chain: fruit result ~02:30 SAST; jobs 5–7 finish ~Thu 10:00 SAST, 8 by ~Thu 13:00 SAST. Backlog (peer, after their plan): remaining 05 chunks on the improved recipe, the 01 45 m re-chunk.

Klapmuts Dec 3D stays `h3dgs_e2` (lane 2, half views): training-view median 29.72, held-out 19.82.
Every 3D shown must measure ≥ 25 dB (training views); each Kenyan segment is shown only if its evaluator says so.

### Not trained before Tuesday
- **Klapmuts April 3D.** Tried and reverted on 30 Sep: the per-block models are the damaged August era (~17 dB, rejected 09-23), and April H3DGS is ~18 dB, below the bar.
- **H3DGS 01 re-chunk.** Parked by Paul until after Tuesday; Citrus A's hero stays chunk 3_1.
- **Berries lit in 3D.** Needs a berry side-car on the Dec H3DGS model: days, not hours.
- **Gemma fine-tune.** G0 failed its generation gate; the demo runs base Gemma 4 12B.
- **Per-tree orange change on Citrus B.** Only ~2 trees were seen well in both surveys, so the demo uses farm totals (154 vs 108).

### Built (no training)

| id | work | when |
|---|---|---|
| **E1** ✅ 441bcc2 | **Adinkra drives the 3D.** Oranges, cabbages and plants are lit by asking in the chat ("show me the tree with the most oranges"), not by a button. Adinkra's Tassili actions gain a fruit target, the app forwards it to the renderer (`{"t":"query","fruit":10000+id}`, already supported), and per-farm facts are added (orange counts, berries, cabbage count) with farm-appropriate example prompts (Klapmuts currently says "trees… row 2") | Thu |
| **B2** | Citrus B Sankofa: 13D ledger from `assoc_04_05_v4` (04 ↔ 05, 128 canonical trees, confirmed fruit per epoch) | Thu |
| **C2** | Klapmuts Sankofa: Apr registry bushes ↔ ledger v5 bags (825 Dec↔Apr pairs), plus Dec berries from C3 | Thu |
| **G1** | Broker keeps every demo farm's 3D warm at once, so switching farms is instant | Thu |
| **G2** | Phone surveys: portrait stage instead of black bands | Fri |
| **E2** | Question bank per farm in 8 languages, answers checked | Fri |
| **F1** | Demo video re-cut from the live app (§4) | Fri → Sat |

The launch app is served on **:8011** (the :8001 server is the old viewer), so the Tuesday tunnel forwards 8011.

Every run is logged in the notebook and committed after it goes green. Known-good states are tagged
Monday (`demo-aicfs-2026-10-06`).

## 3. The live walkthrough (3 minutes, farm picked to suit the visitor) — as built, 30 Sep evening

Switching farms stages that farm's chunk on the render broker in ~12 s; talk over it ("loading the farm in 3D").

| visitor | farm | what you show | ask Adinkra (chat drives the 3D) |
|---|---|---|---|
| fruit / horticulture (**default**) | **Citrus farm B**, chunk 1_0 (from ~Thu 00:00) | walk the row; ask, and tree 5's oranges light by containment | "Which tree has the most oranges? Show me." → tree 5, 61 |
| monitoring / seasons | **Klapmuts** | Bateleur: 912 bags in 14 rows; Sankofa: 3 Dec 2025 → 15 Apr 2026, 825 found again (Dec lane in 3D, 29.7 dB) | "How many bushes were found again in April?"; Afrikaans per row |
| research / scale | **Citrus farm A**, chunk 3_1 | the 3D walk (28.2 dB); three surveys of one farm; a tree's vegetation index over time | "Are any trees getting worse over time?" |
| smallholder / Kenya | **Gwakungu** | only if the cabbage H3DGS clears 25 dB (see §1); otherwise the phone-survey story on the poster and the survey date in chat | "Shamba hili lilipimwa mara ya mwisho lini?" |

**Close (every visitor):** the lesson ("same farm, same data — the gap between languages was far bigger than we expected, even for simple questions") → hand over the keyboard in any of the 8 languages → ASK / OFFER → card / QR.
**Don't do live:** action requests ("mark tree 12…"); asking to see plants outside the 3D section (the app says so, but it is a dead end on stage). Cabbage count is now honest: 29 heads within 4 m of the walk, two rows (3-frame registry, 30 Sep) — say "counted along the survey path".
Question bank with expected answers: `automation/adinkra_www/demo_bank_20260930.json` (14 items, 89 question-language pairs) — run against Gemma when the GPU frees.

## 4a. Globe framing (Paul, 1 Oct night: "show a globe, go down to citrus farms with NDVI, then global, then Klapmuts, then global, then Gwakungu and Kendu Bay… local observations, local language tagline")

A globe (three.js, vendored, with a public-domain NASA Blue Marble texture) that zooms from space to each farm in turn, cross-fading into OUR top-down of that farm (Bateleur / NDVI dots), with that farm's observations and a tagline in its language, then back out to the globe, before the farm's walk. **Video only (Paul, 2 Oct morning): the globe is not part of Tassili or the launch app.** Built as a standalone page under ops `automation/demo/globe/` (three.js + vendored textures in `assets/`), rendered headless to frames for the assembly; the app's landing page stays as it is.

| farm | coordinates we hold | top-down to fade into | observations | tagline language |
|---|---|---|---|---|
| Citrus A / B | per-tree WGS84 in ledger_v2 (748 / 75 obs; Riverside, CA) + Sentinel-2 NDVI per tree (0.07–0.43) | NDVI dot map from the ledger | 290 / 100 trees, 3 surveys, 154 confirmed oranges | English (**location: see question 1**) |
| Klapmuts | ENU0 anchor from RTK (Aug) + WGS84 anchors + NDVI series (30 Aug) — file to confirm | 912 bags by row, Dec→Apr pairs | 912 bags, 825 found again | Afrikaans / isiXhosa |
| Kendu Bay | none in the data (phone clips carry no GPS) — **question 2** | 29-head style registry for the ground crop (none yet) | 61-frame phone walk, 26.1 dB | Swahili (Dholuo?) |
| Gwakungu | none — **question 2** | 29 cabbages in two rows | 29 cabbages counted along the path | Swahili (Kisii?) |

Decided (Paul, 1 Oct night): citrus zooms to the **continent only** (no location on screen); Blue Marble texture downloaded (NASA, public domain) for vendoring. Open: (2) coordinates for the two Kenyan farms (a pin at village scale). (3) taglines need native-speaker checks; Swahili/Afrikaans/isiXhosa drafts from the question set's vocabulary. Build: Friday, CPU + headless browser only, ~3–4 h; frames into the assembly as the transitions between farms.

## 4. The demo video (idle loop, 1080p, silent, captions) — **v1 cut exists (2 Oct 07:xx SAST): `ujamaa_demo_v1.mp4`, 196 s, 720p; `automation/demo/assemble_v1.py`.** Below is the target shape; v2 = Paul's card words, globe transitions, chosen Citrus B cell, re-shot Kenyan/Klapmuts walks at 8 fps, 1080p.

Frames are rendered offline at fixed poses along each farm's walk (render service / hier service), not screen-recorded — headless browser capture runs at 8–11 fps.

| time | shot | poster link |
|---|---|---|
| 0–8 s | landing → "Western Cape and Kenya" → choose a farm (the new agent cards: our own render, registry map, found-again pairs) | status |
| 8–30 s | Citrus B: "ask the orchard" reel — **0_0 for the drive** (Paul's v1 review: 1_0's turn poses are the grey frames); fruit segment per the peer's answer (v11 treatment vs v3 side-car). Was: chunk 1_0 (59 s cut in, `~/logs/demo_chunks/05_1_0/demo.mp4`; its cameras are the eastern turnaround, looking at row ends and open ground) or chunk 0_0 (lane drive like the 27 Sep reel; cut queued → `05_0_0/demo.mp4`) — **Paul picks the cell**; live stage stays on 1_0. **Oranges: fruit v3 (split boundary at the gate) gives FRUIT of 5 IoU 0.301** — real, below the 0.5 block record, far below the 1.000 oracle. Peer re-composites the 1_0 reel WITH fruit (640×360, then 720p → `05_1_0_720/`). **Paul decides from the reel whether oranges light in the video and on the live stage**; until then Adinkra answers 61 and Tassili lights tree 5 | story 1 |
| 30–50 s | Klapmuts: December lane in 3D → Bateleur's 912 bags by row → Sankofa December/April pairs, 825 found again | story 2 |
| 50–70 s | Adinkra: the orange question in Swahili, the found-again question in Afrikaans | story 3 |
| 70–82 s | Citrus A chunk 3_1 walk (28 dB) with a tree lit | scale |
| 82–90 s | ASK / OFFER, contact, ujamaa.ai | 3, 4 |

## 5. Timeline (revised Thu 1 Oct, 22:40 SAST)

**Done Thu:** four farms at the bar + cabbages by Paul's call, all verified live; trees lit on both citrus chunks; mean exposure on; portrait stage; question bank 87/89 grounded (2 output-length failures); detector rebuilt from the question sets; oracle containment 1.000 (fruit failure is the rendered field); densify gate fixed in HiGH; five raw shots captured.

| when | what | who |
|---|---|---|
| Thu night | peer's fruit re-test with the corrected gate on chunk 1_0 (~1 h); card otherwise idle | GPU |
| Thu night → Fri early | peer: fruit re-test v1 (metres gate alone) **changed nothing — the boost WINDOW (`fruit_densify_tail` 2000 on a 2000-step pass) is the operative blocker**; v2 (tail 0 + metres gate) queued after the reels; then the **"ask the orchard" reels** on chunks 1_0 and 3_1 (chunk mode of the 27 Sep chain: one-row drive at 8 fps, questions in Swahili + English, modes over time, map inset, mean exposure; fruit segment only if the re-test is honest) → `~/logs/demo_chunks/{05_1_0,01_3_1}/demo.mp4` | peer |
| **Fri morning** | 0. fruit v2 read: **failed honestly** (third model-unit term — `densify_size_thresh`; see §7). Reels 05_1_0 / 05_0_0 / 01_3_1 **fetched** (`…/demo_video_20261001/reels/`): recommend 0_0 for Citrus B; 640×360 → 720p re-render of the chosen cells; 3_1 blue cast to check against the live stage reels 05_1_0 / 05_0_0 / 01_3_1 (`~/logs/demo_chunks_run.log`) — fetch mp4s + contact sheets to `/Users/paulamayo/data/for_a100/demo_video_20261001/` 1. read the fruit result; oranges in Citrus B's live stage only if honest — Paul decides 2. live-stream walks at 8 fps, one pass each: Klapmuts lane, Kendu Bay, Gwakungu (registry names on screen) 3. landing / picker / two Adinkra exchanges captured from the browser 4. competitive tree scoring measured on 3_1 and 1_0 for the live stage (no thresholds) 5. cabbage side-car (colour bridge + name map + empty-fruit path, lane converter) | me |
| Fri midday | video first draft assembled: landing + picker captures, five walks, two Adinkra answers (Swahili oranges, Afrikaans found-again), captions, ASK/OFFER → Paul | me |
| Fri afternoon | Paul's feedback → v2; screenshot PDF; **Paul tests SSH from a phone hotspot** | me, Paul |
| Sat–Sun | Paul rehearses the four farm paths with a timer | Paul |
| Mon 5 Oct | freeze 12:00; tag `demo-aicfs-2026-10-06` in ujamaa, aru_sil_core, ops; dress rehearsal off campus on the tunnel (8011 + 8024); unstage/stage procedure rehearsed | Paul, me |
| Tue 6 Oct | morning procedure (§6): unstage → no training → stage the first farm → Gemma warm → one question per farm | Paul |

## 6. Tuesday morning and packing

- **06:30 (box):**
  1. `ollama ps` shows Gemma 100 % on the GPU.
  2. All 4 farms report `ready`.
  3. One question per farm.
  4. No training jobs.
- **Venue, 30 min before:**
  1. Connect the tunnel (ports **8011 and 8024**); stage the first farm.
  2. Ask the Swahili question.
  3. Start the video loop.
  4. Run `caffeinate -dis`; turn on Focus mode.
  5. Browser in fullscreen.
- **If it breaks mid-walk:** switch to the video; don't debug in front of a visitor.
- **Pack:**
  - laptop, charger, adapters
  - **a mouse**
  - phone with hotspot data
  - printed question card with the four farm questions
  - business cards / QR sticker
  - video + screenshot PDF saved locally and on a USB stick

## 7. Decisions for Paul (defaults apply unless you say otherwise)

1. **Klapmuts count.** If the Apr 3D was trained on a different registry than gen2's 517, the public
   Klapmuts number becomes that registry's count, so the map, the chat and the 3D agree. *Default: yes.*
2. **Naming Kenyan farms publicly** (Gwakungu, Kendu Bay). *Default: keep, as the site already does.*
3. **Laptop-only fallback** (Gemma on the M1 Pro). *Default: decide Friday, after your hotspot test; the video and screenshots are the fallback until then.*
4. **H3DGS 01 re-chunk** stays parked until after Tuesday. *Default: yes.*
5. **ujamaa.ai (the poster's QR)** currently serves a GoDaddy builder page. *Default: put the video and a contact line there on Monday.*

- **Oranges in 3D (2 Oct, 02:xx SAST):** three model-unit terms now found in the side-car fruit densify — boost window (fixed, tail 0), boost gate (fixed, metres), and gsplat's split threshold `densify_size_thresh` (0.01 model units = 0.28 m on chunk 1_0, 0.11 m on the HiGH blocks) which turns every boost into a clone and blew the card at 9.39 M gaussians. A fourth test (threshold converted through the dataparser scale, boost rounds capped so the count stays under ~7 M) is ~45 min of card and plausible, but it competes with Friday's video shots and nothing guarantees a fourth term isn't behind it. UPDATE 2 Oct ~03:00 SAST: the peer's v3 (split boundary moved to the gate, `FD_SPLIT_M`) ran bounded in 36 min: **FRUIT of 5 IoU 0.301** (6.67 M gaussians, no OOM). That is a real response from the containment field with no fitted cut. **Decision for Paul from the re-composited 1_0 reel (fruit segment) and a live-stage look Friday morning: oranges lit (0.30 — partial, honest) or tree-only (0.73).** Serving the fruit side-car means re-pointing `h3dgs_sidecar_demo_1_0/block_000` at the wm_t0s run (~10 min, same memory class).

