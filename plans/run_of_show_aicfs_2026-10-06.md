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

| farm | H3DGS model | measured | passes 25 dB? | demo 3D |
|---|---|---|---|---|
| **Citrus farm A** (01) | improved recipe (12 M cap, 60k/45k, exposure); **5 of 24 chunks** (7_2, 6_2, 2_2, 2_0, 3_1) | in-cell 31.09 / 30.53 / 28.23 (7_2 / 2_0 / 3_1) | **yes**, on built chunks | the walk stays inside the built chunks; hero 3_1 |
| **Citrus farm B** (05) | original recipe, 6 chunks (09-22); improved recipe on chunk 0_0 only | original 18.97 held-out (whole survey); **0_0 improved 27.54 full / 28.07 sky-masked** in-cell | original no; improved yes | **train the other 5 chunks on the improved recipe** |
| Citrus farm B (04) | original recipe, 6 chunks (09-27) | not scored at the bar | — | none needed: 04 appears in Sankofa only |
| **Klapmuts** Dec lane 2 | arm E, 4 M | training 25.13, held-out 21.82 | borderline | Paul reports a 30 dB Klapmuts H3DGS; waiting on which run |
| Klapmuts Apr | 4 chunks | ~17–18 held-out | **no** | none |
| **Gwakungu / Kendu Bay** | none (phone per-block only) | per-block ns-eval 18–26 | no H3DGS yet | **H3DGS per segment** (single chunk, improved recipe) |

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
| 2 | Citrus B side-car on expo 0_0 (identity) | peer | 25–55 min | running (started 15:09 box) |
| 3 | Citrus A side-car on 01 chunk 3_1 | peer | 25–55 min | queued (peer chain v3) |
| 4 | Citrus B chunk 1_0: improved recipe, in-cell score, side-car, fruit densify, per-fruit containment cuts | peer | ~4.5 h | queued (peer chain v3) |
| 5 | Gwakungu chillies, H3DGS IMG_7990_s1, **8 M** | me | ~2.5 h | queue v5, after the peer's done-file |
| 6 | Kendu Bay ground plants, H3DGS IMG_7975_s0, **8 M** | me | ~2 h | queue v5 |
| 7 | Gwakungu cabbages again at **8 M** (own project `h3dgs_8m`) — Paul 17:1x | me | ~2.5 h | queue v6, after v5 |
| 8 | Klapmuts Dec berries (SAM3) + Kenyan side-cars for any segment ≥ 25 dB + the cabbage depth-scale fix | me | ~2 h | after 7 |

Why 1_0: the peer found that 05 chunk 0_0 holds 12 trees, of which only tree 90 (8 oranges) and tree 92 (2) carry fruit; tree 5 (61) and tree 3 (14) are in cell 1_0.
Tassili guard (ujamaa 27563f3): a plant more than 6 m from the served walk is "outside the 3D section": the answer is kept and points to the map, with no walk into empty space.

The peer chain hands the GPU back ~22:00 box (~00:00 SAST); jobs 5–7 finish ~Thu 09:30 SAST, 8 by ~Thu 11:30 SAST. Backlog (peer, after their plan): remaining 05 chunks on the improved recipe, the 01 45 m re-chunk.

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
**Don't do live:** action requests ("mark tree 12…"); asking to see plants outside the 3D section (the app says so, but it is a dead end on stage); cabbage counts (registry failed its check, 30 Sep).
Question bank with expected answers: `automation/adinkra_www/demo_bank_20260930.json` (14 items, 89 question-language pairs) — run against Gemma when the GPU frees.

## 4. The demo video (idle loop, ~90 s, 1080p, silent, captions)

Frames are rendered offline at fixed poses along each farm's walk (render service / hier service), not screen-recorded — headless browser capture runs at 8–11 fps.

| time | shot | poster link |
|---|---|---|
| 0–8 s | landing → "Western Cape and Kenya" → choose a farm (the new agent cards: our own render, registry map, found-again pairs) | status |
| 8–30 s | Citrus B chunk 1_0: walk the row; "which tree has the most oranges?" → tree 5's oranges lit | story 1 |
| 30–50 s | Klapmuts: December lane in 3D → Bateleur's 912 bags by row → Sankofa December/April pairs, 825 found again | story 2 |
| 50–70 s | Adinkra: the orange question in Swahili, the found-again question in Afrikaans | story 3 |
| 70–82 s | Citrus A chunk 3_1 walk (28 dB) with a tree lit | scale |
| 82–90 s | ASK / OFFER, contact, ujamaa.ai | 3, 4 |

## 5. Timeline

| day | GPU queue | build / capture | Paul |
|---|---|---|---|
| **Wed 30 Sep** | D1 probe + chain; C1 staging; B1 check | plan v2; peer session told | read v2 |
| **Thu 1 Oct** | C3 berries; D2; B1 densify (overnight, if needed) | B2, C2, G1 | **SSH to the A100 from a phone hotspot** |
| **Fri 2 Oct** | — | E1 question bank ×8 languages; G2; video capture → v1; screenshot PDF | review video v1 |
| **Sat–Sun** | — | video v2 | rehearse the 4 farm paths with a timer |
| **Mon 5 Oct** | freeze 12:00 | tag; dress rehearsal off campus | pack |
| **Tue 6 Oct** | — | 06:30: all 4 farms staged + Gemma warm + smoke test | present |

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
