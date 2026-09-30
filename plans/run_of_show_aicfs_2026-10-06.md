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
| 1 | Gwakungu **cabbages**: H3DGS IMG_7993_s0 | me | ~1.5 h left | running |
| 2 | Citrus B side-car on expo 0_0 (identity) | peer | ~30 min | queued (`~/logs/sidecar_demo_queue.sh`) |
| 3 | Citrus A side-car on 01 chunk 3_1 | peer | ~30 min | queued |
| 4 | **Citrus B chunk 1_0 on the improved recipe** (trees 5 and 3: 61 + 14 oranges): train 2.3 h + post-opt 1 h + in-cell score, then side-car + fruit densify; becomes Citrus B's demo chunk (Paul, 14:1x) | peer | ~4.5 h | chained (`~/logs/sidecar_demo_queue_v2.sh`); walk root `h3dgs_demo/expo_chunk_1_0` ready |
| 5 | Gwakungu **chillies**: H3DGS IMG_7990_s1 | me | ~2.5 h | waits for the peer's done-file |
| 6 | Kendu Bay **ground plants**: H3DGS IMG_7975_s0 | me | ~2 h | after 5 |
| 7 | Klapmuts Dec **berries**: SAM3 counts per ledger bag | me | ~1 h | after 6 |
| 8 | side-cars on each Kenyan segment that clears 25 dB | me | ~20 min each | after its H3DGS |

Why 1_0: the peer found that 05 chunk 0_0 holds 12 trees, of which only tree 90 (8 oranges) and tree 92 (2) carry fruit; tree 5 (61) and tree 3 (14) are in cell 1_0.
Tassili guard (ujamaa 27563f3): a plant more than 6 m from the served walk is "outside the 3D section": the answer is kept and points to the map, with no walk into empty space.

The peer chain hands the GPU back ~22:00 box (~00:00 SAST); the Kenya runs, berries and side-cars finish ~Thu 08:00 SAST. Backlog (peer, after their plan): remaining 05 chunks on the improved recipe, the 01 45 m re-chunk.

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

## 3. The live walkthrough (3 minutes, the farm picked to suit the visitor)

All four farms stay warm (G1), so switching is instant.

**Hook (15 s):** "One walk through a farm, by robot or on a phone, becomes a 3D twin
you can question in your own language. Pick a farm."

| visitor | farm | what you show | ask Adinkra |
|---|---|---|---|
| smallholder / Kenya / food security (**default**) | **Gwakungu cabbages** | "filmed on a phone"; Tassili lights each cabbage; Bateleur counts them | Swahili: *Kuna kabichi ngapi shambani?* |
| fruit / horticulture | **Citrus farm B** | ask, and Tassili lights that tree's oranges | "Which tree has the most oranges? Show me." |
| monitoring / seasons | **Klapmuts** | Sankofa: the same bag in December and April; berries in December, none after harvest | Afrikaans or isiXhosa: "Which bushes had berries in December?" |
| research / scale | **Citrus farm A** | three surveys of one farm; a tree's history | "How many trees were in all three surveys?" |

**Close (every visitor):** state the lesson ("same farm, same data: the gap between
languages was far bigger than we expected, even for simple questions"), then hand
over the keyboard ("ask it anything, in any of these eight languages"), then ASK / OFFER, then a card or the QR code.

**Don't do this live:**
- action requests ("mark tree 12…")
- Kendu Bay clips until G2 lands
- berries in 3D

## 4. The demo video (idle loop, ~90 s, 1080p, silent, captions)

| time | shot | poster link |
|---|---|---|
| 0–8 s | landing page, "Western Cape and Kenya", choose a farm | status |
| 8–25 s | Gwakungu: phone walk → each cabbage lit, the count | story 1, OFFER: phones |
| 25–40 s | Citrus farm B: robot row, trees lit, then the oranges | story 1 |
| 40–60 s | Klapmuts: Sankofa, the same bag in Dec and Apr; berries in Dec | story 2 |
| 60–80 s | Adinkra: the Swahili cabbage question and answer; one in English | story 3 |
| 80–90 s | ASK / OFFER, contact, ujamaa.ai | 3, 4 |

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
  1. Connect the tunnel; stage every farm.
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
