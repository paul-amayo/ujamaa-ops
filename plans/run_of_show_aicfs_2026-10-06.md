# Run of show: AIC FS 2026 gallery walk, live demo (Tuesday 6 Oct, morning)

Status: v2 for Paul, 2026-09-30. Supersedes v1, which assumed the GPU was shared with training.
The GPU is now free, so this version adds a **work and training plan** that
gives every demo farm the data each agent needs. The demo accompanies the Pillar 3 poster
(v13); its three stories are **plants individuated in 3D, the same plant tracked over time,
and questions answered in the farmer's language**.

## 1. What each farm has today, and what it will show on Tuesday

Inventory measured on the A100, 2026-09-30.

| farm | 3D (Tassili) | registry (Bateleur) | over time (Sankofa) | fruit / crop | gap to close |
|---|---|---|---|---|---|
| **Citrus farm A** (13B, survey 01) | per-block, 28–33 fps | 290 trees | ledger v2: surveys 01/02/03 (289/177/303 obs) | none (0 fruit trees on 13B) | none. Also the H3DGS chunk 3_1 hero, 28.2 dB (peer session) |
| **Citrus farm B** (13D, surveys 04 + 05) | 05 per-block, 43 blocks | 05 registry, 3D-confirmed **oranges** per tree (154 on 20 trees; 04: 108 on 17) | pairing 04↔05 exists (`assoc_04_05_v4`, 128 canonical trees). **Not wired**: catalogue has 05 only, no ledger survey | oranges | B1 fruit lighting on the served models; B2 Sankofa ledger for 04↔05 |
| **Klapmuts** (blueberries in grow bags) | Dec 2025 H3DGS lane (11 fps). **Apr 2026: 25 trained blocks, not staged** | Apr gen2, 517 | **ledger v5 (today, peer session): Dec↔Apr, 825 bags matched, 0.23 m median** | **berries in Dec** (SAM3 recipe measured 08-08); none in Apr | C1 stage Apr 3D; C2 Sankofa from ledger v5; C3 berry counts per bag in Dec |
| **Gwakungu** (Kenya, phone) | cabbage bed 22.5 dB, open field 25.9 dB | none (SAM3 'tree' finds nothing on vegetables) | single date | **cabbages** | D1 cabbage registry (retrain with crop prompt) |
| **Kendu Bay** (Kenya, phone) | 3 segments (towers, beans, seedlings), ~21 dB | none | single date | kale / towers, beans | D2 one crop registry (towers or kale) |

**Rule for the demo:** each agent quotes the same numbers. Bateleur's count, Sankofa's
matches and Adinkra's answers come from one registry per survey: the one the 3D identity
field was trained on.

## 2. Work plan: what I will train and build

GPU budget: A100 40 GB. The renderer (4 GB) and Gemma (8 GB) stay resident, leaving
about 28 GB for jobs, run **one at a time** in a queue. The peer session's
H3DGS 01 re-chunk (about 36 h) **stays parked until after Tuesday**, and I only read its
Klapmuts ledger files, never change them.

### Training (GPU), status as of 30 Sep 11:30 box

| id | what is trained | why (what the visitor sees) | GPU | status |
|---|---|---|---|---|
| **D1** | Gwakungu cabbages: HiGH identity model with SAM3 prompt "cabbage" (IMG_7993_s0, copy dir `_cabbage`, served model untouched) | each cabbage lit on its own in Tassili; Bateleur counts them; Adinkra answers "how many cabbages" | ~2 h | probe good (956 masks / 72 frames). **Run 1 failed at the lifting gate**: stale SAM3 outputs reused from the copy, and all 654 detections dropped by the 30-unit distance cut. Diagnosing |
| **D2** | Kendu Bay: the same chain on one more crop (towers: "pot" 103 masks; "plant" too dense at 5632) | a second Kenyan farm with a registry | ~2 h | after D1 works |
| **C3** | Klapmuts Dec berries: SAM3 "berry" detection pass over the Dec keyframes (the 08-08 recipe), counted per ledger bag | "which bushes had berries in December" | ~1 h | queued |
| **B1** | Citrus B oranges: fruit densification + reseed on the demo row blocks (018–023), **only if** the fruit-lit test render looks weak. Recorded fruit IoU on the served blocks is 0.48–0.60 | oranges lit in 3D | 0 or ~3 h | test render in progress |

### Not trained before Tuesday
- **Klapmuts April 3D.** Tried and reverted on 30 Sep: the only per-block models are the damaged August era (~17 dB held-out, rejected 09-23), and the H3DGS April work sits at ~18 dB against the 20 dB goal. Klapmuts 3D stays December.
- **H3DGS 01 re-chunk.** Parked by Paul until after Tuesday; Citrus A's hero stays chunk 3_1.
- **Berries lit in 3D.** Needs a berry side-car on the Dec H3DGS model: days, not hours.
- **Gemma fine-tune.** G0 failed its generation gate; the demo runs base Gemma 4 12B.
- **Per-tree orange change on Citrus B.** Only ~2 trees were seen well in both surveys, so the demo uses farm totals (154 vs 108).

### Built (no training)

| id | work | when |
|---|---|---|
| **E1** | **Adinkra drives the 3D.** Oranges, cabbages and plants are lit by asking in the chat ("show me the tree with the most oranges"), not by a button. Adinkra's Tassili actions gain a fruit target, the app forwards it to the renderer (`{"t":"query","fruit":10000+id}`, already supported), and per-farm facts are added (orange counts, berries, cabbage count) with farm-appropriate example prompts (Klapmuts currently says "trees… row 2") | Thu |
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
