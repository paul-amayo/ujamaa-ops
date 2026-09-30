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

| id | work | kind | GPU | when |
|---|---|---|---|---|
| **D1** | Gwakungu cabbages: SAM3 prompt probe ("cabbage" / "cabbage head"), then `image_farm_recipe.sh --prompt <best>` with `IF_MODE=high` on IMG_7993_s0: markers → embedder → HiGH features → registry | **train** | ~2 h | Wed |
| **C1** | Klapmuts Apr 3D: stage manifests for the 25 blocks (`stage_splat_manifests.py`); check the models' identity ids against the registry; add a render entry | stage + verify | minutes | Wed |
| **B1** | Citrus B oranges: fruit queries on the served 05 blocks along the demo row (018–023). If weak, run `fruit_chain.sh` (fruit densify + reseed) on those blocks | verify → **train** | 0 or ~3 h (overnight) | Wed → Thu |
| **C3** | Klapmuts Dec berries: SAM3 'berry' on the Dec keyframes (recipe of 08-08), counts per ledger bag | **SAM3 pass** | ~1 h | Thu |
| **D2** | Kendu Bay: prompt probe on the tower (IMG_7961_s2) and bean segments; the best crop through the same chain as D1 | **train** | ~2 h | Thu |
| **B2** | Citrus B Sankofa: 13D ledger from `assoc_04_05_v4` (presence, canopy, confirmed fruit per epoch); survey 04 added to the catalogue | CPU | — | Thu |
| **C2** | Klapmuts Sankofa: registry bushes ↔ ledger v5 bags (≤ 0.5 m); history "found in Dec and Apr", plus Dec berries | CPU | — | Thu |
| **E1** | Adinkra: per-farm sources (orange counts, berries, cabbage count). Question bank per farm in 8 languages, answers checked | CPU + Gemma | — | Fri |
| **G1** | Broker: keep **every demo farm's 3D warm at once** (one renderer per farm, ~5 GB each), so switching farms is instant instead of ~20 s | code | ~20 GB resident | Thu |
| **G2** | Phone surveys: the stage adopts the clip's portrait aspect instead of black bands | front end | — | Fri |
| **F1** | Demo video re-cut from the live app (§4) | capture | — | Fri → Sat |

Every run is logged in the notebook and committed after it goes green. Known-good states are tagged
Monday (`demo-aicfs-2026-10-06`).

**Out of scope before Tuesday, stated honestly:**
- Berries lit in 3D on Klapmuts Dec. That needs a side-car with berry supervision on the H3DGS model, which is days of work, not hours. Berries appear as counts and history instead.
- Per-tree orange *change* on Citrus B. Only about 2 trees were seen well in both surveys (08-29 v4 verdict), so the honest comparison is farm totals (154 vs 108).

## 3. The live walkthrough (3 minutes, the farm picked to suit the visitor)

All four farms stay warm (G1), so switching is instant.

**Hook (15 s):** "One walk through a farm, by robot or on a phone, becomes a 3D twin
you can question in your own language. Pick a farm."

| visitor | farm | what you show | ask Adinkra |
|---|---|---|---|
| smallholder / Kenya / food security (**default**) | **Gwakungu cabbages** | "filmed on a phone"; Tassili lights each cabbage; Bateleur counts them | Swahili: *Kuna kabichi ngapi shambani?* |
| fruit / horticulture | **Citrus farm B** | "show me the oranges": fruit lit in 3D; the tree with the most | "Which tree has the most oranges?" |
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
