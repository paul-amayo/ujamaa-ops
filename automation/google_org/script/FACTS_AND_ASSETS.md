# UJAMAA — verified facts and available assets for the Google.org Food Security Collaborative film (compiled 30 Oct 2026)

Everything below is from the project record (lab notebook, catalogue, measured runs). A script may use ONLY these facts. Anything not here is not claimable.

## Who / what
- UJAMAA = Unified Joint Agricultural Modelling with Artificial intelligence Agents (expand at first use; never gloss the Kiswahili word). UCT African Robotics Unit (ARU), University of Cape Town. Lead: Paul Amayo. Member of Google.org's Food Security Collaborative.
- The thesis in one line (Paul's): "Robotics opened the path. The phone walks it." The pipeline was built for a ground robot with camera + LiDAR; the same pipeline now runs on a phone walk, with no robot and no LiDAR.
- What UJAMAA does: builds a 3D digital twin of a farm from a survey (robot or phone), gives every plant an identity (which tree, which row), remembers plants across surveys (found again months later), counts fruit, and answers a farmer's questions about the farm in the farmer's own language — spoken to the 3D model, which lights up what the answer refers to.
- Named parts (the "agents"): Tassili = the 3D farm you walk through; Bateleur = the farm map / plant registry; Sankofa = plant history across surveys (found-again, change); Adinkra = the language assistant (answers in English, Afrikaans, isiXhosa, isiZulu, Kiswahili, Hausa, Arabic, Amharic — 8 languages). Adinkra runs on UJAMAA's own GPU machine, not a cloud service.
- How identity works (safe public phrasing): the 3D model carries meaning, not just colour — each point knows which plant and which row it belongs to, so a question lights exactly that plant. Technical name: identity by containment in a hierarchical 3D Gaussian model (HiGH / H3DGS). No per-object thresholds are tuned.

## Farms and numbers (all measured)
- Citrus farm A: ground-robot survey (camera + LiDAR), July 2023; 289 trees in 36 rows in the registry; 3D model of one section (the demo chunk) at 26.9 dB median reconstruction quality; identity IoU median 0.87 against hand-checked masks. Location: a citrus orchard — DO NOT name the place or country.
- Citrus farm B: ground-robot surveys, July 2023, 3 surveys; 100 trees; 154 oranges confirmed by hand across the farm; the tree with the most oranges has 61 (tree 5, "tree 3 of row 6/7" by position). The 3D model rings its oranges when asked. Location withheld as above.
- Klapmuts, Western Cape, South Africa: blueberry shade tunnels, ground-robot surveys (camera + LiDAR for both surveys, as the app's farm cards read; added 2 Nov 2026). December 2025: 911 grow bags; April 2026: 912 bags; 825 bags found again four months later, bag by bag. 3D model quality 29.0 dB median (the best-looking model).
- Kenya, May 2026 — PHONE surveys only (an ordinary smartphone, a walk down the bed; no robot, no LiDAR):
  - Gwakungu, Nyahururu (16 May 2026): a cabbage bed — 29 cabbages counted in two rows; a red-chilli bed — ripe red chillies detected and lit; an open field being prepared.
  - Gendia, Kendu Bay, Homa Bay County (13–22 May 2026): a ground crop in 61 frames (3D model from one short walk); tower planters; bean rows; seedling rows; raised beds with drip irrigation.
- Shown live: AIC Food Systems Summit 2026 gallery walk, 6 October 2026 (Pillar 3 poster); live 3D app driven over SSH from a laptop to the UCT GPU machine.
- A question bank of 89 question/language pairs was answered correctly from the farms' own data (87 of 89 completed; the two failures were output-length, not wrong answers). Example answers: "tree 5 has the most oranges, 61"; "825 of 912 bushes were found again"; honest refusals when the data does not exist (soil moisture, yield forecasts).

## 2023 → 2026 (vision vs execution)
- 22 May 2023, EBE Talks (UCT): Paul proposed "lightweight techniques ... deploy on cheaper and low-cost devices such as mobile phones", "robot-equipped mobile phones", robots' internal 3D models "interpreted by humans to provide 3D quantitative data", merged "with the deep domain knowledge that farmers possess". Transcript with timestamps: _script/ebe_talk_2023_transcript.txt. His AUDIO may be used as narration; his picture must not.
- 2023: the work needed a robot (Jackal with camera + LiDAR) in a citrus orchard. 2025: Klapmuts tunnels, robot. May 2026: the same pipeline from a phone walk on Kenyan smallholdings. October 2026: answering questions in 8 languages, live.

## Honesty rules for the film
- No claim beyond the facts above. No "farmers are using it" (they are not yet — pilots and surveys so far; the Kenyan plots were surveyed by the team with a phone). No yield/income impact claims. Fine to say: "the first phone-only farm models", "answers questions in the farmer's own language", "found again", "counted".
- Faces: field footage has people; blurred versions are the only ones usable. Robot footage from the ICRA video is UNDER REVIEW (anonymous) — do not use unless Paul clears it; the script may call for a robot shot and mark it "pending clearance / substitute: Klapmuts 3D render".
- Tone: Paul's voice (see /Users/paulamayo/.claude/skills/how-paul-writes/SKILL.md): plain, declarative, concrete, no hype, no point–counterpoint, no exclamation. Message over readouts: a few numbers as landmarks, not a dashboard. Write "Section 3" never "§" (irrelevant here, but no symbols in VO). The narration will be RECORDED BY PAUL — write it to be read aloud by him in the first person plural ("we").

## Assets (all on the Mac under /Users/paulamayo/data/for_a100/google_org_2026-11/; durations exact; 1080p unless stated)
FIELD (phone, 4K, no audio, faces blurred where marked):
- 01_field/06_gwakungu_open_field.mp4 — 23.3 s, 2160x3840 vertical, BLURRED. Open red-soil field with drip lines, big sky, 16 May 2026.
- 01_field/08_gwakungu_cabbages.mp4 — 21.0 s, 2160x3840 vertical, BLURRED. Walking along a cabbage bed with drip pipes, heads visible.
- 01_field/07_gwakungu_chillies.mp4 — 40.0 s, 2160x3840 vertical, BLURRED (arriving). Down a chilli bed, red pods.
- 01_field/01_kendu_towers.mp4 — 30.0 s, 2160x3840 vertical, BLURRED (arriving). Tower planters with leafy veg, Kendu Bay 13 May.
- 01_field/02_kendu_beds_landscape.mp4 — 35.0 s, 3840x2160 LANDSCAPE, BLURRED (arriving). Raised beds with drip irrigation.
- 01_field/03_kendu_bean_rows.mp4 — 30.0 s, 1080x1920 vertical (blur pending). Bean rows.
- 01_field/04_gendia_ground_crop.mp4 — 30.0 s, 1080x1920 vertical (blur pending). The ground crop that became the Gendia 3D model.
- 01_field/05_kendu_seedling_rows.mp4 — 30.0 s, 2160x3840 vertical (blur pending). Seedling rows with drip lines.
VOICE (Paul, 2023, with audio; picture not to be shown):
- 02_voices/_source/ebe_talk_2023_amayo.mp4 — 320 s, full talk. Any passage may be cut by timestamp (see transcript).
- 02_voices/01_amayo_collaborative_approach.mp4 (22.0 s, 36.0–58.0), 02_amayo_lowering_the_barrier.mp4 (19.5 s, 156.1–175.6), 03_amayo_robot_equipped_phones.mp4 (23.3 s, 290.0–313.3).
3D RENDERS (clean, no text; coloured highlights = identity):
- 03_3d_renders/01_citrus_a_orchard_query.mp4 — 47.5 s, 1280x720. Drive down a citrus lane; segments in order (re-measured from frames 2 Nov 2026): plain walk 0–5.6 s; one ROW lit blue 5.6–14.6; EVERY tree lit in its own colour 14.6 to at least 33; two colours (purple left, green right) by 36 and through 41.8 — there is NO single-tree segment in this file (the earlier "one TREE lit 37.4–41.8" was wrong); WHOLE ORCHARD lit green 41.8–47.5.
- 03_3d_renders/02_citrus_b_oranges_lane.mp4 — 26.57 s, 1280x720, 4 fps feel. Re-measured from frames 2 Nov 2026: plain 0–2.5; ORANGES ringed magenta on tree 5, faint from 2.5, clear by 2.7, to 8.0 (the hero moment); rows grouped 8.0–12.0; whole orchard green 12–15; one row 15–18.8; one tree orange 18.8 to beyond 22.5 (the only single-tree picture in the renders); every tree by 23.0–26.57. (Note: a few frames are softer; the lane is continuous.)
- 03_3d_renders/03_klapmuts_tunnel_walk.mp4 — 27.75 s, 1280x720. Smooth walk down the blueberry tunnel, bags both sides. Best-looking model. No highlights. Clean 1–21.5 s; exit artefacts after 22 s.
- 03_3d_renders/04_gendia_ground_crop_walk.mp4 — 6.375 s, 1080x1920 vertical. Phone-built 3D of the ground crop. No highlights.
- 03_3d_renders/05_gwakungu_cabbages_query.mp4 — 9.0 s, 1080x1920 vertical. Phone-built 3D cabbage bed; heads/rows light (purple/blue) from ~2 s.
- 03_3d_renders/06_gwakungu_red_chillies.mp4 — 21.6 s, 540x1280 vertical (centre crop). Red chillies ringed magenta throughout.
APP (screen recordings, 1920x1080, silent; the 3D inside moves at 1–3 fps):
- 04_app/01_picker.mp4 — 41.4 s. "Choose a farm" page: Citrus A, Citrus B, Klapmuts, Kendu Bay, Gwakungu cards. Measured 2 Nov 2026: the Adinkra card with all eight language chips is fully in frame from 30.0 s until the page scrolls to the top at about 37.3 s, after which the chips are out of frame.
- 04_app/02_citrus_b_oranges.mp4 — 59.0 s. Citrus B 3D; measured from frames 2 Nov 2026: typing starts at about 11.8 s (box empty at 11.5, "N" at 12.0), the Kiswahili question ("Ni mti gani wenye machungwa mengi zaidi? Nionyeshe.") is complete by 16.5 s and sent at 17.6 s; the assistant's "reading the farm record" wait 17.6–46.65 s; answer in Kiswahili appears at 46.65 s ("Mti wenye machungwa mengi zaidi ni mti 5, wenye machungwa 61.") and the tree 5 record opens at the same moment; 3D walks to tree 5 and lights it 47–59 s.
- 04_app/03_cabbage_query.mp4 — 60.0 s. Phone-built cabbage bed in the app (portrait pane); "How many cabbages are there?" → "29"; "Which cabbage is in front of me? Show me." → measured from frames 2 Nov 2026: the answer lands at 49.85 s and the record that opens is cabbage 12 ("HIGHLIGHTED CABBAGE 12", row 1); one head lights orange at 51.9 s and holds to the end.
- 04_app/04_bateleur_map.mp4 — 25.8 s. Klapmuts map: 911 bushes in 14 rows, coloured by row, a bush record opened. Measured 2 Nov 2026: grey dots 0–4.63 s, row colours from 4.63 s, the bush 3 record opens between 10.3 and 10.5 s (its SEEN IN and FIRST SEEN fields are blank); the page header reads "Shade-tunnel lane · Dec 2025" and the stats card PLANTS 911 (the December count), so a shot under the narration's 912 must crop them out.
GLOBE (1920x1080, no text, NASA Blue Marble):
- 05_globe/earth_idle.mp4 6 s (slow spin, Africa centred); klapmuts.mp4 6 s zoom to Klapmuts (pin); gendia.mp4 6 s zoom to Gwakungu/Nyahururu (pin); kendu_bay.mp4 6 s zoom to Gendia/Kendu Bay (pin); citrus.mp4 6 s — measured 2 Nov 2026: it zooms from Africa to NORTH AMERICA, not to continent scale; NEVER USE; *_out.mp4 = reverse pull-outs; tour.mp4 42 s = idle→klapmuts→out→gendia→out→kendu→out (visible ~9° jump at each join).
FINISHED: 06_finished_video/ujamaa_demo_2026-10-20.mp4 — 181 s, the summit demo film (title cards, question banners). Can be mined for moments but carries text.
NOT AVAILABLE (yet): footage of the robot itself (ICRA video under review; robot's-eye video could be exported from the bags in ~1 h); footage of farmers/people speaking; the Klapmuts Sankofa history page (empty in the app); any 2023 field footage.
