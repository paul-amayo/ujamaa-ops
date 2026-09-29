# UJAMAA website — brief for the next design pass

African Robotics Unit, University of Cape Town · 29 September 2026

We are taking the UJAMAA site to public launch at **ujamaa.ai**, alongside an
open-source release in December 2026. The May design bundle is the starting
point, but the product has moved on since then, and only part of the site is
real. This brief says what is live, what is still a mockup, what the design
can connect to, and what must not change.

## 1. The site today

One server hosts every page on the same origin:

| page | route | state |
|---|---|---|
| Research Site (landing) | `/research` | **Mockup**: the May design, unchanged |
| Workspace (pillar picker) | `/workspace` | **Mockup** |
| Tassili workspace | `/tassili/` | **Live**: see below |
| Adinkra chat (docked in every workspace) | — | **Half-wired**: see below |
| Bateleur, Spoor, Azalai, Hapi | `/bateleur` … | **Mockups**: procedural canvases, no real data |
| Sankofa | — | **No page**: absent from the May design entirely |

**Tassili is the only real workspace.** Its canvas is a live 3D walkthrough of
a real surveyed farm. The GPU renders it server-side and streams it to the
browser at about 10–17 frames per second. Working features:
- Play, pause and reset, plus a walking-pace selector.
- The walkthrough replays the path the survey vehicle or phone actually took.
- A query box. Typing something like "find the tree" lights up matching plants
  in 3D: user-driven segmentation on top of real-time rendering.
- Several surveys can be served: citrus rows, berry shade tunnels, and
  phone-captured vegetable beds with tower planters.

The live Tassili page is the compiled Workspace mockup with the real viewer
wired into its Tassili tab. It's in `live_site_tassili/index.html`, where
`TassiliCanvas` mounts the viewer. If you redesign the Tassili workspace,
design around that live canvas; don't replace it with an illustration.

**Adinkra's chat is only half-connected.** "Find X"-style messages are routed
to Tassili and highlight plants in 3D. Every other question gets a canned
reply ("the digital-twin endpoint isn't connected…"), because the chat was
built to answer through the Claude Design sandbox. The real Adinkra backend
exists and has been evaluated (see §4), but nothing links the chat to it.
**This is the most important functional gap before launch.**

## 2. The pillar model has changed since May — design decision needed

May bundle: 01 Tassili · Reconstruction, 02 Bateleur · State estimation,
03 Spoor · Localisation, 04 Azalai · Navigation, 05 Hapi · Inputs,
06 Adinkra · Language.

What the programme now says publicly (roadmap, summit poster):
- **Tassili**: the farm fixed in time, as an explorable 3D reconstruction.
- **Bateleur**: the registry. Every plant identified, positioned and assigned
  its row, across the whole farm, seen from above.
- **Sankofa**: memory. The same plant recognised across survey dates, with its
  change tracked over time.
- **Adinkra**: language. Farmers ask questions in their own language and it
  routes each one to the right specialist agent.
- **Spoor, Azalai, Hapi** remain design-stage and should be clearly labelled
  "coming".

Asks: add Sankofa as a pillar, reframe Bateleur as the registry, and mark
Spoor, Azalai and Hapi as coming.

## 3. Who it's for, and the voice

- **Users:** farmers, farm workers and extension officers. Assume no GIS or
  3D skills.
- **Scale:** from small gardens to hectare-scale plots. Capture comes from
  ground, legged and humanoid robots, platform-mounted sensors, and mobile
  phones.
- **Vocabulary:** say "farm", not "orchard". Use plain agricultural language
  and no pipeline terms.
- **Privacy:** no farm names, owners or coordinates on public pages.

## 4. What the design can connect to (real endpoints)

**Tassili viewer.** `initViewer({mountEl, embed, params})` returns an object
with `play()`, `pause()`, `reset()`, `setPace()`, `playing` and `on(event)`.
- URL options: `stream_url`, `traj_stride`, `speed_scale`, `start_frac`,
  `query`, `block`.
- Window events that sibling components use: `tassili:relevancy_query`,
  `tassili:relevancy_clear` and `tassili:relevancy_status`.

**Scene data** (all GET):
- `/scene/hierarchy`: plants (`Obj_NN`, 3D position), row membership and sections.
- `/scene/path` and `/scene/trajectory`: the survey route.
- `/scene/highlights`, `/scene/config` and `/scene/tree_to_block`.

**Adinkra**, which answers from the farm's records:
- Request: `POST /query {query, panel, lang}`, where `panel` is one of
  `bateleur | tassili | sankofa | azalai`.
- `lang` is one of `en af xh zu sw ha ar am` (English, Afrikaans, isiXhosa,
  isiZulu, Kiswahili, Hausa, Arabic, Amharic).
- Response: `{reasoning, command: {action, target, coords?}}`. `reasoning` is
  the answer text. `action` is one of `answer | explain | goto | highlight |
  locate | compare | trend`, so an answer can come with a "take me there" or
  "show it" step.

The chat UI must also handle two states:
- **Waiting:** answers take roughly 10–30 seconds depending on language.
  That's by design, because the model runs locally.
- **Honest failure:** when the model can't answer, it says so in the user's
  language. That is a feature to present, not an error to hide.

**Sankofa's data.** Per-plant observations across survey dates: position,
vegetation-index change, and the biggest movers. Canopy volume and height are
not in the data yet, so the design must say so rather than imply them.

## 5. Keep

- **Design tokens** in `colors_and_type.css`: heritage navy, harvest gold,
  terracotta clay, leaf green, sky and the soil neutrals.
- **Fonts:** DM Serif Display, Manrope and JetBrains Mono.
- **The Adinkra chat card:** paper card with a soft shadow, green status dot,
  "Adinkra" in the display serif and a mono "06 · LANGUAGE" badge. Dark user
  bubbles sit on the right, warm-grey twin bubbles on the left. It ends with
  an input reading "Ask the twin in any language…" and a SEND button.
- **Untranslated identifiers:** plant IDs (`Obj_NN`), row numbers and metric
  names stay as they are in every language.

## 6. Suggested scope, in priority order

1. **Adinkra, properly wired.** A language picker (or auto-detect), a waiting
   state, the answer plus action chips such as "Take me there" or "Highlight",
   and a no-answer state. It should work from every workspace.
2. **A Sankofa workspace.** Pick a plant or a row and see how it has changed
   between surveys.
3. **Bateleur as the registry.** A top-down farm view where you click a plant
   to see its record, and one click opens it in Tassili or Sankofa.
4. **A Tassili survey switcher**, so visitors can move between the farms and
   crops we've captured.
5. **Landing-page refresh for launch.** Launch messaging, "coming" labels, and
   links to the open-source docs (Quickstart, Install) and publications.

Paul will confirm or reorder these.

## 7. Files in this handover

- `live_site_tassili/`: the **current live front end**, copied from the
  server on 29 Sep. It's the Workspace shell with the real viewer wired in.
  - `index.html`: the Workspace shell; the wiring is in `TassiliCanvas`,
    `buildTassiliViewerParams` and `AdinkraChat`.
  - `main.js`, `walkthrough.js`, `stream_layer.js`: the 3D viewer.
  - `styles.css`: the viewer's styles.
- `design_source/`: the May design bundle sources, including tokens, CSS,
  `workspace-mockups.jsx`, `components/`, `workspace/`, `assets/` and the
  i18n seed. **`live_site_tassili/index.html` is newer than these** wherever
  they overlap.

**On handback:** export the full bundle with file names kept. For anything in
the Tassili tab, say what should change and we'll port it into the live viewer.
