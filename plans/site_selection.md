# Site selection: every demo starts at the UJAMAA landing page

Status: DRAFT for Paul, 2026-09-29. Follows the launch-workspace wiring
(ujamaa 840db86, aru_sil_core 00dbed67), which pins every service to one
survey (citrus farm B, survey 01).

## What Paul asked for

The final website lets a visitor choose a site. The demo should do the same:
start at the UJAMAA landing page, choose a site (Klapmuts takes you to the
Klapmuts surveys, and so on), then open the workspaces scoped to that
site. The right checkpoint is brought up on demand instead of being spun up
by hand beforehand.

## Flow

```
ujamaa landing page (/)        ->  "Explore a farm" (site cards)
  -> site page (/app/#/site/klapmuts)   survey list for that site, what each has
     -> workspace picker scoped to one survey
        -> Tassili / Bateleur / Sankofa, all on that survey; Adinkra reads the same survey
```

Every URL carries the site and the survey, so any demo can be deep-linked
and shared.

## What each site can show today (measured 2026-09-29)

| site | survey | registry | 3D checkpoints | change over time |
|---|---|---|---|---|
| Citrus farm B | 01 (Jul 2023) | 290 plants / 40 rows | 142 per-block | ledger, 3 surveys |
| | 02, 03 | 182 / 309 plants | 27 / 61 per-block | same ledger |
| Citrus farm D | 04, 05 | 107 / 100 plants | 1 / 88 per-block, H3DGS | ledger, 1 survey |
| Klapmuts berry farm | Apr 2026 | 481 plants / 16 rows | 51 per-block, 5 H3DGS runs | none yet |
| | Dec 2025 (ten rows) | none | H3DGS lanes (latest 21.8 dB) | none yet |
| Smallholder plots (phone) | image_farm, 44 segments | none | per segment | none |
| | Kenya (osuga, sukuma) | none (sukuma: 28 towers, v2) | per survey | none |

A workspace appears for a survey only if the data exists: no registry means
no Bateleur, and a single survey means no Sankofa. The page says so rather
than showing an empty panel.

## Build

1. **Site catalogue.** `surveys.json` becomes `sites.json`: sites contain
   surveys, and each survey declares its registry, ledger, data root and 3D
   backend (per-block checkpoint glob, or H3DGS project). Only public fields
   leave the server.
2. **Per-survey APIs.** `/api/*` already takes `?survey=`. Add
   `/api/path?survey=` so the walking path no longer depends on which data
   root the viewer server was started with.
3. **Adinkra per survey.** The request carries `survey`, and Adinkra loads
   that survey's registry, scores and ledger from the catalogue. No
   per-survey restart.
4. **3D on demand: the render broker.** `/api/stage?survey=` makes sure a
   render backend for that survey is running, starting one if needed and
   stopping the least-recently-used one when VRAM runs low, then returns the
   stream address. The stage shows "Loading this farm in 3D…" while a block
   loads (~20 s).
5. **Landing page.** The Research Site mockup gets a real "Explore a farm"
   section: one card per site, each with a real render frame instead of the
   stand-in photos.

## Decisions for Paul

1. **GPU policy for on-demand 3D.** One A100 is shared with training.
   Options: (a) the demo may start one render backend at a time (~2–4 GB),
   which can slow training; (b) 3D only when no training job is running,
   otherwise the stage says it's busy; (c) a separate serving GPU (the
   deployment plan's P2).
2. **Which sites are public at launch.** The citrus farms are in the US
   (Riverside, CA, per the ledger coordinates). Show them as "Citrus
   farm B/D", with no location?
3. **Landing page scope.** Just a site-choice section on the existing
   Research Site mockup, or a Claude Design pass on the landing page first?
