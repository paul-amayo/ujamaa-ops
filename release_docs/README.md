# UJAMAA

**A digital twin for the orchard — reconstruction, monitoring, and a
multilingual language interface, built for African agriculture.**

UJAMAA turns a vehicle-mounted camera + LiDAR survey of an orchard into a
queryable 3D digital twin: walk the orchard as a gaussian-splat
reconstruction, see every plant's recorded state from above, track change
across survey dates, and ask questions about the farm in plain language —
in English, Afrikaans, isiXhosa, isiZulu, Kiswahili, Hausa, Arabic or
Amharic.

Developed by the [African Robotics Unit](https://www.aru.uct.ac.za),
University of Cape Town.

```
survey (camera + LiDAR)
   │  ingest → poses → splat training → plant census
   ▼
digital twin ──────────────┬──────────────────────────────┐
   │                       │                              │
   TASSILI                 BATELEUR            SANKOFA
   3D walkthrough          top-down farm state change over time
   (gaussian splats)       (plants, rows, GPS) (per-tree ledger)
   └───────────────────────┴──────────────────────────────┘
                           │
                        ADINKRA
             ask about your farm, in your language
```

## Choose your journey

| I want to… | Start here | Needs |
|---|---|---|
| **See it work on real data** (an afternoon) | [QUICKSTART.md](QUICKSTART.md) | one NVIDIA GPU, ~15 GB disk |
| **Run it on my own orchard** | [docs/pipeline/](docs/pipeline/) after the quickstart | camera+LiDAR rig, GPU, patience |
| **Extend or study it** | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | the above |

## Hardware floor

Measured on an NVIDIA A100-40GB; the serving stack (viewer + panels +
language model) is expected to fit a single 24 GB GPU (the language model
alone needs ~9 GB in its served 4-bit build). Pipeline stages (splat
training) run one block at a time in ~16 GB.
<!-- VERIFY-BEFORE-LAUNCH: run the quickstart on a 24 GB card and replace
"expected to fit" with the measurement. -->

## What the language interface will and won't do

Adinkra answers from the farm's *recorded* data and is built to fail
honestly — "I could not work out an answer" in the user's language — rather
than guess. It is measured, not just designed: our
[evaluation of 524 interactions across eight languages](docs/EVALUATION.md)
reports where it is faithful (registry facts, refusing non-existent
plants) and where it still isn't (honesty under missing data varies by
language). Read that page before trusting it with decisions.

## Status

Research software, launched December 2026 from an active programme. The
pipeline has run end-to-end on two farms (a Western Cape citrus orchard
and a berry farm); expect sharp edges beyond the documented paths, and
[open an issue](../../issues) when you hit one.

## Citing

See [CITATION.cff](CITATION.cff). The technical report describing the
system and its evaluation: *(arXiv link at launch)*.

## Licence

Apache-2.0 (see [LICENSE](LICENSE)). Third-party components keep their own
licences ([NOTICE](NOTICE)); the language model (Google Gemma) is
downloaded by you at install time under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) — it is not
redistributed in this repository.
