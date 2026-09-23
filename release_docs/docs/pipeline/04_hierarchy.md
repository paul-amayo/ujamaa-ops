# 4 · The plant registry — census to hierarchy

The *marker hierarchy* is the twin's registry: every plant as a first-class
object with a permanent ID, a position, and a place in the farm's
structure. It is the single file every panel and the language interface
read — the data contract of the whole system.

## What it is

```
marker_hierarchy.json
├── _provenance      # survey, date, code version — who made this and from what
├── objects[]        # one per plant: id (Obj_NN), xyz centre, per-plant detail
├── rows[]           # row id + member object ids (membership ONLY — see note)
├── super_rows[]     # groups of rows
└── sections[]       # groups of super-rows (farm subdivisions)
```

Optionally attached per plant: 3D-confirmed fruit counts
(`n_fruit3d` — deduplicated physical fruit, ≥2 sightings) vs raw
detection-event counts, kept distinct on purpose.

## Build

```bash
pipeline/hierarchy/build_hierarchy.sh data/<survey_id>
```
<!-- CONSOLIDATION-CONTRACT: today the marker/census chain in
aru_sil_core/src/scripts + attach_fruit3d_counts.py; one released entry
point. -->

Stages inside: per-frame plant detections (segmentation in **image mode,
frame by frame** — video-tracking mode was measurably unreliable on row
crops) → 3D marker association across frames → census (plant list + row
assignment from the geometry) → hierarchy assembly with provenance.

## The design rule that surprises people

The hierarchy records **what is there, not what should be**. Row entries
are membership lists; there is no "expected plant count", no gap
detection, no derived analysis. That is deliberate: derived judgements
(what's missing, what's unhealthy) belong to the consumers — and our
published evaluation measures whether the language interface reasons
about such ill-defined questions honestly from the recorded facts. If
you add derived fields, you change what "honest" means downstream; do it
knowingly.

## Verify

```bash
pixi run python pipeline/hierarchy/check_hierarchy.py data/<survey_id>
```

- Object count vs your row plan (our citrus site: 107 plants / 8 rows;
  the berry site: 517 / 17 — the census landed exact on both).
- Every object belongs to exactly one row; positions plot inside the
  survey bounds (the tool renders a top-down check figure).
- `_provenance` filled — a hierarchy without provenance will haunt you at
  the ledger stage.
- IDs are permanent from here on: panels, transcripts and the ledger all
  speak `Obj_NN`. Rebuilding the hierarchy re-numbers the farm — treat a
  rebuild as a new registry version, not an update.

Next: [05_ledger.md](05_ledger.md) (needs two surveys) or skip to
[06_serving.md](06_serving.md) to see this one.
