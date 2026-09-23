# 5 · The change ledger — one farm across many surveys

Everything so far describes one survey. The ledger is where UJAMAA becomes
*monitoring*: the same physical tree, recognised across survey dates, with
observations accumulating against a stable identity.

## What it is

`ledger.json` — a list of observations:

```json
{ "canonical_id": "13B:139",      // the physical tree, stable across epochs
  "epoch": "2026-07-16",          // survey date
  "ndvi_sentinel2": 0.178,        // vegetation index at that date
  ... }
```

- **canonical_id ≠ Obj_NN**: `Obj_NN` is a plant *in one survey's
  registry*; the canonical ID names the physical tree those per-survey
  objects were associated to. The ledger stores the mapping.
- Observations carry whatever was measured that epoch. Our shipped ledger
  carries positions and a satellite vegetation index (NDVI); structural
  change (canopy volume/height from the splats) is the designed next
  column — the schema has the slot, the released pipeline does not fill
  it yet, and the serving layer says exactly that when asked.

## Build (two or more surveys of the same site)

```bash
pipeline/ledger/associate.sh data/<site>/<survey_A> data/<site>/<survey_B>
```
<!-- CONSOLIDATION-CONTRACT: today the sankofa_substrate association
chain (assoc_topdown.py lineage + NDVI attach); released as one driver. -->

Association is geometric (registered top-down positions across epochs)
with conservative matching: a tree seen in only one epoch stays a
single-epoch entry rather than being force-matched. Reference scale: our
citrus ledger holds 854 observations / 452 canonical trees / 272 seen in
more than one epoch.

## Verify

```bash
pixi run python pipeline/ledger/check_ledger.py data/<site>
```

- Multi-epoch fraction: on consecutive surveys of the same rows expect
  most trees matched; a low fraction means the epochs aren't registered
  into the same frame (fix at §2, not by loosening matching).
- Change distribution: the tool prints the per-tree index deltas
  (median/min/max) and the extreme movers — eyeball the extremes against
  imagery before believing them; they are also exactly what the serving
  layer will quote.
- No duplicate (canonical_id, epoch) pairs.

## The honesty contract, again

The serving layer summarises the ledger *as it is* — including "structural
metrics are not in the ledger yet" when asked about growth. Keep that
statement true: if your deployment adds columns, the summary follows the
data; if it lacks epochs, "I have one survey" is the right answer, not an
extrapolation.

Next: [06_serving.md](06_serving.md).
