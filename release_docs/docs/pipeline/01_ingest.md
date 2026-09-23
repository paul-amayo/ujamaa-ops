# 1 · Ingest — recording → monolithic streams

**Environment: system `python3`** (the robotics reader/writer bindings do
not live in the pixi env — see INSTALL.md's two-environment note).

The pipeline's working format is the *monolithic*: one append-only file per
stream (images, odometry, LiDAR frames), indexed by timestamp. Everything
downstream reads monolithics, never the original bag.

## Command

```bash
python3 pipeline/ingest/bag_to_monolithics.py \
    --bag  /path/to/survey.mcap \
    --rig  /path/to/rig.json \
    --out  data/<survey_id>/monolithics/
```
<!-- CONSOLIDATION-CONTRACT: today `zed_bag_to_monolithics` + the
tenrows_ingest.sh / tenrows_post_ingest.sh pair; released as one entry
point. -->

Outputs (names are the downstream contract):

| file | content |
|---|---|
| `image_left.monolithic` | rectified left camera stream |
| `zed_transform.monolithic` | camera odometry (body frame) |
| `laser_*.monolithic` | LiDAR frames |

## Verify before continuing

- **Counts**: the tool prints frames written per stream; a stream that
  ended early (recording hiccup) shows here, not three stages later.
- **Image integrity**: decode a sample through the end of the file —
  interrupted dumps produce truncated final images that crash training
  much later. (Our tooling validates with an actual image `load()`, not
  just file presence; if you interrupt an ingest, re-run it.)
- **Timestamps monotonic** per stream (printed as a check).

## Common failures

- *Reader and writer in one process*: the reader and writer bindings
  cannot be loaded together — the released tool runs them as separate
  processes internally. If you script against the bindings yourself,
  keep that split, and import your imaging library before the binding.
- *Wrong python*: this stage in the pixi env fails on binding imports;
  it's a system-python stage.

Next: [02_poses.md](02_poses.md).
