# 0 · Capturing a survey

Everything downstream is decided in the field. This guide is the distilled
cost of our own mistakes — each rule below traces to a measured failure and
its fix.

## The rig

- **Stereo camera** (we use a ZED, recorded at full frame rate) and a
  **spinning LiDAR** (we use an Ouster), rigidly co-mounted on a ground
  vehicle. The camera faces the direction of travel.
- The **camera↔LiDAR extrinsic (L2C)** must be calibrated once per rig and
  recorded (`rig.json`). Single-frame projection of LiDAR into the image is
  the check — do it before the first survey, not after.
- INS/GPS is optional. It corrects long-run yaw drift (§2 of the poses
  guide); without it, expect a few degrees of heading drift per long leg.

## The drive

- **One steady pass per row**, both sides of a row face the camera across
  the two adjacent passes. Walking pace (~1–1.5 m/s).
- Keep rows in frame at a roughly constant lateral distance; the
  reconstruction is built from ~**20 cm spacing** along the path — faster
  driving doesn't break selection, it just thins the views.
- Avoid harsh backlight passes where you can; glare frames survive
  selection and become floaters.

## The three pre-flight checks (five minutes, saves days)

1. **Image orientation vs odometry frame.** Record 30 s driving forward;
   verify that image content flows OUTWARD from centre (approaching) and
   that the odometry translation is +forward in the frame you think it is.
   *Why: our ZED was physically inverted with image auto-flip on — images
   upright, odometry rolled 180° about the optical axis. Every
   reconstruction was silently 3–5 dB worse for weeks; found only by
   checking against an independent photogrammetric solve.*
2. **LiDAR self-return band.** Histogram the ranges of a static scan: rig
   parts show up as a band (ours: 0.16–0.39 m). Note its upper edge — the
   pipeline's min-range filter must clear it without eating canopy
   (we ship 0.45 m as the default; 0.6 m measurably deleted overhanging
   branches).
3. **Timestamps.** Camera and LiDAR clocks in the same timebase (or the
   offset recorded). Association downstream assumes it.

## What to record per survey (the metadata that becomes provenance)

Site name/ID, date, rig ID + `rig.json` version, weather/light notes, and
the row plan (which rows, which direction). This ends up in every derived
file's `_provenance` block; twins without provenance are not auditable.

## Deliverable of this stage

One bag/MCAP recording (camera stream + LiDAR packets + odometry [+ INS])
per survey, plus `rig.json`. Continue to [01_ingest.md](01_ingest.md).
