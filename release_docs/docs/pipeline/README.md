# The pipeline — your own orchard, end to end

Survey in, digital twin out. Six stages, each with the same shape:
what it does, the command, how to verify, what usually breaks.

| stage | doc | environment | GPU time (A100 reference) |
|---|---|---|---|
| 0 | [Capture](00_capture.md) | the field | — |
| 1 | [Ingest](01_ingest.md) | system python | minutes |
| 2 | [Poses](02_poses.md) | pixi (+ bundled SfM) | ~2–3 min/block refine |
| 3 | [Train splats](03_train_splats.md) | pixi | ~6 min/block ×2 stages |
| 4 | [Plant registry](04_hierarchy.md) | pixi + system | ~1 h/survey |
| 5 | [Change ledger](05_ledger.md) | system python | minutes |
| 6 | [Serving](06_serving.md) | both | resident |

Reference scale: a 23-block survey (one citrus orchard pass) costs about
**2.5 h of GPU for stage 3** plus ~1 h of stage-2 refinement — an
afternoon on one A100, a day on a 24 GB card.

Two principles run through every stage:

1. **Verify with numbers, not eyeballs.** Every stage ends in a check
   with expected values printed beside yours. The expensive failures are
   silent ones (a frame convention, a truncated stream) — the checks
   exist because each caught one for us.
2. **The twin records what is there, not what should be.** Derived
   judgements live in the consumers, where they can be evaluated —
   see the [evaluation](../EVALUATION.md) for why.
