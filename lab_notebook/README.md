# Lab notebook

Append-only record of experiments and runs — the *record*, distinct from memory files
(which hold *conclusions*). One markdown file per month (`2026-07.md`, …), newest entry
at the top of each file.

Entry template:

```markdown
## 2026-07-13 · <short title>
- **Goal:** what question this run answers
- **Command/config:** exact invocation + key flags (or path to script/config)
- **Data:** survey/block ids, input paths
- **Result:** metrics (PSNR, counts, match rates…), output paths
- **Verdict:** keep / discard / follow-up — one sentence of interpretation
```

Rules:
- Every ns-train / pipeline / association run gets an entry, even failures — failures
  especially.
- Metrics come from the psnr skill / tensorboard events, not eyeballs, where possible.
- When an entry produces a durable conclusion, promote it to a memory file and link it.
