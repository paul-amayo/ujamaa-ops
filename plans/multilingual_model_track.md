# Multilingual model track — mapping Gemma 4's gaps before anyone fine-tunes

Status: DRAFT v1, 2026-08-26. Own track — **not** Adinkra, **not** the WWW
readout. Adinkra's dossier measures the product interaction; this track
measures the MODEL, so the two never share a harness or a headline.

**Premise (Paul):** Gemma 4 stays the base. Fine-tuning the Gemma family for
African languages demonstrably works — AfriqueGemma-12B (McGill, Gemma 3-12B +
26B-token CPT) gains +4.0 overall (7.3% rel) on the African suite. The question
this track answers is not "which model instead" but **"where exactly is Gemma 4
short, and is the gap worth a fine-tune"** — along three axes:

1. **Size** — what does parameter count (and precision) buy, per language?
2. **Multilingualism** — which languages, which capabilities, which scripts?
3. **Agriculture** — the domain axis, driven by Way With Words' terminology
   and request work (their Stage 2 output feeds this directly).

## 1. The anchor experiment — decompose the gap

CPT on Gemma 3 is the verification that adaptation works. Run the same probes
over a small matrix and subtract:

| model | role |
|---|---|
| `google/gemma-3-12b-pt` | Gemma 3 baseline (McGill's own baseline) |
| `McGill-NLP/AfriqueGemma-12B` | what African CPT **buys** (CC BY 4.0, ollama-ready quants) |
| Gemma 4 12B (pt if published, else it) | what a **generation** buys |
| `hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M` | what we actually serve (deployment plane) |

Readout: `adaptation_gain` (Afrique − g3) vs `generation_gain` (g4 − g3) per
language per capability. **If generation_gain ≥ adaptation_gain in our
languages, Gemma 4 already ate the fine-tune's lunch and the case weakens; the
remaining per-language deficit is the fine-tune's scope if not.** Base models
are probed few-shot (McGill's own protocol); instruct behaviour is a separate
plane, never mixed in one table.

## 2. Axis: size (and precision)

- Enumerate the released Gemma 4 size ladder at kit-freeze (we know 12B; list
  siblings from the official card then — do not assume Gemma 3's 1/4/12/27).
- Probe the ladder at fixed quantization; then our serving quant (Q4_K_M) vs a
  higher-precision build at 12B to price the quantization tax per language —
  low-resource languages plausibly pay more of it.
- Output: capability-per-parameter curves per language; the size at which each
  language's floor (script fidelity, instruction-following) is reached.

## 3. Axis: multilingualism

Public suite, chosen to be comparable with McGill's table: **FLORES** (MT),
**Belebele** (reading comprehension), **AfriMMLU**, **SIB-200** (topic),
AfriXNLI — restricted to our eight (en af xh zu sw ha ar am) plus any
UJAMAA-relevant additions (yo for InkubaLM overlap; nso/tsn from the MzansiLM
eleven when data exists).

Own probes, from what r3 taught us the public suites miss:
- **Tokenizer fertility** per language (tokens/char) — the measured driver of
  the xh/zu latency gradient; pure tokenizer arithmetic, no GPU.
- **Instruction-following in-language**: does a constraint stated in isiZulu
  bind as tightly as in English?
- **Structured-output compliance** per language (JSON contract hold rate) —
  the deployability number the public suites never report.
- **Reasoning consistency**: the B3-style ill-defined probe — same facts, all
  languages — scored on whether the *verdict* agrees, not the wording.
- **Script robustness**: Ge'ez and Arabic-script generation + RTL identifier
  survival (Obj_12 through an RTL sentence).

## 4. Axis: agriculture (WWW-driven)

Blocked-by-design on Way With Words' Stage 2 terminology/request resource —
their validated term lists become the probe vocabulary. Until that lands, an
interim v0 from what we own: the frozen question bank, crop/symptom/operation
term lists drawn from the two farms, and agronomic terms from DATASETS.md
surveys. Probes: term comprehension per language (definition match), term
survival through translation, and agricultural sense disambiguation ("sukkel"
vs generic struggle, "sack/pot/bag" across epochs — the scene.json prompt
lesson). This axis is where the track and WWW's Phase 2 quotation meet:
their data work, our measurement.

## 5. Mechanics

- **Harness**: new `automation/multiling_eval/` — direct model calls (ollama /
  transformers), few-shot templates, log-prob or exact-match scoring. No
  Adinkra server, no panels, no farm grounding. Separate output roots
  (`experimental_multiling_*`).
- **Compute**: A100 first (bf16 12B ≈ 24 GB fits; check fleet contention and
  the 245 GB disk — the model zoo is ~25 GB per checkpoint, gate pulls on
  Paul's go). HPC a100 partition is the overflow once access lands.
- **No training in this track.** Its deliverable is the *scoping* of training:
  a gap map per (language × capability × size), the decomposition table, and
  a go/no-go recommendation with data-size reference points (AfriqueLLM used
  ~26B tokens; an instruct layer needs its own estimate).
- **Collaboration surface**: McGill (AfriqueGemma recipe + their live instruct
  pipeline — AfriqueQwen3.5-Instruct appeared 2026-08-26), UCT/Buys
  (MzansiText for the seven untested SA languages, Apache-2.0), WWW
  (terminology + native review of probe translations).

## 6. Milestones

- **M0 — kit**: harness skeleton, tokenizer-fertility table (free, no pulls),
  Gemma 4 size-ladder enumeration, disk/contention check. *Gate: Paul's go
  before any checkpoint pull (~100 GB for the matrix).*
- **M1 — anchor**: the §1 matrix on FLORES + Belebele + our probes, eight
  languages. The decomposition table is the first readout.
- **M2 — size ladder** at fixed quant + the Q4 tax measurement.
- **M3 — agriculture v0** (interim probes now; re-run when WWW terminology
  lands).
- **M4 — gap map + fine-tune scoping memo** — the track's product.

## Open questions for Paul

1. Does M0 get its go now (disk + pulls on the A100), or wait for the readout?
2. Is yo in scope (InkubaLM/AfriqueLLM overlap, not a UJAMAA deployment lang)?
3. Should the fine-tune scoping memo assume UCT compute (HPC a100) or cloud?
