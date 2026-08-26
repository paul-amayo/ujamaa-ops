# Note for the future — the English-pivot experiment (and why the African LMs are not it)

Status: **NOTE, not scheduled.** Written 2026-08-26 after the r3 dossier.
Trigger: revisit after the WWW readout (moved to mid week of 2026-08-31).

## The idea

Adinkra today is **native**: the question arrives in isiXhosa, Gemma reasons and
answers in isiXhosa. The alternative is a **pivot**: translate the question to
English, let Gemma reason in English, translate the answer back.

## Why it is worth testing

r3 (524 interactions, 8 languages) produced two findings that a pivot speaks to
directly:

1. **Conclusions diverge by language.** The ill-defined B3 ("are any plants
   missing along row 7?") returned *no* with census reasoning in English, *yes*
   with a flawed rationale in isiZulu, *yes* from ID-sequence gaps in Kiswahili,
   and a decline in Afrikaans — same farm, same data. A pivot makes every
   language share ONE reasoning pass, so the divergence should collapse **by
   construction**. If it does not, the divergence has moved into the translation
   layer, which is itself the finding.
2. **The equity gradient is a GENERATION cost, not a thinking cost.** xh/zu run
   ~3x English while clean answers are the same length in every language. A
   pivot has Gemma generate English (fast) and a translator handle the rest — so
   the pivot could be FASTER than native for xh/zu. That would inverst the
   equity story and is worth knowing before any channel/speech decision.

## The design (two arms, not four)

| arm | path | new deps |
|---|---|---|
| A — native (already measured) | question(xx) -> Gemma -> answer(xx) | none, r3 is the baseline |
| B — Gemma-pivot | question(xx) -> Gemma translate -> Gemma reason(en) -> Gemma translate -> answer(xx) | none |

Same question bank, same scoring, same grounding — only the transport changes.
The harness is path-agnostic, so this is a runner change, not a new eval.

**Free automatic metric, needs no native speaker:** identifier and number
survival. Does `Obj_12` and "107 plants" come back intact through the round
trip? Translation layers mangle exactly those, and the registry-count check
already measures it.

Scope: Pass B subset x 8 languages x 2 arms, ~130 queries, half a day A100.

## Why InkubaLM / MzansiLM are NOT arms in this (2026-08-26, model cards read)

Both were considered as the translator and both are ruled out **for now**:

| | InkubaLM-0.4B | MzansiLM-125m |
|---|---|---|
| licence | **CC BY-NC 4.0** — research only | **Apache 2.0** — permissive, commercial OK |
| size | 422M | 125M |
| languages | zu, yo, sw, xh, ha + en, fr | all 11 official SA languages |
| type | **base, NOT instruction-tuned** | **base, NOT instruction-tuned** |
| context | 2,048 tokens | 2,048 tokens |
| origin | Lelapa AI | **UCT CS — Anri Lombard, Jan Buys** |

- **Neither can translate zero-shot.** Both cards say base model, plain
  continuation prompts. Using either as a translator means FINE-TUNING, which
  Paul ruled out for this round.
- **2,048 context is too small anyway.** Bateleur citrus context is ~2.6k chars;
  klapmuts ~7.6k. Even fine-tuned, the context would need chunking.
- **InkubaLM's card lists "sometimes switches between languages during
  generation"** — a poor property for a translation layer specifically.
- Licence surprise worth remembering: **MzansiLM is the LESS encumbered one**
  (Apache 2.0). The earlier assumption that both were non-commercial was wrong.

**The real opportunity is a collaboration, not a download.** MzansiLM is
Apache-2.0, UCT-built, and covers all 11 official languages — which is exactly
the gap the dossier declares (we tested af/xh/zu; seven untested). A fine-tuning
collaboration with Buys' group, measured against our existing harness (frozen
question bank + 524-interaction baseline + scoring), is a stronger move than any
drop-in comparison. Contact: Dr Jan Buys / Anri Lombard, UCT Computer Science;
paper arXiv 2603.20732.

## McGill's AfriqueLLM — the most promising external lead (2026-08-26)

McGill-NLP's **AfriqueLLM** family adapts open backbones to African languages by
**continued pre-training** (~26B tokens: native-language data, code, maths,
synthetic translation), covering **20-24 African languages**. Relevant because
the family INCLUDES GEMMA BACKBONES — Gemma 3-4B and **Gemma 3-12B**, the same
size class we serve — so it is the closest thing to a drop-in comparator we
have found: same family, same scale, African-adapted.

Published siblings: `McGill-NLP/AfriqueLlama-8B`, `McGill-NLP/AfriqueQwen-14B`
(HF collection). Reported: AfriqueQwen-8B beats Gemma 3 12B in aggregate at half
the parameters; largest gains on seen languages.

**Caveats to check before counting on it:**
- CPT, **not instruction tuning** — no chat/instruct variant is documented. Our
  panels need structured JSON command output, so a CPT-only checkpoint may need
  the same instruct-layer work as the African LMs above. VERIFY on the model
  card before designing an arm around it.
- We run **Gemma 4**-12B; the Afrique adaptation is on **Gemma 3**. A comparison
  would confound "African adaptation" with "generation of base model".
- Licence per checkpoint not yet read.
- Language overlap with our eight needs checking (they list 20-24; we run
  af/xh/zu/sw/ha/ar/am + en).

**If it checks out** it is a better third arm than either African LM: a
Gemma-family, 12B-class, African-adapted model that could be compared natively
(arm A) rather than only as a translator — which is the comparison WWW's report
actually asks for (same test set, same rubric, credible comparator).

Sources: AfriqueLLM overview (emergentmind.com/topics/afriquellm); AfroBench
arXiv 2311.07978; HF McGill-NLP collection.

## Also parked

- Higher-precision Gemma build (we serve Q4_K_M) as a quality lever.
- Raising the 4,000-token generation cap for the expensive-tokenizer languages.
