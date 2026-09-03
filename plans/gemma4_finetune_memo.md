# M4 — Scoping memo: an African-language fine-tune of Gemma 4 for UJAMAA

Status: DRAFT for Paul, 2026-09-03. The multilingual model track's final
deliverable (plans/multilingual_model_track.md). Every capability number in
here was measured on our own harness (experimental_multiling_m1/, frozen
slices, summary.txt); estimates are labelled as estimates.

## 1. The decision, and the recommendation

**Decision:** commission a Gemma-4 fine-tune for UJAMAA's languages, or keep
stock Gemma 4 with product-side mitigations?

**Recommendation: a two-stage fine-tune is justified, feasible in stages, and
now precisely targeted — but gate it on a cheap PEFT proof-of-lever first.**
The evidence says the gap is real, the lever demonstrably works on this model
family, and the missing stage (delivery-preserving instruction tuning) is
novel work nobody has published — which makes it a research contribution, not
just an ops fix.

## 2. Evidence base (all measured, 2026-08-25 → 09-03)

| finding | number |
|---|---|
| Stock g4-it fails to DELIVER in xh/zu | 38–49% silent at any budget (2k/4k/8k) |
| g4-it understands better than g3 base | conditional Belebele xh 82% vs 58 |
| g4 BASE has no regression — modest gains | Belebele am +9, xh +6; chrF +2..+5 |
| The instruct layer alone destroys delivery | g4-it vs g4-base: xx→en −30 chrF, all langs |
| African CPT buys generation (proven on g3) | AfriqueGemma: en→xx +14.5 zu, +9.5 xh, +8 ha, +7 am |
| CPT ≫ generation upgrade for this capability | +9.5/+14.5 (CPT) vs +2.8/+4.1 (g3→g4 base) |
| Tokenizer prices Ge'ez at byte-fallback | am 1.59 chars/token vs en 3.96 |
| Product-side symptom (r3, 524 interactions) | xh/zu 3–4× en latency; verdicts diverge by language |

**The one-line thesis:** Gemma 4 base already knows these languages better
than anything we can adapt from Gemma 3; what must be built is (a) stronger
GENERATION via CPT and (b) an instruct layer that does not lose it — because
the stock instruct layer measurably does.

## 3. The recipe — two stages

### Stage A — continued pre-training (CPT) on `gemma-4-12b` base

AfriqueLLM's corpus was never released, but its recipe is fully documented
(arXiv 2601.06395) and ~97% of it comes from public sources. Reconstruct at
their proportions, scoped to OUR languages:

| slice | Afrique source (public) | their share | ours |
|---|---|---|---|
| African monolingual | FineWeb2, WURA, MADLAD-400 | ~22.8B tok / 20 langs | xh zu ha am (+af sw replay) — est. 4–8B tok |
| English/French replay | (forgetting mitigation) | high-resource mix | en + af replay; add **sw replay** (see §7 risk) |
| Code | CornStack-Python | ~1B | keep, scaled |
| Maths | FineMath-4+ | ~1B | keep, scaled |
| Synthetic domain | GPT-4.1 translations, 10 domains | ~324M | **replace with agricultural domain — the WWW pipeline (§6)** |
| SA-language depth | — | — | **MzansiText** (Apache-2.0, UCT/Buys) for the seven untested SA languages |

Base checkpoint: `google/gemma-4-12B` (gated; accept licence) or the ungated
`unsloth/gemma-4-12b` republish we validated in bf16. Per-language token
availability in FineWeb2/WURA/MADLAD to be verified at kit-freeze —
order-of-magnitude, xh/zu are hundreds of millions of web tokens each,
af/sw/ar multi-billion; upsample low-resource per Afrique's temperature
sampling rather than pretending the corpora are balanced.

### Stage B — delivery-preserving instruction tuning (the novel stage)

This is the stage AfriqueLLM does not have and our table says is decisive.
Objective is explicitly *delivery*: the acceptance metric is the silence
rate and in-language answer quality on OUR harness, not English benchmarks.

Data sources, in order of leverage:
1. **WWW ladder output** (their Stages C–E): validated agricultural
   request→response pairs in xh/af (+zu/sw), exactly the interaction shape
   Adinkra serves. Doubles as their Phase-2 work package — one pipeline,
   two purposes.
2. **r3-harness-shaped synthetic**: our frozen question bank pattern
   (user-voice agri interactions) generated at scale in-language and
   human-filtered — the cheap bulk.
3. **Inkuba-Instruct** (Lelapa) for general instruct coverage of zu/xh/sw/ha
   — licence is CC-BY-NC lineage: fine for research training, RE-CHECK
   before any commercial path.
4. Standard multilingual instruct sets (Aya, translated FLAN-style) as the
   backbone so general ability doesn't collapse.

Include *thinking-behaviour* data deliberately: short-deliberation exemplars
in low-resource languages, because the measured failure is the reasoning
layer deliberating itself into silence precisely there.

## 4. Compute (assumptions stated; 6·N·D FLOPs, ~40% MFU on A100)

| option | tokens | hardware | wall-clock (est.) | purpose |
|---|---|---|---|---|
| **P0 — PEFT proof-of-lever (QLoRA)** | ~1–2B, xh/zu-weighted | our A100-40 | **~1–2 weeks**, interleaves with fleet | does the lever move OUR metrics? gate for everything below |
| P1 — scoped full CPT | 4–8B | UCT HPC `a100` ×8 | ~1–2 weeks queue-time | the real Stage A |
| P1′ — same on cloud | 4–8B | 8×A100 rental | days; ~$2–4k order | if HPC access slips |
| full Afrique-scale | 26B | multi-node | ~180 A100-days | NOT recommended; scoped beats it for our langs |
| Stage B instruct | ~50–200M | our A100 (QLoRA) or HPC | days | cheap; data is the constraint, not compute |

P0 is the only spend needed to de-risk the whole memo: if QLoRA on 1–2B
tokens doesn't move en→xx chrF and the silence rate on our harness, stop.

## 5. Evaluation = the harness we already froze (acceptance gates)

Same slices, same scorer, same planes (automation/multiling_eval/ +
adinkra_www/). Proposed gates for a successful fine-tune, against stock
g4-it deployed-plane numbers:

- **Silence**: xh/zu no-answer rate < 10% (from 38–49%) at the 4k budget.
- **Generation**: en→xx chrF xh/zu ≥ AfriqueGemma's (50.2 / 54.7) — i.e.
  match on Gemma 4 what CPT achieved on Gemma 3.
- **No forgetting**: en/af Belebele within 2 pts of stock; **sw within 2 pts
  of stock** (named gate — see §7).
- **Product plane**: r3 question bank re-run — verdict-consistency on the
  ill-defined class, identifier survival 100%, latency medians ≤ stock.

## 6. Collaboration surfaces (who brings what)

- **WWW**: Stage-B data via their quoted ladder; native review of synthetic
  instruct data; the terminology resource as vocabulary for both stages.
- **UCT / Buys**: MzansiText for SA-language depth (Apache-2.0, the least
  encumbered corpus in this space); possible co-supervision of the instruct
  stage; our harness as the shared evaluation.
- **McGill**: the CPT recipe is theirs — reproduced, credited (CC-BY-4.0
  models; paper arXiv 2601.06395); the delivery-preservation finding and
  instruct stage is the natural joint-paper hook (their models stop at base;
  our evidence says the instruct layer is where African languages are lost).
- **HPC**: plans/hpc_migration.md P0 access request finally has its killer
  workload.

## 7. Risks

- **Swahili forgetting is real, not hypothetical**: AfriqueGemma lost 9
  Belebele pts on sw. Mitigate with sw in the replay mix + the named gate.
- **Instruct data scarcity** in xh/zu is the actual bottleneck (compute is
  not). The WWW/synthetic/human-filter pipeline is the plan; its throughput
  sets the timeline.
- **Quantised deployment**: we serve Q4_K_M. Gates must be re-measured on
  the quantised fine-tune, not the bf16 checkpoint (fertility/quant tax may
  interact; unmeasured).
- **Gemma licence**: fine-tuned derivatives are permitted under the Gemma
  Terms with use restrictions; fine for research and the UJAMAA demo;
  re-read before any broader release. Inkuba-Instruct is NC.
- **Tokenizer floor**: no fine-tune fixes am's byte-fallback Ge'ez pricing —
  vocabulary extension is a separate, heavier intervention; out of scope
  here, noted for honesty.

## 8. Staged go/no-go

1. **G0 (now)**: approve P0. Cost: ~2 weeks of interleaved A100 + the data
   prep. No external spend.
2. **G1 (post-P0)**: PEFT moved the gates' needles → request HPC window +
   commission WWW Stage-B data with real numbers in hand. Else stop with a
   negative result worth publishing anyway.
3. **G2 (post-P1+B)**: gates pass on quantised checkpoint → swap into
   Adinkra behind the same freeze/eval discipline as every recipe change.

## Appendix — provenance

Measured inputs: experimental_multiling_m1/ (4 arms × 2 tasks × 8 langs,
three quarantined g4 attempts documenting the method traps), r3
(experimental_adinkra_www_20260827_r3/, 524 interactions), fertility table
(served-tokenizer prompt_eval_count). External: AfriqueLLM arXiv 2601.06395;
model cards read 2026-08-26/09-02 (AfriqueGemma-12B CC-BY-4.0;
InkubaLM-0.4B CC-BY-NC; MzansiLM-125m Apache-2.0; unsloth/gemma-4-12b
ungated republish of gated google/gemma-4-12B).
