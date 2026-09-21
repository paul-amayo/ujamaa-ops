# Negative result: LoRA-rank continued pre-training does not buy generation on Gemma 4

**G0/P0 write-up — UJAMAA multilingual track.** Run 2026-09-05 → 09-20 on the lab A100-40GB;
verdict 2026-09-20. Companion to `plans/gemma4_finetune_memo.md` (the scoping memo whose §4
stop rule this run executes) and the frozen M1 harness (`automation/multiling_eval/`).

## TL;DR

1.28 B tokens of QLoRA continued pre-training on `gemma-4-12b` base **halved held-out
perplexity on isiXhosa and isiZulu (ratio 0.45) while moving generation quality nowhere**:
en→xh chrF +0.5, en→zu −1.6 against stock on the frozen harness, flat at the 25 %, 50 %
and 100 % checkpoints. No forgetting occurred either (Belebele en +0.0, af −1.0, sw +3.0)
— the adapter changed next-token statistics without changing task behaviour in any
direction. The memo's stop rule ("doesn't move en→xx chrF → stop") is met. The contrast
with AfriqueGemma — the same corpus recipe applied **full-parameter** to Gemma 3 gained
+9.5/+14.5 chrF — localises the failure in the update mechanism (low-rank adaptation),
not in the data or the languages. LM fit is not a proxy for few-shot generation at PEFT
rank.

## 1. Why this run existed

The memo's evidence base (measured 2026-08-25 → 09-03, all numbers in
`experimental_multiling_m1/summary.txt`):

- Stock `gemma-4-12b-it` fails to *deliver* in xh/zu (38–49 % silent on the product
  plane) and its instruct layer destroys translation (−20 to −27 chrF vs its own base).
- Gemma 4 **base** is the strongest starting point we have (above Gemma 3 base by
  +2.8/+4.1 en→xx chrF; Belebele xh 82 vs 58 conditional).
- AfriqueGemma proved the data recipe on Gemma 3: full-parameter CPT on ~22.8 B
  African-language tokens bought en→xh +9.5 and en→zu +14.5 chrF.

P0 asked one question before any external spend: **does a cheap PEFT proxy of that
recipe (QLoRA, ~1–2 B tokens, our own A100) move our frozen metrics on Gemma 4?**
Stop-on-flat was a designed outcome, priced at ~2 weeks of interleaved GPU.

## 2. What was trained

| item | value |
|---|---|
| base | `unsloth/gemma-4-12b` (bf16-validated ungated republish of the gated Google base) |
| method | QLoRA: base frozen in 4-bit nf4, bf16 compute, LoRA adapter trained |
| trainable | 262,275,072 params = **2.15 %** of 12.22 B |
| tokens | **1.28 B** (622,592 × 2048-token seqs; 262 k tokens/step; 4864 steps) |
| schedule | cosine, peak 1e-4, warmup 100, effective batch 128 seqs |
| hardware | single A100-40GB, 32.3 GB resident, 239–241 s/step |
| wall | 13.5 days of stepping; 15.2 days wall (two eval pauses + one 23.6 h ops incident) |

Data mix (Afrique proportions scoped to our languages, temperature-upsampled):
xh 294 M, zu 176 M, ha 151 M, am/af/sw/ar 101 M each, en 252 M replay.

Training was healthy throughout: loss 2.37 (first 100 steps) → 1.955 (step 300) →
plateau ≈1.93 → 1.858 (2400) → 1.834 (4800), final 1.845; gradient spikes absorbed by
clipping; the two pause/resume cycles restored optimizer state exactly (post-resume loss
continued the pre-pause trend).

## 3. Evaluation protocol

The **frozen M1 harness** — the same slices, prompts and scorer that produced the August
anchor table, byte-identical across arms: raw 3-shot prompts, greedy decoding,
newline truncation; FLORES-plus devtest n=32 per language-direction; Belebele n=100 per
language; chrF via the local character-6-gram implementation. The finished adapter was
merged into the bf16 base (`/data/g0_run/merged`, 24 GB) and run through `hf_run.py`
exactly as the stock `gemma-4-12b-pt-bf16` arm was. Mid-run gates (25 %, 50 %) used
4-bit **paired** evaluation on identical indices, so the quantisation confound cancels
in the delta.

## 4. Results

**FLORES chrF en→xx** (n=32/lang; G0-CPT vs stock g4 base, identical protocol):

| lang | stock | G0-CPT | Δ | Afrique bar |
|---|---|---|---|---|
| xh | 43.5 | 44.0 | **+0.5** | 50.2 (short by 6.2) |
| zu | 44.3 | 42.7 | **−1.6** | 54.7 (short by 12.0) |
| af | 69.7 | 67.6 | −2.1 | |
| sw | 67.0 | 66.9 | −0.1 | |
| ha | 50.9 | 50.6 | −0.3 | |
| ar | 58.7 | 56.3 | −2.5 | |
| am | 33.5 | 32.5 | −1.0 | |

**FLORES chrF xx→en:** xh −1.8, zu +0.1, af −0.5, sw −1.2, ha +1.1, ar −3.2, am −4.2.
Blank-generation rate 0.0 % in both arms (base models don't go silent; the silence
pathology is instruct-plane and untestable here).

**Belebele accuracy** (n=100/lang): en 94→94 (+0.0), af 89→88 (−1.0), sw 83→86
(**+3.0** — the memo's named forgetting-risk gate, passed with a gain), xh 64→60,
zu 69→67, ha 71→68, ar 90→87, am 84→86. All within the ±4 noise band of n=100.

**Mid-run gates** (paired 4-bit, same indices):

| gate | tokens seen | en→xh Δ | en→zu Δ | ppl ratio (xh / zu) |
|---|---|---|---|---|
| 25 % (ckpt-1200) | 315 M | +0.16 | −0.44 | — |
| 50 % (ckpt-2400) | 630 M | +0.64 | −0.37 | **0.451 / 0.452** |
| 100 % (merged, bf16) | 1.28 B | +0.5 | −1.6 | — |

**Gates:** no-forgetting en/af/sw — PASS. Generation movement en→xh/zu — **FAIL**.

## 5. The finding: LM fit dissociates from few-shot generation at LoRA rank

The adapter unambiguously learned the languages: held-out perplexity on xh/zu text fell
from 43.8/39.2 to 19.8/17.7 — **halved by the 630 M-token mark** — and training loss on
the mix kept its slow descent to the end. Yet no downstream cell moved beyond noise in
either direction at any checkpoint. A 262 M-parameter low-rank update can reshape the
next-token distribution over these languages while leaving the machinery that few-shot
translation draws on untouched. Full-parameter CPT on the same recipe family (Gemma 3 →
AfriqueGemma) restructured it: +9.5/+14.5 chrF.

Two secondary factors, stated for honesty:

- **Token budget.** 1.28 B vs Afrique's 22.8 B. We cannot exclude that full-rank at
  1.28 B would also read flat. What the run *does* exclude is "LoRA + our budget", and
  the early perplexity saturation argues that more LoRA tokens would improve LM fit
  further without converting.
- **Headroom.** Gemma 4 base already sits above Gemma 3 base on these languages, so part
  of Afrique's gain was catch-up g4 doesn't need. But the Afrique bar remains 7–12 chrF
  above stock g4 — headroom exists; LoRA didn't reach it.

## 6. Threats to validity

- **Sample size:** every chrF cell rests on the frozen 32-sentence devtest slice. It was
  measured ≈0 three ways (two 4-bit paired gates, one bf16 protocol-matched final), which
  bounds any true effect well below the +6.7/+10.4 needed to matter, but a planned n=128
  paired supplement is **blocked**: `openlanguagedata/flores_plus` is hub-gated and the
  box token lacks access. Accepting the gate unblocks a ~2 h run.
- **Single configuration:** one LoRA config, one seed, one token budget. No rank sweep —
  "insufficient rank" is the leading interpretation, not a demonstrated dose-response.
- **Quantised base during training** (nf4): mitigated by paired mid-run evals (confound
  cancels in deltas) and by merging to bf16 for the final protocol-matched arm.
- **Scorer:** local chrF implementation, consistent across all arms and anchors; absolute
  values are comparable only within this harness.

## 7. Implications and options

1. **The cheap path to Stage A is closed.** Scaling LoRA tokens (cloud rental, ~$2–4 k)
   is now expected waste; don't do it.
2. **G1 decision (memo §8):** either stop here — this document is the negative result —
   or take the case to the HPC committee for **P1: full-parameter scoped CPT, 4–8 B
   tokens, 8×A100-class**. The P0 evidence sharpens that ask: the data pipeline is built
   and validated (perplexity halves; no forgetting with the replay mix), the harness and
   gates are frozen, and the mechanism argument (rank, not data) is measured.
3. **The product-plane problem is untouched.** The xh/zu silence pathology lives in the
   instruct layer (Stage B); it never depended on this run. Stage B remains cheap in
   compute and blocked on data (WWW commissioning), not on GPUs.
4. Assets that persist: the 1.28 B-token corpus, tokenized shards, paired-eval machinery
   (`automation/g0_qlora/eval_ckpt.py`), the merged model (24 GB), and the harness
   extensions.

## 8. Provenance

Run dir `/data/g0_run/` (final_adapter, merged, ckpts 4650/4800/4864, eval{25,50}.json);
harness rows `experimental_multiling_m1/g0-cpt-merged_{belebele,flores}.jsonl`; scorer
`~/logs/g0_verdict.py`; training driver `automation/g0_qlora/` (post-incident: default
interpreter `hfeval_ft`, torch 2.7.1 — the torch<2.6 `torch.load` guard cost 23.6 h on
09-09, notebook 2026-09.md). Notebook entries: 09-06 (pause), 09-07 (resume trap),
09-11 (incident + 25 %), 09-13 (50 % gate), 09-20 (completion, verdict). Key commits:
`b6eaf3e`, `45ccad6`, `c55b698`.
