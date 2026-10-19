# Publications plan — 19 Oct → 31 Dec 2026 (and the JIRS collection by Feb 2027)

Paul, 19 Oct: "there's a special issue open … let's plan publications towards the end of the year."
Skills: how-paul-writes (voice), paper-latex (build/checks/packaging), reviewer-response.

## The three outputs

| # | Output | Venue | Content (contribution boundary) | Target | Status 19 Oct |
|---|---|---|---|---|---|
| 1 | **HiGH** | IEEE RA-L (rolling; 6 + 2 paid pages; double-anonymous) | The method: hierarchically grounded hyperbolic embeddings for open-vocabulary queries in 3DGS farm models; containment, void check, fruit registry | **submit by Fri 13 Nov** | Draft 31 in the Overleaf repo (`Hyperbolic-Hierarchies-for-Precision-Agriculture`), 8 pages, builds clean |
| 2 | **UJAMAA release report** | arXiv (cs.RO) with the public release | The system as released: pipeline, data, demo, how to reproduce | **mid-Dec**, with the roadmap's Phase 5 launch | Not started; roadmap Phase 4 item |
| 3 | **UJAMAA 4D twin** | *Journal of Intelligent & Robotic Systems*, collection "AI and Soft Robotics in Agriculture: From Sensing to Harvesting and Beyond" (deadline 31 May 2027) | The system and its use over time: robot + phone surveys → H3DGS models → identity (cites HiGH, does not re-claim it) → per-plant ledger across surveys (found again, change, NDVI) → multilingual questions answered from the twin | **submit Feb 2027**; outline + gap experiments in Nov–Dec | Not started |

Overlap rule (both RA-L and JIRS forbid substantially overlapping submissions under review): HiGH owns the embedding
and its evaluation; the JIRS paper owns the temporal ledger, the multi-platform survey pipeline (Jackal, A300, phone),
deployment, and the interface. The JIRS paper cites HiGH (arXiv or "under review") for identity and reports identity
only as a system component, never with new ablations of the embedding.

## JIRS collection — facts read 19 Oct (link.springer.com/collections/cfacggejfe; journal 10846 guidelines)

- Journal of Intelligent & Robotic Systems, fully open access (article processing charge applies — check the amount
  and whether UCT/SA has a Springer Nature transformative agreement before committing), IF 3.1 (2025), median 12 days
  to first decision; single-blind review.
- Scope: agricultural robotics, sensing, imaging, phenotyping, "farm of the future", SDG 15. Two papers published so far
  (teleoperation HMI; active perception in vineyards).
- Springer Nature LaTeX template (sn-jnl) recommended; abstract 150–250 words; 4–6 keywords; MSC codes; category
  number(s) on page 1 (likely 7 Robot Vision and 10 Intelligent Interfaces); references numbered in brackets with full
  DOI links; mandatory "Statements and Declarations" (funding, competing interests, author contributions, data
  availability); figures named Fig1…, 300 dpi halftones, RGB.
- **LLM policy**: generative use of an LLM must be documented in the Methods (copy-editing need not be). Drafts written
  with Claude count as generative; the same disclosure question applies to RA-L (IEEE asks for disclosure in the
  acknowledgements). Paul to decide the wording for both papers.

## Schedule

| Week of | HiGH (RA-L) | JIRS 4D twin | Release report |
|---|---|---|---|
| 19 Oct | close open items: scene-graph figure (placeholder names Paul — anonymity), Acknowledgement withheld, last `\todo`, 19 unsourced numbers, reference fixes (types, published versions, DOIs) | read the collection; decide title + contribution list | — |
| 26 Oct | Paul's claim-by-claim pass; co-author list and order; LLM disclosure wording | outline (sn-jnl skeleton in its own Overleaf repo); list the measurements the paper needs that do not exist | — |
| 2 Nov | co-author read; final build, anon check, arXiv decision (post at submission or after) | gap experiments planned and queued on the A100 (see below) | — |
| 9 Nov | **submit Fri 13 Nov** via PaperPlaza | gap experiments run; numbers files | outline from the release README |
| 16 Nov – 7 Dec | reviews pending (RA-L first decision typically ~2–3 months) | figures + method and system sections drafted | draft with the release |
| 14 Dec | — | results drafted | **post with the launch** |
| Jan 2027 | — | full draft, co-author pass, Paul's claim pass | — |
| Feb 2027 | (revision if returned; reviewer-response skill) | **submit** (three months before the 31 May deadline, leaving room for a revision) | — |

## JIRS gap measurements (to confirm in the outline)

- Ledger over time: found-again rates with their protocol (Klapmuts Dec→Apr 825 of 912 bags; citrus 79 trees two days
  apart), with association error measured against hand-checked pairs, not only counts.
- Cross-platform reconstruction quality on one protocol: robot (Jackal citrus, A300 Klapmuts) vs phone (Kendu Bay,
  Gwakungu), training and held-out views, with the frame bar used for the demo stated as such.
- Interface: the 89-pair question bank across 8 languages (87 grounded) and the Way With Words field-interaction review,
  reported as answer correctness and grounding, with the failure cases.
- Cost: survey-to-twin time and GPU hours per hectare, serving cost per farm.

## Decisions needed from Paul

1. JIRS paper scope and author list (does Way With Words / the Kenyan field team co-author?).
2. APC: who pays, or whether a transformative agreement covers it.
3. arXiv timing for HiGH (with the RA-L submission, or after the first decision).
4. LLM-use disclosure wording for both papers.
