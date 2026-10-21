# Publications plan — 19 Oct → 31 Dec 2026 (and the JIRS collection by Feb 2027)

Paul, 19 Oct: "there's a special issue open … let's plan publications towards the end of the year."
Skills: how-paul-writes (voice), paper-latex (build/checks/packaging), reviewer-response.

## Decisions, 20 Oct (Paul)

- **No CVPR submission.** The super-hierarchy idea (parts below trees, text-pinned addresses + image-anchored part
  nodes; A100 spec `~/code/plans/cvpr_super_hierarchy_spec.md`) merges into the RSS paper as part-level targets.
- **Three submissions by 31 Dec 2026:** HiGH (RA-L, brought forward to Fri 23 Oct; Paul is finishing the draft himself), RSS Azalai (Stage 1, 4 Dec), JIRS 4D twin (pulled
  forward from Feb to December; the schedule table below still shows Feb and needs re-planning).
- **Then a full UJAMAA systems paper with all the agents** — venue **Scientific African** (Paul, 21 Oct; Elsevier, open access,
  sponsored by the Next Einstein Forum, pan-African and multidisciplinary; APC listed at USD 720 excl. tax with a waiver
  policy, to confirm on the guide for authors). It subsumes the release report. Content: the whole system (surveys → twins →
  agents → answers), the eight-language farmer-facing evaluation, the agents without papers of their own (Bateleur, Spoor,
  Hapi), deployment cost and what broke; Tassili / Sankofa / Azalai cited to their papers. Timing: write once the RA-L and RSS
  decisions are in (Feb–Apr 2027); the December arXiv release report is its preprint.

## The outputs

| # | Output | Venue | Content (contribution boundary) | Target | Status 19 Oct |
|---|---|---|---|---|---|
| 1 | **HiGH** | IEEE RA-L (rolling; 6 + 2 paid pages; double-anonymous) | The method: hierarchically grounded hyperbolic embeddings for open-vocabulary queries in 3DGS farm models; containment, void check, fruit registry | **submit by Fri 13 Nov** | Draft 31 in the Overleaf repo (`Hyperbolic-Hierarchies-for-Precision-Agriculture`), 8 pages, builds clean |
| 2 | **UJAMAA release report** | arXiv (cs.RO) with the public release | The system as released: pipeline, data, demo, how to reproduce | **mid-Dec**, with the roadmap's Phase 5 launch | Not started; roadmap Phase 4 item |
| 3 | **UJAMAA 4D twin** | *Journal of Intelligent & Robotic Systems*, collection "AI and Soft Robotics in Agriculture: From Sensing to Harvesting and Beyond" (deadline 31 May 2027) | The system and its use over time: robot + phone surveys → H3DGS models → identity (cites HiGH, does not re-claim it) → per-plant ledger across surveys (found again, change, NDVI) → multilingual questions answered from the twin | **submit Feb 2027**; outline + gap experiments in Nov–Dec | Not started |
| 4 | **Azalai** (robot navigation to plants the twin names) | RSS 2027 (Athens, 6–11 Jul), two-stage: Stage 1 extended abstract (5 + 1 pages, anonymised) **4 Dec 2026**; decision 5 Feb; Stage 2 (8 pages) 16 Apr; final 30 Apr | A robot that monitors the farm over time by localising against the twin: the scene graph + robot layer + VT&R (ARU, ICRA 2026) and SANA (under review) brings the robot back to plants HiGH's hierarchy and the registry name; on arrival it confirms which plant it is at (twin render vs live frame) and re-observes it | Stage 1 **4 Dec** | Not started; Paul 20 Oct: one of the new agents, Azalai (navigation; first said Spoor) |

Overlap rule (both RA-L and JIRS forbid substantially overlapping submissions under review): HiGH owns the embedding
and its evaluation; the JIRS paper owns the temporal ledger, the multi-platform survey pipeline (Jackal, A300, phone),
deployment, and the interface. The JIRS paper cites HiGH (arXiv or "under review") for identity and reports identity
only as a system component, never with new ablations of the embedding.

## Papers ↔ Adinkra agents (Paul, 20 Oct)

Each paper is the science behind one agent, which also sets the contribution boundaries:
- **HiGH (RA-L) → Tassili**, the 3D field view: the embedding, containment, void check, fruit registry.
- **JIRS 4D twin → Sankofa**, change over time: surveys → twins → the per-plant ledger across surveys. Goes in Feb
  as planned; it does not wait for RSS.
- **RSS → Azalai**, navigation: an instruction or registry entry names a plant; the robot goes back to it over time
  and confirms it arrived (the arrival check is localisation, a component, not the claim). Builds on the ARU language-enabled scene graph + robot layer + VT&R3 paper (Mukuddem, Speed-Andrews, Amayo,
  published ICRA 2026) and on SANA row following (Travi et al., under review at ICRA 2027) and on HiGH; cites them, re-claims none of them.
  Azalai owns the robot in the loop (where is that plant, go to it, confirm it); Sankofa owns survey-to-survey
  change. Azalai may *use* a registry entry as a target but reports no farm-scale found-again rates (that is JIRS).
  Spoor (localisation) stays a later agent.

### RSS Stage 1 sketch (superseded by the full plan: `plans/rss2027_azalai.md`)
- Hypotheses: (H1) HiGH grounding matches the enumerated scene graph on numbered instructions and extends to attribute
  and count instructions it cannot express; (H2) a goal keyframe chosen as the target's best view (HiGH census
  weights = robot-layer edges) beats the first-observed keyframe on an arrival check; (H3) the arrival check (twin
  render vs live frame, E1 IoU protocol) catches wrong-plant arrivals that teach/repeat path RMSE misses.
- Before field time: offline on existing logs (Klapmuts Dec twin, Apr bags as the revisit; citrus).
- Feb–Apr: field runs to carry out the Extension Charter.
- Site: Klapmuts (Paul, 20 Oct). Open: first author (Adam
  Mukuddem owns the robot layer and VT&R); the ICRA decisions land ~late Jan, before Stage 2.

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
| 19 Oct | remaining after re-check: the scene-graph figure (still a placeholder box), four head counts (1,444 / 5,893 / 1,557 / 1,317) to add to `numbers/`, `% num-ok` on three design constants (512/256 layer widths, 1,024-word vocabulary), reference metadata fixes (entry types, published versions, DOIs). Acknowledgement and `\todo` were false alarms of the new checks (fixed) | read the collection; decide title + contribution list | — |
| 26 Oct | Paul's claim-by-claim pass; co-author list and order; LLM disclosure wording | outline (sn-jnl skeleton in its own Overleaf repo); list the measurements the paper needs that do not exist | — |
| 2 Nov | co-author read; final build, anon check, arXiv decision (post at submission or after) | gap experiments planned and queued on the A100 (see below) | — |
| 9 Nov | **submit Fri 13 Nov** via PaperPlaza | gap experiments run; numbers files | outline from the release README |
| 16 Nov – 7 Dec | reviews pending (RA-L first decision typically ~2–3 months) | figures + method and system sections drafted | draft with the release |
| 14 Dec | — | results drafted | **post with the launch** |
| Jan 2027 | — | full draft, co-author pass, Paul's claim pass | — |
| Feb 2027 | (revision if returned; reviewer-response skill) | **submit** (three months before the 31 May deadline, leaving room for a revision) | — |

Paper repositories (Paul, 21 Oct): RSS `~/Documents/UCT/SIL/Publications/Azalai_RSS` (GitHub `paul-amayo/Azalai_RSS`), JIRS
`~/Documents/UCT/SIL/Publications/Sankofa_JIRS` (GitHub `paul-amayo/Sankofa_JIRS`), both Overleaf-synced. Titles name what the
system does; the agent names stay internal.

## JIRS method update from the STPG pilot (21 Oct)

- The ledger's Dec→Apr link is one global yaw + translation; the 21 Oct trajectory-ICP alias (one lane off on a drifted
  stream) shows how fragile that is. The offline STPG gives per-keyframe spatial edges between epochs (0.15 m on strong
  edges, inlier count as the flag): re-associate the ledger through those edges, plant by plant, in locally registered
  frames; keep the global transform only as the prior. This is the 4D-twin contribution: change measured where the
  surveys are locally registered, with a declared site frame (`klapmuts/site_frame.json`, December FP_ENU0) and one
  splat per epoch (fusing would average the change away).
- Related work to sit between: per-plant temporal splats (GrowSplat, CASE 2025, arXiv:2505.10923; Grow with the Flow,
  arXiv:2602.08958; Riemannian growth fields, arXiv:2608.13007; Chebrolu et al. non-rigid 4D plant registration,
  PLOS ONE 2021, Pheno4D) and scene map updates (LT-Gaussian arXiv:2508.01704; GaussianUpdate arXiv:2508.08867; GaME
  arXiv:2506.06909; SI-Update). None is farm-scale, robot-collected, months apart, with identity kept per plant.
- Third epoch: Feb 2026 (two sessions) once the shared-drive transfer completes; April day one / day two as a same-week
  pair. Hand-checked association pairs become cheap to label from the STPG's matched keyframe images.

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
5. RSS (Azalai): first author and team (site decided: Klapmuts).
