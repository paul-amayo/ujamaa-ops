# UJAMAA open-source launch — adopter documentation plan

Status: DRAFT v2 for Paul, 2026-09-22. v1 (site-visitor copy) was the wrong
frame — this plan is for the person who finds UJAMAA at launch and wants to
RUN it: on our data first, then on their own orchard. Companion to
plans/web_deployment.md (hosting) and roadmap Phase 4 (release
engineering); this is the documentation half of Phase 4, planned now
because docs decide the repo cleanup, not the other way round.

## 1. The adopter, and the three supported journeys

Docs exist to answer one question: **"I want to use this — what exactly do
I do?"** Three tiers, each with an explicit promise:

- **T1 — Try it (an afternoon, one GPU):** clone, install, download the
  sample survey bundle, and stand up the full demo stack (viewer + panels +
  Adinkra) on OUR data. No robot, no farm. This is the tier that makes or
  breaks adoption: if T1 doesn't work on a clean machine, nothing else
  matters.
- **T2 — Your own orchard (the real product):** capture a survey with your
  own rig, run the pipeline end to end (ingest → poses → splat → hierarchy
  → ledger → serve). Requires sensors and patience; docs must say exactly
  how much of each.
- **T3 — Extend it (researchers/contributors):** architecture, data
  contracts, how to add a panel/language/pipeline stage, how to run the
  eval harnesses.

## 2. The documentation set

### T1 — Try it
1. **README (front door)**: what UJAMAA is in ten lines, one architecture
   figure, the three tiers, hardware floor (measured: the demo stack fits
   one 24 GB GPU — ollama Q4 ~9 GB + render service; VERIFY the floor on a
   smaller card before we print it), links down the ladder.
2. **INSTALL.md**: the honest environment story — pixi env for
   nerfstudio/training, system python for ingest, CUDA/driver floor,
   ollama + model pull (7.3 GB). Every step copy-pasteable, no "adjust to
   taste". Target: fresh Ubuntu 22.04 + one GPU → working in under an hour.
3. **QUICKSTART.md + sample survey bundle**: the load-bearing artifact.
   One small real survey (e.g. one citrus block: images, poses, LiDAR
   init, trained splat, hierarchy, ledger slice) packaged for download so
   T1 needs zero pipeline runs. Size target ≤ ~2 GB. Needs farm
   permission (§5) and a hosting home (release asset / HF dataset).
   Ends with: viewer walking the orchard + one Adinkra question answered
   in each of two languages.

### T2 — Your own orchard
4. **Capture guide**: what a survey IS — rig (ZED + LiDAR), drive pattern,
   speed, spacing, lighting; the field lessons the lab notebook already
   paid for (fg-mask regression, pose conventions, the Sep-6 roll bug)
   distilled into "do this, check that". Include a pre-flight checklist
   and a "is my capture usable?" validation step.
5. **Pipeline how-tos, one per stage, same template** (inputs → command →
   outputs → how to verify → common failures):
   a. Ingest (rosbag → monolithics; system python)
   b. Poses (LIO; when and how to refine with GLOMAP — the measured
      +2.9 dB recipe, incl. "use GLOMAP, not incremental COLMAP, on
      row-crop imagery")
   c. Splat training (two-stage census-init recipe, per-block; the psnr
      verification skill)
   d. Hierarchy build (markers → census → marker_hierarchy.json; the data
      contract every panel reads)
   e. Ledger (sankofa: association across epochs, NDVI)
   f. Serving (viewer 8001 / query 8002 / adinkra 8003 / render 8004 +
      ollama; systemd units from web_deployment P1 double as the docs)
6. **Configuration reference**: every env var (ADINKRA_*, data roots,
   ports), directory conventions (`<survey>/prod/<pillar>/…`), file
   formats (marker_hierarchy schema, ledger schema, monolithics) — the
   contracts, written once, linked everywhere.
7. **Troubleshooting + FAQ**: seeded from our own lab notebook incidents
   (env splits, CUDA/torch pinning, ollama not running, f-string/py3.10
   traps are ours not theirs — pick the user-facing ones).

### T3 — Extend
8. **ARCHITECTURE.md**: pillars, repos, data flow, where each contract
   lives; why splat + hierarchy + ledger rather than a monolith.
9. **Extension how-tos**: add a language (the panel_agents recipe — one
   dict entry + NO_ANSWER_MSG + validation caveat), add a panel agent, run
   the eval harnesses (adinkra_www, multiling_eval) on your own deployment.
10. **CONTRIBUTING.md + code of conduct + issue templates.**

### Publications & citing
11. **CITATION.cff + a citable anchor**: the roadmap's optional arXiv tech
    report becomes non-optional at launch — it is the answer to "how do I
    cite this?" and the natural home for the measured results (census-init
    recipe, refined-pose fleet numbers, the Adinkra eval, the multilingual
    findings). Decide scope: one systems report vs report + separate
    language-eval note.
12. **Publications page**: the tech report, the dossier (or its public
    summary — governance §5), Paul's list of related ARU papers/theses.

## 3. Doc-driven release engineering (docs first, cleanup follows)

The Phase 4 repo cleanup should be DRIVEN by T1/T2 docs: write the doc,
then make the repo match it, not vice versa.

- **Repo shape decision (blocking, week 1):** what is released, under
  which names — high/, aru_sil_core/, nerf_new/ (env), ujamaa/ (app),
  parts of ujamaa-ops/automation as the pipeline drivers? One umbrella
  repo with subpackages vs multiple repos with an index README. Docs
  can't name paths until this lands.
- **Path/secret scrub**: nothing may reference /home/paperspace, box
  hostnames, or farm coordinates (the §9-style host detail stays in
  private readouts).
- **The T1 gate = a clean-machine dry run**: docs are tested artifacts.
  Before launch, execute INSTALL + QUICKSTART verbatim on a fresh cloud
  GPU instance (not our boxes) — every deviation is a doc bug. Repeat
  once more after freeze. This is the docs' equivalent of
  commit-after-green.
- **Licences**: per-repo LICENSE (decide MIT/Apache-2.0 vs research-only),
  third-party notices (nerfstudio, gaussian-splats-3d, Gemma terms for
  the served model — users pull Gemma themselves, our docs just say how).

## 4. What exists vs what is genuinely new writing

Adaptable: the recipes already live in automation/ scripts, lab-notebook
entries, SANKOFA_SPEC, the splat_viewer README, and CLAUDE.md environment
facts — the pipeline how-tos are largely *transcription with verification*,
not invention. The glossary/limits material from the Verbamore briefing
reuses directly (T1 users need the same honesty about Adinkra). Genuinely
new: README/INSTALL/QUICKSTART, the capture guide, the sample bundle, the
config reference, ARCHITECTURE, the tech report.

## 5. Governance gate (unchanged from v1, now with the sample bundle)

Farm permission now covers a REDISTRIBUTABLE sample survey (imagery +
positions), not just web copy — get it in writing early, or plan a
fallback (synthetic/anonymised block). Verbamore naming, dossier
public-vs-summary, Gemma serving terms: as v1. No internal
hostnames/paths anywhere public.

## 6. To-do (owner · target)

**Week 1 — decisions that unblock everything:**
- [ ] Repo shape + naming (Paul + Claude, one sitting)
- [ ] Licence choice per repo (Paul)
- [ ] Sample-bundle farm permission requested (Paul)
- [ ] Hardware floor verified on a 24 GB card (Claude, after G0 frees the
      A100 — or a cheap cloud instance)

**Weeks 2–3 — Claude drafts, Paul batch-reviews:**
- [ ] README + INSTALL + QUICKSTART (against the sample bundle)
- [ ] Sample bundle built + hosted (block choice: small, pretty, permitted)
- [ ] Pipeline how-tos a–f (transcribed from automation/ + notebook,
      each verified by running it)
- [ ] Configuration reference + capture guide
- [ ] ARCHITECTURE + extension how-tos + CONTRIBUTING

**Week 4 — the gate:**
- [ ] Clean-machine dry run of T1 verbatim; fix doc bugs; repeat
- [ ] CITATION.cff + tech-report scope decision (Paul)
- [ ] Docs land in the released repos; site (/docs) serves the same files

Site-visitor copy (v1 of this plan: landing page, publications page,
demo-user help) is now a thin layer over these docs — it survives as a
1-page landing + links, drafted after the adopter docs freeze.
