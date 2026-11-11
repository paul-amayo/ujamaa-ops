# The State of Robotics in Africa as a paper — plan, draft 11 Nov 2026

Paul, 11 Nov: "another planned paper, Science Robotics or RA-M. Let's plan." Source: *The State of Robotics in Africa 2026*
(SORA), African Robotics Network (AfROB), ed. K. C. Mbanisi, contributing editor P. Amayo, June 2026, CC BY-SA 4.0,
doi 10.5281/zenodo.20749780; 142 pages, image-only PDF in `~/Documents/UCT/SIL/Publications/` (OCR text of this session in the
scratchpad; re-OCR before quoting). This paper is outside the UJAMAA agent split (HiGH = Tassili, JIRS = Sankofa, RSS = Azalai,
systems paper = Scientific African); it must not re-claim any of their results.

## 1. What the report already holds

| Pillar | Evidence in the report | Method behind it |
|---|---|---|
| Industry | ~80 use cases (Fig 2.1–2.7): aerial 41.8 %, ground 28.6 %, fixed-base 23.1 %, marine 5.5 %; agriculture and environmental monitoring lead; Southern Africa ~40 % (South Africa 22 of 35); 71 % private-led; >60 % integrator-led; robot density ~10 per 10,000 workers (IFR) | desk research + 15 interviews; coding of sector, robot type, region, actor type (integrator / innovator / implementer), organisation type |
| Talent | ~60 universities with robotics-related programmes; >85 % mechatronics; 47.5 % undergraduate-only, ~16 % to doctoral level; competitions reach >95 % of countries (55 % in two, 30 % in all three); ILO/UNESCO attainment data | desk research, public datasets |
| Research | ~60 labs (70 % university, 20 % government, 10 % private); ICRA + IROS 2014–2025: 71 papers with African affiliations (<1 %), South Africa 29, Egypt 17, Libya 10; ~two-thirds with non-African co-authors (100 % Libya, 94 % Egypt, 87 % Algeria, ~31 % South Africa); one paper co-authored across two African countries; R&D/GDP (only Egypt ≥ 1 %) | Paper Search Interface over two venues; World Bank; THE rankings |
| Policy | 13 national AI frameworks (2025); drone rules in 23 %+ of countries (2017 figure); data protection in 65 %; benchmarking against global leaders and emerging adopters (four levers) | desk research |
| Synthesis | three cross-cutting themes (fragmented ecosystem; robotics serves development when local priorities lead; talent both asset and constraint); four global trends; stakeholder recommendations | interviews + synthesis |

The report's central empirical observation, stated in the industry chapter: **in Africa robots are adopted to fill gaps in
productivity, data, safety and access, not to substitute labour** (low wages, high capital cost: a UR arm at US$50–100k vs
monthly earnings < US$500), which is why drones in agriculture and environmental monitoring lead rather than manipulators
in manufacturing.

## 2. Why a paper cannot be the report re-typeset

- The report is published with a DOI. Both venues require original content; a paper that repeats it is redundant
  publication. The paper must cite it, say what is new, and add **peer-reviewable method and new analysis**.
- Most counts rest on a desk-research coding with no stated codebook, inclusion rule or inter-coder agreement, and two
  bibliometric venues. Reviewers will ask for exactly these.
- Several numbers are dated or thin for a journal (drone rules from 2017; ~80 cases with uneven country coverage that
  the report itself calls a finding).

## 3. The claim

**Robotics in Africa is adopted where it fills gaps rather than replaces labour, and the research system that should serve
that adoption is linked outward, not inward.** Two measured halves:
1. *Adoption:* coded on an explicit motivation scheme (labour substitution / productivity gap / data gap / safety / access
   and logistics), the African use cases are dominated by gap-filling, with drones as the platform that makes it affordable;
   contrasted with the global picture (IFR industrial stock, drone use-mix in the US).
2. *Research:* African robotics output at the field's main venues is small, concentrated in three institutions, and
   collaborates mostly outside the continent (one intra-African co-authorship in a decade), so the knowledge base is not
   organised around the adoption pattern in half 1.
The consequence the paper argues: the research agenda and the collaboration structure should follow the gap-filling pattern
(field robotics, drones, agricultural and environmental autonomy, teleoperation services) and be built regionally.

## 4. Venue (one, with a cheap check of the other)

| | Science Robotics | IEEE Robotics and Automation Magazine |
|---|---|---|
| Form that fits | Review (editor-commissioned in practice) or a short Viewpoint/Focus on the thesis | Survey/review feature (RA-M "welcomes" these; a critical or subjective viewpoint is allowed) |
| Length | to confirm on the AAAS author page (403 to our fetch) | ≤ 9 magazine pages, ~4,500 words, ≤ 10 figures; survey exempt from the 20-reference cap; US$250/page beyond 9 |
| Entry | presubmission inquiry / editor interest | proposal first: timeliness, relevance, authors' qualifications |
| Audience | broad science; wants one claim of general interest | IEEE RAS, which already backs the Robotics in Africa forums (ICAR 2023, IROS 2024, ICRA 2025) and ICRA@40 Africa |
| Risk | high: an ecosystem survey is unusual; commissioned Reviews are by invitation | moderate: good fit, data-rich figures, community support |

**Recommendation: RA-M survey feature as the paper, with a one-paragraph Science Robotics inquiry sent first.** The inquiry
costs a day; if the editors want a Review, the same analysis serves it and RA-M waits; if not (the likely outcome), the RA-M
proposal goes the same week. Do not split the content into two papers: one dataset, one claim, one venue.

## 5. What has to be added (the paper's own contribution)

1. **Use-case dataset v2 with a codebook.** Inclusion rule (deployed or piloted in an African country, 2015–2026, robot by
   the ISO definition the report uses), fields as the report's plus *motivation* (section 3), deployment status, local vs
   imported hardware, scale. Two coders on a 30 % overlap, Cohen's kappa reported; update the ~80 cases to 2026 and add the
   under-covered regions (Central Africa) by a stated search. Release on Zenodo (CC BY-SA, as the report).
2. **Bibliometrics over the field, not two conferences.** ICRA, IROS, RSS, CoRL, RA-L, T-RO, IJRR, Science Robotics, plus
   field venues (JFR, Computers and Electronics in Agriculture for agricultural robotics) and the regional ones (RobMech,
   AFRICON), 2014–2025, from OpenAlex/DBLP with affiliation-country matching; per-country and per-institution output,
   co-authorship network (intra-African vs external), topic mix of African papers against the adoption mix of half 1.
   **Comparators:** Latin America and South/South-East Asia, so "small and outward-linked" is measured against peers, not
   only against the global total.
3. **The two halves joined.** The figure the paper exists for: the sector/platform mix of adoption (left) beside the topic
   mix of African research (right), with the mismatch (or match) read off.
4. **Talent and policy compressed** to what bears on the claim (programme levels, competition reach, AI-policy coverage);
   update the 2017 drone-regulation figure to 2026 (ICAO/national CAA registers).
5. **Interviews**: 15 already done; cite them as the report's and use them only for interpretation, unless ethics clearance
   covers re-use (check with AfROB / Olin).

## 6. Figures (RA-M, ≤ 10)

1. Map: use cases by country with platform glyphs (from report Fig 2.1, rebuilt from dataset v2).
2. Adoption motivation × sector (the gap-filling result).
3. Platform mix in Africa vs global references (IFR, drone use-mix).
4. Research output over time by country and institution concentration (Lorenz curve).
5. Co-authorship network: African institutions, their external partners, intra-African links.
6. Adoption mix vs research topic mix (the joined figure).
7. Talent pipeline: programme levels and competition reach.
8. Policy coverage map (AI frameworks, drone rules 2026, data protection).
9. One photograph of a deployed system (RA-M asks for a high-quality colour photo), with permission.

## 7. Authorship and permissions (decide before writing)

- The report is Mbanisi's (editor, AfROB executive director); Paul is contributing editor. Natural order: Mbanisi first,
  Amayo second or corresponding, research contributors who build dataset v2 and the bibliometrics as co-authors; Expert
  Advisory Group acknowledged. Paul to agree this with Ken before any proposal goes out.
- AfROB/Olin funding acknowledgement; CC BY-SA means derived figures must carry the licence and attribution.
- No conflict with the UJAMAA papers; UCT's ARU appears in the report's lab spotlight, so the paper must describe its own
  authors' institutions in the same neutral tone as any other.

## 8. Schedule (fits around Azalai 3 Dec and JIRS)

| When | Work | Who |
|---|---|---|
| by 20 Nov | Paul + Ken: authorship, venue, go; Science Robotics inquiry sent | Paul, Ken |
| 20 Nov–18 Dec | codebook written and frozen; bibliometric pipeline (OpenAlex) built and run; dataset v2 search | agents + research contributors |
| Dec | RA-M proposal (1–2 pages) if Science Robotics declines | Paul |
| Jan 2027 | double coding, kappa; comparators; the joined figure | contributors + agents |
| Feb 2027 | draft (how-paul-writes; RA-M house style), figures | Paul, Ken |
| Mar 2027 | submit; dataset v2 on Zenodo | |
JIRS (Feb 2027) and Azalai Stage 2 (Feb–Apr 2027) take priority; this paper uses contributors' time, not the A100.

## 9. Decisions for Paul

1. Venue: RA-M survey, with a Science Robotics inquiry first (recommended), or Science Robotics only.
2. Authorship with Ken Mbanisi and AfROB, and who builds dataset v2.
3. Whether the paper carries the claim of section 3 (recommended) or stays a descriptive survey.
4. Interview re-use: needs the ethics position of the original study.
