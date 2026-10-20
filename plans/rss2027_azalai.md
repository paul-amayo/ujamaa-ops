# RSS 2027 — Azalai: sending a robot back to the plant the twin names (plan, draft 20 Oct 2026)

Paul, 20 Oct: "there is much to be explored combining HiGH with something else"; the paper is one of the new agents,
**Azalai** (navigation). Papers ↔ agents: HiGH = Tassili, JIRS 4D twin = Sankofa, RSS = Azalai; Spoor (the live
in-field tracker) is a later agent. Programme plan: `plans/publications_2026Q4.md`.

## 1. Venue (CfP read 20 Oct, roboticsconference.org/information/cfp)

- RSS 2027, Athens, 6–11 July 2027. Two stages, both anonymised.
- **Stage 1: extended abstract, 5 pages + 1 of references, due Fri 4 Dec 2026 23:59 AoE.** Judged on the hypothesis
  and the methodology; complete experiments are not required.
- 5 Feb 2027: reject or invitation to rebuttal; rebuttal 5–12 Feb; 26 Feb the area chairs issue the **Extension
  Charter** (the clarifications, evaluations and extensions Stage 2 must carry out).
- **Stage 2: 8 pages excluding references, due 16 Apr 2027**; supplementary (zip, ≤ 100 MB) 19 Apr; decision 30 Apr.
  Papers are checked against the Charter and "will not be rejected for yielding negative experimental results".
- Double submission: a paper under submission to another conference is forbidden; arXiv/tech reports are fine.
  LLM use permitted (disclosure encouraged).

Consequence for how we write it: Stage 1 is a pre-registration. State hypotheses, metrics, baselines and the decision
rule now; Stage 2 reports them whichever way they fall.

## 2. Where it sits (boundaries)

| Work | Status | What Azalai takes from it | What Azalai must not re-claim |
|---|---|---|---|
| HiGH (Tassili) | RA-L, submit 13 Nov | open-vocabulary identity, containment, argmax query rule, census weights, fruit registry, E1 IoU protocol | the embedding or any new ablation of it |
| Language-enabled scene graph + robot layer + VT&R3 (Mukuddem, Speed-Andrews, Amayo) | ICRA, under review | robot layer (keyframe ↔ object edges), LLM instruction decomposition, VT&R3 multi-experience repeat | enumerated addressing, the repeat results |
| SANA row following (Travi et al.) | ICRA, under review | optional in-row driving between keyframes | row following itself |
| JIRS 4D twin (Sankofa) | JIRS, submit Feb 2027 | a registry entry (e.g. "missing since December") as a *target* | farm-scale found-again rates, the survey-to-survey ledger |
| Spoor | later agent | — | continuous live tracking; Azalai checks position once, at the goal |

All prior ARU work is cited in the third person (anonymised).

## 3. The gap and the claim

The ICRA system can be told "go to plant 4 in line 4" and repeat a taught route there. It cannot be told *which*
plant by what it is or what happened to it (rows are homogeneous, so addresses are numbers); its goal is the first
keyframe that ever saw the plant; and nothing checks that the robot reached the right plant, only that it stayed on
the taught path (RMSE). On a farm monitored over months the questions are of the other kind: "go to the bag that was
missing in April", "the tree in row 3 with the most fruit".

**Claim (one sentence for the abstract):** a robot can be sent back, across sessions, to a plant named by what it is,
how many, or what changed, by grounding the instruction in a hierarchical twin, choosing the goal as the plant's best
view, and confirming arrival against the twin, without a planner or thresholds tuned per plant.

Working titles: "Go to the one that changed: sending field robots back to plants named by a hierarchical twin" /
"Azalai: instruction-to-plant navigation over a hierarchical agricultural twin".

## 4. Method (what exists, what is new)

1. **Twin and registry** (exists): H3DGS model with HiGH identity; plant registry (Klapmuts ledger v5 grow bags; citrus
   fruit registry). Robot-layer keyframes are the survey's own training views.
2. **Grounding** (new combination): the LLM only decomposes the instruction into sub-tasks (as in the ICRA system);
   each sub-task's target is resolved by HiGH containment + argmax, or by a registry lookup for count/change terms.
   No per-object thresholds (memory: relevancy-not-fitted-cuts).
3. **Goal keyframe** (new): HiGH's census weights already say how much each training view sees of each object;
   read as robot-layer edges, the goal is the keyframe that sees the target best, not the first that saw it.
4. **Driving** (exists): VT&R3 multi-experience repeat to the goal keyframe (SANA in-row optional, not claimed).
5. **Arrival check** (new): render the twin's target mask at the repeat pose, segment the live frame, score with the
   E1 IoU protocol; arrived = target's IoU is the argmax over the target and its row neighbours.

6. **Part-level targets** (merged from the CVPR super-hierarchy idea, Paul 20 Oct): parts below the trees, anchored
   by image embeddings of their masks while trees keep pinned addresses, so a target can be "the trunk of tree 36".
   Status on the A100 (20 Oct): parts embed and identity holds, but the composed trunk query is at IoU 0.07–0.10
   against a 0.50 bar (a trunk is carried by ~13 Gaussians); the pre-registered next step is part-share seeding.
   In this paper parts are a goal level for navigation (go to and view the part), so the bar to meet is the arrival
   check at part level, stated in Stage 1 as an extension, not a claim.

## 5. Hypotheses (pre-registered)

| | Hypothesis | Metric | Baseline | Decision rule |
|---|---|---|---|---|
| H1 | Twin grounding matches enumerated addressing on numbered instructions and extends to identity / count / change instructions | correct target per instruction, by category (the ICRA paper's 7 categories + identity, count, change) | the enumerated scene graph + LLM (ICRA system) | H1 holds if numbered categories are within 1 instruction of the baseline per category and the new categories score where the baseline cannot express them |
| H2 | Best-view goals beat first-sighting goals | target mask IoU in the frame at the goal; robot-to-plant distance | first-sighting keyframe (ICRA rule) | paired comparison over the same targets, sign test |
| H3 | The arrival check catches wrong-plant arrivals that path RMSE misses | wrong-plant detection rate and false alarms; arrivals labelled from lidar ground truth | path RMSE threshold (ICRA's verification) | report the confusion matrix at the argmax rule, no tuned threshold |

## 6. Data and sites

- **Klapmuts (primary):** December 2025 A300 survey = the twin; April 2026 ZED survey = the revisit; ledger v5
  (911 / 912 bags, 825 pairs) gives "missing", "new" and "moved" targets with lidar-registered positions. Field runs
  Feb–Apr 2027 here (the A300 + ZED 2i rig; same farm as SANA, to confirm).
- **Citrus 05 (offline only, public data):** identity and fruit-count instructions, H1 and the arrival check on logged
  frames; no field runs (we have no access to that orchard).
- **The ICRA paper's farm** (14 rows, ~60 m, Husky A200): to confirm with Adam whether a twin can be built from its
  logs; it would give the baseline on its home ground.

Checks before relying on Klapmuts (this week): stereo pairs present in the Klapmuts bags (VT&R3 needs them); the
Klapmuts H3DGS held-out quality is 16.8–17.8 dB per chunk (09-23 notebook), below the demo bar; the arrival check
uses masks, not photometric match, but the masks must be scored first.

## 7. Stage 1 evidence (offline, from logs, by 27 Nov)

Stage 1 does not need results; a small pilot makes the hypotheses credible.
- **Instruction bank v0:** ~10 per category × 10 categories, English, user voice, frozen before scoring; ground truth
  from the registry. Run grounding on Klapmuts and citrus. (H1 pilot.)
- **Best view vs first sighting:** for registry targets, score the target mask in both keyframes of the logged
  survey. (H2 pilot, no robot.)
- **Arrival check on logs:** take April frames at each bag and at its row neighbour; does the check pick the right
  one? (H3 pilot.)
Every number in the abstract resolves to a provenance file (paper-latex number audit).

## 8. Stage 2 (Feb–Apr, carried out against the Charter)

- Teach once in the first session; repeat across ≥ 4 sessions, varied light (the ICRA protocol: sunny midday,
  afternoon, overcast).
- Per session: the frozen instruction bank on the robot; arrivals labelled from lidar; interventions counted.
- Ablations: goal rule (best view vs first sighting), arrival check on/off, grounding (twin vs enumerated).
- If the Charter asks for it: a twin made months before the run (December twin, March run) as the stress case.

## 9. Schedule

| Week of | Azalai | Note |
|---|---|---|
| 20 Oct | this plan; Paul + Adam: lead author, team, farm | HiGH has priority until 13 Nov |
| 27 Oct | data checks (stereo, Klapmuts masks); instruction bank v0 written and frozen | |
| 3 Nov | grounding harness on the A100 (reuse Adinkra panel + HiGH query) | |
| 9 Nov | — | **HiGH submits 13 Nov** |
| 16 Nov | pilots H1–H3 on logs; numbers files | |
| 23 Nov | 5-page draft (how-paul-writes, paper-latex, RSS template) | |
| 30 Nov | Paul's claim pass; build, anon check; **submit Fri 4 Dec** | |
| Feb 2027 | 5 Feb decision; rebuttal by 12 Feb; Charter 26 Feb | JIRS submits in Feb |
| Mar–Apr | field sessions at Klapmuts; Stage 2 by 16 Apr | |

## 10. Risks

- Three weeks after HiGH for a new paper: the pilots must be small; the abstract can stand on hypotheses.
- The ICRA decisions (expected around late January, to check) land before Stage 2; if either is rejected the RSS
  paper still cites it as a preprint, and nothing from it is re-claimed.
- Klapmuts twin quality and the stereo check (section 6) could move the field site.
- Field access Feb–Apr (berry season, farm operations) — to confirm with the farm.

## 11. Decisions for Paul

1. Lead author and team (Adam Mukuddem owns the robot layer and VT&R; Travi for SANA if in-row driving is used).
2. Field site: Klapmuts, or the ICRA farm.
3. Whether the Stage 1 abstract carries the pilot numbers or hypotheses only.
4. Title.
