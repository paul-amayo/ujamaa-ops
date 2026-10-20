# Super hierarchy on citrus 05: spec with one result as the goal (2026-10-19, Paul: "spec out and have a single result as a goal")

## The goal, one sentence
On held-out frames of citrus 05 cell 1_0, the composed query **"the trunk of tree 36"** resolves to the right pixels: a held-out IoU
of **>= 0.50** against the SAM 3 "tree trunk" mask inside tree 36's mask, from one feature per Gaussian, with the cell's tree and row
identity unchanged (held-out argmax medians within 0.02 of E1's 0.773 / 0.859 for survey 05), where neither existing field can do it:
HiGH alone has no "trunk", and the LangSplat-style field has no "tree 36" (it lights every trunk in view).

That is the whole claim of the super hierarchy in one number: identity from the explicit graph and appearance from masks of masks,
read from the same point. If the composed IoU is below 0.50 while the parts are separable, the goal is missed and the paper's
story changes; if the parts of neighbouring trees are not separable (test 0 below), the paper is "explicit pins implicit" and the
composition is not attempted.

## Trunk visibility on the cell (measured 10-20, trunk_probe.py)
SAM 3 "tree trunk" on the 15 verdict frames finds 108 masks; 38 lie inside a near tree (>= 3 % of the frame), with areas of 0.1-0.3 %
of the frame (quartiles 965 / 1,362 / 2,940 px), 16 of them above the paper's 0.2 % head floor; every held-out frame has at least one,
and tree 36's trunk on the hero frame is 2,993 px, wholly inside its mask. Trunks are therefore present but small, at the size where E1
showed identity collapsing, so the goal is scored on the trunk masks >= 0.2 % of the frame (about one per held-out frame) and reported
beside an instance-recall readout (any lit pixel inside the right trunk), as E4 did for fruit.

## The decisive test
- Cell: citrus 05, chunk 1_0 (the hero cell: 8 rows, 38 trees, 7 fruit instances; splat, census W, maps, protocol files on disk).
- Frames: the same 14 verdict frames as E1 (4 census, 10 held-out); the composed query is scored on the 10 held-out frames only.
- Query: for every held-out tree head >= 3 % of the frame (E1's list), the composed query "trunk of tree k". Reference: the SAM 3 "tree
  trunk" mask of that frame intersected with tree k's supervision mask. Readout: the argmax rule over the part level inside the
  tree's containment (a pixel belongs to "trunk of tree k" when its tree-level argmax is k and its part-level argmax is a trunk node),
  no threshold. Number reported: median IoU over the heads whose trunk mask covers >= 0.2 % of the frame; the goal bar is 0.50,
  with instance recall reported beside it.
- Comparators on the same frames, same reference: (a) HiGH as published, "trunk" scored by the open-vocabulary cosine inside tree k
  (expected near 0, the field carries no appearance); (b) the LangSplat-style field, "trunk" at relevancy 0.5 without identity
  (expected to light all trunks: precision low); (c) the super hierarchy without pinned addresses (OpenHype-style, implicit only,
  identity by mask containment alone).
- Guard: E1 tree and row argmax medians of cell 1_0 before and after (unchanged within 0.02), nesting consistency across both
  ancestries (part -> tree -> row) reported.

## Test 0, run first (the thing that can fail cheaply)
Are the parts of neighbouring trees separable in CLIP image space inside pinned tree cones? On the 14 verdict frames: for each tree
mask, the sub-masks' CLIP crop embeddings; measure the nearest-neighbour tree of each sub-mask embedding among all trees in the frame
(does a trunk crop of tree 36 sit nearer tree 36's other crops than tree 60's?). The hero figure says whole-tree crops do not
separate (pairwise cosines 0.69-0.89). If sub-mask crops do not either, the implicit nodes will be placed by their explicit parent
only, and the composed query reduces to "trunk" x "tree 36" with the trunk read from the part level regardless of appearance
separability; the test tells us which story we are writing before the embedder is trained. One afternoon, CPU + one CLIP pass.

## Construction
1. **Explicit trunk of the hierarchy** (exists): section -> row -> tree -> fruit, addresses pinned, census-init, containment.
2. **Masks of masks** (new): inside every SAM 3 tree mask of a keyframe, automatic masks from SAM 3 (point-grid "everything" mode on the
   tree crop) kept when >= 90 % of their area lies inside the parent mask; two levels by area, coarse parts >= 5 % of the tree mask
   and fine parts >= 1 %; plus the prompted "tree trunk" masks (new prompt on 05; the prompt worked on Lindendhof and Fuji) and the
   existing fruit masks (sam3_fruit clips) as the finest level. Nesting by construction; every implicit node has exactly one parent.
3. **3D part instances** (new, reuse): per-view sub-masks are linked across views by the fruit registry's vote mechanism (Gaussians
   visible in a view vote for the mask they land in; links at S >= 8 and S/(S+C) >= 0.95; transitive-safe merge), so an implicit node
   is a 3D object with a set of views, not a per-image mask. Each node's anchor is the mean CLIP image embedding of its mask crops
   (sam3_clip_targets.py already computes mask-crop CLIP embeddings; lite_targets hold the lifted per-pixel version for 290 frames).
4. **Embedding** (extend): addresses pinned as now; implicit nodes free, anchored by their image embedding (MERU: image beyond text);
   the same distance, angle, alignment and reconstruction terms with the explicit graph's edges plus the mask-nesting edges; the
   decoder maps to CLIP space, so a text query reaches a part node by cosine as it reaches an address. Containment walk: one waypoint
   per level of the super hierarchy (section, row, tree, part, fruit) instead of four fixed steps.
5. **Seeding** (reuse): the census from the finest-level id maps (part ids below the tree ids); a Gaussian takes its finest node;
   tree and row are read by containment as now, parts at the own point.
6. **Query composition**: the pixel belongs to "part p of tree k" when its tree-level argmax is k and its part-level argmax is p's
   node (or, for an open-vocabulary part word, when the word's cosine at the part level is the maximum among part words).

## What is on disk for cell 1_0 (checked 10-19)
splat + census W + maps_v3 + protocol files; supervision/trees_only 659 frames (tree ids); sam3_v2 tree masks (clips 000-004),
sam3_fruit masks (clips 000-002, 175 confirmed oranges); arms/lite_targets 290 frames of lifted mask-crop CLIP (PCA-32) +
basis; the hero and lite relevancy dumps for kf_002411; the per-apple registry code (fruit_registry.py family) for the linking.
Not on disk: SAM 3 automatic sub-masks, "tree trunk" masks for 05, per-node CLIP anchors, the extended embedder.

## Budget (estimates, GPU = the A100 here)
SAM 3 automatic masks inside tree crops, 294 training views: ~1 s per tree crop x ~6 trees per view ~ 30 min; trunk prompt 294 views
~ 5 min; CLIP crops ~ 10 min; linking (registry votes) ~ 20 min; embedder with ~1,000 implicit nodes ~ 15 min; census + render +
score ~ 5 min. About half a day of compute, two days of work including test 0 and the comparators.

## Kill criteria and honesty
- Test 0 negative (sub-mask crops of neighbouring trees not separable): the composed-query goal is not pursued; report it.
- Composed IoU < 0.50 with identity unchanged: the parts are placed but the trunk is not read; try the trunk prompt masks as the only
  part level (no automatic masks) once, then stop.
- Identity moves by more than 0.02: the implicit levels disturb the explicit ones; the embedder's pinning is at fault, not the idea.
The reference for parts is held-out SAM 3 masks, the same status as E1's reference; no hand truth exists and the write-up says so.

## Order of work
1. Test 0 (separability of sub-mask crops inside tree masks) on the 14 verdict frames.
2. Masks of masks + trunk prompt on the 294 views; nesting check (every node one parent).
3. Linking into 3D part instances; count of nodes per level.
4. Embedder with pinned addresses + image-anchored parts; retrieval check extended to parts (each part's nearest tree node is its tree).
5. Census, render, the decisive test with the three comparators and the identity guard.
6. One figure: the held-out frame of Fig. 1 with "trunk of tree 36" under the three fields; one table: the composed IoU, the guard,
   the nesting.


## Result (2026-10-20, citrus 05 cell 1_0) — goal missed, mechanism measured
**Composed query "trunk of tree k", held-out, 13 heads with a trunk >= 0.2 % of the frame:** median IoU **0.071** (precision 0.17, recall
0.23; instance recall 0.85) under the final design (v4: trunk-only part nodes, part-view census, fruit-recipe seeding, embedder
`05_13D_super_v4`, waypoints at the embedder's level radii); best heads 0.47 / 0.40 / 0.34 / 0.27. Comparators on the same heads: HiGH as
published 0.000 (no part level; "trunk" by relevancy lights nothing), LangSplat-style 0.000 ("trunk" at relevancy 0.5 lights nothing; "bark"
lights everything at precision 0.00). Identity guard: held-out trees 0.736 / rows 0.827 / nesting 0.89 (published field on this cell
0.757 / 0.872) — identity survives the extra level once the walk's waypoints follow the embedder's radii (with the production 11.1 the rows
fell to 0.54-0.67).

**What holds.** The super hierarchy embeds: 22 trunk nodes beyond their trees on their trees' rays (angle median 0.5 deg, part -> nearest
tree 22/22, tree -> row 37/38), radii rows 1.96 / trees 3.91 / parts 6.76; the part level reads correctly wherever the splat carries it:
Spearman 0.83 (p < 0.001) between the trunk-node share of the opacity at the trunk's depth and the composed IoU over the 17 heads; the six
heads with share >= 0.5 score a median 0.30, the eleven below 0.03.

**Why the median fails.** (1) Geometry: a trunk (0.1-1 % of the frame) is carried by 1-40 Gaussians (median 13, opacity sum median 5.8),
with 300-9,000 other Gaussians projecting into its mask from other depths; the hero frame's trunk has 5. (2) Labels: those few primitives are
withheld by the canopy-era seeding rules (mass floor, background competes) unless promoted (v4), and a canopy part label ("leaves") in the
views without a trunk detection absorbs them (v1). (3) Anchors: part image embeddings are near-duplicates (trunk vs branch of one tree 0.96,
trunks across trees 0.97, trunk vs the tree's own address 0.21), so the part decision is fine where the tree decision is coarse. (4) The word
path (LERF relevancy at 0.5) is not a readout for parts this small on any field.

**Readings for the CVPR idea.** "Explicit pins implicit" stands: identity never comes from appearance at any level (test 0: parts of
neighbouring trees are 0.86 alike), the explicit graph supplies it and the hyperbolic structure holds with parts added. Composition fails on
primitives, not on the hierarchy: a part needs its own primitives (the fruit finding again: level-of-detail at the part) or must be a part
the splat already models (a canopy half, a sunny side, a row segment). Next step if pursued: the LoD split at the trunk masks (guided split,
as for fruit) before re-seeding; or choose parts of >= 3 % of the frame for the decisive test.

Artifacts: `<cell>/parts/{verdict,train,v2,v3,v4}` (masks, nodes, maps, protocol files, super_eval*.json, proj_heads.json, figures);
scripts `automation/h3dgs/v3/{parts_masks,parts_test0,parts_nodes,parts_hierarchy,build_super_seed,super_eval,super_fig,
super_retrieval_check,super_proj_diag,trunk_probe}.py`, chains under `automation/h3dgs/v3/chains/`; notebook 2026-10-20.
