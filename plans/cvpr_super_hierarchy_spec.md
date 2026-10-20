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

## The decisive test
- Cell: citrus 05, chunk 1_0 (the hero cell: 8 rows, 38 trees, 7 fruit instances; splat, census W, maps, protocol files on disk).
- Frames: the same 14 verdict frames as E1 (4 census, 10 held-out); the composed query is scored on the 10 held-out frames only.
- Query: for every held-out tree head >= 3 % of the frame (E1's list), the composed query "trunk of tree k". Reference: the SAM 3 "tree
  trunk" mask of that frame intersected with tree k's supervision mask. Readout: the argmax rule over the part level inside the
  tree's containment (a pixel belongs to "trunk of tree k" when its tree-level argmax is k and its part-level argmax is a trunk node),
  no threshold. Number reported: median IoU over those heads; the goal bar is 0.50.
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
