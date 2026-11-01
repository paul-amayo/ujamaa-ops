#!/usr/bin/env python3
"""smac_patch_mc_params.py (30 Oct 2026) — scratch-clone patch: the MapClosures constants that were tuned for KITTI-scale local maps
(ORB on the density image, the self-similarity filter) become environment overrides with the upstream defaults unchanged:
SMAC_MC_SELFSIM (35: a feature is kept only if its second-nearest descriptor in the same image is further than this; set 0 to keep all),
SMAC_MC_FAST (35), SMAC_MC_EDGE (31), SMAC_MC_PATCH (31), SMAC_MC_NLEVELS (1), SMAC_MC_SCALE (1.0), SMAC_MC_NFEAT (500).
Run: python3 smac_patch_mc_params.py <clone root>"""
import sys
p = sys.argv[1] + "/thirdparty/map_closures_vendored/cpp/map_closures/MapClosures.cpp"; s = open(p).read()
old = """static constexpr int min_no_of_matches = 2;
static constexpr int no_of_local_maps_to_skip = 3;
static constexpr double ground_alignment_resolution = 5.0;
static constexpr double self_similarity_threshold = 35.0;

// fixed parameters for OpenCV ORB Features
static constexpr float scale_factor = 1.00f;
static constexpr int n_levels = 1;
static constexpr int first_level = 0;
static constexpr int WTA_K = 2;
static constexpr int nfeatures = 500;
static constexpr int edge_threshold = 31;
static constexpr int score_type = 0;
static constexpr int patch_size = 31;
static constexpr int fast_threshold = 35;"""
assert s.count(old) == 1
new = """static constexpr int min_no_of_matches = 2;
static constexpr int no_of_local_maps_to_skip = 3;
static constexpr double ground_alignment_resolution = 5.0;
// scratch-clone patch: KITTI-tuned constants overridable from the environment (upstream values are the defaults)
static double envd(const char *k, double d) { const char *v = std::getenv(k); return v ? std::atof(v) : d; }
static const double self_similarity_threshold = envd("SMAC_MC_SELFSIM", 35.0);

// parameters for OpenCV ORB Features
static const float scale_factor = static_cast<float>(envd("SMAC_MC_SCALE", 1.0));
static const int n_levels = static_cast<int>(envd("SMAC_MC_NLEVELS", 1));
static constexpr int first_level = 0;
static constexpr int WTA_K = 2;
static const int nfeatures = static_cast<int>(envd("SMAC_MC_NFEAT", 500));
static const int edge_threshold = static_cast<int>(envd("SMAC_MC_EDGE", 31));
static constexpr int score_type = 0;
static const int patch_size = static_cast<int>(envd("SMAC_MC_PATCH", 31));
static const int fast_threshold = static_cast<int>(envd("SMAC_MC_FAST", 35));"""
s = s.replace(old, new)
if "#include <cstdlib>" not in s: s = s.replace("#include <iostream>", "#include <cstdlib>\n#include <iostream>", 1)
open(p, "w").write(s); print("map_closures: env parameters")
