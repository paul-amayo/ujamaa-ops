#!/usr/bin/env python3
"""smac_patch_centroids.py (30 Oct 2026) — scratch-clone patch for the SegMatch descriptor arm: the recognised matches are written with the
centroids of both segments (matching_centroids.csv: keyframe timestamp, source id, target id, source centroid xyz in the keyframe frame,
target centroid xyz in the target-cloud frame) and the app writes keyframes.csv (map_idx, session, first_scan, last_scan) so that a
keyframe timestamp (map_idx seconds) maps back to its scans. Run: python3 smac_patch_centroids.py <clone root>"""
import sys
root = sys.argv[1]
p = root + "/src/segmatch_worker.cpp"; s = open(p).read()
old = """    LOG(INFO) << "[diag] matchLocalMap: recognised " << recognized_matches.size();"""
assert s.count(old) == 1
s = s.replace(old, old + """
    {   // scratch-clone patch: recognised matches with both centroids, for the descriptor-arm score
      std::ofstream cc("matching_centroids.csv", std::ios::app);
      const segmatch::SegmentedCloud target_segments = segmatch_.getTargetAsSegmentedCloud();
      for (const auto &m : recognized_matches) {
        segmatch::Segment a, b;
        if (curr_segments_.findValidSegmentById(m.ids_.first, &a) && target_segments.findValidSegmentById(m.ids_.second, &b)) {
          const auto &ca = a.getLastView().centroid; const auto &cb = b.getLastView().centroid;
          cc << latest_pose.time_ns << "," << m.ids_.first << "," << m.ids_.second << "," << ca.x << "," << ca.y << "," << ca.z << "," << cb.x << "," << cb.y << "," << cb.z << "\\n";
        }
      }
    }""")
open(p, "w").write(s); print("worker: centroid export")
p = root + "/apps/segmapper_node.cpp"; s = open(p).read()
old_t = """      map_idx_to_id[map_idx] = map_id;

      //Take into SegMap
      //UNCOMMENT TO USE SEGMENTED CLOUD"""
assert s.count(old_t) == 1, s.count(old_t)
s = s.replace(old_t, """      map_idx_to_id[map_idx] = map_id;
      { std::ofstream kf(FLAGS_out_dir + "/keyframes.csv", std::ios::app); kf << map_idx << ",teach," << candidate_first_idx_teach << "," << i << "\\n"; }   // scratch-clone patch

      //Take into SegMap
      //UNCOMMENT TO USE SEGMENTED CLOUD""")
old_r = """      map_idx_to_id[map_idx] = map_id;

      //Take into SegMap
      //Uncomment to use segemneted cloud"""
assert s.count(old_r) == 1, s.count(old_r)
s = s.replace(old_r, """      map_idx_to_id[map_idx] = map_id;
      { std::ofstream kf(FLAGS_out_dir + "/keyframes.csv", std::ios::app); kf << map_idx << ",repeat," << candidate_first_idx << "," << i << "\\n"; }   // scratch-clone patch

      //Take into SegMap
      //Uncomment to use segemneted cloud""")
open(p, "w").write(s); print("app: keyframes.csv")
