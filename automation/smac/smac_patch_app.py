#!/usr/bin/env python3
"""smac_patch_app.py (30 Oct 2026) — make the SMaC app (aru_segmap test/smac-evaluation, apps/segmapper_node.cpp) runnable on the A100
without touching the branch: its hard-coded /home/quad paths become gflags, the python viewer becomes optional, and the raw-cloud
arm (MapClosures on the whole keyframe instead of the SegMatch segments) becomes --arm raw. Applied to the scratch clone
/home/paperspace/code/aru_segmap_smac only. Run: python3 smac_patch_app.py <path to segmapper_node.cpp>"""
import re, sys
p = sys.argv[1]; s = open(p).read(); n0 = s.count('/home/quad')
flags = '''#include <gflags/gflags.h>
#include <boost/filesystem.hpp>
DEFINE_string(config_dir, "", "directory with icp_dynamic_outdoor.yaml and input_filters_outdoor.yaml");
DEFINE_string(python_dir, "", "directory with SegMapView.py (only used with --viz)");
DEFINE_string(teach_dir, "", "teach split: <dir>/laser/laser_<i>.ply, <dir>/transforms/transform_<i>.txt");
DEFINE_string(repeat_dir, "", "repeat split, same layout");
DEFINE_string(out_dir, "", "results directory");
DEFINE_int32(n_teach, 899, "number of teach scans to read");
DEFINE_int32(n_repeat, 900, "number of repeat scans to read");
DEFINE_string(arm, "segments", "what MapClosures sees: segments (SegMatch segments of the keyframe) or raw (the whole keyframe cloud)");
DEFINE_bool(viz, false, "embed the python viewer");
'''
assert s.count('int main(int argc, char **argv) {') == 1
s = s.replace('int main(int argc, char **argv) {', flags + 'int main(int argc, char **argv) {\n  gflags::ParseCommandLineFlags(&argc, &argv, true);\n  boost::filesystem::create_directories(FLAGS_out_dir + "/density_maps");', 1)
s, c1 = re.subn(r'"/home/quad/code/aru_segmap/src/"\s*"config/kitti/icp_dynamic_outdoor.yaml"', 'FLAGS_config_dir + "/icp_dynamic_outdoor.yaml"', s)
s, c2 = re.subn(r'"/home/quad/code/aru_segmap/src/config/kitti/"\s*"input_filters_outdoor.yaml"', 'FLAGS_config_dir + "/input_filters_outdoor.yaml"', s)
s = s.replace('"/home/quad/code/aru_segmap/src/python"', 'FLAGS_python_dir')
s, c3 = re.subn(r'py::object viewer =\s*py::module_::import\("SegMapView"\)\.attr\("SegMapViewer"\)\("path",\s*"path"\);',
                'py::object viewer;\n  if (FLAGS_viz) viewer = py::module_::import("SegMapView").attr("SegMapViewer")("path", "path");', s)
s = s.replace('for (int i = 1; i <= 899; i += 1) {', 'for (int i = 1; i <= FLAGS_n_teach; i += 1) {', 1)
s = s.replace('for (int i = 1; i <= 900; i += 1) {', 'for (int i = 1; i <= FLAGS_n_repeat; i += 1) {', 1)
# input paths in order of appearance: teach laser, teach transform, repeat init pose, repeat laser, repeat transform, reference pose (teach)
order = ['FLAGS_teach_dir', 'FLAGS_teach_dir', 'FLAGS_repeat_dir', 'FLAGS_repeat_dir', 'FLAGS_repeat_dir', 'FLAGS_teach_dir']
pat = re.compile(r'"/home/quad/data/laser_splits/split_1/(laser/laser_|transforms/transform_)"'); k = [0]
def sub(m):
    d = order[k[0]] if k[0] < len(order) else 'FLAGS_teach_dir'; k[0] += 1; return f'{d} + "/{m.group(1)}"'
s = pat.sub(sub, s)
s = s.replace('"/home/quad/results/mapclosures/split_maps_seg_xy/', 'FLAGS_out_dir + "/').replace('"/home/quad/results/density_maps/', 'FLAGS_out_dir + "/density_maps/')
old1 = '''      std::vector<Eigen::Vector3d> eigen_local_map_points;
      for (const auto& pcloud : segment_points){
        for (const auto &point : pcloud.points) {
          eigen_local_map_points.emplace_back(point.x, point.y, point.z);
        }
      } //Uncomment to this point'''
new1 = '''      std::vector<Eigen::Vector3d> eigen_local_map_points;
      if (FLAGS_arm == "raw") {
        for (const auto &point : keyframe_cloud->points) eigen_local_map_points.emplace_back(point.x, point.y, point.z);
      } else {
      for (const auto& pcloud : segment_points){
        for (const auto &point : pcloud.points) {
          eigen_local_map_points.emplace_back(point.x, point.y, point.z);
        }
      } }'''
assert s.count(old1) == 1, 'teach arm anchor'; s = s.replace(old1, new1)
old2 = old1.replace('segment_points', 'repeat_segment_points'); new2 = new1.replace('segment_points', 'repeat_segment_points').replace('keyframe_cloud->points', 'keyframe_match_cloud->points')
assert s.count(old2) == 1, 'repeat arm anchor'; s = s.replace(old2, new2)
open(p, 'w').write(s)
print(f'patched: config {c1}+{c2}, viewer {c3}, input paths {k[0]}, remaining /home/quad {s.count("/home/quad")} of {n0}')
