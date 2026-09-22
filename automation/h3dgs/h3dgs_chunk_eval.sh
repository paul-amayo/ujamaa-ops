#!/bin/bash
# Held-out PSNR of ONE finished H3DGS chunk (its optimised hierarchy + the scaffold as context) on the
# every-10th keyframes that fall inside it, then per-block comparison with the per-block fleet numbers.
#   usage: h3dgs_chunk_eval.sh <chunk name, e.g. 0_0>
C=${1:?chunk}; REPO=/home/paperspace/code/hierarchical-3d-gaussians
PROJ=/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs
OUT=$PROJ/output; CH=$PROJ/camera_calibration/chunks; SC=$OUT/scaffold/point_cloud/iteration_30000
EV=$OUT/eval_chunks/$C; mkdir -p $EV
export PATH=/home/paperspace/miniconda3/envs/h3dgs/bin:$PATH
cd $REPO
t0=$(date +%s)
[ -n "$(ls $EV/render_0.0 2>/dev/null)" ] || python render_hierarchy.py -s $CH/$C -i ../../rectified/images --model_path $OUT/trained_chunks/$C --hierarchy $OUT/trained_chunks/$C/hierarchy.hier_opt \
  --scaffold_file $SC --out_dir $EV --eval --taus 0 > $EV/render.log 2>&1
echo "render rc=$? in $(( $(date +%s)-t0 ))s"; grep -aE "tau:|Error|Traceback" $EV/render.log | tail -3
python - "$C" "$EV" << 'EOF'
import sys, json, csv, numpy as np
from pathlib import Path
from PIL import Image
C, EV = sys.argv[1], Path(sys.argv[2])
PROJ = Path("/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs")
GT = PROJ / "camera_calibration/rectified/images"; FG = Path("/home/paperspace/data/citrus_all/05_13D_Jackal/prod/tassili/fg_masks")
BL = Path("/home/paperspace/data/citrus_all/05_13D_Jackal/prod/tassili/blocks_ns/lio_row100")
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess")
from read_write_model import read_images_binary, qvec2rotmat
c = np.loadtxt(PROJ / f"camera_calibration/chunks/{C}/center.txt"); e = np.loadtxt(PROJ / f"camera_calibration/chunks/{C}/extent.txt")
ims = read_images_binary(str(PROJ / "camera_calibration/aligned/sparse/0/images.bin"))
pos = {im.name: -qvec2rotmat(im.qvec).T @ im.tvec for im in ims.values()}
inside = lambda n: abs(pos[n][0] - c[0]) <= e[0] / 2 and abs(pos[n][1] - c[1]) <= e[1] / 2
block_of, frames_of = {}, {}
for tj in sorted(BL.glob("block_*/transforms.json")):
    b = int(tj.parent.name.split("_")[1]); names = [Path(f["file_path"]).name for f in json.load(open(tj))["frames"]]
    frames_of[b] = names
    for n in names: block_of[n] = b
def psnr(a, b, m=None):
    d = (a - b) ** 2
    if m is not None: d = d[m]
    return 10 * np.log10(1.0 / max(d.mean(), 1e-12))
rows = []
for r in sorted((EV / "render_0.0").glob("*.png")):
    n = r.stem + ".png"
    ren = np.asarray(Image.open(r).convert("RGB"), np.float32) / 255; gt = np.asarray(Image.open(GT / n).convert("RGB"), np.float32) / 255
    if ren.shape != gt.shape: gt = np.asarray(Image.open(GT / n).convert("RGB").resize((ren.shape[1], ren.shape[0])), np.float32) / 255
    m = np.asarray(Image.open(FG / n).convert("L").resize((ren.shape[1], ren.shape[0]))) > 0
    rows.append({"name": n, "block": block_of.get(n), "inside": bool(inside(n)), "psnr": float(psnr(ren, gt)), "psnr_fg": float(psnr(ren, gt, m))})
json.dump(rows, open(EV / "scores.json", "w"), indent=1)
fleet = {}
for row in csv.DictReader(open("/home/paperspace/logs/citrus_fleet_glref.tsv"), delimiter="\t"):
    if row["survey"] == "05_13D_Jackal": fleet[int(row["block"].split("_")[1])] = (float(row["eval_fg"]), float(row["eval_psnr"]))
ins = [r for r in rows if r["inside"]]
print(f"[chunk {C}] {len(rows)} held-out views rendered, {len(ins)} inside the cell: full-frame PSNR mean {np.mean([r['psnr'] for r in ins]):.2f} dB, FG-masked {np.mean([r['psnr_fg'] for r in ins]):.2f} dB (all {len(rows)}: {np.mean([r['psnr'] for r in rows]):.2f} / {np.mean([r['psnr_fg'] for r in rows]):.2f})")
print(f"{'block':>6} {'kf in cell':>10} {'n test':>6} {'H3DGS full':>11} {'fleet full':>11} {'H3DGS FG':>9} {'fleet FG':>9}")
for b in sorted(set(r["block"] for r in ins)):
    rb = [r for r in ins if r["block"] == b]; frac = np.mean([inside(n) for n in frames_of[b]])
    ff, fp = fleet.get(b, (float("nan"), float("nan")))
    print(f"{b:>6} {frac*100:>9.0f}% {len(rb):>6} {np.mean([r['psnr'] for r in rb]):>11.2f} {fp:>11.2f} {np.mean([r['psnr_fg'] for r in rb]):>9.2f} {ff:>9.2f}")
full = [b for b in set(r["block"] for r in ins) if np.mean([inside(n) for n in frames_of[b]]) > 0.99]
if full:
    h = [np.mean([r["psnr"] for r in ins if r["block"] == b]) for b in full]; hf = [np.mean([r["psnr_fg"] for r in ins if r["block"] == b]) for b in full]
    f = [fleet[b][1] for b in full if b in fleet]; ffg = [fleet[b][0] for b in full if b in fleet]
    print(f"[chunk {C}] blocks fully inside the cell ({len(full)}): H3DGS full-frame {np.mean(h):.2f} vs fleet {np.mean(f):.2f} dB; FG-masked {np.mean(hf):.2f} vs fleet {np.mean(ffg):.2f} dB")
EOF
