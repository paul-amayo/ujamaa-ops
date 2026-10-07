"""Few-shot SAM3 config for citrus oranges from the repo's ODinW 10-shot recipe (odinw_text_only_train.yaml): same transforms,
losses, lr (x0.1), schedule; dataset = our pseudo-label COCO (make_pseudo_coco.py), category 'fruit' as the prompt, no job array,
one local GPU, checkpoints kept. python3 make_config.py <data root with train/ and val/> <experiment log dir> [num_images] [epochs=40] [epoch_size=1500] [name] -> prints the yaml path"""
import re, sys
from pathlib import Path
SRC = Path("/home/paperspace/code/sam3/sam3/train/configs/odinw13/odinw_text_only_train.yaml"); OUT = Path("/home/paperspace/code/sam3/sam3/train/configs/citrus_fruit_fewshot.yaml")
data, logdir = sys.argv[1], sys.argv[2]; n_img = sys.argv[3] if len(sys.argv) > 3 else "null"; epochs = sys.argv[4] if len(sys.argv) > 4 else "40"; epoch_size = sys.argv[5] if len(sys.argv) > 5 else "1500"; name = sys.argv[6] if len(sys.argv) > 6 else "citrus_fruit_fewshot"; s = SRC.read_text()
OUT = OUT.with_name(name + ".yaml")
rep = {"odinw_data_root: <YOUR_DATA_DIR>": f"odinw_data_root: {data}", "experiment_log_dir: <YOUR EXPERIMENET LOG_DIR>": f"experiment_log_dir: {logdir}",
       "bpe_path: <BPE_PATH> # This should be under sam3/assets/bpe_simple_vocab_16e6.txt.gz": "bpe_path: /home/paperspace/code/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
       "  num_images: null\n  supercategory_tuple: ${all_odinw_supercategories.${string:${submitit.job_array.task_index}}}": f"  num_images: {n_img}\n  supercategory_tuple:\n    name: citrus_fruit\n    train: {{img_folder: train/images/, json: train/_annotations.coco.json}}\n    val: {{img_folder: val/images/, json: val/_annotations.coco.json}}",
       "prompts: ${odinw35_prompts.${odinw_train.supercategory_tuple.name}} #${odinw_train.supercategory_tuple.name)": "prompts: null",
       "prompts: ${odinw35_prompts.${odinw_train.supercategory_tuple.name}}": "prompts: null",
       "  skip_saving_ckpts: true\n  # _target_": "  skip_saving_ckpts: false\n  # _target_",
       "  gpus_per_node: 2\n  experiment_log_dir: null #${paths.experiment_log_dir}": "  gpus_per_node: 1\n  experiment_log_dir: ${paths.experiment_log_dir}",
       "  use_cluster: True\n  cpus_per_task: 10": "  use_cluster: False\n  cpus_per_task: 10",
       "  max_data_epochs: 40\n  target_epoch_size: 1500": f"  max_data_epochs: {epochs}\n  target_epoch_size: {epoch_size}"}
for a, b in rep.items():
    assert a in s, a[:60]; s = s.replace(a, b)
s = re.sub(r"\n  job_array:\n(    .*\n)+", "\n", s)   # no dataset sweep
s = s.replace("${odinw_train.supercategory_tuple.train.img_folder}", "${odinw_train.supercategory_tuple.train.img_folder}").replace("log_dir: ${launcher.experiment_log_dir}/logs/${odinw_train.supercategory_tuple.name}", "log_dir: ${launcher.experiment_log_dir}/logs")
OUT.write_text(s); left = [l for l in s.splitlines() if "job_array" in l or "<YOUR" in l]; print(OUT, "| unresolved placeholders:", left or "none")
