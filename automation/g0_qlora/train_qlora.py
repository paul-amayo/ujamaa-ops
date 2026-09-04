#!/usr/bin/env python3
"""G0 QLoRA CPT on gemma-4-12b base (plans/gemma4_finetune_memo.md §4 P0).

4-bit NF4 base + bf16 LoRA adapters over packed 2048-token shards from
prep_data.py. HF Trainer for checkpoint/resume; shard order shuffled with a
fixed seed so the language mix interleaves. VRAM-guarded, fleet-aware.

  ~/envs/hfeval/bin/python train_qlora.py --data /data/g0_tokens \
      --out /data/g0_run [--max-steps N] [--resume]
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MODEL = "unsloth/gemma-4-12b"
SEQ = 2048


class PackedShards(Dataset):
    def __init__(self, data_dir, seed=17):
        self.files = sorted(Path(data_dir).glob("*_[0-9]*.npy"))
        assert self.files, f"no shards in {data_dir}"
        rng = random.Random(seed)
        rng.shuffle(self.files)
        self.per = np.load(self.files[0], mmap_mode="r").shape[0]

    def __len__(self):
        return len(self.files) * self.per

    def __getitem__(self, i):
        arr = np.load(self.files[i // self.per], mmap_mode="r")
        row = arr[i % self.per].astype(np.int64)
        ids = torch.from_numpy(row)
        return dict(input_ids=ids, labels=ids.clone())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--micro-bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=128)
    a = ap.parse_args()

    free, _ = torch.cuda.mem_get_info()
    if free / 1e9 < 20:
        raise SystemExit(f"only {free/1e9:.1f} GB VRAM free — fleet on the "
                         "card? aborting loudly (relaunch resumes).")

    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model

    ds = PackedShards(a.data)
    tokens_per_step = a.micro_bs * a.accum * SEQ
    max_steps = a.max_steps or len(ds) * SEQ // tokens_per_step
    print(f"dataset: {len(ds)} seqs = {len(ds)*SEQ/1e9:.2f}B tok | "
          f"{tokens_per_step/1e3:.0f}k tok/step | max_steps {max_steps}",
          flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, dtype=torch.bfloat16,
        device_map="cuda:0")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(r=64, lora_alpha=16, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=a.out, max_steps=max_steps,
        per_device_train_batch_size=a.micro_bs,
        gradient_accumulation_steps=a.accum,
        learning_rate=1e-4, lr_scheduler_type="cosine", warmup_steps=100,
        bf16=True, logging_steps=10, save_steps=150, save_total_limit=3,
        report_to=[], dataloader_num_workers=2, seed=17)
    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train(resume_from_checkpoint=a.resume or None)
    trainer.save_model(str(Path(a.out) / "final_adapter"))
    Path(a.out, "TRAIN_DONE").write_text("done")
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
