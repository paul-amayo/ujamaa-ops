#!/usr/bin/env python3
"""G0 mid-run eval: paired base-vs-checkpoint chrF on en->xx generation.

Loads the 4-bit base and, separately, base+LoRA adapter, and runs the SAME
few-shot FLORES en->xx protocol as the M1 harness on the focus languages.
Pairing under identical loading/protocol removes the quant confound: the
number that matters is the DELTA, not the absolute.

  ~/envs/hfeval/bin/python eval_ckpt.py --ckpt /path/checkpoint-1200 \
      --out /path/eval25.json [--langs xh,zu] [--n 32]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "multiling_eval"))
import anchor_run as ar          # noqa: E402  (data + mt_prompt + NAMES)
import score as sc               # noqa: E402  (chrf)

import torch                     # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer,  # noqa: E402
                          BitsAndBytesConfig)

BASE = "unsloth/gemma-4-12b"


def load_base():
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map={"": 0})
    model.eval()
    return tok, model


def gen(tok, model, prompt, max_new):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True)
    return text.split("\n")[0].strip()


def run_arm(tok, model, langs, n):
    en = json.loads((ar.DATA / "flores_en.json").read_text())
    res = {}
    for lang in langs:
        xx = json.loads((ar.DATA / f"flores_{lang}.json").read_text())
        scores = []
        for i in range(n):
            p = ar.mt_prompt(en["dev"], xx["dev"], "English",
                             ar.NAMES[lang], en["devtest"][i])
            hyp = gen(tok, model, p, 130)
            scores.append(sc.chrf(hyp, xx["devtest"][i]))
        res[lang] = round(sum(scores) / len(scores), 2)
        print(f"  {lang}: chrF {res[lang]}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", default="xh,zu")
    ap.add_argument("--n", type=int, default=32)
    a = ap.parse_args()
    langs = a.langs.split(",")

    t0 = time.time()
    tok, model = load_base()
    print("BASE arm:", flush=True)
    base_res = run_arm(tok, model, langs, a.n)

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.ckpt)
    model.eval()
    print("CKPT arm:", flush=True)
    ckpt_res = run_arm(tok, model, langs, a.n)

    out = dict(ckpt=a.ckpt, n=a.n, base=base_res, ckpt_scores=ckpt_res,
               delta={l: round(ckpt_res[l] - base_res[l], 2) for l in langs},
               secs=round(time.time() - t0))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["delta"]), flush=True)
    print("EVAL_CKPT_DONE", flush=True)


if __name__ == "__main__":
    main()
