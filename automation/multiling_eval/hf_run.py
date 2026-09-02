#!/usr/bin/env python3
"""bf16 transformers runner for the base-vs-base column (gemma-4 base).

Reuses anchor_run's prompt builders and data files so the slices are
byte-identical to the ollama arms; raw few-shot, greedy, newline-truncated —
the same protocol the base models got. Resumable on (task, lang, idx).

  ~/envs/hfeval/bin/python hf_run.py --model unsloth/gemma-4-12b \
      --task belebele --out .../gemma-4-12b-pt-bf16_belebele.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import torch

import anchor_run as ar

TAG_SUFFIX = "#bf16"


def load(model_id):
    free, total = torch.cuda.mem_get_info()
    if free / 1e9 < 26:
        raise SystemExit(f"only {free/1e9:.1f} GB free VRAM — need ~26; "
                         "fleet running? aborting loudly.")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = None
    err = []
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            import transformers
            cls = getattr(transformers, cls_name)
            model = cls.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                        device_map="cuda:0")
            print(f"loaded via {cls_name}", flush=True)
            break
        except Exception as e:
            err.append(f"{cls_name}: {e}")
    if model is None:
        raise SystemExit("could not load model:\n" + "\n".join(err))
    model.eval()
    return tok, model


def gen(tok, model, prompt, max_new):
    ids = tok(prompt, return_tensors="pt").to("cuda:0")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True)
    return text.split("\n")[0], round(time.time() - t0, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/gemma-4-12b")
    ap.add_argument("--task", required=True, choices=["belebele", "flores"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", default="")
    a = ap.parse_args()
    langs = a.langs.split(",") if a.langs else ar.LANGS
    done = ar.done_keys(a.out)
    out = open(a.out, "a")
    name = a.model + TAG_SUFFIX
    tok, model = load(a.model)

    def emit(row):
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()

    if a.task == "belebele":
        for lang in langs:
            items = json.loads((ar.DATA / f"belebele_{lang}.json").read_text())
            for idx in range(100):
                if ("belebele", lang, idx) in done:
                    continue
                resp, secs = gen(tok, model, ar.mcq_prompt(items, idx), 4)
                pred = ar.parse_letter(resp)
                emit(dict(task="belebele", lang=lang, idx=idx, model=name,
                          mode="raw-bf16", pred=pred, gold=items[idx]["gold"],
                          ok=pred == items[idx]["gold"], secs=secs,
                          raw=resp.strip()[:40]))
            print(f"[bf16] belebele {lang} done", flush=True)
    else:
        en = json.loads((ar.DATA / "flores_en.json").read_text())
        for lang in langs:
            if lang == "en":
                continue
            xx = json.loads((ar.DATA / f"flores_{lang}.json").read_text())
            for direction, (sd, td, sname, tname, src, ref) in {
                "en-xx": (en["dev"], xx["dev"], "English", ar.NAMES[lang],
                          en["devtest"], xx["devtest"]),
                "xx-en": (xx["dev"], en["dev"], ar.NAMES[lang], "English",
                          xx["devtest"], en["devtest"]),
            }.items():
                for i in range(32):
                    idx = f"{direction}:{i}"
                    if ("flores", lang, idx) in done:
                        continue
                    p = ar.mt_prompt(sd, td, sname, tname, src[i])
                    resp, secs = gen(tok, model, p, 130)
                    emit(dict(task="flores", lang=lang, idx=idx, model=name,
                              mode="raw-bf16", hyp=resp.strip(), ref=ref[i],
                              secs=secs))
            print(f"[bf16] flores {lang} done", flush=True)


if __name__ == "__main__":
    main()
