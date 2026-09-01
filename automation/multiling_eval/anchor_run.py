#!/usr/bin/env python3
"""M1 anchor runner: one model x one task over the eight languages.

Few-shot, base-model-fair probing per the plan: every model is called with
raw:true (no chat template), so instruct and base models see the identical
prompt. Resumable on (task, lang, idx); one JSONL row per query.

  python3 anchor_run.py --model <ollama-name> --task belebele|flores \
      --out <run.jsonl> [--langs en,xh]
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
LANGS = ["en", "af", "xh", "zu", "sw", "ha", "ar", "am"]
NAMES = {"en": "English", "af": "Afrikaans", "xh": "isiXhosa",
         "zu": "isiZulu", "sw": "Swahili", "ha": "Hausa",
         "ar": "Arabic", "am": "Amharic"}
LETTERS = "ABCD"


def gen(model, prompt, n_predict, stop):
    body = json.dumps({
        "model": model, "prompt": prompt, "raw": True, "stream": False,
        "keep_alive": "30m",
        "options": {"num_predict": n_predict, "temperature": 0, "stop": stop},
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return r.get("response", ""), round(time.time() - t0, 2)


def mcq_prompt(items, idx):
    """2-shot in-language MCQ; shots are the items past the eval slice."""
    shots = items[100:102]
    parts = []
    for it in shots + [items[idx]]:
        opts = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(it["opts"]))
        parts.append(f"Passage: {it['passage']}\nQuestion: {it['q']}\n{opts}\nAnswer:")
        if it is not items[idx]:
            parts[-1] += f" {LETTERS[it['gold']]}\n"
    return "\n".join(parts)


def parse_letter(text):
    for ch in text.strip().upper():
        if ch in LETTERS:
            return LETTERS.index(ch)
    return -1


def mt_prompt(src_dev, tgt_dev, src_name, tgt_name, src_sent):
    parts = []
    for s, t in zip(src_dev, tgt_dev):
        parts.append(f"{src_name}: {s}\n{tgt_name}: {t}\n")
    parts.append(f"{src_name}: {src_sent}\n{tgt_name}:")
    return "\n".join(parts)


def done_keys(path):
    keys = set()
    if Path(path).exists():
        for ln in open(path):
            try:
                d = json.loads(ln)
                keys.add((d["task"], d["lang"], d["idx"]))
            except Exception:
                pass
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=["belebele", "flores"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", default="")
    a = ap.parse_args()
    langs = a.langs.split(",") if a.langs else LANGS
    done = done_keys(a.out)
    out = open(a.out, "a")

    def emit(row):
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()

    if a.task == "belebele":
        for lang in langs:
            items = json.loads((DATA / f"belebele_{lang}.json").read_text())
            for idx in range(100):
                if ("belebele", lang, idx) in done:
                    continue
                resp, secs = gen(a.model, mcq_prompt(items, idx), 4, ["\n"])
                pred = parse_letter(resp)
                emit(dict(task="belebele", lang=lang, idx=idx, model=a.model,
                          pred=pred, gold=items[idx]["gold"],
                          ok=pred == items[idx]["gold"], secs=secs,
                          raw=resp.strip()[:40]))
            print(f"[{a.model.split('/')[-1]}] belebele {lang} done", flush=True)
    else:
        en = json.loads((DATA / "flores_en.json").read_text())
        for lang in langs:
            if lang == "en":
                continue
            xx = json.loads((DATA / f"flores_{lang}.json").read_text())
            for direction, (sd, td, sname, tname, src, ref) in {
                "en-xx": (en["dev"], xx["dev"], "English", NAMES[lang],
                          en["devtest"], xx["devtest"]),
                "xx-en": (xx["dev"], en["dev"], NAMES[lang], "English",
                          xx["devtest"], en["devtest"]),
            }.items():
                for i in range(32):
                    idx = f"{direction}:{i}"
                    if ("flores", lang, idx) in done:
                        continue
                    p = mt_prompt(sd, td, sname, tname, src[i])
                    resp, secs = gen(a.model, p, 130, ["\n"])
                    emit(dict(task="flores", lang=lang, idx=idx, model=a.model,
                              hyp=resp.strip(), ref=ref[i], secs=secs))
            print(f"[{a.model.split('/')[-1]}] flores {lang} done", flush=True)


if __name__ == "__main__":
    main()
