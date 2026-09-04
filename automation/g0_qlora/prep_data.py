#!/usr/bin/env python3
"""G0 data prep (plans/gemma4_finetune_memo.md §4 P0).

Streams per-language FineWeb-2 (FineWeb classic for the English replay),
tokenizes with the gemma-4 tokenizer, packs into fixed 2048-token sequences,
and writes uint32 .npy shards + a manifest. Deterministic order (no shuffle
at prep; the trainer shuffles shards). Resumable per language: a language
with a complete manifest entry is skipped.

  ~/envs/hfeval/bin/python prep_data.py --out /data/g0_tokens [--langs xh,zu]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

SEQ = 2048
SHARD_SEQS = 4096          # 8.4M tokens per shard file (~34 MB uint32)
MODEL = "unsloth/gemma-4-12b"

# P0 mix: 1.5B tokens, xh/zu-weighted, with en/af/sw/ar replay (memo §4).
MIX = {  # lang -> (hf dataset, config, target tokens)
    "xh": ("HuggingFaceFW/fineweb-2", "xho_Latn", 350_000_000),
    "zu": ("HuggingFaceFW/fineweb-2", "zul_Latn", 350_000_000),
    "ha": ("HuggingFaceFW/fineweb-2", "hau_Latn", 150_000_000),
    "am": ("HuggingFaceFW/fineweb-2", "amh_Ethi", 100_000_000),
    "af": ("HuggingFaceFW/fineweb-2", "afr_Latn", 100_000_000),
    "sw": ("HuggingFaceFW/fineweb-2", "swh_Latn", 100_000_000),
    "ar": ("HuggingFaceFW/fineweb-2", "arb_Arab", 100_000_000),
    "en": ("HuggingFaceFW/fineweb", "sample-10BT", 250_000_000),
}


def prep_lang(tok, lang, dataset, config, target, out):
    from datasets import load_dataset
    manifest_f = out / f"manifest_{lang}.json"
    if manifest_f.exists():
        m = json.loads(manifest_f.read_text())
        if m.get("complete"):
            print(f"[{lang}] already complete ({m['tokens']:,} tok)", flush=True)
            return
    ds = load_dataset(dataset, name=config, split="train", streaming=True)
    buf, shards, total, docs = [], 0, 0, 0
    t0 = time.time()
    batch = []
    for ex in ds:
        batch.append(ex["text"])
        if len(batch) < 256:
            continue
        for ids in tok(batch, add_special_tokens=False)["input_ids"]:
            buf.extend(ids + [tok.eos_token_id])
            docs += 1
        batch = []
        while len(buf) >= SHARD_SEQS * SEQ:
            arr = np.array(buf[:SHARD_SEQS * SEQ], dtype=np.uint32)
            arr = arr.reshape(SHARD_SEQS, SEQ)
            np.save(out / f"{lang}_{shards:04d}.npy", arr)
            del buf[:SHARD_SEQS * SEQ]
            total += SHARD_SEQS * SEQ
            shards += 1
            rate = total / max(time.time() - t0, 1)
            print(f"[{lang}] shard {shards} | {total/1e6:.0f}M tok "
                  f"| {rate/1e3:.0f}k tok/s", flush=True)
            manifest_f.write_text(json.dumps(dict(
                lang=lang, source=f"{dataset}:{config}", shards=shards,
                tokens=total, docs=docs, complete=False)))
        if total >= target:
            break
    manifest_f.write_text(json.dumps(dict(
        lang=lang, source=f"{dataset}:{config}", shards=shards, tokens=total,
        docs=docs, complete=True)))
    print(f"[{lang}] COMPLETE {total/1e6:.0f}M tok, {shards} shards", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    langs = a.langs.split(",") if a.langs else list(MIX)
    for lang in langs:
        d, c, t = MIX[lang]
        prep_lang(tok, lang, d, c, t, out)
    print("PREP_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
