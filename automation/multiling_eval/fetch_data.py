#!/usr/bin/env python3
"""M1 data fetch (plans/multilingual_model_track.md).

Pulls small, fixed slices of Belebele (reading comprehension MCQ) and
FLORES-200 (MT) for the eight UJAMAA languages via the HF datasets-server
rows API — no `datasets` dependency, plain HTTPS, deterministic slices.
Writes automation-independent JSON under DATA_DIR and fails LOUDLY on any
language that comes back short: a silent gap here would skew the
decomposition table.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
LANGS = {  # ujamaa code -> FLORES-200 / Belebele code
    "en": "eng_Latn", "af": "afr_Latn", "xh": "xho_Latn", "zu": "zul_Latn",
    "sw": "swh_Latn", "ha": "hau_Latn", "ar": "arb_Arab", "am": "amh_Ethi",
}
N_MCQ, N_MCQ_SHOTS = 100, 2      # eval items + few-shot items (disjoint)
N_MT, N_MT_SHOTS = 32, 3         # devtest eval + dev shots


def rows(dataset, config, split, offset, length):
    url = ("https://datasets-server.huggingface.co/rows?"
           f"dataset={urllib.parse.quote(dataset, safe='')}&config={config}"
           f"&split={split}&offset={offset}&length={length}")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return [x["row"] for x in json.load(r)["rows"]]
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"{dataset}/{config}/{split}: {e}")
            time.sleep(5)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    problems = []
    for code, fl in LANGS.items():
        # Belebele: passage/question/4 answers/correct index
        try:
            rs = rows("facebook/belebele", fl, "test", 0, N_MCQ + N_MCQ_SHOTS)
            items = [dict(passage=r["flores_passage"], q=r["question"],
                          opts=[r["mc_answer1"], r["mc_answer2"],
                                r["mc_answer3"], r["mc_answer4"]],
                          gold=int(r["correct_answer_num"]) - 1) for r in rs]
            if len(items) < N_MCQ + N_MCQ_SHOTS:
                raise RuntimeError(f"short: {len(items)}")
            (DATA_DIR / f"belebele_{code}.json").write_text(
                json.dumps(items, ensure_ascii=False))
            print(f"belebele {code}: {len(items)}")
        except Exception as e:
            problems.append(f"belebele {code}: {e}")
        # FLORES: parallel by row index across languages
        try:
            dev = rows("Muennighoff/flores200", fl, "dev", 0, N_MT_SHOTS)
            devtest = rows("Muennighoff/flores200", fl, "devtest", 0, N_MT)
            sents = dict(dev=[r["sentence"] for r in dev],
                         devtest=[r["sentence"] for r in devtest])
            if len(sents["devtest"]) < N_MT:
                raise RuntimeError(f"short: {len(sents['devtest'])}")
            (DATA_DIR / f"flores_{code}.json").write_text(
                json.dumps(sents, ensure_ascii=False))
            print(f"flores   {code}: dev {len(sents['dev'])} devtest {len(sents['devtest'])}")
        except Exception as e:
            problems.append(f"flores {code}: {e}")
    if problems:
        print("\nFETCH INCOMPLETE — fix before any run:", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        sys.exit(1)
    print("\nall languages complete")


if __name__ == "__main__":
    import urllib.parse  # noqa: E402  (used in rows())
    main()
