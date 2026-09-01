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


FLORES_URL = "https://dl.fbaipublicfiles.com/flores101/dataset/flores200_dataset.tar.gz"


def fetch_flores_tarball():
    """Download+extract the official FLORES-200 tarball once; reuse after."""
    import tarfile
    root = DATA_DIR / "flores200_dataset"
    if (root / "devtest").exists():
        return root
    DATA_DIR.mkdir(exist_ok=True)
    tgz = DATA_DIR / "flores200_dataset.tar.gz"
    if not tgz.exists():
        print("downloading FLORES-200 tarball (~25 MB)...")
        urllib.request.urlretrieve(FLORES_URL, tgz)
    with tarfile.open(tgz) as t:
        t.extractall(DATA_DIR)
    return root


def main():
    DATA_DIR.mkdir(exist_ok=True)
    problems = []
    flores_root = fetch_flores_tarball()
    for code, fl in LANGS.items():
        # Belebele: passage/question/4 answers/correct index.
        # The rows API caps length at 100, so paginate.
        try:
            rs = (rows("facebook/belebele", fl, "test", 0, 100)
                  + rows("facebook/belebele", fl, "test", 100, N_MCQ + N_MCQ_SHOTS - 100))
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
        # FLORES-200 from the original Meta tarball: plain per-language text
        # files, parallel by line index — no HF gating, no API caps.
        try:
            dev = (flores_root / "dev" / f"{fl}.dev").read_text().splitlines()
            devtest = (flores_root / "devtest" / f"{fl}.devtest").read_text().splitlines()
            sents = dict(dev=dev[:N_MT_SHOTS], devtest=devtest[:N_MT])
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
