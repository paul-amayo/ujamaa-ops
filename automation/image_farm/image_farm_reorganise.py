"""Reorganise image_farm by farm and recording date (2026-09-29).

  image_farm/IMG_7993[_s0]  ->  image_farm/<farm>/<date>/IMG_7993[_s0]

Dates are the iPhone's own recording dates (com.apple.quicktime.creationdate,
UTC+3), not the September export date. Farms per Paul: 16 May = Gwakungu,
every other session = Kendu Bay.

Paths stored inside the segments are rewritten in both forms they take:
plain strings (".../image_farm/IMG_7993_s0/...") in transforms/splats/index
json, and nerfstudio's split-path YAML lists ("- image_farm" / "- IMG_7993_s0")
in config.yml. Everything is recorded in MOVE_MANIFEST_<stamp>.tsv with an
undo script beside it; original text files are kept in a tarball.

  python3 image_farm_reorganise.py            # dry run
  python3 image_farm_reorganise.py --apply
"""
import re
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path("/home/paperspace/data/image_farm")
SESSIONS = {  # clip number -> (farm, recording date)
    **{n: ("kendu_bay", "2026-05-13") for n in (7959, 7960, 7961, 7962, 7963, 7964)},
    **{n: ("kendu_bay", "2026-05-14") for n in (7967, 7970, 7971, 7972, 7973, 7974, 7975)},
    **{n: ("gwakungu", "2026-05-16") for n in (7984, 7985, 7986, 7987, 7990, 7992, 7993)},
    **{n: ("kendu_bay", "2026-05-22") for n in (7994, 7995, 7996, 7997, 7998, 7999, 8001)},
    **{n: ("kendu_bay", "2026-05-23") for n in (8019,)},
}
TEXT_SUFFIXES = {".json", ".yml", ".yaml", ".txt", ".tsv", ".csv"}
MAX_TEXT = 20 * 2**20
ENTRY = re.compile(r"^IMG_(\d+)(_s\d+)?$")


def dest_of(name):
    m = ENTRY.match(name)
    farm, date = SESSIONS[int(m.group(1))]
    return f"{farm}/{date}/{name}"


def rewrite(text):
    """Old top-level entry paths -> new nested ones, both path forms."""
    n = 0

    def plain(m):
        nonlocal n
        n += 1
        return f"/image_farm/{dest_of(m.group(1))}"
    text = re.sub(r"/image_farm/(IMG_\d+(?:_s\d+)?)(?=[/\"'\s]|$)", plain, text)

    def listed(m):
        nonlocal n
        n += 1
        ind, name = m.group(1), m.group(2)
        farm, date = dest_of(name).split("/")[:2]
        return f"{ind}- image_farm\n{ind}- {farm}\n{ind}- '{date}'\n{ind}- {name}"  # quoted: YAML reads a bare date as a date
    text = re.sub(r"^([ \t]*)- image_farm\n[ \t]*- (IMG_\d+(?:_s\d+)?)$", listed, text, flags=re.M)
    return text, n


def main(apply):
    entries = sorted(p for p in ROOT.iterdir() if p.is_dir() and ENTRY.match(p.name))
    unknown = [p.name for p in entries if int(ENTRY.match(p.name).group(1)) not in SESSIONS]
    if unknown:
        sys.exit(f"no session for {unknown}; refusing to move anything")
    plan, edits = [], []
    for p in entries:
        plan.append((p, ROOT / dest_of(p.name)))
        for f in p.rglob("*"):
            if f.is_file() and not f.is_symlink() and f.suffix in TEXT_SUFFIXES and f.stat().st_size < MAX_TEXT:
                try:
                    t = f.read_text()
                except UnicodeDecodeError:
                    continue
                new, n = rewrite(t)
                if n:
                    edits.append((f, new, n))
    by_dest = {}
    for _, d in plan:
        by_dest.setdefault(str(d.parent.relative_to(ROOT)), []).append(d.name)
    for k in sorted(by_dest):
        print(f"{k}: {len(by_dest[k])} entries ({', '.join(sorted(by_dest[k])[:4])}{'…' if len(by_dest[k]) > 4 else ''})")
    print(f"moves: {len(plan)}; files to rewrite: {len(edits)} ({sum(n for *_, n in edits)} path references)")
    kinds = {}
    for f, _, n in edits:
        kinds[f.name] = kinds.get(f.name, 0) + 1
    print("rewritten file kinds:", dict(sorted(kinds.items(), key=lambda kv: -kv[1])))
    if not apply:
        print("dry run: nothing changed. Re-run with --apply.")
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    with tarfile.open(ROOT / f"MOVE_originals_{stamp}.tar.gz", "w:gz") as tar:
        for f, _, _ in edits:
            tar.add(f, arcname=str(f.relative_to(ROOT)))
    for f, new, _ in edits:
        f.write_text(new)
    man = ROOT / f"MOVE_MANIFEST_{stamp}.tsv"
    undo = ROOT / f"MOVE_UNDO_{stamp}.sh"
    with open(man, "w") as m, open(undo, "w") as u:
        m.write("old\tnew\n")
        u.write(f"#!/bin/bash\n# undo {stamp}: move back, then restore original text files\nset -e\ncd {ROOT}\n")
        for src, dst in plan:
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            m.write(f"{src.relative_to(ROOT)}\t{dst.relative_to(ROOT)}\n")
            u.write(f"mv {dst.relative_to(ROOT)} {src.relative_to(ROOT)}\n")
        u.write(f"tar xzf MOVE_originals_{stamp}.tar.gz\n")
    print(f"applied: manifest {man.name}, undo {undo.name}, originals MOVE_originals_{stamp}.tar.gz")


if __name__ == "__main__":
    main("--apply" in sys.argv)
