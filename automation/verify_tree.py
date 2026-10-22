"""Exact per-entry check of a copied tree: every source file exists on the share with the same size, every source symlink
exists there as a symlink with the same target (lstat, not readdir d_type — CIFS mfsymlinks report as regular files in d_type)."""
import os, sys
src, dst = sys.argv[1], sys.argv[2]; nf = nl = bad = 0
for root, dirs, files in os.walk(src):
    for n in dirs + files:
        s = os.path.join(root, n); d = os.path.join(dst, os.path.relpath(s, src)); st = os.lstat(s)
        try: dt = os.lstat(d)
        except FileNotFoundError: bad += 1; print("MISSING", d) if bad < 5 else None; continue
        if os.path.islink(s):
            nl += 1
            if not os.path.islink(d) or os.readlink(d) != os.readlink(s): bad += 1; print("LINK DIFF", d) if bad < 5 else None
        elif os.path.isfile(s):
            nf += 1
            if dt.st_size != st.st_size: bad += 1; print("SIZE DIFF", d) if bad < 5 else None
print(f"files {nf} symlinks {nl} problems {bad}"); sys.exit(1 if bad else 0)
