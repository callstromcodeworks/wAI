#!/usr/bin/env python3
"""Combine per-source piece counts, damping pieces that are frequent in only one source.

  robust_counts.py KAPPA OUT.counts SRC_COUNTS_DIR SRC_SHARD_DIR

For each piece, let f_s = count_s / bytes_s be its frequency in source s. The source with
the highest frequency may contribute at most KAPPA times the second-highest frequency
(scaled to its own size), plus 1. Every other source contributes its raw count. Pieces
common across sources are unaffected; repo-specific macros, product names and
book-specific character names are damped. KAPPA=inf reproduces plain summed counts.
"""
import glob
import math
import os
import struct
import sys

kappa = float(sys.argv[1])
depth = int(os.environ.get("DEPTH", "1"))  # 2: also cap the 2nd source by the 3rd
out_path, cdir, sdir = sys.argv[2], sys.argv[3], sys.argv[4]

# piece -> [total, top1_f, top1_c, top2_f, top2_c, top3_f]
acc = {}
for cpath in sorted(glob.glob(os.path.join(cdir, "*.counts"))):
    name = os.path.basename(cpath)[:-7]
    size = os.path.getsize(os.path.join(sdir, name + ".shard"))
    data = open(cpath, "rb").read()
    (n,) = struct.unpack_from("<Q", data, 0)
    pos = 8
    for _ in range(n):
        c, l = struct.unpack_from("<QH", data, pos)
        pos += 10
        b = data[pos:pos + l]
        pos += l
        f = c / size
        e = acc.get(b)
        if e is None:
            acc[b] = [c, f, c, 0.0, 0, 0.0]
        else:
            e[0] += c
            if f > e[1]:
                e[5] = e[3]
                e[3], e[4] = e[1], e[2]
                e[1], e[2] = f, c
            elif f > e[3]:
                e[5] = e[3]
                e[3], e[4] = f, c
            elif f > e[5]:
                e[5] = f

items = []
for b, (total, f1, c1, f2, c2, f3) in acc.items():
    if math.isinf(kappa):
        r = total
    else:
        f2c = f2
        r = total
        if depth >= 2 and c2:
            size2 = c2 / f2
            capped2 = min(c2, kappa * f3 * size2 + 1)
            r = r - c2 + capped2
            f2c = capped2 / size2
        size1 = c1 / f1
        r = r - c1 + min(c1, kappa * f2c * size1 + 1)
    r = int(round(r))
    if r > 0:
        items.append((r, b))
items.sort(key=lambda x: (-x[0], len(x[1]), x[1]))
with open(out_path, "wb") as o:
    o.write(struct.pack("<Q", len(items)))
    for c, b in items:
        o.write(struct.pack("<QH", c, len(b)) + b)
print(f"kappa={kappa}: {len(acc)} pieces in, {len(items)} out", file=sys.stderr)
