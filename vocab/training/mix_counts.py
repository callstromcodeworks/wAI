#!/usr/bin/env python3
"""Combine counts files with weights: mix_counts.py OUT IN1:W1 IN2:W2 ..."""
import struct, sys
acc = {}
for spec in sys.argv[2:]:
    path, w = spec.rsplit(":", 1); w = float(w)
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            c, l = struct.unpack("<QH", f.read(10)); b = f.read(l)
            acc[b] = acc.get(b, 0) + c * w
items = sorted(((max(1, round(c)), b) for b, c in acc.items()), key=lambda x: (-x[0], len(x[1]), x[1]))
with open(sys.argv[1], "wb") as o:
    o.write(struct.pack("<Q", len(items)))
    for c, b in items:
        o.write(struct.pack("<QH", c, len(b)) + b)
