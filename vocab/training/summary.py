#!/usr/bin/env python3
"""Run ft_eval and summarise: English, code, and a 50/50 English+code mix (bytes/token)."""
import os
import subprocess
import sys

ROOT = os.environ.get("VOCAB_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work"))

specs = sys.argv[1:]
EN = ["held_en_books", "held_en_news", "held_en_docs"]
CODE = ["held_code_c", "held_code_csharp", "held_code_go", "held_code_js", "held_code_jvm", "held_code_php",
        "held_code_python", "held_code_ruby", "held_code_rust", "held_code_swift", "held_code_config", "held_code_web"]
shards = [f"{ROOT}/shards/{s}.shard" for s in EN + CODE]
out = subprocess.run([f"{ROOT}/bin/ft_eval", *specs, "--", *shards],
                     capture_output=True, text=True, check=True).stdout.splitlines()
names = out[0].split()[2:-4]
rows = {}
for line in out[1:]:
    parts = line.split()
    rows[parts[0].replace(".shard", "")] = (float(parts[1]), [float(x) for x in parts[2:]])


def agg(keys, i):
    b = sum(rows[k][0] for k in keys)
    t = sum(rows[k][0] / rows[k][1][i] for k in keys)
    return b / t


print(f"{'vocab':18s} {'books':>7s} {'news':>7s} {'docs':>7s} {'EN':>7s} {'CODE':>7s} {'50/50':>7s}")
for i, n in enumerate(names):
    en, code = agg(EN, i), agg(CODE, i)
    mix = 1 / (0.5 / en + 0.5 / code)
    print(f"{n:18s} {rows['held_en_books'][1][i]:7.3f} {rows['held_en_news'][1][i]:7.3f} "
          f"{rows['held_en_docs'][1][i]:7.3f} {en:7.3f} {code:7.3f} {mix:7.3f}")
