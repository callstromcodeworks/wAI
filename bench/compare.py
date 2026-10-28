#!/usr/bin/env python3
"""Throughput comparison: fasttok (C++ CLI, timed in-process) vs tiktoken (Python API).

  python3 bench/compare.py --bin build/fasttok --vocab-dir DIR FILE [FILE...] [--iters 5]

Reports MB/s of UTF-8 input, best of N runs, for single-threaded encode_ordinary and
for 2-thread encoding (fasttok's encode_ordinary_parallel vs tiktoken's
encode_ordinary_batch over 64 chunks).
"""
import argparse
import os
import re
import subprocess
import time

import tiktoken

ap = argparse.ArgumentParser()
ap.add_argument("--bin", required=True)
ap.add_argument("--vocab-dir", required=True)
ap.add_argument("--encodings", default="o200k_base,cl100k_base")
ap.add_argument("--iters", type=int, default=5)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("files", nargs="+")
args = ap.parse_args()


def fasttok_mbps(enc, path, threads):
    out = subprocess.run([args.bin, enc, os.path.join(args.vocab_dir, enc + ".tiktoken"), "bench", path,
                          str(args.iters), str(threads)], capture_output=True, text=True, check=True).stdout
    assert "roundtrip ok" in out, out
    return float(re.search(r"= ([0-9.]+) MB/s", out).group(1))


def best(fn, text):
    b = 1e30
    for _ in range(args.iters):
        t = time.perf_counter()
        fn(text)
        b = min(b, time.perf_counter() - t)
    return b


rows = []
for enc_name in args.encodings.split(","):
    enc = tiktoken.get_encoding(enc_name)
    for path in args.files:
        text = open(path, encoding="utf-8").read()
        nbytes = len(text.encode("utf-8"))
        lines = text.split("\n")
        k = max(1, len(lines) // 64)
        docs = ["\n".join(lines[i:i + k]) for i in range(0, len(lines), k)]
        tk1 = nbytes / best(enc.encode_ordinary, text) / 1e6
        tknp = nbytes / best(lambda t: enc.encode_to_numpy(t, disallowed_special=()), text) / 1e6
        tkb = nbytes / best(lambda _: enc.encode_ordinary_batch(docs, num_threads=args.threads), text) / 1e6
        ft1 = fasttok_mbps(enc_name, path, 1)
        ftn = fasttok_mbps(enc_name, path, args.threads)
        rows.append((enc_name, os.path.basename(path), nbytes, tk1, tknp, ft1, ft1 / max(tk1, tknp), tkb, ftn, ftn / tkb))
        print(rows[-1], flush=True)

print()
print(f"| encoding | corpus | size | tiktoken 1T | tiktoken 1T (numpy) | fasttok 1T | speedup | "
      f"tiktoken {args.threads}T batch | fasttok {args.threads}T | speedup |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for e, f, nb, tk1, tknp, ft1, s1, tkb, ftn, sn in rows:
    print(f"| {e} | {f} | {nb / 1e6:.1f} MB | {tk1:.1f} | {tknp:.1f} | {ft1:.1f} | {s1:.1f}x | {tkb:.1f} | {ftn:.1f} | {sn:.1f}x |")
