#!/usr/bin/env python3
"""Slow, obviously-correct reference BPE trainer (same rules as bpe_train.cpp).

  ref_train.py COUNTS VOCAB_SIZE MIN_WORD_COUNT OUT.tiktoken
"""
import base64
import collections
import struct
import sys


def load_counts(path, min_count):
    words = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            c, l = struct.unpack("<QH", f.read(10))
            b = f.read(l)
            if c < min_count:
                break
            if l >= 2:
                words.append([list(b), c])
    return words


def main():
    path, vocab_size, min_count, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    words = load_counts(path, min_count)
    tok = [bytes([b]) for b in range(256)]
    tok_id = {t: i for i, t in enumerate(tok)}
    while len(tok) < vocab_size:
        stats = collections.Counter()
        for s, c in words:
            for x, y in zip(s, s[1:]):
                stats[(x, y)] += c
        if not stats:
            break
        (a, b), cnt = min(stats.items(), key=lambda kv: (-kv[1], kv[0]))
        if cnt <= 0:
            break
        t = tok[a] + tok[b]
        if t in tok_id:
            z = tok_id[t]
        else:
            z = len(tok)
            tok.append(t)
            tok_id[t] = z
        for w in words:
            s, out_s, i = w[0], [], 0
            while i < len(s):
                if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                    out_s.append(z)
                    i += 2
                else:
                    out_s.append(s[i])
                    i += 1
            w[0] = out_s
    with open(out, "w") as f:
        for r, t in enumerate(tok):
            f.write(f"{base64.b64encode(t).decode()} {r}\n")


main()
