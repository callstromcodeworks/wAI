#!/usr/bin/env python3
"""Differential test: fasttok must produce exactly tiktoken's token ids.

  python3 tests/test_vs_tiktoken.py --lib build/libfasttok.so --vocab-dir VOCAB_DIR \
      [--corpus-dir DIR] [--encodings o200k_base,cl100k_base] [--fuzz N] [--seed S]

VOCAB_DIR must contain <encoding>.tiktoken files. tiktoken itself must be able to load
the same encodings (set TIKTOKEN_CACHE_DIR if you are offline).

"ec200k_base:N" tests the vocabulary made of ec200k_base's first N ranks, with the specials
renumbered to N..N+4 (fasttok's Tokenizer::ec_prefix); tiktoken gets the same truncated ranks.
"""
import argparse
import ctypes
import glob
import os
import random
import sys
import time

import tiktoken
from tiktoken.load import load_tiktoken_bpe

CL100K_PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""


EC_SPECIALS = ["<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|endofprompt|>"]


def split_name(name):
    """"ec200k_base:N" -> ("ec200k_base", N); other names -> (name, None)."""
    base, _, n = name.partition(":")
    return base, int(n) if n else None


def get_encoding(name, vocab_dir):
    """tiktoken's own encoder; ec200k_base (and its prefixes) are built from the same .tiktoken file."""
    base, n = split_name(name)
    if base != "ec200k_base":
        return tiktoken.get_encoding(name)
    n = n or 200000
    ranks = load_tiktoken_bpe(os.path.join(vocab_dir, "ec200k_base.tiktoken"))
    return tiktoken.Encoding(
        name=name, pat_str=CL100K_PAT, mergeable_ranks={k: v for k, v in ranks.items() if v < n},
        special_tokens={t: n + i for i, t in enumerate(EC_SPECIALS)})


class FastTok:
    def __init__(self, lib_path, vocab_path, encoding):
        lib = ctypes.CDLL(lib_path)
        lib.ft_new.restype = ctypes.c_void_p
        lib.ft_new.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.ft_new_ec_prefix.restype = ctypes.c_void_p
        lib.ft_new_ec_prefix.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
        lib.ft_last_error.restype = ctypes.c_char_p
        lib.ft_encode_batch.restype = ctypes.c_size_t
        lib.ft_encode_batch.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64),
                                        ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t,
                                        ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_uint]
        lib.ft_decode.restype = ctypes.c_size_t
        lib.ft_decode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t,
                                  ctypes.c_char_p, ctypes.c_size_t]
        self.lib = lib
        base, n = split_name(encoding)
        if n is None:
            self.h = lib.ft_new(vocab_path.encode(), encoding.encode())
        else:
            self.h = lib.ft_new_ec_prefix(vocab_path.encode(), n)
        if not self.h:
            raise RuntimeError(lib.ft_last_error().decode())

    def encode_batch(self, texts, mode=0, threads=0):
        bufs = [t.encode("utf-8") for t in texts]
        offs = (ctypes.c_uint64 * (len(bufs) + 1))()
        pos = 0
        for k, b in enumerate(bufs):
            offs[k] = pos
            pos += len(b)
        offs[len(bufs)] = pos
        cap = max(pos, 1)
        ids = (ctypes.c_uint32 * cap)()
        counts = (ctypes.c_uint64 * max(len(bufs), 1))()
        total = self.lib.ft_encode_batch(self.h, b"".join(bufs), offs, len(bufs), ids, cap, counts, mode, threads)
        if total == ctypes.c_size_t(-1).value:
            raise RuntimeError(self.lib.ft_last_error().decode())
        out, p = [], 0
        flat = ids[:total]
        for k in range(len(bufs)):
            out.append(flat[p:p + counts[k]])
            p += counts[k]
        return out

    def decode(self, ids):
        arr = (ctypes.c_uint32 * max(len(ids), 1))(*ids)
        cap = len(ids) * 256 + 16
        buf = ctypes.create_string_buffer(cap)
        n = self.lib.ft_decode(self.h, arr, len(ids), buf, cap)
        return buf.raw[:n]


class Checker:
    def __init__(self, ft, enc):
        self.ft, self.enc = ft, enc
        self.cases = 0
        self.failures = 0

    def check(self, label, texts, mode=0):
        if mode == 1:
            want = [self.enc.encode(t, allowed_special="all") for t in texts]
        else:
            want = self.enc.encode_ordinary_batch(texts, num_threads=8)
        got = self.ft.encode_batch(texts, mode=mode)
        bad = [k for k in range(len(texts)) if got[k] != want[k]]
        self.cases += len(texts)
        if bad:
            self.failures += len(bad)
            print(f"  FAIL {label}: {len(bad)}/{len(texts)} mismatches")
            for k in bad[:5]:
                t = texts[k]
                print(f"    text={t[:120]!r}")
                print(f"    want={want[k][:40]}")
                print(f"    got ={got[k][:40]}")
                # show first diverging piece
                w = [self.enc.decode_single_token_bytes(x) for x in want[k]]
                g = [self.enc.decode_single_token_bytes(x) for x in got[k]]
                i = next((i for i in range(min(len(w), len(g))) if w[i] != g[i]), min(len(w), len(g)))
                print(f"    diverge at token {i}: want {w[i:i+4]} got {g[i:i+4]}")
        return not bad


# Characters chosen to hit every class boundary in the regexes.
SPECIAL_CHARS = (
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") * 3
    + list("0123456789") * 2
    + list(" " * 20 + "\t\n\r" * 4 + "\x0b\x0c")
    + list("'\"!?.,;:/\\-_()[]{}<>@#$%^&*+=|~`") * 2
    + ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'T", "'RE", "'Ve", "'M", "'LL", "'D", "'ſ"]
    + ["ſ", "é", "É", "ß", "İ", "ı", "ǅ", "ǈ",  # Ll Lu Lt
       "ʰ", "ˆ", "々",  # Lm
       "中", "文", "あ", "א", "ا", "क", "ก",  # Lo
       "́", "̈", "ि", "ั", "⃝", "ः",  # Mn Mc Me
       "٠", "०", "Ⅰ", "²", "½", "１",  # Nd Nl No
       " ", " ", " ", " ", " ", " ", " ", "　", "\u0085", "​",
       "­", "‍", "﻿", "�", "\U0001f600", "\U0001f468‍\U0001f469", "©",
       "\U0001d400", "\U0001d41a", "\U00020000", "\U0001e900", "Ⴀ", "ⴀ",
       "￿", "\U0010ffff", "\U000e0001", ""]
)


def random_text(rng, max_len):
    n = rng.randint(0, max_len)
    out = []
    for _ in range(n):
        r = rng.random()
        if r < 0.85:
            out.append(rng.choice(SPECIAL_CHARS))
        elif r < 0.95:
            out.append(chr(rng.randint(0x80, 0x2FFF)))
        else:
            cp = rng.randint(0, 0x10FFFF)
            if 0xD800 <= cp <= 0xDFFF:
                cp = 0x41
            out.append(chr(cp))
    return "".join(out)


def all_codepoint_texts():
    texts = []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        texts.append(f"a{c}b {c}A{c}'s 1{c}2\n{c}\t{c}  {c}x{c}{c}'LL{c}.{c}\r\n{c}HeLLo{c}{c}{c}9{c} ")
    return texts


def load_corpus_docs(corpus_dir, rng):
    docs = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
        text = open(path, encoding="utf-8").read()
        # The whole file, plus many random slices.
        docs.append((os.path.basename(path) + " (whole)", [text]))
        slices = []
        for _ in range(3000):
            a = rng.randrange(len(text))
            slices.append(text[a:a + rng.choice([5, 50, 500, 5000])])
        docs.append((os.path.basename(path) + " (slices)", slices))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--vocab-dir", required=True)
    ap.add_argument("--corpus-dir")
    ap.add_argument("--encodings", default="o200k_base,cl100k_base")
    ap.add_argument("--fuzz", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--skip-codepoints", action="store_true")
    args = ap.parse_args()

    ok = True
    for name in args.encodings.split(","):
        t0 = time.time()
        enc = get_encoding(name, args.vocab_dir)
        ft = FastTok(args.lib, os.path.join(args.vocab_dir, split_name(name)[0] + ".tiktoken"), name)
        ck = Checker(ft, enc)
        rng = random.Random(args.seed)
        print(f"== {name}")

        ck.check("hand-picked", [
            "", " ", "  ", "\n", " \n ", "a", "Hello world", "HELLO world", "HeLLo", "don't", "DON'T",
            "they're we've I'm you'll he'd", "'s'S'ſ", "x'ſ", "123456789", "1 2  3   4", "ÀÉÎÕÜ",
            "é", "́́a", "Á", "́A", "ʰA", "ǅa", "ǅA", "  hello  ",
            "\t\tfoo", "foo\n\n\nbar", "foo \n bar", "a/b//c\n/", "!!!\n\n", " !!!", "  !!!",
            "\r\n\r\n", "x y", "　　中文", "<|endoftext|>", "a<|endoftext|>b",
            " " * 1000, "\n" * 1000, " \n" * 500, "a" * 5000, "ab" * 3000, "€" * 2000,
            "\U0001f600" * 500, "1" * 999, "😀👍🏽🇺🇸", "Ⅻ", "x²", "١٢٣٤٥", "ﾟ･✿ヾ╲(｡◕‿◕｡)╱✿･ﾟ",
        ])
        ck.check("specials (allowed)", [
            "<|endoftext|>", "hello<|endoftext|>world", " <|endoftext|> ", "<|endoftext|><|endoftext|>",
            "a <|endofprompt|>\n\n b", "<|endoftext", "<|fim_prefix|>x<|fim_suffix|>y<|fim_middle|>",
            "text   <|endoftext|>   ", "\n<|endoftext|>\n",
        ], mode=1)

        if args.corpus_dir:
            for label, docs in load_corpus_docs(args.corpus_dir, rng):
                ck.check(label, docs)

        if not args.skip_codepoints:
            texts = all_codepoint_texts()
            for k in range(0, len(texts), 100000):
                ck.check(f"codepoints {k}..", texts[k:k + 100000])

        # Random fuzzing, short and long.
        for k in range(0, args.fuzz, 20000):
            ck.check(f"fuzz {k}..", [random_text(rng, 40) for _ in range(20000)])
        ck.check("fuzz long", [random_text(rng, 5000) for _ in range(300)])
        ck.check("fuzz specials", ["".join(rng.choice(["<|endoftext|>", "<|endofprompt|>", random_text(rng, 8)])
                                           for _ in range(rng.randint(0, 12))) for _ in range(5000)], mode=1)

        # Parallel encoder must equal the sequential one on large inputs.
        if args.corpus_dir:
            for path in sorted(glob.glob(os.path.join(args.corpus_dir, "*.txt"))):
                text = open(path, encoding="utf-8").read()
                seq = ft.encode_batch([text], mode=0)[0]
                par = ft.encode_batch([text], mode=2, threads=4)[0]
                ck.cases += 1
                if seq != par:
                    ck.failures += 1
                    print(f"  FAIL parallel != sequential on {path}")
                if ft.decode(seq) != text.encode("utf-8"):
                    ck.failures += 1
                    print(f"  FAIL decode roundtrip on {path}")

        status = "PASS" if ck.failures == 0 else "FAIL"
        print(f"   {status}: {ck.cases} cases, {ck.failures} mismatches ({time.time() - t0:.0f}s)")
        ok &= ck.failures == 0
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
