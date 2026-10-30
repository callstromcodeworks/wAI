"""Continual-learning benchmark: a sequence of narrow specialties the base model saw little of, from the raw
data already downloaded (rare languages in github-code-clean, topic sites in FineWeb-Edu by URL).

    python model/build_domains.py            # -> data/ec32k/domains/<name>/{train,val}.npy

Each domain: up to --train_tokens of training text and --val_tokens of held-out text, split by document."""
import argparse
import glob
import os
import re

import numpy as np
import pyarrow.parquet as pq

from tokenizer import VOCABS, Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
CODE = {"haskell": "Haskell", "lua": "Lua", "fortran": "FORTRAN", "rust": "Rust"}
WEB = {"medical": r"(nih\.gov|ncbi|mayoclinic|webmd|medlineplus|cdc\.gov|who\.int)",
       "law": r"(law\.|\.gov/.*(law|code|regulat)|uscode|justia|findlaw|legislation)",
       "music": r"(music|guitar|piano)",
       "space": r"(nasa\.gov|space\.com|physics|astronomy)"}
ORDER = ["haskell", "medical", "lua", "law", "fortran", "music", "rust", "space"]


def docs_for(name):
    if name in CODE:
        for f in sorted(glob.glob(os.path.join(RAW, "data", "*.parquet"))):
            t = pq.read_table(f, columns=["code", "language"])
            for c, l in zip(t["code"].to_pylist(), t["language"].to_pylist()):
                if l == CODE[name]:
                    yield c
    else:
        rx = re.compile(WEB[name])
        for f in sorted(glob.glob(os.path.join(RAW, "sample", "10BT", "*.parquet"))):
            t = pq.read_table(f, columns=["text", "url"])
            for x, u in zip(t["text"].to_pylist(), t["url"].to_pylist()):
                if rx.search(u):
                    yield x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="ec32k")
    ap.add_argument("--train_tokens", type=int, default=3_000_000)
    ap.add_argument("--val_tokens", type=int, default=200_000)
    a = ap.parse_args()
    tok = Tokenizer(VOCABS[a.vocab])
    rng = np.random.default_rng(0)
    for name in ORDER:
        docs = list(docs_for(name))
        order = rng.permutation(len(docs))
        out = {"val": [], "train": []}
        n = {"val": 0, "train": 0}
        cap = {"val": a.val_tokens, "train": a.train_tokens}
        for c in range(0, len(order), 256):  # first fill val, then train, from a shuffled order (split by document)
            if n["train"] >= cap["train"]:
                break
            for ids in tok.encode_batch([docs[i] for i in order[c:c + 256]], prepend_eot=True):
                split = "val" if n["val"] < cap["val"] else "train"
                if n["train"] >= cap["train"]:
                    break
                out[split].append(ids)
                n[split] += len(ids)
        d = os.path.join(HERE, "data", a.vocab, "domains", name)
        os.makedirs(d, exist_ok=True)
        for split in out:
            np.save(os.path.join(d, f"{split}.npy"), np.concatenate(out[split]).astype(np.uint16))
        print(f"{name:8s} {len(docs):7d} docs -> train {n['train']:,} tokens, val {n['val']:,}", flush=True)


if __name__ == "__main__":
    main()
