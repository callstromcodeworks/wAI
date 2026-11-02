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
       "space": r"(nasa\.gov|space\.com|physics|astronomy)",
       "philosophy": r"(plato\.stanford\.edu|iep\.utm\.edu|philosoph)"}
# logic: chosen by text, not URL (URL matches are mostly "biological", "ecological", ...): a page must use at
# least LOGIC_MIN different terms of formal logic and argument
LOGIC = re.compile(r"\b(syllogisms?|modus ponens|modus tollens|propositional (?:logic|calculus)|predicate (?:logic|calculus)|"
                   r"first-order logic|truth tables?|tautolog\w*|logical(?:ly)? (?:valid|equivalent|fallac\w*)|"
                   r"deductive(?:ly)?|inductive reasoning|valid arguments?|sound arguments?|premises?|conclusion follows|"
                   r"contrapositive|counterexample|necessary and sufficient|quantifiers?|inference rules?|proof by contradiction)\b", re.I)
LOGIC_MIN = 4
ORDER = ["haskell", "medical", "lua", "philosophy", "law", "fortran", "logic", "music", "rust", "space"]
# instruction and conversation data (not in ORDER: build with --only instruct,chat), written by people, in
# data/raw_chat: instruct = Dolly-15k (databricks-dolly-15k.jsonl, CC BY-SA 3.0, sha256 2df90833...), single-turn;
# chat = OpenAssistant oasst2 English (2023-11-05_oasst2_ready.trees.jsonl.gz, Apache-2.0, sha256 7a886a16...),
# one conversation per tree along the best-ranked reply. Both as "User: ...\n\nAssistant: ..." turns.
CHAT = {"instruct": ("dolly", "databricks-dolly-15k.jsonl"), "chat": ("oasst2", "2023-11-05_oasst2_ready.trees.jsonl.gz")}


def chat_docs(name):
    import gzip
    import json
    sub, f = CHAT[name]
    path = os.path.join(HERE, "data", "raw_chat", sub, f)
    if name == "instruct":
        for line in open(path, encoding="utf-8"):
            x = json.loads(line)
            ask = x["instruction"].strip() + ("\n\n" + x["context"].strip() if x["context"].strip() else "")
            yield f"User: {ask}\n\nAssistant: {x['response'].strip()}"
        return
    for line in gzip.open(path, "rt", encoding="utf-8"):
        m = json.loads(line)["prompt"]
        if m["lang"] != "en" or m.get("deleted"):
            continue
        turns = []
        while m is not None:
            turns.append(("User" if m["role"] == "prompter" else "Assistant") + ": " + m["text"].strip())
            live = [r for r in m.get("replies") or [] if not r.get("deleted")]
            m = min(live, key=lambda r: (r.get("rank") is None, r.get("rank") or 0)) if live else None
        if len(turns) >= 2:
            yield "\n\n".join(turns)


def docs_for(name):
    if name in CHAT:
        yield from chat_docs(name)
    elif name in CODE:
        for f in sorted(glob.glob(os.path.join(RAW, "data", "*.parquet"))):
            t = pq.read_table(f, columns=["code", "language"])
            for c, l in zip(t["code"].to_pylist(), t["language"].to_pylist()):
                if l == CODE[name]:
                    yield c
    elif name == "logic":
        for f in sorted(glob.glob(os.path.join(RAW, "sample", "10BT", "*.parquet"))):
            for x in pq.read_table(f, columns=["text"])["text"].to_pylist():
                if len({m.lower() for m in LOGIC.findall(x)}) >= LOGIC_MIN:
                    yield x
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
    ap.add_argument("--only", default="", help="build just these domains (comma list; instruct and chat only so)")
    a = ap.parse_args()
    tok = Tokenizer(VOCABS[a.vocab])
    rng = np.random.default_rng(0)
    for name in (a.only.split(",") if a.only else ORDER):
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
