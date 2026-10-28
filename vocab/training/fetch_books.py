#!/usr/bin/env python3
"""Fetch English Project Gutenberg books from GITenberg mirrors on GitHub.

Writes raw/books/train/<id>.txt and raw/books/heldout/<id>.txt with the PG license
header/footer removed. Books are sampled (fixed seed) from the GITenberg repo list.
"""
import concurrent.futures as cf
import os
import random
import re
import sys
import urllib.request

ROOT = os.environ.get("VOCAB_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work"))

LIST = f"{ROOT}/raw/gb/gitberg-0.8.8/gitenberg/data/GITenberg_repo_list.tsv"
OUT = f"{ROOT}/raw/books"
N_TRAIN, N_HELD = int(sys.argv[1]), int(sys.argv[2])

STOP = set("the of and to a in that is was he for it with as his on be at by i had not are but from or have an they which one you were her all she there would their we him been has when who will more no if out so said what up its about into than them can only other new some could time these two may then do first any my now such like our over man me even most made after also did many before must through back years where much your way well down should because each just those people how too little state good very make world still own see men work long get here between both life being under never day same another know while last might us great old year off come since against go came right used take three".split())


def fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read()
    except Exception:
        return None


def strip_pg(text):
    s = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG[^\n]*\n", text, re.I)
    e = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", text, re.I)
    if not s or not e or e.start() <= s.end():
        return None
    body = text[s.end():e.start()]
    # drop a trailing transcriber/producer credit block at the very start, if short
    return body.strip("\r\n ") + "\n"


def english(text):
    words = re.findall(r"[a-z]+", text[:200000].lower())
    if len(words) < 2000:
        return False
    return sum(w in STOP for w in words) / len(words) > 0.35


def get(book_id, repo):
    for name in (f"{book_id}.txt", f"{book_id}-0.txt", f"{book_id}-8.txt"):
        raw = fetch(f"https://raw.githubusercontent.com/GITenberg/{repo}/master/{name}")
        if raw:
            break
    else:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    text = text.replace("\r\n", "\n")
    body = strip_pg(text)
    if not body or len(body) < 20000 or not english(body):
        return None
    return body


def main():
    rows = [l.rstrip("\n").split("\t") for l in open(LIST)]
    rng = random.Random(20261025)
    rng.shuffle(rows)
    for d in ("train", "heldout"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    got = {"train": 0, "heldout": 0}
    total = {"train": 0, "heldout": 0}
    with cf.ThreadPoolExecutor(24) as ex:
        it = iter(rows)
        futs = {}
        def submit():
            try:
                bid, repo = next(it)
            except StopIteration:
                return False
            futs[ex.submit(get, bid, repo)] = bid
            return True
        for _ in range(48):
            submit()
        while futs and (got["train"] < N_TRAIN or got["heldout"] < N_HELD):
            done, _ = cf.wait(list(futs), return_when=cf.FIRST_COMPLETED)
            for f in done:
                bid = futs.pop(f)
                body = f.result()
                if body:
                    # Held-out books are the first N_HELD successes; the rest train.
                    split = "heldout" if got["heldout"] < N_HELD else "train"
                    if got[split] < (N_HELD if split == "heldout" else N_TRAIN):
                        open(os.path.join(OUT, split, f"{bid}.txt"), "w").write(body)
                        got[split] += 1
                        total[split] += len(body.encode())
                        if sum(got.values()) % 50 == 0:
                            print(got, {k: f"{v/1e6:.0f}MB" for k, v in total.items()}, flush=True)
                submit()
    print("FINAL", got, {k: f"{v/1e6:.0f}MB" for k, v in total.items()}, flush=True)


main()
