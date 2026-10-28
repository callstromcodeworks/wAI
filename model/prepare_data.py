"""Download pretraining text and tokenize it with fasttok into token shards.

    python model/prepare_data.py --source fineweb-edu --files 1 --vocab ec64k
    python model/prepare_data.py --source code --files 4 --vocab ec64k
    python model/prepare_data.py --source code --files 2 --first_file 4 --vocab ec64k   # append more

Shards go to data/<vocab>/<source>/: val_000000.bin (the first --val_tokens tokens) and
train_NNNNNN.bin. Every document is preceded by <|endoftext|>. Vocabularies with at most
65,536 ids are stored as uint16, larger ones as uint32. Upstream files are kept in data/raw,
so tokenizing the same text for another vocabulary needs no new download.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from tokenizer import VOCABS, Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MAGIC, VERSION, HEADER_INTS = 20261025, 2, 256

SOURCES = {
    # English: educational web pages (ODC-By).
    "fineweb-edu": dict(repo="HuggingFaceFW/fineweb-edu", column="text",
                        files=[f"sample/10BT/{i:03d}_00000.parquet" for i in range(14)]),
    # Source code in ~30 languages from GitHub (Apache-2.0 dataset card).
    "code": dict(repo="codeparrot/github-code-clean", column="code",
                 files=[f"data/train-{i:05d}-of-00880.parquet" for i in range(880)]),
}


def token_dtype(n_vocab):
    return np.uint16 if n_vocab <= 65536 else np.uint32


def write_shard(path, tokens, n_ranks, dtype):
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[:5] = MAGIC, VERSION, len(tokens), n_ranks, np.dtype(dtype).itemsize
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(dtype, copy=False).tobytes())
    os.replace(tmp, path)


def load_shard(path):
    """memmap of a shard's tokens, plus its n_ranks."""
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    if header[0] != MAGIC or header[1] != VERSION:
        raise ValueError(f"{path} is not a version-{VERSION} token shard; re-run prepare_data.py")
    dtype = {2: np.uint16, 4: np.uint32}[int(header[4])]
    return np.memmap(path, dtype=dtype, mode="r", offset=HEADER_INTS * 4, shape=(int(header[2]),)), int(header[3])


class ShardWriter:
    def __init__(self, out_dir, shard_tokens, val_tokens, n_ranks, dtype):
        self.out_dir, self.shard_tokens, self.n_ranks, self.dtype = out_dir, shard_tokens, n_ranks, dtype
        self.val_left = val_tokens
        self.buf = np.empty(shard_tokens, dtype=dtype)
        self.n = 0
        self.val = []
        self.shard_idx = 0
        self.total = 0

    def add(self, doc):
        self.total += len(doc)
        if self.val_left > 0:  # whole documents go to validation until it is full
            self.val.append(doc)
            self.val_left -= len(doc)
            if self.val_left <= 0:
                write_shard(os.path.join(self.out_dir, "val_000000.bin"), np.concatenate(self.val), self.n_ranks,
                            self.dtype)
                self.val = []
            return
        while len(doc):
            take = min(len(doc), self.shard_tokens - self.n)
            self.buf[self.n:self.n + take] = doc[:take]
            self.n += take
            doc = doc[take:]
            if self.n == self.shard_tokens:
                self.flush()

    def flush(self):
        if self.n:
            write_shard(os.path.join(self.out_dir, f"train_{self.shard_idx:06d}.bin"), self.buf[:self.n],
                        self.n_ranks, self.dtype)
            self.shard_idx += 1
            self.n = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=sorted(SOURCES), required=True)
    ap.add_argument("--vocab", choices=sorted(VOCABS), default="ec200k")
    ap.add_argument("--files", type=int, default=1, help="number of upstream parquet files to use")
    ap.add_argument("--first_file", type=int, default=0,
                    help="first upstream file; above 0, appends train shards to an existing directory")
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--shard_tokens", type=int, default=100_000_000)
    ap.add_argument("--val_tokens", type=int, default=5_000_000)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()

    src = SOURCES[args.source]
    out_dir = os.path.join(args.out, args.vocab, args.source)
    raw_dir = os.path.join(args.out, "raw")
    os.makedirs(out_dir, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(out_dir, "train_*.bin")))
    if args.first_file == 0 and existing:
        raise SystemExit(f"{out_dir} already has shards; delete them first to rebuild, or append with --first_file")
    tok = Tokenizer(VOCABS[args.vocab])
    dtype = token_dtype(tok.n_vocab)
    writer = ShardWriter(out_dir, args.shard_tokens, args.val_tokens if args.first_file == 0 else 0, tok.n_ranks,
                         dtype)
    writer.shard_idx = len(existing)
    n_bytes = n_docs = 0
    t0 = time.time()
    for name in src["files"][args.first_file:args.first_file + args.files]:
        path = os.path.join(raw_dir, name)
        if not os.path.exists(path):
            t_dl = time.time()
            path = hf_hub_download(src["repo"], name, repo_type="dataset", local_dir=raw_dir)
            print(f"{name}: downloaded in {time.time() - t_dl:.0f}s", flush=True)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=16384, columns=[src["column"]]):
            texts = [t for t in batch.column(0).to_pylist() if t]
            n_bytes += sum(len(t.encode("utf-8")) for t in texts)
            n_docs += len(texts)
            for doc in tok.encode_batch(texts, threads=args.threads, prepend_eot=True):
                writer.add(doc)
            el = time.time() - t0
            print(f"  {n_docs:,} docs  {writer.total / 1e6:,.0f}M tokens  {n_bytes / 1e6 / el:.0f} MB/s", end="\r", flush=True)
        print()
    writer.flush()
    stats = dict(source=args.source, vocab=args.vocab, n_ranks=tok.n_ranks, files=[args.first_file, args.files],
                 docs=n_docs, bytes=n_bytes, tokens=writer.total,
                 bytes_per_token=n_bytes / max(writer.total - n_docs, 1), seconds=round(time.time() - t0))
    name = "stats.json" if args.first_file == 0 else f"stats_from_{args.first_file}.json"
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
