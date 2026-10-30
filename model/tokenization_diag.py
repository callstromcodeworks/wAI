"""Is the prose difficulty about how words are split into tokens, or about the words themselves?

    python model/tokenization_diag.py runs/sweep_s_med_plain runs/sweep_s_med_plain_ec64k

1. Fragmentation (first model): per text type, how many words are split into 2+ / 3+ tokens, and how the loss
   divides between single-token words, the first piece of a split word, and its continuation pieces.
2. Vocabulary comparison: each model scores the SAME text (each specialty's held-out set decoded and re-encoded
   with that model's vocabulary), in bits per byte, so a 32k and a 64k vocabulary compare directly."""
import argparse
import json
import math
import os

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel

from forgetting import HERE, T, load_model, rows_from
from prepare_data import load_shard
from tokenizer import VOCABS, Tokenizer
from train import SDPA_BACKENDS, token_byte_lengths


def texts_of(tok32, n_tokens):
    """Each text type's held-out documents as strings (decoded from the ec32k sets, split at document boundaries)."""
    out = {}
    ddir = os.path.join(HERE, "data", "ec32k", "domains")
    srcs = {n: np.load(os.path.join(ddir, n, "val.npy")) for n in sorted(os.listdir(ddir))}
    for s in ("fineweb-edu", "code"):
        srcs["general_" + s] = load_shard(os.path.join(HERE, "data", "ec32k", s, "val_000000.bin"))[0]
    for name, ids in srcs.items():
        ids = ids[:n_tokens].astype(np.int64)
        cuts = np.flatnonzero(ids == tok32.eot)
        docs = [tok32.decode(ids[a + 1:b].tolist()) for a, b in zip(cuts, list(cuts[1:]) + [len(ids)]) if b > a + 1]
        out[name] = docs
    return out


@torch.no_grad()
def token_losses(model, ids):
    """Per-token loss (nats) over a token stream, in windows of T."""
    model.eval()
    rows = rows_from(ids, 10 ** 9)
    losses = []
    for i in range(0, len(rows), 8):
        r = torch.from_numpy(rows[i:i + 8]).cuda()
        with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPA_BACKENDS, set_priority=True):
            h = model(r[:, :-1])
        losses.append(model.token_losses(h, r[:, 1:]).float().cpu().numpy().reshape(-1))
    return np.concatenate(losses), rows[:, 1:].reshape(-1)


def word_starts(tok, ids):
    """True where a token begins a new word: it starts with a non-letter/digit (space, newline, punctuation),
    or the previous token ended with one."""
    pieces = [tok.decode([int(t)]) for t in ids]
    starts = np.ones(len(ids), dtype=bool)
    for i in range(1, len(ids)):
        p, c = pieces[i - 1], pieces[i]
        starts[i] = not (c[:1].isalnum() and p[-1:].isalnum())
    return starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs=2, help="a 32k-vocabulary model, then a 64k-vocabulary model")
    ap.add_argument("--tokens", type=int, default=200_000)
    a = ap.parse_args()
    tok32 = Tokenizer(VOCABS["ec32k"])
    texts = texts_of(tok32, a.tokens)
    report = {"fragmentation": {}, "vocab": {}}

    # 1. fragmentation, with the 32k model
    model, ck = load_model(a.ckpts[0])
    print("== fragmentation (32k vocabulary)")
    for name, docs in texts.items():
        ids = np.concatenate(tok32.encode_batch(docs, prepend_eot=True)).astype(np.int64)
        loss, tgt = token_losses(model, ids)
        st = word_starts(tok32, tgt)
        wid = np.cumsum(st) - 1                                   # word index of each token
        n_pieces = np.bincount(wid)
        piece_of = n_pieces[wid]
        single = piece_of == 1
        first = st & ~single
        cont = ~st
        tot = loss.sum()
        r = dict(words=int(len(n_pieces)), split2=float((n_pieces >= 2).mean()), split3=float((n_pieces >= 3).mean()),
                 tokens_per_word=float(len(tgt) / len(n_pieces)),
                 loss_share_single=float(loss[single].sum() / tot), loss_share_first=float(loss[first].sum() / tot),
                 loss_share_cont=float(loss[cont].sum() / tot),
                 loss_per_word_single=float(loss[single].mean()),
                 loss_per_word_split=float(np.bincount(wid, loss)[n_pieces >= 2].mean()))
        report["fragmentation"][name] = r
        print(f"  {name:20s} | words split 2+ {r['split2']:.1%}, 3+ {r['split3']:.1%} | tokens/word {r['tokens_per_word']:.2f} | "
              f"loss share: whole words {r['loss_share_single']:.1%}, first pieces {r['loss_share_first']:.1%}, "
              f"continuations {r['loss_share_cont']:.1%} | nats per word: whole {r['loss_per_word_single']:.2f}, "
              f"split {r['loss_per_word_split']:.2f}", flush=True)

    # 2. the same text under each model's own vocabulary, bits per byte
    print("== vocabulary comparison (same text, bits per byte)")
    for path in a.ckpts:
        if path != a.ckpts[0]:
            del model
            torch.cuda.empty_cache()
            model, ck = load_model(path)
        vocab = ck["args"]["vocab"]
        tok = Tokenizer(VOCABS[vocab])
        tb = token_byte_lengths(VOCABS[vocab], model.cfg.vocab_size)
        res = {}
        for name, docs in texts.items():
            ids = np.concatenate(tok.encode_batch(docs, prepend_eot=True)).astype(np.int64)
            loss, tgt = token_losses(model, ids)
            res[name] = float(loss.sum() / math.log(2) / tb[tgt].sum())
        report["vocab"][vocab] = res
        print(f"  {vocab}: " + " | ".join(f"{k} {v:.4f}" for k, v in res.items()), flush=True)
    v32, v64 = report["vocab"]["ec32k"], report["vocab"]["ec64k"]
    print("== 64k gain over 32k (bpb, positive = 64k better)")
    for name in texts:
        print(f"  {name:20s} {v32[name] - v64[name]:+.4f}  ({(v32[name] - v64[name]) / v32[name]:+.1%})")
    path = os.path.join(HERE, "runs", "tokenization_diag.json")
    json.dump(report, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
