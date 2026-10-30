"""Meaning vs word choice: how much of a model's loss is picking the meaning, and how much is picking the word.

    python model/register_diag.py runs/sweep_b32k_sparseB32c2_t1b

Tokens are grouped by the model's own output embeddings (k-means on cosine similarity) into "meaning groups" of
near-synonyms and variants. Each token's loss splits exactly:
    -log p(w) = -log p(group(w))  +  -log p(w | group(w))
                 meaning              word choice within the meaning
Reported per text type (10 specialties, general English, code) in bits per byte, at several group counts."""
import argparse
import json
import math
import os

import numpy as np
import torch

from forgetting import HERE, T, load_model, rows_from
from prepare_data import load_shard
from tokenizer import VOCABS, Tokenizer
from train import token_byte_lengths


def kmeans(emb, k, iters=20, seed=0):
    """Spherical k-means: groups of tokens whose output embeddings point the same way."""
    g = torch.Generator(device=emb.device).manual_seed(seed)
    x = torch.nn.functional.normalize(emb.float(), dim=-1)
    cent = x[torch.randperm(len(x), generator=g, device=emb.device)[:k]].clone()
    for _ in range(iters):
        assign = (x @ cent.t()).argmax(-1)
        cent = torch.zeros_like(cent).index_add_(0, assign, x)
        empty = cent.norm(dim=-1) == 0
        cent[empty] = x[torch.randint(len(x), (int(empty.sum()),), generator=g, device=emb.device)]
        cent = torch.nn.functional.normalize(cent, dim=-1)
    return (x @ cent.t()).argmax(-1)


@torch.no_grad()
def split_loss(model, rows, group, k, token_bytes, batch=4, chunk=1024):
    """Summed (meaning, word-choice) loss in nats and bytes over the rows."""
    model.eval()
    meaning = choice = 0.0
    n_bytes = 0
    for i in range(0, len(rows), batch):
        r = torch.from_numpy(rows[i:i + batch]).cuda()
        x, y = r[:, :-1], r[:, 1:]
        with torch.autocast("cuda", torch.bfloat16):
            h = model(x)
        h, y = h.flatten(0, 1), y.flatten()
        for s in range(0, len(y), chunk):
            logp = torch.log_softmax(model.logits(h[s:s + chunk]), -1)            # [n, V] (padding at -inf)
            p = logp.exp()
            gid = group[y[s:s + chunk]]
            mass = torch.zeros(len(gid), k, device=p.device).scatter_add_(1, group.expand(len(gid), -1), p)  # p(group)
            lg = torch.log(mass.gather(1, gid[:, None]).squeeze(1).clamp_min(1e-30))
            lw = logp.gather(1, y[s:s + chunk, None]).squeeze(1)
            meaning += (-lg).sum().item()
            choice += (lg - lw).sum().item()
        n_bytes += token_bytes[y].sum().item()
    return meaning, choice, n_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--groups", default="1024,4096,8192")
    ap.add_argument("--tokens", type=int, default=200_000, help="tokens scored per text type")
    a = ap.parse_args()
    model, ck = load_model(a.ckpt)
    vocab = ck["args"]["vocab"]
    tok = Tokenizer(VOCABS[vocab])
    n_vocab = model.cfg.n_vocab
    token_bytes = torch.from_numpy(token_byte_lengths(VOCABS[vocab], model.cfg.vocab_size)).cuda()
    texts = {}
    ddir = os.path.join(HERE, "data", vocab, "domains")
    for name in sorted(os.listdir(ddir)):
        texts[name] = rows_from(np.load(os.path.join(ddir, name, "val.npy")), a.tokens // T)
    for src in ("fineweb-edu", "code"):
        texts["general_" + src] = rows_from(load_shard(os.path.join(HERE, "data", vocab, src, "val_000000.bin"))[0],
                                           a.tokens // T)
    emb = model.wte.weight[:n_vocab]
    out = {}
    for k in [int(v) for v in a.groups.split(",")]:
        group = kmeans(emb, k)
        group = torch.cat([group, torch.zeros(model.wte.weight.size(0) - n_vocab, dtype=group.dtype, device=group.device)])
        if k == 4096:  # sanity: what the meaning groups look like
            for probe in (" hello", " casual", " therefore", " said", " big"):
                ids = tok.encode(probe)
                if len(ids) == 1:
                    members = (group[:n_vocab] == group[ids[0]]).nonzero().flatten()[:14].tolist()
                    print(f"  group of {probe!r}: " + ", ".join(repr(tok.decode([m])) for m in members), flush=True)
        res = {}
        for name, rows in texts.items():
            m, c, b = split_loss(model, rows, group, k, token_bytes)
            res[name] = dict(meaning_bpb=m / math.log(2) / b, choice_bpb=c / math.log(2) / b,
                             choice_share=c / (m + c))
            print(f"groups {k:5d} | {name:20s} | total {res[name]['meaning_bpb'] + res[name]['choice_bpb']:.4f} bpb = "
                  f"meaning {res[name]['meaning_bpb']:.4f} + word choice {res[name]['choice_bpb']:.4f} "
                  f"({res[name]['choice_share']:.1%})", flush=True)
        out[str(k)] = res
    path = os.path.join(HERE, "runs", "register_diag.json")
    json.dump(dict(ckpt=a.ckpt, tokens=a.tokens, results=out), open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
