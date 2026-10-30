"""Hippocampal episodic memory beside a trained model: remember moments of text it has read, recall similar moments
while reading new text, and blend what came next in them with the model's own prediction.

    python model/memory_lm.py runs/sweep_b32k_sparseB32c2_t1b --store_tokens 10e6 [--domains 1]

Episode = (the model's final hidden state at a position, the token that actually came next). Recall = the k stored
states nearest the current one (L2); the memory's guess p_mem(v) = sum of softmax(-distance / tau) over the
neighbours whose next token was v. Prediction = (1 - lam) p_model + lam p_mem (the cortex/hippocampus blend; cf.
kNN-LM). Nothing is trained: new knowledge enters memory by reading it once. Reports bits per byte per text type
without memory, with the best fixed blend, and with a blend that leans on memory when the model is unsure."""
import argparse
import json
import math
import os

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel

from forgetting import HERE, T, load_model, rows_from
from prepare_data import load_shard
from train import SDPA_BACKENDS, token_byte_lengths
from tokenizer import VOCABS


@torch.no_grad()
def states(model, ids, batch=16):
    """Final hidden states and next tokens for a token stream, in windows of T."""
    model.eval()
    rows = rows_from(ids, 10 ** 9)
    hs, ys = [], []
    for i in range(0, len(rows), batch):
        r = torch.from_numpy(rows[i:i + batch]).cuda()
        with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPA_BACKENDS, set_priority=True):
            h = model(r[:, :-1])
        hs.append(h.reshape(-1, h.size(-1)).half())
        ys.append(r[:, 1:].reshape(-1))
    return torch.cat(hs), torch.cat(ys)


@torch.no_grad()
def recall(keys, key_sq, q, k, chunk=1 << 20, qb=1024):
    """k nearest stored states (squared L2) for each query: distances [n, k] and indices [n, k]."""
    out_d, out_i = [], []
    for s in range(0, len(q), qb):
        qq = q[s:s + qb].float()
        best_d = torch.full((len(qq), k), float("inf"), device=q.device)
        best_i = torch.zeros((len(qq), k), dtype=torch.long, device=q.device)
        for c in range(0, len(keys), chunk):
            d = key_sq[c:c + chunk][None, :] - 2 * (qq.half() @ keys[c:c + chunk].t()).float()   # + |q|^2 later
            dd, ii = torch.topk(d, min(k, d.size(1)), dim=1, largest=False)
            cat_d, cat_i = torch.cat([best_d, dd], 1), torch.cat([best_i, ii + c], 1)
            best_d, sel = torch.topk(cat_d, k, dim=1, largest=False)
            best_i = cat_i.gather(1, sel)
        out_d.append(best_d + qq.square().sum(-1, keepdim=True))
        out_i.append(best_i)
    return torch.cat(out_d), torch.cat(out_i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--store_tokens", type=float, default=5e6, help="general text read into memory (60% English, 40% code)")
    ap.add_argument("--domain_tokens", type=float, default=1e6, help="training text per specialty read into memory")
    ap.add_argument("--domains", type=int, default=1, help="also read each specialty's training text into memory")
    ap.add_argument("--eval_tokens", type=int, default=100_000, help="tokens scored per text type")
    ap.add_argument("--k", type=int, default=32)
    a = ap.parse_args()
    model, ck = load_model(a.ckpt)
    vocab = ck["args"]["vocab"]
    tb = torch.from_numpy(token_byte_lengths(VOCABS[vocab], model.cfg.vocab_size)).cuda()

    # evaluation texts (held out from everything stored)
    evals = {}
    ddir = os.path.join(HERE, "data", vocab, "domains")
    names = sorted(os.listdir(ddir))
    for n in names:
        evals[n] = np.load(os.path.join(ddir, n, "val.npy"))[:a.eval_tokens + 1]
    for src in ("fineweb-edu", "code"):
        evals["general_" + src] = load_shard(os.path.join(HERE, "data", vocab, src, "val_000000.bin"))[0][:a.eval_tokens + 1]

    def run_store(sources, label):
        keys, vals = [], []
        for ids in sources:
            h, y = states(model, ids)
            keys.append(h)
            vals.append(y)
        keys, vals = torch.cat(keys), torch.cat(vals)
        print(f"memory '{label}': {len(keys):,} episodes ({keys.numel() * 2 / 2**30:.1f} GiB)", flush=True)
        return keys, vals, keys.float().square().sum(-1)

    general = [load_shard(os.path.join(HERE, "data", vocab, "fineweb-edu", "train_000000.bin"))[0][:int(a.store_tokens * 0.6)],
               load_shard(os.path.join(HERE, "data", vocab, "code", "train_000000.bin"))[0][:int(a.store_tokens * 0.4)]]
    stores = {"general": general}
    if a.domains:
        stores["general+specialties"] = general + [np.load(os.path.join(ddir, n, "train.npy"))[:int(a.domain_tokens)]
                                                   for n in names]

    report = {}
    for label, sources in stores.items():
        keys, vals, key_sq = run_store(sources, label)
        report[label] = {}
        for name, ids in evals.items():
            h, y = states(model, ids)
            lm, ent = [], []
            with torch.no_grad():
                for c in range(0, len(h), 8192):
                    logp = torch.log_softmax(model.logits(h[c:c + 8192].float()), -1)
                    lm.append(logp.gather(1, y[c:c + 8192, None]).squeeze(1))     # log p_model(next)
                    ent.append(-(logp.exp() * logp).nan_to_num(0.0).sum(-1))      # model's uncertainty
            lm, ent = torch.cat(lm), torch.cat(ent)
            d, i = recall(keys, key_sq, h, a.k)
            match = (vals[i] == y[:, None]).float()
            # settings are chosen on the first half and reported on the second (no tuning on the reported text)
            half = torch.arange(len(y), device=y.device) >= len(y) // 2
            nb = {False: tb[y][~half].sum().item(), True: tb[y][half].sum().item()}
            bpb = lambda lp, part=False: (-lp[half == part]).sum().item() / math.log(2) / nb[part]
            best = dict(lam=0.0, tau=None, bpb=bpb(lm), make=lambda: lm)
            base = bpb(lm, True)
            for tau in (1.0, 3.0, 10.0, 30.0, 100.0):
                w = torch.softmax(-d / tau, -1)
                pm = (w * match).sum(-1)                                          # p_mem(next)
                for lam in (0.05, 0.1, 0.2, 0.3, 0.5):
                    mix = torch.log((1 - lam) * lm.exp() + lam * pm + 1e-12)
                    v = bpb(mix)
                    if v < best["bpb"]:
                        best = dict(lam=lam, tau=tau, bpb=v, make=lambda m=mix: m)
                # lean on memory when unsure: lam_t = lam_max * sigmoid(ent - c)
                for lam_max in (0.3, 0.5, 0.8):
                    for c in (1.0, 2.0, 3.0):
                        lt = lam_max * torch.sigmoid(ent - c)
                        mix = torch.log((1 - lt) * lm.exp() + lt * pm + 1e-12)
                        v = bpb(mix)
                        if v < best["bpb"]:
                            best = dict(lam=f"adaptive {lam_max}/{c}", tau=tau, bpb=v, make=lambda m=mix: m)
            test = bpb(best["make"](), True)
            report[label][name] = dict(base=base, memory=test, lam=best["lam"], tau=best["tau"])
            print(f"  [{label}] {name:20s} model {base:.4f} -> with memory {test:.4f} "
                  f"({(base - test) / base:+.1%}; lam {best['lam']}, tau {best['tau']})", flush=True)
        del keys, vals, key_sq
        torch.cuda.empty_cache()
    path = os.path.join(HERE, "runs", "memory_lm.json")
    json.dump(dict(ckpt=a.ckpt, store_tokens=a.store_tokens, k=a.k, results=report), open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
