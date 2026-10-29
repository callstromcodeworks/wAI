"""Does a model use long context? Bits per byte by position in long documents.

    python model/long_context.py runs/sweep_m_full/ckpt.pt runs/sweep_m_concept/ckpt.pt [--length 8192]
    python model/long_context.py runs/sweep_m32k_full runs/sweep_m32k_concept   # exported runs (safetensors)

Takes the validation documents of at least --length tokens from each source (English, code),
scores their first --length tokens, and reports bits per byte for positions 0-1k, 1-2k, 2-4k and
4k+. A model that uses distant context gets better further in; a model that can't see far
plateaus. Positions past the training length (4,096 for the memory runs) test extrapolation.

Also reports what each model reads per query at that length: keys attended per token, and the
keys it must keep on the GPU if old blocks only come back when expanded (concept memory can keep
the raw history in CPU RAM and fetch the few blocks it expands; full attention can't).
"""
import argparse
import json
import math
import os

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel

import safe_checkpoint
from brain import BrainConfig, BrainGPT
from gpt import GPT, GPTConfig
from prepare_data import load_shard
from tokenizer import VOCABS
from train import SDPA_BACKENDS, token_byte_lengths

HERE = os.path.dirname(os.path.abspath(__file__))
BUCKETS = [(0, 1024), (1024, 2048), (2048, 4096), (4096, 1 << 30)]


def long_docs(vocab, source, length, max_docs):
    ids, _ = load_shard(os.path.join(HERE, "data", vocab, source, "val_000000.bin"))
    ids = np.asarray(ids)
    eot = VOCABS[vocab]  # <|endoftext|> follows the regular tokens
    cuts = np.flatnonzero(ids == eot)
    docs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a - 1 >= length + 1:
            docs.append(ids[a + 1:a + 1 + length + 1].astype(np.int64))
            if len(docs) == max_docs:
                break
    return docs


def reads_per_query(cfg, L):
    if cfg.attn_mode == "full":
        return L, L
    local = cfg.window + cfg.concept_block
    if cfg.attn_mode == "window":
        return local, local
    reads = local + L // cfg.concept_block + cfg.concept_topk * cfg.concept_block
    return reads, reads  # GPU-resident if the raw history lives in CPU RAM and only expanded blocks come back


@torch.no_grad()
def evaluate(path, docs_by_source, length):
    if path.endswith(".pt"):  # a checkpoint made on this machine
        ck = torch.load(path, map_location="cpu", weights_only=False)
        is_brain = ck["config"].get("arch") == "brain"
        cfg = (BrainConfig if is_brain else GPTConfig)(**{**ck["config"], "seq_len": length})  # RoPE for the eval length
        with torch.device("cuda"):
            model = BrainGPT(cfg) if is_brain else GPT(cfg)
        model.load_state_dict(ck["model"])
    else:  # an exported run (safetensors + JSON): nothing is unpickled, safe for runs from elsewhere
        model, ck = safe_checkpoint.load_model(path, seq_len=length)
        cfg = model.cfg
    model.eval()
    vocab = ck["args"]["vocab"]
    token_bytes = torch.from_numpy(token_byte_lengths(VOCABS[vocab], cfg.vocab_size)).cuda()
    run = path if os.path.isdir(path) else os.path.dirname(path)
    res = {"run": os.path.basename(os.path.normpath(run)), "train_len": ck["config"]["seq_len"],
           "attn_mode": cfg.attn_mode, "fast_heads": cfg.fast_heads, "arch": getattr(cfg, "arch", "gpt")}
    res["keys_per_query"], res["gpu_keys"] = reads_per_query(cfg, length)
    for source, docs in docs_by_source.items():
        loss_b, bytes_b = np.zeros(len(BUCKETS)), np.zeros(len(BUCKETS))
        for d in docs:
            r = torch.from_numpy(d).cuda()[None]
            x, y = r[:, :-1], r[:, 1:]
            with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPA_BACKENDS, set_priority=True):
                losses = model.token_losses(model(x), y)[0].float()
            nb = token_bytes[y[0]].float()
            for i, (a, b) in enumerate(BUCKETS):
                loss_b[i] += losses[a:b].sum().item()
                bytes_b[i] += nb[a:b].sum().item()
        res[source] = [lb / math.log(2) / bb if bb else float("nan") for lb, bb in zip(loss_b, bytes_b)]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--length", type=int, default=8192)
    ap.add_argument("--docs", type=int, default=40, help="documents per source")
    ap.add_argument("--vocab", default="ec32k")
    a = ap.parse_args()
    docs = {s: long_docs(a.vocab, s, a.length, a.docs) for s in ("fineweb-edu", "code")}
    print(f"documents of {a.length}+ tokens: " + ", ".join(f"{s} {len(d)}" for s, d in docs.items()))
    rows = [evaluate(p, docs, a.length) for p in a.ckpts]
    names = [f"{x[0] // 1024}-{x[1] // 1024 if x[1] < 1 << 30 else a.length // 1024}k" for x in BUCKETS]
    print(f"\n| run | keys/query at {a.length // 1024}k | " + " | ".join(f"Eng {n}" for n in names) + " | "
          + " | ".join(f"code {n}" for n in names) + " |")
    print("|---|---:|" + "---:|" * (2 * len(BUCKETS)))
    for r in rows:
        print(f"| {r['run']} | {r['keys_per_query']:,} | " + " | ".join(f"{v:.4f}" for v in r["fineweb-edu"]) + " | "
              + " | ".join(f"{v:.4f}" for v in r["code"]) + " |")
    out = os.path.join(HERE, "runs", "long_context.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
