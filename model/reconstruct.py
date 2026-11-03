"""Replay by reconstruction (wai_struct STRUCT_PLAN item 3b): the model writes its own replay text.

    python model/reconstruct.py check  [--grown runs/main_forgetting/C2s_spec_instruct_chat]  # decoding = forward?
    python model/reconstruct.py bench  [--grown ...] [--batch 32]                              # tokens/s, memory
    python model/reconstruct.py sample [--grown ...] [--n 4] [--cue haskell --cue_len 32]      # read what it writes

generate() samples windows from the model at temperature 1 (its own distribution), each from a document start
(<|endoftext|>) or from a cue (the first tokens of a kept window), with BrainCache incremental decoding. Training on
its own samples changes, in expectation, nothing in the model that wrote them (the gradient of log p over samples
drawn from p averages to zero). So replaying them later pulls the model back towards what it knew when it wrote
them: a sampled estimate of KL(then || now), with no old text stored. Patterns it knows well come back close to their
source, rare ones loosely."""
import argparse
import copy
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import BrainCache
from gpt import rope_tables
from forgetting import HERE, T, load_grown
from prepare_data import load_shard
from tokenizer import VOCABS, Tokenizer

VOCAB = "ec32k"


class Decoder:
    """Incremental decoding of a BrainGPT: a bf16 copy of the model (RoPE tables and router bias kept in fp32), a
    BrainCache, and one decoding step captured as a CUDA graph and replayed for every token."""

    @torch.no_grad()
    def __init__(self, model, batch, length, graph=True):
        m = copy.deepcopy(model).bfloat16().eval()
        c = m.cfg
        m.rope_cos, m.rope_sin = rope_tables(c.seq_len, c.head_dim, c.rope_base, "cuda")
        m.rope_cos_m, m.rope_sin_m = rope_tables(c.seq_len, c.module_dim // c.module_heads, c.rope_base, "cuda")
        if hasattr(model, "route_bias"):
            m.route_bias = model.route_bias.detach().float().clone()
        self.m, self.cache = m, BrainCache(m, batch, length)
        self.tok = torch.zeros(batch, 1, dtype=torch.long, device="cuda")
        self.graph = None
        if graph:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):  # warm-up (allocates the banks' buffers) before capture
                for _ in range(2):
                    self._step()
            torch.cuda.current_stream().wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self._step()
        self.cache.reset()

    def _step(self):
        return self.m.logits(self.m(self.tok, self.cache)[:, -1].float())

    @torch.no_grad()
    def step(self, tok):
        """Logits [batch, vocab] for the token after tok [batch, 1] at the current position; then moves on one."""
        self.tok.copy_(tok)
        if self.graph is not None:
            self.graph.replay()
            out = self.out
        else:
            out = self._step()
        self.cache.advance(1)
        return out

    def reset(self):
        self.cache.reset()


@torch.no_grad()
def generate(model, n, length, eot, batch=32, cues=None, temperature=1.0, seed=0, graph=True):
    """[n, length] uint16 windows written by the model. cues [n, L]: the first L tokens of each window are given and
    the rest is written; without cues each window starts at <|endoftext|>."""
    torch.cuda.empty_cache()  # inside a training run: hand cached blocks back before the decoder's buffers
    dec = Decoder(model, batch, length, graph)
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = np.empty((n, length), np.uint16)
    for i in range(0, n, batch):
        B = min(batch, n - i)
        seq = torch.full((batch, length), eot, dtype=torch.long, device="cuda")  # fixed shape: spare rows idle
        given = 1
        if cues is not None:
            given = cues.shape[1]
            seq[:B, :given] = torch.from_numpy(cues[i:i + B].astype(np.int64)).cuda()
        dec.reset()
        for t in range(length - 1):
            logits = dec.step(seq[:, t:t + 1])
            if t + 1 >= given:
                seq[:, t + 1] = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=g)[:, 0]
        out[i:i + B] = seq[:B].cpu().numpy()
    del dec
    torch.cuda.empty_cache()
    return out


def load(a):
    path = a.grown if os.path.isabs(a.grown) else os.path.join(HERE, a.grown)
    model, meta = load_grown(path)
    return model, len(model.grown_at)


def val_rows(name, n, length, rng):
    ids = (np.load(os.path.join(HERE, "data", VOCAB, "domains", name, "val.npy")) if name not in ("fineweb-edu", "code")
           else load_shard(os.path.join(HERE, "data", VOCAB, name, "val_000000.bin"))[0])
    starts = rng.integers(0, len(ids) - length - 1, n)
    return np.stack([ids[s:s + length] for s in starts]).astype(np.int64)


@torch.no_grad()
def check(model, rows, graph=True):
    """Decoding (bf16 copy, CUDA graph) against one full forward of the model over the same windows."""
    x = torch.from_numpy(rows).cuda()
    B, L = x.shape
    with torch.autocast("cuda", torch.bfloat16):
        full = torch.log_softmax(model.logits(model(x[:, :-1]).float()), -1)
    dec = Decoder(model, B, L - 1, graph)
    step = torch.stack([torch.log_softmax(dec.step(x[:, t:t + 1]), -1) for t in range(L - 1)], 1)
    y = x[:, 1:, None]
    lf, ls = full.gather(-1, y)[..., 0], step.gather(-1, y)[..., 0]
    d = (lf - ls).abs()
    agree = (full.argmax(-1) == step.argmax(-1)).float().mean().item()
    print(f"log p(next), decoding vs full forward over {B} x {L - 1} tokens: mean |diff| {d.mean():.4f}, "
          f"max {d.max():.4f}; loss {-lf.mean():.4f} vs {-ls.mean():.4f}; top-1 agree {agree:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["check", "bench", "sample"])
    ap.add_argument("--grown", default="runs/main_forgetting/C2s_spec_instruct_chat")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--length", type=int, default=T + 1)
    ap.add_argument("--cue", default="", help="sample: cues from this specialty's held-out text (or fineweb-edu, code)")
    ap.add_argument("--cue_len", type=int, default=32)
    ap.add_argument("--chars", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    model, parts = load(a)
    tok = Tokenizer(VOCABS[VOCAB])
    print(f"model: {a.grown} ({parts} grown parts)")
    if a.what == "check":
        for name in ("fineweb-edu", "code", "haskell"):
            print(name, end=": ")
            check(model, val_rows(name, 4, T + 1, rng))
    elif a.what == "bench":
        torch.cuda.reset_peak_memory_stats()
        generate(model, a.batch, 64, tok.eot, a.batch)  # warm-up
        torch.cuda.synchronize()
        t0 = time.time()
        generate(model, a.batch, a.length, tok.eot, a.batch, seed=a.seed)
        torch.cuda.synchronize()
        s = time.time() - t0
        print(f"batch {a.batch} x {a.length} tokens: {s:.1f} s, {a.batch * a.length / s:,.0f} tokens/s, "
              f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")
    else:
        cues = val_rows(a.cue, a.n, a.cue_len, rng) if a.cue else None
        rows = generate(model, a.n, a.length, tok.eot, a.batch, cues=cues, seed=a.seed)
        for i, r in enumerate(rows):
            text = tok.decode(r.tolist())
            cut = len(tok.decode(cues[i].tolist())) if cues is not None else 0
            print(f"--- {i + 1}" + (f" (cue: {text[:cut]!r})" if cut else ""))
            print(text[cut:cut + a.chars])


if __name__ == "__main__":
    main()
