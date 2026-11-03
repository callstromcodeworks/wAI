"""Selective hippocampus (wai_struct work queue item 2): is memory worth more per stored episode when it keeps only
what is new and surprising instead of everything that arrives?

    python model/selective_memory.py [--grown runs/main_forgetting/C2s_spec_instruct_chat] [--cap 500000]

The cortex is a model that has already learned the 10 specialties (default: main's C2s, base + grown modules), so
memory is scored on top of consolidated modules, as M2's "final" numbers were. Each specialty arrives as a day:
--day windows of its training text mixed 50/50 with familiar text (fineweb-edu/code, 60/40), unlabeled. Every
policy stores the same number of episodes per day (--cap, M2 stored 500k tokens per specialty):
- all: windows of the day as they come (half familiar text);
- buffer: windows the day's detector kept (today@6: over-represented today compared with familiar text, threshold
  at the 0.9 quantile of familiar windows held outside the day; forgetting.py --stream's buffer);
- buffer p<P: from the buffer's windows, only the tokens the model predicts with p(next) < P (more windows read
  for the same number of episodes);
- labeled, labeled p<P: the specialty's own windows (a perfect detector), all tokens / surprising tokens only;
- buffer_rR, labeled_rR: a random share R of the tokens, the control for p<P (as much text covered, without choosing
  by surprise; R=0.6 matches p<0.5's share);
- buffer_mMpP: tokens with M <= p(next) < P, a band of interest: surprising, but not so surprising it looks like
  noise (people are most curious about what they half know).
Each memory is keyed by the model and scored like M2 (episodic.py score: blend chosen on the first half of each
held-out text, reported on the second): gain in bpb over the model alone, per specialty and on familiar English
and code. Philosophy is left out of means (18% of its held-out text recurs verbatim in logic's training text)."""
import argparse
import math
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.nn.attention import sdpa_kernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from episodic import EpisodicMemory
from forgetting import HERE, T, load_grown
from prepare_data import load_shard
from surprise_probe import today_share
from train import SDPA_BACKENDS, token_byte_lengths
from tokenizer import VOCABS

DOMAINS = "haskell,medical,lua,philosophy,law,fortran,logic,music,rust,space"


@torch.no_grad()
def states(model, rows, depth, batch=8):
    """Mean normalised state after block depth, per window [n, d] (what today@D compares)."""
    out = {}
    hk = model.blocks[depth].register_forward_hook(lambda m, i, o: out.__setitem__("h", o))
    model.eval()
    res = []
    for i in range(0, len(rows), batch):
        r = torch.from_numpy(rows[i:i + batch].astype(np.int64)).cuda()
        with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPA_BACKENDS, set_priority=True):
            model(r[:, :-1])
        res.append(nn.functional.rms_norm(out["h"].float(), (out["h"].size(-1),)).mean(1))
    hk.remove()
    return torch.cat(res)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grown", default="runs/main_forgetting/C2s_spec_instruct_chat")
    ap.add_argument("--domains", default=DOMAINS)
    ap.add_argument("--day", type=int, default=1500, help="specialty windows per day (as many familiar ones)")
    ap.add_argument("--cap", type=int, default=500_000, help="episodes stored per day, every policy")
    ap.add_argument("--policies", default="all,buffer,buffer_p0.9,buffer_p0.5,labeled,labeled_p0.5")
    ap.add_argument("--depth", type=int, default=6, help="today@D: state compared")
    ap.add_argument("--k", type=int, default=30, help="today@D: neighbours")
    ap.add_argument("--ref", type=int, default=2000, help="familiar windows: reference, and as many for the null")
    ap.add_argument("--eval_tokens", type=int, default=100_000, help="held-out tokens scored per text")
    ap.add_argument("--memory_k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    model, meta = load_grown(a.grown if os.path.isabs(a.grown) else os.path.join(HERE, a.grown))
    vocab = "ec32k"
    token_bytes = torch.from_numpy(token_byte_lengths(VOCABS[vocab], model.cfg.vocab_size)).cuda()
    shard = lambda src, split: load_shard(os.path.join(HERE, "data", vocab, src, f"{split}_000000.bin"))[0]
    general = [shard("fineweb-edu", "train"), shard("code", "train")]

    def fam():  # one window of familiar text, English or code 60/40
        src = general[0] if rng.random() < 0.6 else general[1]
        st = rng.integers(0, len(src) - T - 1)
        return src[st:st + T + 1]
    names = a.domains.split(",")
    ddir = os.path.join(HERE, "data", vocab, "domains")

    ref = states(model, np.stack([fam() for _ in range(a.ref)]), a.depth)
    held = states(model, np.stack([fam() for _ in range(a.ref)]), a.depth)
    days = {}
    print(f"cortex: {a.grown} ({len(model.grown_at)} grown parts); day = {a.day} specialty + {a.day} familiar windows")
    for n in names:
        ids = np.load(os.path.join(ddir, n, "train.npy"))
        off = int(rng.integers(0, max(1, len(ids) - a.day * T - 1)))
        new = np.stack([ids[off + i * T: off + i * T + T + 1] for i in range(min(a.day, (len(ids) - off - 1) // T))])
        rows = np.concatenate([new, np.stack([fam() for _ in range(len(new))])])
        is_new = np.r_[np.ones(len(new), bool), np.zeros(len(new), bool)]
        perm = rng.permutation(len(rows))
        rows, is_new = rows[perm], is_new[perm]
        share, null = today_share(states(model, rows, a.depth), ref, held, a.k)
        keep = share > np.quantile(null, 0.9)
        days[n] = dict(rows=rows, buffer=rows[keep], labeled=rows[is_new])
        print(f"  {n:11} buffer {keep.sum():5}/{len(rows)} | precision {(keep & is_new).sum() / max(1, keep.sum()):.3f} "
              f"recall {(keep & is_new).sum() / is_new.sum():.3f}", flush=True)

    texts = {n: np.load(os.path.join(ddir, n, "val.npy")) for n in names}
    texts.update({"english": shard("fineweb-edu", "val"), "code": shard("code", "val")})
    results = {}
    for pol in a.policies.split(","):
        src, _, v = pol.partition("_")
        opt = {c: float(x) for c, x in re.findall(r"([pmr])([0-9.]+)", v)}
        p, r, m = (opt.get(c) for c in "prm")
        mem = EpisodicMemory(model, token_bytes, k=a.memory_k)
        stored = read = 0
        for n in names:
            rows = days[n]["rows" if src == "all" else src]
            rows = rows[rng.permutation(len(rows))]
            s, rd = mem.store(n, rows, max_p=p, cap=a.cap, rate=r, min_p=m)
            stored, read = stored + s, read + rd
        res = {t: mem.score(ids, a.eval_tokens) for t, ids in texts.items()}
        gain = {t: (r["model"] - r["memory"]) / r["model"] for t, r in res.items()}
        mean = np.mean([gain[n] for n in names if n != "philosophy"])
        results[pol] = dict(gain=gain, stored=stored, read=read, mean=mean)
        print(f"{pol:16} stored {stored / 1e6:.2f}M episodes (read {read / 1e6:.2f}M tokens) | mean gain {mean:.2%} "
              f"({mean / (stored / 1e6):.3%} per M episodes) | " +
              " | ".join(f"{t[:7]} {g:+.2%}" for t, g in gain.items()), flush=True)
        del mem
        torch.cuda.empty_cache()
    print("\nsummary (mean over the specialties, philosophy excluded):")
    for pol, r in results.items():
        print(f"  {pol:16} {r['mean']:.2%} with {r['stored'] / 1e6:.2f}M episodes = {r['mean'] / (r['stored'] / 1e6):.3%} "
              f"per M; English {r['gain']['english']:+.2%}, code {r['gain']['code']:+.2%}")


if __name__ == "__main__":
    main()
