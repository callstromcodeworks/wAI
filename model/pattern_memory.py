"""Pattern hippocampus (wai_struct STRUCT_PLAN item 2b): memory keeps patterns, not copies.

    python model/pattern_memory.py [--grown runs/main_forgetting/C2s_spec_instruct_chat]

The owner's account of human memory: recall isn't verbatim, the brain keeps patterns and reconstructs them, and a
pattern used more is recalled closer to its source. Here each episode's key (the model's 768-number state, 1,536
bytes at 16 bits) becomes pointers into prototypes learned from what was stored (product quantization):
- coarse, 32 bytes: the state cut into 32 pieces of 24 numbers, each stored as the nearest of 256 prototypes;
- fine, 96 bytes: coarse plus a correction, the leftover cut into 64 pieces of 12 numbers, 256 prototypes each.
What is stored is item 2's best policy: the day's buffer (today@6), tokens with p(next) < 0.5, 500k episodes a day.
Every episode arrives fine (a recent memory is detailed). Then a period of use: more of each specialty's training
text (windows outside its day) and familiar text query memory, and each episode counts how often it was among the
k recalled. Precision then follows use:
- use: the most-recalled quarter keep their correction, the rest drop to coarse (48 bytes an episode on average);
- random: a random quarter keep it, the same bytes (the control for use);
- use, never-recalled released: score's "held" column on the use memory.
Scored like item 2 (episodic.py score): gain over the model alone on held-out text; philosophy left out of means.
Search decodes the codes back to 16-bit vectors (simpler than searching on codes; the same neighbours), so bytes are
counted from the codes, not from what this script holds."""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from episodic import EpisodicMemory, encode
from forgetting import HERE, T, load_grown
from prepare_data import load_shard
from selective_memory import DOMAINS, states
from surprise_probe import today_share
from tokenizer import VOCABS
from train import token_byte_lengths


@torch.no_grad()
def kmeans(x, k=256, iters=15, chunk=32768, seed=0):
    """Centroids [M, k, ds] for each of M sub-vector sets x [M, N, ds] (float32, GPU)."""
    M, N, ds = x.shape
    g = torch.Generator(device="cuda").manual_seed(seed)
    c = x[:, torch.randperm(N, device="cuda", generator=g)[:k]].clone()
    for _ in range(iters):
        s = torch.zeros(M, k, ds, device="cuda")
        n = torch.zeros(M, k, device="cuda")
        for i in range(0, N, chunk):
            xc = x[:, i:i + chunk]
            a = assign(xc, c)
            s.scatter_add_(1, a[..., None].expand(-1, -1, ds), xc)
            n.scatter_add_(1, a, torch.ones_like(a, dtype=torch.float32))
        empty = n == 0  # a prototype nobody chose: restart it on a random vector
        c = torch.where(empty[..., None], x[:, torch.randint(0, N, (k,), device="cuda", generator=g)],
                        s / n.clamp(min=1)[..., None])
    return c


def assign(x, c):
    """Nearest centroid per sub-vector: x [M, n, ds], c [M, k, ds] -> [M, n] (long)."""
    return (c.square().sum(-1)[:, None, :] - 2 * torch.bmm(x, c.transpose(1, 2))).argmin(-1)


class PQ:
    """Coarse codes (M1 bytes) and a correction (M2 bytes) for 768-number states."""

    def __init__(self, sample, m1=32, m2=64):
        d = sample.size(1)
        self.m1, self.m2, self.d = m1, m2, d
        self.c1 = kmeans(sample.view(-1, m1, d // m1).transpose(0, 1).contiguous())
        res = sample - self.decode1(self.encode1(sample))
        self.c2 = kmeans(res.view(-1, m2, d // m2).transpose(0, 1).contiguous(), seed=1)

    def _enc(self, x, c, m, chunk=16384):
        return torch.cat([assign(x[i:i + chunk].reshape(-1, m, self.d // m).transpose(0, 1).contiguous(), c).t()
                          for i in range(0, len(x), chunk)]).to(torch.uint8)  # [n, m]

    def _dec(self, codes, c):
        m = c.size(0)
        return c[torch.arange(m, device="cuda")[None, :], codes.long()].reshape(len(codes), self.d)

    def encode1(self, x):
        return self._enc(x, self.c1, self.m1)

    def decode1(self, codes):
        return self._dec(codes, self.c1)

    def encode(self, x):
        """(coarse [n, m1], correction [n, m2]) uint8 for states x [n, d]."""
        a = self.encode1(x)
        return a, self._enc(x - self.decode1(a), self.c2, self.m2)

    def decode(self, a, b=None):
        x = self.decode1(a)
        return x if b is None else x + self._dec(b, self.c2)


def with_keys(mem, sizes, keys, held=None):
    """A memory holding mem's episodes (same next tokens) under other keys: keys {tag: function returning its [n, d]
    keys on the GPU}; held {tag: bool [n]} marks episodes still within reach."""
    m = EpisodicMemory(mem.model, mem.tb, k=mem.k)
    m.rows, m.sel = mem.rows, mem.sel
    at = 0
    for tag, n in zip(mem.rows, sizes):
        h = keys[tag]()
        m.keep[tag] = held[tag] if held is not None else torch.ones(n, dtype=torch.bool)
        m._add(tag, h.half(), mem.vals[at:at + n])
        at += n
        del h
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grown", default="runs/main_forgetting/C2s_spec_instruct_chat")
    ap.add_argument("--domains", default=DOMAINS)
    ap.add_argument("--day", type=int, default=1500, help="specialty windows per day (as many familiar ones)")
    ap.add_argument("--cap", type=int, default=500_000, help="episodes stored per day")
    ap.add_argument("--max_p", type=float, default=0.5, help="store tokens predicted with p(next) below this")
    ap.add_argument("--m1", type=int, default=32, help="coarse code bytes")
    ap.add_argument("--m2", type=int, default=64, help="correction bytes")
    ap.add_argument("--fine_share", type=float, default=0.25, help="use / random: share that keeps the correction")
    ap.add_argument("--use_windows", type=int, default=64, help="use period: windows per specialty (as many familiar)")
    ap.add_argument("--variants", default="exact,fine,coarse,use,random")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--ref", type=int, default=2000)
    ap.add_argument("--eval_tokens", type=int, default=100_000)
    ap.add_argument("--memory_k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
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

    # the days, as in selective_memory.py: each specialty's windows mixed 50/50 with familiar text, buffer by today@6
    ref = states(model, np.stack([fam() for _ in range(a.ref)]), a.depth)
    held_ref = states(model, np.stack([fam() for _ in range(a.ref)]), a.depth)
    days = {}
    print(f"cortex: {a.grown} ({len(model.grown_at)} grown parts); day = {a.day} specialty + {a.day} familiar windows; "
          f"stored: buffer tokens with p < {a.max_p}, {a.cap} a day")
    for n in names:  # built exactly as selective_memory.py builds them (same seed: the same days and buffers)
        ids = np.load(os.path.join(ddir, n, "train.npy"))
        off = int(rng.integers(0, max(1, len(ids) - a.day * T - 1)))
        new = np.stack([ids[off + i * T: off + i * T + T + 1] for i in range(min(a.day, (len(ids) - off - 1) // T))])
        rows = np.concatenate([new, np.stack([fam() for _ in range(len(new))])])
        is_new = np.r_[np.ones(len(new), bool), np.zeros(len(new), bool)]
        perm = rng.permutation(len(rows))
        rows, is_new = rows[perm], is_new[perm]
        share, null = today_share(states(model, rows, a.depth), ref, held_ref, a.k)
        keep = share > np.quantile(null, 0.9)
        days[n] = dict(buffer=rows[keep], ids=ids, span=(off, off + len(new) * T))
        print(f"  {n:11} buffer {keep.sum():5}/{len(rows)} | precision {(keep & is_new).sum() / max(1, keep.sum()):.3f}",
              flush=True)
    mem = EpisodicMemory(model, token_bytes, k=a.memory_k)
    use_rows = []
    for n in names:
        buf, ids, (lo, hi) = days[n]["buffer"], days[n]["ids"], days[n]["span"]
        s, r = mem.store(n, buf[rng.permutation(len(buf))], max_p=a.max_p, cap=a.cap)
        # the use period's text: windows of this specialty outside its day, and as many familiar ones
        out = [i for i in range(0, len(ids) - T - 1, T) if i + T < lo or i > hi]
        pick = rng.choice(len(out), size=min(a.use_windows, len(out)), replace=False)
        use_rows += [ids[out[j]:out[j] + T + 1] for j in pick] + [fam() for _ in range(a.use_windows)]
        print(f"  {n:11} stored {s:,} episodes from {r:,} tokens | use-period windows from outside the day: {len(pick)}",
              flush=True)
    sizes = [len(b) for b in mem.blocks]
    n_ep = sum(sizes)

    # prototypes learned from what was stored, then every episode's codes
    g = torch.Generator().manual_seed(a.seed)
    allk = torch.cat([b[torch.randint(0, len(b), (min(len(b), 262144 // len(sizes)),), generator=g)]
                      for b in mem.blocks]).cuda().float()
    pq = PQ(allk, a.m1, a.m2)
    del allk
    codes = {}
    err = {"coarse": 0.0, "fine": 0.0, "norm": 0.0}
    for tag, blk in zip(mem.rows, mem.blocks):
        cs = []
        for c in range(0, len(blk), 131072):
            x = blk[c:c + 131072].cuda().float()
            ca, cb = pq.encode(x)
            cs.append((ca.cpu(), cb.cpu()))
            err["coarse"] += (x - pq.decode(ca)).square().sum().item()
            err["fine"] += (x - pq.decode(ca, cb)).square().sum().item()
            err["norm"] += x.square().sum().item()
        codes[tag] = (torch.cat([c[0] for c in cs]), torch.cat([c[1] for c in cs]))
    print(f"codes: {n_ep:,} episodes; relative squared error coarse {err['coarse'] / err['norm']:.3f}, "
          f"fine {err['fine'] / err['norm']:.3f} ({time.time() - t0:.0f} s)", flush=True)

    def decoded(tag, fine_mask=None):
        """The tag's keys from its codes: fine everywhere (fine_mask True), coarse everywhere (None), or fine where the
        mask [n] is set and coarse elsewhere."""
        ca, cb = codes[tag]
        out = []
        for c in range(0, len(ca), 131072):
            a_, b_ = ca[c:c + 131072].cuda(), cb[c:c + 131072].cuda()
            x = pq.decode(a_, b_) if fine_mask is True else pq.decode(a_)
            if fine_mask is not None and fine_mask is not True:
                m = fine_mask[c:c + 131072].cuda()
                x = torch.where(m[:, None], pq.decode(a_, b_), x)
            out.append(x.half())
        return torch.cat(out)

    texts = {n: np.load(os.path.join(ddir, n, "val.npy")) for n in names}
    texts.update({"english": shard("fineweb-edu", "val"), "code": shard("code", "val")})
    avg = a.m1 + a.fine_share * a.m2
    variants = a.variants.split(",")
    results = {}

    def report(var, m, nbytes):
        res = {t: m.score(ids, a.eval_tokens) for t, ids in texts.items()}
        gain = {t: (r["model"] - r["memory"]) / r["model"] for t, r in res.items()}
        gain_held = {t: (r["model"] - r["held"]) / r["model"] for t, r in res.items()}
        mean = np.mean([gain[n] for n in names if n != "philosophy"])
        mean_held = np.mean([gain_held[n] for n in names if n != "philosophy"])
        mb = n_ep * nbytes / 2 ** 20
        results[var] = dict(gain=gain, gain_held=gain_held, mean=mean, mean_held=mean_held, bytes=nbytes, mb=mb)
        print(f"{var:7} {nbytes:6.0f} B/episode ({mb:,.0f} MB of keys) | mean gain {mean:.2%} ({mean / (mb / 1024):.2%} "
              f"per GB)" + (f", never-recalled released: {mean_held:.2%}" if var == "use" else "") + " | " +
              " | ".join(f"{t[:7]} {v:+.2%}" for t, v in gain.items()) + f" ({time.time() - t0:.0f} s)", flush=True)

    if "exact" in variants:
        report("exact", mem, 2 * model.cfg.d_model)
    mem.blocks = []  # free the exact keys (pinned RAM); the codes stand in for them from here
    torch.cuda.empty_cache()

    # the period of use: every episode arrives fine; count how often each is among the k recalled
    fine_mem = with_keys(mem, sizes, {t: (lambda t=t: decoded(t, True)) for t in mem.rows})
    hits = torch.zeros(n_ep, dtype=torch.long, device="cuda")
    use_rows = np.stack(use_rows).astype(np.int64)
    for i in range(0, len(use_rows), 64):
        h, _, _ = encode(model, use_rows[i:i + 64])
        _, idx = fine_mem.search(h)
        hits += torch.bincount(idx.flatten(), minlength=n_ep)
        del h, idx
    order = torch.argsort(hits.float() + torch.rand(n_ep, device="cuda"), descending=True)  # random tie-break
    n_fine = int(a.fine_share * n_ep)
    used = torch.zeros(n_ep, dtype=torch.bool, device="cuda")
    used[order[:n_fine]] = True
    rand = torch.zeros(n_ep, dtype=torch.bool, device="cuda")
    rand[torch.randperm(n_ep, device="cuda")[:n_fine]] = True
    recalled = (hits > 0).cpu()
    print(f"use period: {len(use_rows):,} windows ({len(use_rows) * T:,} queries); recalled at least once "
          f"{recalled.float().mean():.1%} of episodes, the most-used quarter >= {int(hits[order[n_fine - 1]])} times "
          f"({time.time() - t0:.0f} s)", flush=True)
    split = lambda v: dict(zip(mem.rows, torch.split(v.cpu(), sizes)))
    used_t, rand_t, rec_t = split(used), split(rand), split(recalled)
    if "fine" in variants:
        report("fine", fine_mem, a.m1 + a.m2)
    del fine_mem
    torch.cuda.empty_cache()
    for var in variants:
        if var in ("exact", "fine"):
            continue
        if var == "coarse":
            m, nbytes = with_keys(mem, sizes, {t: (lambda t=t: decoded(t)) for t in mem.rows}), a.m1
        elif var == "use":  # ... and those never recalled are released (the held column)
            m, nbytes = with_keys(mem, sizes, {t: (lambda t=t: decoded(t, used_t[t])) for t in mem.rows},
                                  held=rec_t), avg
        elif var == "random":
            m, nbytes = with_keys(mem, sizes, {t: (lambda t=t: decoded(t, rand_t[t])) for t in mem.rows}), avg
        else:
            raise SystemExit(f"unknown variant {var}")
        report(var, m, nbytes)
        del m
        torch.cuda.empty_cache()
    print("\nsummary (mean over the specialties, philosophy excluded):")
    for var, r in results.items():
        print(f"  {var:7} {r['bytes']:6.0f} B/episode, {r['mb']:7,.0f} MB: {r['mean']:.2%}; English "
              f"{r['gain']['english']:+.2%}, code {r['gain']['code']:+.2%}"
              + (f"; never-recalled released: {r['mean_held']:.2%}" if var == "use" else ""))


if __name__ == "__main__":
    main()
