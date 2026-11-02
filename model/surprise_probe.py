"""Which signal tells new material from familiar text, without labels? (wai_struct, label-free arrival, step 0)

    python model/surprise_probe.py runs/sweep_b32k_sparseB32c2_t1b [--per 300]

The base model reads windows of each specialty (data/ec32k/domains/<n>/train.npy, what a day's stream would show)
and of familiar text (held-out fineweb-edu and code, 60/40). For each window it computes candidate surprise signals:
- loss: bits per byte (prediction error);
- dissonance@d: KL(final prediction || prediction read out after block d), the model's internal conflict;
- maha@d: Mahalanobis distance of the window's mean hidden state after block d from familiar text (a reference set
  of training-shard windows), i.e. "this doesn't look like anything I usually read";
- knn@d: cosine distance to the 10 nearest reference windows at block d (a hippocampus-style novelty check);
- areas: Mahalanobis distance of the routing pattern (share of tokens each bank module fired on) from familiar
  text: "an unusual set of areas lit up".
Each signal is scored by AUC per specialty (1 = every specialty window more surprising than every familiar one,
0.5 = chance). Labels are used for scoring only.

Result on the base (Nov 2): none of these works. The English specialties (medical, law, philosophy, music, space)
score *below* familiar text on every one (AUC 0.2-0.5): one window of them is no stranger than familiar text. What
makes them new is that they keep coming up. So a second table scores
- today@d: a simulated day (a specialty's windows mixed 50/50 with familiar ones). For each window, the share of its
  k nearest neighbours (state after block d) that come from today's stream rather than from the familiar reference:
  "over-represented today compared with usual". Familiar windows held out of the day give the threshold (their 0.9
  quantile), and precision/recall are reported at it."""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.nn.attention import sdpa_kernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forgetting import HERE, T, load_model
from prepare_data import load_shard
from tokenizer import VOCABS
from train import SDPA_BACKENDS, token_byte_lengths

DOMAINS = "haskell,medical,lua,philosophy,law,fortran,logic,music,rust,space"


def auc(pos, neg):
    """P(a positive scores above a negative), ties counted half (Mann-Whitney)."""
    allv = np.concatenate([pos, neg])
    ranks = np.empty(len(allv))
    order = np.argsort(allv, kind="mergesort")
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv[np.diff(np.sort(allv), prepend=np.nan) == 0]):  # average ranks of ties
        ranks[allv == v] = ranks[allv == v].mean()
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


@torch.no_grad()
def features(model, rows, token_bytes, depths, batch=4):
    """Per window: bpb, dissonance at each depth, mean hidden state at each depth, routing pattern (share of tokens
    each bank module fired on)."""
    outs, fired = {}, []
    route = model.route  # a method: wrap it to see which modules fire

    def seen_route(*args, **kw):
        res = route(*args, **kw)
        fired.append(res[2].float().mean(1).cpu())
        return res
    model.route = seen_route
    hooks = [model.blocks[d].register_forward_hook(lambda m, i, o, d=d: outs.__setitem__(d, o)) for d in depths]
    model.eval()
    bpb, dis, hid = [], [], []
    for i in range(0, len(rows), batch):
        r = torch.from_numpy(rows[i:i + batch]).cuda()
        x, y = r[:, :-1], r[:, 1:]
        with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPA_BACKENDS, set_priority=True):
            h = model(x)
            late = torch.log_softmax(model.logits(h).float(), -1)
            tl = model.token_losses(h, y).float()
            bpb.append((tl.sum(1) / token_bytes[y].sum(1) / np.log(2)).cpu())
            dd, hh = [], []
            for d in depths:
                o = outs[d]
                early = torch.log_softmax(model.logits(nn.functional.rms_norm(o, (o.size(-1),))).float(), -1)
                dd.append((late.exp() * (late - early)).nan_to_num(0.0).sum(-1).mean(1).cpu())
                hh.append(nn.functional.rms_norm(o.float(), (o.size(-1),)).mean(1).cpu())
        dis.append(torch.stack(dd, 1))
        hid.append(torch.stack(hh, 1))
    for hk in hooks:
        hk.remove()
    del model.route
    return dict(bpb=torch.cat(bpb).numpy(), dis=torch.cat(dis).numpy(), hid=torch.cat(hid).numpy(),
                areas=torch.cat(fired).numpy())


class Gaussian:
    """Familiar text as a Gaussian (shrunk covariance); distance = Mahalanobis."""
    def __init__(self, ref, shrink=0.1):
        self.mu = ref.mean(0)
        c = np.cov(ref - self.mu, rowvar=False)
        c = (1 - shrink) * c + shrink * np.trace(c) / len(c) * np.eye(len(c))
        self.prec = np.linalg.inv(c)

    def __call__(self, x):
        z = x - self.mu
        return np.sqrt(np.einsum("nd,de,ne->n", z, self.prec, z))


def knn_dist(ref, x, k=10):
    a = torch.nn.functional.normalize(torch.from_numpy(ref).cuda(), dim=-1)
    b = torch.nn.functional.normalize(torch.from_numpy(x).cuda(), dim=-1)
    return (1 - (b @ a.T).topk(k, dim=-1).values.mean(-1)).cpu().numpy()


def today_share(day, ref, extra, k):
    """For each day window: the share of its k nearest neighbours (cosine, itself excluded) among day + ref that are
    from the day. Same for the extra windows (not in the day): the null for familiar text."""
    day, ref, extra = (torch.nn.functional.normalize(torch.as_tensor(v).float().cuda(), dim=-1)
                       for v in (day, ref, extra))
    pool = torch.cat([day, ref])
    from_day = torch.cat([torch.ones(len(day)), torch.zeros(len(ref))]).cuda()

    def share(x, self_rows):
        out = []
        for i in range(0, len(x), 1024):
            sims = x[i:i + 1024] @ pool.T
            if self_rows:
                j = torch.arange(len(sims), device=sims.device)
                sims[j, i + j] = -2.0
            out.append(from_day[sims.topk(k, dim=-1).indices].mean(-1))
        return torch.cat(out).cpu().numpy()
    return share(day, True), share(extra, False)


def windows(ids, n, rng, lo=0):
    return np.stack([ids[s:s + T + 1] for s in rng.integers(lo, len(ids) - T - 1, n)]).astype(np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--per", type=int, default=300, help="windows per specialty")
    ap.add_argument("--familiar", type=int, default=1000, help="familiar test windows")
    ap.add_argument("--ref", type=int, default=2000, help="familiar reference windows (fit the novelty models)")
    ap.add_argument("--domains", default=DOMAINS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="", help="also save the per-window features (.npz)")
    ap.add_argument("--load", default="", help="read the features from a --save file instead of the model")
    ap.add_argument("--k", type=int, default=30, help="today@d: neighbours")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    model, ck = load_model(a.ckpt if os.path.isabs(a.ckpt) else os.path.join(HERE, a.ckpt))
    vocab = ck["args"]["vocab"]
    token_bytes = torch.from_numpy(token_byte_lengths(VOCABS[vocab], model.cfg.vocab_size)).cuda()
    depths = list(range(1, len(model.blocks) - 1))
    shard = lambda src, split: load_shard(os.path.join(HERE, "data", vocab, src, f"{split}_000000.bin"))[0]

    def familiar(n, split, lo=0):
        ne = int(round(n * 0.6))
        return np.concatenate([windows(shard("fineweb-edu", split), ne, rng, lo),
                               windows(shard("code", split), n - ne, rng, lo)])

    ref_rows = familiar(a.ref, "train")
    fam_rows = familiar(a.familiar, "val", lo=400_000)  # past the text forgetting.py scores ("keep")
    names = a.domains.split(",")
    spec_rows = {n: windows(np.load(os.path.join(HERE, "data", vocab, "domains", n, "train.npy")), a.per, rng)
                 for n in names}

    if a.load:
        z = np.load(a.load)
        feat = {k: {f: z[f"{k}.{f}"] for f in ("bpb", "dis", "hid", "areas")} for k in ["ref", "fam"] + names}
    else:
        feat = {k: features(model, r, token_bytes, depths) for k, r in
                [("ref", ref_rows), ("fam", fam_rows)] + list(spec_rows.items())}
    if a.save:
        np.savez(a.save, **{f"{k}.{f}": v for k, d in feat.items() for f, v in d.items()})

    signals = {"loss": lambda k: feat[k]["bpb"]}
    for j, d in enumerate(depths):
        signals[f"dissonance@{d}"] = lambda k, j=j: feat[k]["dis"][:, j]
    signals["dissonance mean"] = lambda k: feat[k]["dis"].mean(1)
    for j, d in enumerate(depths):
        g = Gaussian(feat["ref"]["hid"][:, j])
        signals[f"maha@{d}"] = lambda k, g=g, j=j: g(feat[k]["hid"][:, j])
    for j, d in enumerate(depths):
        signals[f"knn@{d}"] = lambda k, j=j: knn_dist(feat["ref"]["hid"][:, j], feat[k]["hid"][:, j])
    ga = Gaussian(feat["ref"]["areas"])
    signals["areas"] = lambda k: ga(feat[k]["areas"])

    print(f"AUC: specialty windows more surprising than familiar ones ({a.per} per specialty, {a.familiar} familiar)")
    print(f"{'signal':16}" + "".join(f" {n[:7]:>7}" for n in names) + "    mean   min")
    best = []
    for s, f in signals.items():
        neg = f("fam")
        row = [auc(f(n), neg) for n in names]
        best.append((np.mean(row), s))
        print(f"{s:16}" + "".join(f" {v:7.3f}" for v in row) + f"   {np.mean(row):.3f} {min(row):.3f}")
    print("best by mean AUC: " + ", ".join(f"{s} {m:.3f}" for m, s in sorted(best, reverse=True)[:5]))
    print("mean bpb: familiar " + f"{feat['fam']['bpb'].mean():.3f} | " +
          " | ".join(f"{n} {feat[n]['bpb'].mean():.3f}" for n in names))

    half = min(a.per, len(feat["fam"]["bpb"]) // 2)
    print(f"\ntoday@d (k {a.k}): a day = {a.per} specialty + {half} familiar windows; AUC within the day, then "
          f"precision / recall at the 0.9 quantile of {len(feat['fam']['bpb']) - half} familiar windows outside it")
    print(f"{'signal':16}" + "".join(f" {n[:7]:>7}" for n in names) + "    mean   min | precision recall")
    for j, d in enumerate(depths):
        row, prec, rec = [], [], []
        for n in names:
            day_, null = today_share(np.concatenate([feat[n]["hid"][:, j], feat["fam"]["hid"][:half, j]]),
                                     feat["ref"]["hid"][:, j], feat["fam"]["hid"][half:, j], a.k)
            new = np.r_[np.ones(len(feat[n]["hid"]), bool), np.zeros(half, bool)]
            row.append(auc(day_[new], day_[~new]))
            keep = day_ > np.quantile(null, 0.9)
            prec.append((keep & new).sum() / max(1, keep.sum()))
            rec.append((keep & new).sum() / new.sum())
        print(f"{'today@' + str(d):16}" + "".join(f" {v:7.3f}" for v in row) +
              f"   {np.mean(row):.3f} {min(row):.3f} | {np.mean(prec):.3f} (min {min(prec):.3f}) "
              f"{np.mean(rec):.3f} (min {min(rec):.3f})")


if __name__ == "__main__":
    main()
