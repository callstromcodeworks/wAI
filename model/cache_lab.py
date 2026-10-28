"""Inference-time cache experiments on a trained checkpoint, scored in bits per byte.

    python model/cache_lab.py runs/<run>/ckpt.pt --trim --hybrid

1. KV-cache trimming ("conversation compression"). Text is fed in turns of --turn tokens.
   After each turn the KV cache is compressed back to a fixed budget, and the next turn
   attends to the compressed cache plus itself. Policies:
     window  keep the most recent tokens (plain truncation)
     sinks   keep the first 4 tokens (attention sinks, StreamingLLM) + the most recent
     snap    keep sinks + a recent window + the tokens the turn's last queries attended to
             most, chosen per KV head (H2O / SnapKV style)
     pool    keep sinks + a recent window and average older K/V into fewer slots. A merged
             slot gets a log(count) attention bias, so it stands in for all its tokens
   The cache budget is the memory saving: budget 256 of a 1024-token context is 4x less KV.

2. Neural cache (Grave et al. 2017), mixed word by word. p_cache(w) puts mass on earlier
   positions whose next token was w, weighted by the cosine similarity of their final hidden
   states to the current one. The prediction is (1 - g) p_model + g p_cache, with the gate g
   chosen per token from features that don't see the answer (how similar the best match is,
   how concentrated the cache distribution is). The gate is fit on one half of the
   validation text and scored on the other half.

3. Hybrid (--hybrid): trim the KV cache, but keep the neural cache over the full history.
   The neural cache stores one vector per token (d_model values, or --nc_dim after a random
   projection) against the 2 x layers x kv_heads x head_dim values of K/V, so it can bring
   back long-range copying that trimming loses, at a small fraction of the memory.
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from generate import load_model
from prepare_data import load_shard
from train import token_byte_lengths

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ("fineweb-edu", "code")
N_SINK = 4
THETAS = (5.0, 10.0, 20.0, 40.0, 80.0)


def val_rows(vocab, source, T, n_bytes, token_bytes):
    val, _ = load_shard(os.path.join(HERE, "data", vocab, source, "val_000000.bin"))
    bpt = token_bytes.cpu().numpy()[val[:1_000_000]].mean()
    n = min(int(math.ceil(n_bytes / (bpt * T))), (len(val) - 1) // T)
    rows = np.stack([val[r * T:r * T + T + 1] for r in range(n)]).astype(np.int64)
    return torch.from_numpy(rows)


# ----------------------------------------------------------------------------- KV trimming

class Past:
    """Compressed cache for one K/V layer: keys/values [B, KV, S, D] plus a per-slot log-count bias."""

    def __init__(self, k, v, bias):
        self.k, self.v, self.bias = k, v, bias


class TrimCache:
    """KV cache for turn-by-turn evaluation. Each turn attends to the compressed past plus
    itself (causally); extend() records the turn's K/V and importance scores (attention mass
    from the turn's last `window` queries) so compress() can trim the cache afterwards."""

    def __init__(self, n_kv_layers, window=32):
        self.pasts = [None] * n_kv_layers
        self.new = [None] * n_kv_layers
        self.scores = [None] * n_kv_layers
        self.window = window

    def extend(self, i, q, k, v):
        B, KV, t, D = k.shape
        H = q.size(1)
        causal = torch.ones(t, t, dtype=torch.bool, device=k.device).tril()
        cur = torch.zeros(t, t, device=k.device).masked_fill(~causal, float("-inf")).expand(B, KV, t, t)
        p = self.pasts[i]
        if p is None:
            K, V, mask = k, v, cur
        else:
            K, V = torch.cat([p.k, k], 2), torch.cat([p.v, v], 2)
            mask = torch.cat([p.bias[:, :, None, :].expand(B, KV, t, p.k.size(2)), cur], dim=-1)
        mask = mask.repeat_interleave(H // KV, dim=1).to(q.dtype)  # one bias row per q head
        w = min(self.window, t)
        logits = q[:, :, -w:].float() @ K.repeat_interleave(H // KV, 1).float().transpose(-1, -2)
        probs = torch.softmax(logits * self.attn_scale + mask[:, :, -w:].float(), dim=-1)
        self.scores[i] = probs.view(B, KV, H // KV, w, -1).sum(dim=(2, 3))
        self.new[i] = (k, v)
        return K, V, mask, False

    def compress(self, policy, budget):
        self.pasts = [compress(policy, p, k, v, s, budget)
                      for p, (k, v), s in zip(self.pasts, self.new, self.scores)]


def gather_slots(t, idx):
    """t: [B, KV, S, ...], idx: [B, KV, n] -> [B, KV, n, ...]."""
    if t.dim() == 3:
        return t.gather(2, idx)
    return t.gather(2, idx[..., None].expand(-1, -1, -1, t.size(-1)))


def compress(policy, past, k, v, score, budget):
    """Append this turn's k/v to the cache, then compress it back to `budget` slots (None = keep all)."""
    B, KV, t, D = k.shape
    zeros = torch.zeros(B, KV, t, device=k.device)
    if past is None:
        K, V, bias = k, v, zeros
    else:
        K, V, bias = torch.cat([past.k, k], 2), torch.cat([past.v, v], 2), torch.cat([past.bias, zeros], 2)
    S = K.size(2)
    if budget is None or S <= budget:
        return Past(K, V, bias)
    ar = torch.arange(S, device=k.device)
    if policy == "window":
        idx = ar[S - budget:]
    elif policy == "sinks":
        idx = torch.cat([ar[:N_SINK], ar[S - (budget - N_SINK):]])
    elif policy == "snap":
        recent = budget // 4
        keep = torch.zeros(B, KV, S, dtype=torch.bool, device=k.device)
        keep[..., :N_SINK] = True
        keep[..., S - recent:] = True
        s = score.masked_fill(keep, float("inf"))  # forced slots sort first
        idx = s.topk(budget, dim=-1).indices.sort(dim=-1).values
        return Past(gather_slots(K, idx), gather_slots(V, idx), gather_slots(bias, idx))
    elif policy == "pool":
        recent = budget // 2
        n_old, m = S - N_SINK - recent, budget - N_SINK - recent
        group = torch.arange(n_old, device=k.device) * m // n_old  # contiguous groups of old slots
        w = bias[..., N_SINK:N_SINK + n_old].exp()  # how many tokens each old slot already stands for
        cnt = torch.zeros(B, KV, m, device=k.device).index_add_(2, group, w)

        def pool(x):
            num = torch.zeros(B, KV, m, D, device=k.device).index_add_(
                2, group, x[:, :, N_SINK:N_SINK + n_old].float() * w[..., None])
            return (num / cnt[..., None]).to(x.dtype)
        return Past(torch.cat([K[:, :, :N_SINK], pool(K), K[:, :, S - recent:]], 2),
                    torch.cat([V[:, :, :N_SINK], pool(V), V[:, :, S - recent:]], 2),
                    torch.cat([bias[..., :N_SINK], cnt.log(), bias[..., S - recent:]], 2))
    else:
        raise ValueError(policy)
    idx = idx.view(1, 1, -1).expand(B, KV, -1)
    return Past(gather_slots(K, idx), gather_slots(V, idx), gather_slots(bias, idx))


@torch.no_grad()
def run_turns(model, rows, policy, budget, turn, batch=8, window=32):
    """Yield (final hidden [B, T, d], target log-probs [B, T], targets [B, T]) per batch of rows,
    with the KV cache compressed to `budget` after every turn (policy "full" keeps everything)."""
    T = rows.size(1) - 1
    for b0 in range(0, rows.size(0), batch):
        r = rows[b0:b0 + batch].cuda()
        x, y = r[:, :-1], r[:, 1:]
        cache = TrimCache(model.cfg.n_kv_layers, window)
        cache.attn_scale = model.cfg.attn_scale
        hs, lps = [], []
        with torch.autocast("cuda", torch.bfloat16):
            for s in range(0, T, turn):
                h = model(x[:, s:s + turn], cache, s)
                hs.append(h)
                lps.append(-model.token_losses(h, y[:, s:s + turn]))
                cache.compress(policy, None if policy == "full" else budget)
        yield torch.cat(hs, 1), torch.cat(lps, 1), y


# ----------------------------------------------------------------------------- neural cache

def nc_features(h, logp, y, proj=None):
    """Neural-cache features for one batch: target log-prob under the model, best-match cosine,
    cache top-token mass, and the cache probability of the target for each theta."""
    hn = F.normalize(h.float() if proj is None else h.float() @ proj, dim=-1)
    T = y.size(1)
    sim = hn @ hn.transpose(1, 2)  # [B, T, T]; earlier position i's entry is (h_i, y_i)
    past = torch.ones(T, T, dtype=torch.bool, device=y.device).tril(-1)
    sim = sim.masked_fill(~past, float("-inf"))
    match = (y[:, :, None] == y[:, None, :]) & past  # earlier position's next token == my target
    pc = torch.stack([(torch.softmax(sim * th, -1).nan_to_num(0.0) * match).sum(-1) for th in THETAS], -1)
    p = torch.softmax(sim * 20.0, dim=-1).nan_to_num(0.0)
    top_tok = y.gather(1, p.argmax(-1))
    top = (p * ((y[:, None, :] == top_tok[:, :, None]) & past)).sum(-1)
    return dict(logp=logp.flatten(), maxcos=sim.amax(-1).clamp(min=-1).flatten(), top=top.flatten(),
                pcache=pc.flatten(0, 1))


def collect(model, rows, token_bytes, policy="full", budget=None, turn=128, nc=True, nc_dim=0):
    """bpb of the plain model under a KV policy, plus neural-cache features over the full history."""
    proj = None
    if nc_dim:
        g = torch.Generator(device="cuda").manual_seed(0)
        proj = torch.randn(model.cfg.d_model, nc_dim, device="cuda", generator=g) / math.sqrt(nc_dim)
    feats, loss_sum, byte_sum = [], 0.0, 0
    for h, logp, y in run_turns(model, rows, policy, budget, turn):
        loss_sum -= logp.sum().item()
        byte_sum += token_bytes[y].sum().item()
        if nc:
            f = nc_features(h, logp, y, proj)
            f["bytes"] = token_bytes[y].flatten()
            feats.append(f)
    bpb = loss_sum / math.log(2) / byte_sum
    return bpb, ({k: torch.cat([f[k] for f in feats]) for k in feats[0]} if nc else None)


def neural_cache_eval(feats):
    """Fit fixed and adaptive mixing on the first half of positions; report bpb on the second half."""
    n = feats["logp"].size(0)
    fit, test = slice(0, n // 2), slice(n // 2, n)
    p_model = feats["logp"].exp()
    nbytes = feats["bytes"].float()

    def bpb(p, sl):
        return (-(p[sl].clamp(min=1e-30).log()).sum() / math.log(2) / nbytes[sl].sum()).item()

    best = None
    for j in range(len(THETAS)):
        for lam in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3):
            v = bpb((1 - lam) * p_model + lam * feats["pcache"][:, j], fit)
            if best is None or v < best[0]:
                best = (v, j, lam)
    _, j, lam = best
    pc = feats["pcache"][:, j]
    # Adaptive gate: g = sigmoid(a * maxcos + b * top + c), fit by gradient descent on the fit half.
    params = torch.tensor([0.0, 0.0, math.log(lam / (1 - lam))], device=p_model.device, requires_grad=True)
    opt = torch.optim.Adam([params], lr=0.05)
    f1, f2 = feats["maxcos"], feats["top"]
    for _ in range(300):
        g = torch.sigmoid(params[0] * f1[fit] + params[1] * f2[fit] + params[2])
        loss = -((1 - g) * p_model[fit] + g * pc[fit]).clamp(min=1e-30).log().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        g = torch.sigmoid(params[0] * f1 + params[1] * f2 + params[2])
        return dict(base_bpb=bpb(p_model, test), fixed_bpb=bpb((1 - lam) * p_model + lam * pc, test),
                    theta=THETAS[j], lam=lam, adaptive_bpb=bpb((1 - g) * p_model + g * pc, test),
                    gate=[round(v, 3) for v in params.tolist()], mean_gate=g[test].mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--trim", action="store_true", help="compare KV trimming policies")
    ap.add_argument("--hybrid", action="store_true", help="neural cache on top of full and trimmed KV")
    ap.add_argument("--eval_bytes", type=int, default=1_000_000, help="text per source")
    ap.add_argument("--turn", type=int, default=128)
    ap.add_argument("--budgets", default="128,256,512")
    ap.add_argument("--policies", default="window,sinks,snap,pool")
    ap.add_argument("--hybrid_policy", default="snap")
    ap.add_argument("--nc_dims", default="0,128", help="neural-cache vector sizes to try (0 = d_model)")
    ap.add_argument("--save_gate", type=int, default=-1,
                    help="fit one neural-cache gate on both sources at this nc_dim (0 = d_model) and save it "
                         "next to the checkpoint as neural_cache.json, for fastdecode")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    model, ckpt = load_model(a.ckpt)
    vocab, c = ckpt["args"]["vocab"], model.cfg
    token_bytes = torch.from_numpy(token_byte_lengths(c.n_ranks, c.vocab_size)).cuda()
    kv_tok = model.kv_bytes_per_token()
    if c.mla_rank:  # the cache holds latents, not keys, so attention-score selection doesn't apply
        a.policies = ",".join(p for p in a.policies.split(",") if p != "snap")
        a.hybrid_policy = "window" if a.hybrid_policy == "snap" else a.hybrid_policy
    results = dict(ckpt=a.ckpt, vocab=vocab, kv_bytes_per_token=kv_tok, d_model=c.d_model)
    for src in SOURCES:
        rows = val_rows(vocab, src, c.seq_len, a.eval_bytes, token_bytes)
        res = results.setdefault(src, {})
        full, full_feats = collect(model, rows, token_bytes, "full", nc=a.hybrid)
        res["full"] = full
        print(f"{src}: full KV cache {full:.4f} bpb ({rows.size(0)} rows of {c.seq_len} tokens, turns of {a.turn})",
              flush=True)
        if a.trim:
            for budget in map(int, a.budgets.split(",")):
                for pol in a.policies.split(","):
                    v, _ = collect(model, rows, token_bytes, pol, budget, a.turn, nc=False)
                    res[f"{pol}@{budget}"] = v
                    print(f"  {pol:6s} KV budget {budget:4d}: {v:.4f} bpb ({(v / full - 1) * 100:+.2f}%)", flush=True)
        if a.hybrid:
            for nc_dim in map(int, a.nc_dims.split(",")):
                nc_bytes = (nc_dim or c.d_model) * 2
                feats = full_feats if nc_dim == 0 else collect(model, rows, token_bytes, "full", nc_dim=nc_dim)[1]
                r = neural_cache_eval(feats)
                res[f"full+nc{nc_dim or c.d_model}"] = r
                print(f"  full KV + neural cache (d={nc_dim or c.d_model}, {nc_bytes} B/token): "
                      f"{r['base_bpb']:.4f} -> fixed {r['fixed_bpb']:.4f}, adaptive {r['adaptive_bpb']:.4f} "
                      f"({(r['adaptive_bpb'] / r['base_bpb'] - 1) * 100:+.2f}%), mean gate {r['mean_gate']:.3f}",
                      flush=True)
                for budget in map(int, a.budgets.split(",")):
                    _, tf = collect(model, rows, token_bytes, a.hybrid_policy, budget, a.turn, nc_dim=nc_dim)
                    r2 = neural_cache_eval(tf)
                    res[f"{a.hybrid_policy}@{budget}+nc{nc_dim or c.d_model}"] = r2
                    mem = (budget * kv_tok + c.seq_len * nc_bytes) / (c.seq_len * kv_tok)
                    print(f"  {a.hybrid_policy}@{budget} + neural cache: {r2['base_bpb']:.4f} -> {r2['adaptive_bpb']:.4f} "
                          f"vs full KV {r['base_bpb']:.4f} ({(r2['adaptive_bpb'] / r['base_bpb'] - 1) * 100:+.2f}%) "
                          f"at {mem:.1%} of full-KV memory @ {c.seq_len} tokens", flush=True)
    if a.save_gate >= 0:
        feats = [collect(model, val_rows(vocab, src, c.seq_len, a.eval_bytes, token_bytes), token_bytes, "full",
                         nc_dim=a.save_gate)[1] for src in SOURCES]
        # interleave the sources so the fit and test halves both contain English and code
        n = min(f["logp"].size(0) for f in feats) // 2
        both = {k: torch.cat([feats[0][k][:n], feats[1][k][:n], feats[0][k][n:2 * n], feats[1][k][n:2 * n]])
                for k in feats[0]}
        r = neural_cache_eval(both)
        gate = dict(nc_dim=a.save_gate, theta=r["theta"], gate=r["gate"], base_bpb=r["base_bpb"],
                    adaptive_bpb=r["adaptive_bpb"])
        path = os.path.join(os.path.dirname(a.ckpt), "neural_cache.json")
        json.dump(gate, open(path, "w"), indent=1)
        print(f"neural cache gate (both sources, held-out half): {r['base_bpb']:.4f} -> {r['adaptive_bpb']:.4f} bpb; "
              f"saved {path}")
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
