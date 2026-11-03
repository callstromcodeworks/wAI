"""Episodic memory for continued learning (the hippocampus beside the neurogenesis cortex in forgetting.py).

Episode = (the model's final hidden state at a position, the token that came next), as in memory_lm.py. New text
enters memory by reading it once; recall blends what came next in the k nearest stored moments with the model's own
prediction. Keys live in pinned CPU memory, one block per text (a 12 GB card can't hold millions of episodes beside
training), and stream to the GPU chunk by chunk, each chunk once per scoring pass.

Used from forgetting.py continual --memory_tokens N:
- arrive: when a specialty arrives its training text is read in (N tokens), before any training. Only the new text
  is encoded: the model that keyed the rest hasn't changed (checked; if it has, everything is re-keyed);
- reencode: after the stage's wake and sleep the model has changed, so stored keys are recomputed from the stored
  token windows (recall compares like with like);
- consolidate (--memory_prune P): episodes the model now predicts with p(next) >= P are released: what the cortex
  has learned no longer needs the hippocampus (systems consolidation). Released episodes stay stored, so each score
  compares the full memory with the consolidated one;
- score: bits per byte without memory, with all episodes, and with only those still held. The blend is chosen on
  the first half of each text and reported on the second: a fixed lam, lam leaning on memory when the model is unsure
  (entropy), or when it is conflicted (dissonance, the judge's target). When nothing a text's previous search
  covered has changed (same model, same keys and held flags: a new specialty has just arrived), that search is
  reused and only the newly arrived episodes are searched: the same neighbours as a full search, far cheaper."""
import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel

from train import SDPA_BACKENDS

T = 1024


@torch.no_grad()
def encode(model, rows, batch=16, dissonance_depth=None):
    """Final hidden states [n, d] (fp16), next tokens [n] and log p_model(next) [n] for token windows [r, T + 1].
    With dissonance_depth: also each position's dissonance [n], KL(final prediction || prediction read out after
    that block), the model's conflict between first impression and considered view."""
    model.eval()
    hs, ys, lps, dis = [], [], [], []
    early = {}
    hook = (model.blocks[dissonance_depth].register_forward_hook(lambda m, i, o: early.__setitem__("h", o))
            if dissonance_depth is not None else None)
    for i in range(0, len(rows), batch):
        r = torch.from_numpy(rows[i:i + batch]).cuda()
        with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPA_BACKENDS, set_priority=True):
            h = model(r[:, :-1])
        h, y = h.reshape(-1, h.size(-1)), r[:, 1:].reshape(-1)
        e = early["h"].reshape(-1, h.size(-1)) if hook else None
        for c in range(0, len(h), 8192):
            late = torch.log_softmax(model.logits(h[c:c + 8192].float()), -1)
            lps.append(late.gather(1, y[c:c + 8192, None])[:, 0])
            if hook:
                first = torch.log_softmax(model.logits(F.rms_norm(e[c:c + 8192].float(), (h.size(-1),))), -1)
                dis.append((late.exp() * (late - first)).nan_to_num(0.0).sum(-1))
        hs.append(h.half())
        ys.append(y)
    if hook:
        hook.remove()
    model.train()
    out = torch.cat(hs), torch.cat(ys), torch.cat(lps)
    return out + (torch.cat(dis),) if hook else out


class EpisodicMemory:
    def __init__(self, model, token_bytes, k=32, chunk=1 << 19, qb=512):
        self.model, self.tb, self.k, self.chunk, self.qb = model, token_bytes, k, chunk, qb
        self.dis_depth = getattr(model.cfg, "trunk_layers", 3) - 1  # the routing depth (as forgetting.py's sleep tags)
        self.rows = {}  # tag -> token windows [r, T + 1] (uint16): the raw episodes, kept to re-encode
        self.keep = {}  # tag -> bool mask over that tag's positions [r * T]: still held (not consolidated)
        self.sel = {}  # tag -> bool mask over positions [r * T]: stored at all (store(); absent = every position)
        self.blocks = []  # pinned fp16 keys, one block per tag in self.rows order (arrivals never copy old keys)
        self.key_sq = self.vals = self.held = None  # on the GPU, over all episodes in the same order
        self.keyed_by = None  # fingerprint of the model that made the current keys
        self.version = 0  # bumped whenever keys or held flags may have changed (each re-keying)
        self.cache = {}  # text -> its last search: version, model fingerprint, episodes covered, neighbours found
        self.searched = 0  # episodes searched, summed over passes (the checks use it to see the cache at work)

    def __len__(self):
        return 0 if self.vals is None else len(self.vals)

    def n_held(self):
        return 0 if self.vals is None else int(self.held.sum().item())

    @torch.no_grad()
    def fingerprint(self):
        """Sum and sum of squares of every weight (float64), and their count: reuse only while this is unchanged."""
        s = torch.zeros(2, dtype=torch.float64, device="cuda")
        n = 0
        for p in self.model.parameters():
            s[0] += p.sum(dtype=torch.float64)
            s[1] += p.float().square().sum(dtype=torch.float64)
            n += p.numel()
        return tuple(s.tolist()) + (n,)

    def _add(self, tag, h, y):
        """Append one tag's keys (a new pinned block), next tokens, squared key norms and held flags."""
        blk = torch.empty(h.shape, dtype=torch.float16, pin_memory=True)
        blk.copy_(h)
        self.blocks.append(blk)
        sq = torch.cat([h[c:c + 65536].float().square().sum(-1) for c in range(0, len(h), 65536)])
        cat = lambda old, new: new if old is None else torch.cat([old, new])
        self.key_sq, self.vals, self.held = cat(self.key_sq, sq), cat(self.vals, y), cat(self.held, self.keep[tag].cuda())

    def arrive(self, tag, ids, n_tokens):
        n_rows = min(int(n_tokens) // T, (len(ids) - 1) // T)
        self.rows[tag] = np.stack([ids[r * T: r * T + T + 1] for r in range(n_rows)]).astype(np.uint16)
        self.keep[tag] = torch.ones(n_rows * T, dtype=torch.bool)
        fp = self.fingerprint()
        if self.blocks and fp != self.keyed_by:  # the model changed since the keys were made: re-key everything
            self.reencode()
            return
        h, y, _ = encode(self.model, self.rows[tag].astype(np.int64))
        self._add(tag, h, y)
        self.keyed_by = fp

    @torch.no_grad()
    def store(self, tag, rows, max_p=None, cap=None, chunk_rows=128):
        """Selective arrival (wai_struct): read windows rows [r, T + 1] in order and store only the episodes the model
        predicts with p(next) < max_p (every one with max_p None), until cap are stored; rows after the last stored
        episode are dropped. keep (and so consolidation) then runs over the stored episodes only. Returns how many
        episodes were stored and how many tokens were read to find them."""
        hs, ys, sels, n = [], [], [], 0
        for c in range(0, len(rows), chunk_rows):
            if cap is not None and n >= cap:
                break
            h, y, lp = encode(self.model, rows[c:c + chunk_rows].astype(np.int64))
            sel = torch.ones_like(y, dtype=torch.bool) if max_p is None else lp < math.log(max_p)
            if cap is not None and n + int(sel.sum()) > cap:
                sel[int(torch.nonzero(sel)[cap - n - 1]) + 1:] = False
            hs.append(h[sel])
            ys.append(y[sel])
            sels.append(sel.cpu())
            n += int(sel.sum())
            del h, y, lp
        sel = torch.cat(sels) if sels else torch.zeros(0, dtype=torch.bool)
        n_rows = int(torch.nonzero(sel)[-1]) // T + 1 if bool(sel.any()) else 0
        self.rows[tag] = rows[:n_rows].astype(np.uint16)
        self.sel[tag] = sel[:n_rows * T]
        self.keep[tag] = torch.ones(n, dtype=torch.bool)
        self._add(tag, torch.cat(hs), torch.cat(ys))
        self.keyed_by = self.fingerprint()
        return n, n_rows * T

    def reencode(self, prune=None):
        """Recompute every key with the current model; with prune=P mark episodes it now predicts with p >= P as
        released. Returns {tag: share of that tag's episodes still held}."""
        self.blocks, self.key_sq, self.vals, self.held = [], None, None, None  # free the old keys first (CPU RAM)
        self.version += 1
        for tag, rows in self.rows.items():
            h, y, lp = encode(self.model, rows.astype(np.int64))
            if tag in self.sel:  # stored selectively: re-key only what was stored
                m = self.sel[tag].cuda()
                h, y, lp = h[m], y[m], lp[m]
            if prune is not None:
                self.keep[tag] &= (lp < math.log(prune)).cpu()
            self._add(tag, h, y)
            del h, y, lp
        self.keyed_by = self.fingerprint()
        return {tag: round(self.keep[tag].float().mean().item(), 4) for tag in self.rows}

    @torch.no_grad()
    def search(self, q, held_only=False, start=0, init=None):
        """k nearest stored states for each query among episodes [start:], merged with init = (dist, idx) of the
        k nearest among [:start] from an earlier search. Distances are squared L2 less |q|^2: [n, k], idx [n, k].
        held_only: released (consolidated) episodes are out of reach."""
        k = min(self.k, self.n_held() if held_only else len(self))
        key_sq = self.key_sq.masked_fill(~self.held, float("inf")) if held_only else self.key_sq
        if init is not None and init[0].shape == (len(q), k):
            best_d, best_i = init[0].cuda(), init[1].cuda().long()
        else:
            best_d = torch.full((len(q), k), float("inf"), device="cuda")
            best_i = torch.zeros((len(q), k), dtype=torch.long, device="cuda")
            start = 0
        at = 0
        for blk in self.blocks:
            for c in range(max(start - at, 0), len(blk), self.chunk):
                kc, g = blk[c:c + self.chunk].cuda(non_blocking=True), at + c
                sq = key_sq[g:g + len(kc)]
                for s in range(0, len(q), self.qb):  # distances [qb, chunk]: 512 x 512k fp32 = 1 GiB
                    e = s + self.qb
                    d = sq[None, :] - 2 * (q[s:e] @ kc.t()).float()
                    dd, ii = torch.topk(d, min(k, d.size(1)), dim=1, largest=False)
                    cd, ci = torch.cat([best_d[s:e], dd], 1), torch.cat([best_i[s:e], ii + g], 1)
                    best_d[s:e], sel = torch.topk(cd, k, dim=1, largest=False)
                    best_i[s:e] = ci.gather(1, sel)
            at += len(blk)
        self.searched += len(self) - start
        return best_d, best_i

    def recall(self, q, held_only=False):
        """k nearest stored states (squared L2) for each query: distances [n, k] and indices [n, k]."""
        d, i = self.search(q, held_only)
        return d + q.float().square().sum(-1, keepdim=True), i

    @torch.no_grad()
    def score(self, ids, n_tokens=100_000):
        """Bits per byte on held-out text (second half reported): dict(model, memory, held, blend, blend_held,
        by_gate). memory/held: with all episodes / only those still held, each with its own chosen blend; by_gate:
        the best blend of each kind (fixed, entropy, dissonance) on the full memory, each chosen on the first half."""
        rows = np.stack([ids[r * T: r * T + T + 1] for r in range(min(n_tokens // T, (len(ids) - 1) // T))])
        text = (hashlib.sha1(np.ascontiguousarray(rows).tobytes()).hexdigest(), n_tokens)
        fp = self.fingerprint()
        prev = self.cache.get(text)
        if prev is not None and not (prev["version"] == self.version and prev["model"] == fp and prev["n"] <= len(self)):
            prev = None
        found = {}
        h, y, lm, dis = encode(self.model, rows.astype(np.int64), dissonance_depth=self.dis_depth)
        ent = torch.cat([(lambda lp: -(lp.exp() * lp).nan_to_num(0.0).sum(-1))(
            torch.log_softmax(self.model.logits(h[c:c + 8192].float()), -1)) for c in range(0, len(h), 8192)])
        half = torch.arange(len(y), device="cuda") >= len(y) // 2
        nb = {p: self.tb[y][half == p].sum().item() for p in (False, True)}
        bpb = lambda lp, part: (-lp[half == part]).sum().item() / math.log(2) / nb[part]
        base = bpb(lm, True)
        res = dict(model=base, memory=base, held=base, blend=None, blend_held=None, by_gate={})
        dq = torch.quantile(dis[~half].float()[:200_000], torch.tensor([0.5, 0.75, 0.9], device="cuda")).tolist()
        dsd = dis[~half].float().std().item() + 1e-6

        def blended(held_only, by_gate=None):
            init = prev["found"].get(held_only) if prev is not None else None
            d, i = self.search(h, held_only, prev["n"] if init is not None else 0, init)
            found[held_only] = (d.cpu(), i.int().cpu())
            d = d + h.float().square().sum(-1, keepdim=True)
            match = (self.vals[i] == y[:, None]).float()
            best = {g: (bpb(lm, False), lm, None) for g in ("fixed", "entropy", "dissonance")}
            for tau in (1.0, 3.0, 10.0, 30.0, 100.0):
                pm = (torch.softmax(-d / tau, -1) * match).sum(-1)  # p_mem(next)
                blends = [("fixed", lam, torch.full_like(lm, lam)) for lam in (0.05, 0.1, 0.2, 0.3, 0.5)]
                blends += [("entropy", f"{mx}/{c}", mx * torch.sigmoid(ent - c)) for mx in (0.3, 0.5, 0.8) for c in (1.0, 2.0, 3.0)]
                blends += [("dissonance", f"{mx}/q{q}", mx * torch.sigmoid((dis - c) / dsd))
                           for mx in (0.3, 0.5, 0.8) for q, c in zip((50, 75, 90), dq)]
                for gate, name, lt in blends:
                    mix = torch.log((1 - lt) * lm.exp() + lt * pm + 1e-12)
                    v = bpb(mix, False)
                    if v < best[gate][0]:
                        best[gate] = (v, mix, f"{gate} {name} tau {tau}")
            if by_gate is not None:
                by_gate.update({g: round(bpb(b[1], True), 5) for g, b in best.items()})
            win = min(best.values(), key=lambda b: b[0])  # all three kinds compete on the first half
            return bpb(win[1], True), win[2]

        if len(self):
            res["memory"], res["blend"] = blended(False, res["by_gate"])
            if self.n_held() == len(self):
                res["held"], res["blend_held"] = res["memory"], res["blend"]
                found[True] = found[False]
            elif self.n_held():
                res["held"], res["blend_held"] = blended(True)
            self.cache[text] = dict(version=self.version, model=fp, n=len(self), found=found)
        return res
