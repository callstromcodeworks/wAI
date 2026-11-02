"""Episodic memory (episodic.py): scoring right after an arrival reuses the previous search and searches only the
new episodes, with exactly the scores of a full search; a changed model or re-keyed memory forces a full search;
an arrival with an unchanged model leaves the old keys alone, with a changed one re-keys everything.
Runs on a stand-in model (a few MB of GPU memory), so it can run beside a training job."""
import os
import sys
MODEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MODEL)
import numpy as np
import torch
import episodic

ok = lambda c, name, extra="": print(("PASS " if c else "FAIL ") + name + (f"  ({extra})" if extra else ""))
torch.manual_seed(0)
rng = np.random.default_rng(0)
V, D, T = 512, 64, episodic.T


class Stand(torch.nn.Module):
    """Stands in for the GPT: a hidden state from the last three tokens and the position; logits from it."""
    def __init__(self):
        super().__init__()
        self.cfg = type("Cfg", (), {"d_model": D, "trunk_layers": 3})()
        self.emb = torch.nn.Parameter(torch.randn(V, D, device="cuda"))
        self.pos = torch.nn.Parameter(torch.randn(T, D, device="cuda") * 0.3)
        self.mix = torch.nn.Parameter(torch.randn(3 * D, D, device="cuda") / (3 * D) ** 0.5)
        self.out = torch.nn.Parameter(torch.randn(D, V, device="cuda") / D ** 0.5)

    def logits(self, h):
        return h.float() @ self.out


@torch.no_grad()
def stand_encode(model, rows, batch=16, dissonance_depth=None):
    r = torch.from_numpy(rows).cuda()
    x = r[:, :-1]
    p1, p2 = torch.roll(x, 1, 1), torch.roll(x, 2, 1)
    h = torch.tanh(torch.cat([model.emb[x], model.emb[p1], model.emb[p2]], -1) @ model.mix + model.pos).reshape(-1, D)
    y = r[:, 1:].reshape(-1)
    lp = torch.log_softmax(model.logits(h), -1)
    out = (h.half(), y, lp.gather(1, y[:, None])[:, 0])
    return out + (-(lp.exp() * lp).sum(-1),) if dissonance_depth is not None else out


episodic.encode = stand_encode


def stream(n, trans):
    """n tokens from a sparse Markov chain (so memory of the same source helps)."""
    cum, out, t = np.cumsum(trans, 1), np.empty(n, np.int64), 0
    for i, u in enumerate(rng.random(n)):
        t = min(int(np.searchsorted(cum[t], u)), V - 1)
        out[i] = t
    return out


src = [rng.dirichlet(np.full(V, 0.03), size=V) for _ in range(3)]
data = {tag: stream(6 * T + 1, src[j]) for j, tag in enumerate("abc")}
texts = {"x": stream(4 * T + 1, src[0]), "z": stream(4 * T + 1, src[2])}
tb = torch.randint(1, 6, (V,), device="cuda")
model = Stand()
mem = episodic.EpisodicMemory(model, tb, k=8, chunk=2500, qb=1000)


def learn():  # a stage of "training": the model changes
    with torch.no_grad():
        for p in (model.mix, model.out):
            p.add_(torch.randn_like(p) * 0.05)


def score_all(cache=True):
    """Scores of both texts and the episodes searched; cache=False: a full search (the reuse store is untouched)."""
    saved, n0 = mem.cache, mem.searched
    if not cache:
        mem.cache = {}
    r = {n: mem.score(t, 4 * T) for n, t in texts.items()}
    if not cache:
        mem.cache = saved
    return r, mem.searched - n0


def same(r1, r2):
    keys = ("model", "memory", "held", "blend", "blend_held", "by_gate")
    return all(r1[n][k] == r2[n][k] for n in r1 for k in keys)


mem.arrive("a", data["a"], 6 * T)
r, _ = score_all()
ok(r["x"]["memory"] < r["x"]["model"], "memory of the same source helps", f"{r['x']['model']:.4f} -> {r['x']['memory']:.4f}")

learn()
held = mem.reencode(prune=0.003)  # the stand-in predicts weakly: release p >= 0.003
r, used = score_all()
ok(0 < held["a"] < 1 and used == 2 * 2 * 6 * T, "after re-keying with release: full search, both passes",
   f"held {held['a']}, searched {used:,}")

first = mem.blocks[0]
version = mem.version
mem.arrive("b", data["b"], 6 * T)
ok(mem.blocks[0] is first and mem.version == version and len(mem.blocks) == 2,
   "arrival with the model unchanged keeps the old keys (no re-keying)")
r_reuse, used = score_all()
r_full, used_full = score_all(cache=False)
ok(used == 2 * 2 * 6 * T and used_full == 2 * 2 * 12 * T, "after an arrival only the new episodes are searched",
   f"{used:,} vs {used_full:,}")
ok(same(r_reuse, r_full), "reused search gives exactly the full search's scores",
   f"x memory {r_reuse['x']['memory']:.6f} / {r_full['x']['memory']:.6f}")

learn()
mem.arrive("c", data["c"], 6 * T)
ok(mem.blocks[0] is not first and mem.version == version + 1 and len(mem.blocks) == 3,
   "arrival after the model changed re-keys everything")
r_reuse, used = score_all()
r_full, _ = score_all(cache=False)
ok(used == 2 * 2 * 18 * T and same(r_reuse, r_full), "after re-keying the old search is not reused",
   f"searched {used:,}")

learn()  # model changes with no re-keying (misuse): its searches must not be reused either
r_reuse, used = score_all()
ok(used == 2 * 2 * 18 * T, "a changed model is caught by the weight fingerprint", f"searched {used:,}")

ok(all(b.is_pinned() for b in mem.blocks) and sum(len(b) for b in mem.blocks) == len(mem) == len(mem.held),
   "keys sit in pinned blocks, one per text, aligned with values and held flags")
