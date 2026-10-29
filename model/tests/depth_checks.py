"""Checks for signal / no-signal routing through depth (gpt.py depth_every)."""
import math
import os
import sys
import types
MODEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MODEL)
import torch
import torch.nn.functional as F
from gpt import GPT, GPTConfig, norm
import train

torch.manual_seed(0)
ok = lambda c, name, extra="": print(("PASS " if c else "FAIL ") + name + (f"  ({extra})" if extra else ""))
T = 256
cfg = GPTConfig(n_ranks=32763, seq_len=T, n_layer=4, n_head=4, d_model=256, n_kv_head=2, value_residual=True,
                unet=True, depth_every=2, depth_capacity=0.25)
with torch.device("cuda"):
    m = GPT(cfg)
routed = [i for i, b in enumerate(m.blocks) if b.router is not None]
ok(routed == [1, 3], "every 2nd block after the first is routed", str(routed))
with torch.no_grad():
    for b in m.blocks:
        for n, p in b.named_parameters():
            if n.endswith("proj.weight"):
                p.normal_(0, 0.02)  # zero-initialised outputs would hide routing errors

# 1. a routed block = running the plain block on exactly its routed tokens
blk = m.blocks[1]
x = torch.randn(2, T, 256, device="cuda")
cos, sin = m.rope_cos[:T], m.rope_sin[:T]
with torch.no_grad():
    blk.depth_bias.fill_(0.0)
    out = blk(x, cos, sin, {"cache": None, "v0": None, "kv": None, "rope_mla": None})
    w = blk.router.weight.float()
    s = (norm(x).float() @ (w / (w.norm() + 1e-6)).t()).squeeze(-1) * 4.0
    C = math.ceil(0.25 * T)
    ref = x.clone()
    for bi in range(2):
        toks = (s[bi] > 0).nonzero().flatten()[:C]
        xs = x[bi:bi + 1, toks]
        ctx = {"cache": None, "v0": None, "kv": None, "rope_mla": None}
        y = xs + blk.attn(norm(xs), cos[toks][None], sin[toks][None], ctx)
        y = y + blk.mlp(norm(y))
        ref[bi, toks] += (y - xs)[0] * torch.sigmoid(s[bi, toks])[:, None]
    err = (out - ref).abs().max().item()
    ok(err < 1e-3, "routed block = plain block on its routed tokens only", f"max diff {err:.1e}, {blk.fired.item():.0%} processed")
    untouched = (s[0] <= 0)
    ok(torch.equal(out[0, untouched], x[0, untouched]), "tokens whose router didn't fire pass through unchanged")

# 2. causal, including which tokens fit the capacity
m.eval()
ids = torch.randint(0, 32763, (2, T), device="cuda")
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    a = m(ids).float()
    for pos in (30, 200):
        ids2 = ids.clone(); ids2[:, pos] = (ids2[:, pos] + 1) % 32763
        ok((a[:, :pos] - m(ids2).float()[:, :pos]).abs().max().item() == 0, f"causal at token {pos}")
m.train()

# 3. training: every parameter learns (router included), compiled, thresholds steer toward capacity
args = types.SimpleNamespace(lr_muon=0.05, lr_embed=0.006, lr_scalar=0.02, wd=0.0, ce_chunk=2048)
muon, adam = train.build_optimizers(m, args)
fwd = torch.compile(m)
VAL = os.path.join(MODEL, "data", "ec32k", "fineweb-edu", "val_000000.bin")
text = None
if os.path.exists(VAL):  # the controller needs real text: on random tokens the router has nothing to learn
    import numpy as np
    from prepare_data import load_shard
    text = np.asarray(load_shard(VAL)[0][:2_000_000])
rng = torch.Generator().manual_seed(0)
hist = []
for step in range(200 if text is not None else 3):
    if text is not None:
        st = torch.randint(0, len(text) - T - 1, (8,), generator=rng).tolist()
        ids = torch.from_numpy(np.stack([text[i:i + T + 1] for i in st]).astype("int64")).cuda()
    else:
        ids = torch.randint(0, 32763, (4, T + 1), device="cuda")
    with torch.autocast("cuda", torch.bfloat16):
        loss = m.loss(fwd(ids[:, :-1]), ids[:, 1:])
    loss.backward()
    if step == 2:
        dead = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
        ok(not dead, "every parameter gets gradient (router included)", str(dead[:4]))
    muon.step(); adam.step(); m.zero_grad(set_to_none=True)
    m.update_after_step({})
    hist.append([b.wanted.item() for b in m.blocks if b.router is not None])
if text is not None:
    wanted = torch.tensor(hist[-50:]).mean(0).tolist()  # what the controller regulates: the average share
    ok(all(0.17 < w < 0.33 for w in wanted) and torch.isfinite(loss).item(),
       "on real text, thresholds hold the average firing share near capacity (25%)",
       f"{[round(w, 2) for w in wanted]}, loss {loss.item():.2f}")
else:
    print("SKIP threshold control check: no validation text in model/data")
ok("depth routing" in m.depth_stats(), "training log line", m.depth_stats().strip())

# 4. mlp mode: every token gets attention, only routed tokens get the MLP
cfg2 = GPTConfig(n_ranks=32763, seq_len=T, n_layer=4, n_head=4, d_model=256, n_kv_head=2, depth_every=2,
                 depth_capacity=0.25, depth_mode="mlp")
with torch.device("cuda"):
    m2 = GPT(cfg2)
blk = m2.blocks[1]
with torch.no_grad():
    for n, p in blk.named_parameters():
        if n.endswith("proj.weight"):
            p.normal_(0, 0.02)
    x = torch.randn(2, T, 256, device="cuda")
    ctx = lambda: {"cache": None, "v0": None, "kv": None, "rope_mla": None}
    out = blk(x, cos, sin, ctx())
    xa = x + blk.attn(norm(x), cos, sin, ctx())
    w = blk.router.weight.float()
    s = (norm(xa).float() @ (w / (w.norm() + 1e-6)).t()).squeeze(-1) * 4.0  # router sees the token after attention
    ref = xa.clone()
    for bi in range(2):
        toks = ((s[bi] + blk.depth_bias) > 0).nonzero().flatten()[:math.ceil(0.25 * T)]
        ref[bi, toks] += blk.mlp(norm(xa[bi, toks])) * torch.sigmoid(s[bi, toks])[:, None]
    err = (out - ref).abs().max().item()
    ok(err < 1e-3, "mlp mode: attention on all tokens, MLP on routed tokens only", f"max diff {err:.1e}")
m2.eval()
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    ids = torch.randint(0, 32763, (2, T), device="cuda")
    a = m2(ids).float()
    ids2 = ids.clone(); ids2[:, 100] = (ids2[:, 100] + 1) % 32763
    ok((a[:, :100] - m2(ids2).float()[:, :100]).abs().max().item() == 0, "mlp mode: causal at token 100")
