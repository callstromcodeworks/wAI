"""Checks for the spiking MLP (gpt.py mlp_mode="spiking")."""
import os
import sys
import types
MODEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MODEL)
import torch
from gpt import GPT, GPTConfig, SpikingMLP
import train

torch.manual_seed(0)
ok = lambda c, name, extra="": print(("PASS " if c else "FAIL ") + name + (f"  ({extra})" if extra else ""))
T = 128
cfg = GPTConfig(n_ranks=32763, seq_len=T, n_layer=4, n_head=4, d_model=256, n_kv_head=2, mlp_mode="spiking",
                spike_window=8)
with torch.device("cuda"):
    m = GPT(cfg)
mlp = m.blocks[0].mlp
ok(isinstance(mlp, SpikingMLP), "blocks use the spiking MLP")

# 1. membrane = leaky sum over the last K tokens; spikes are exactly 0/1 and fire past the threshold
with torch.no_grad():
    mlp.decay.normal_(0, 1); mlp.threshold.normal_(0.5, 0.2); mlp.proj.weight.normal_(0, 0.02)
    x = torch.randn(2, T, 256, device="cuda")
    u = mlp.fc(x).float()
    lam = torch.sigmoid(mlp.decay)
    v = torch.zeros_like(u)
    for k in range(8):
        v[:, k:] += (lam ** k) * u[:, :T - k]
    spikes = (v > mlp.threshold).float()
    ref = mlp.proj(spikes)
    out = mlp(x).float()
    err = (out - ref).abs().max().item()
    ok(err < 1e-3, "output = projection of 0/1 spikes from the leaky membrane", f"max diff {err:.1e}, firing {mlp.rate.item():.1%}")

# 2. causal through the whole model
m.eval()
ids = torch.randint(0, 32763, (2, T), device="cuda")
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    a = m(ids).float()
    for pos in (10, 90):
        ids2 = ids.clone(); ids2[:, pos] = (ids2[:, pos] + 1) % 32763
        ok((a[:, :pos] - m(ids2).float()[:, :pos]).abs().max().item() == 0, f"causal at token {pos}")
m.train()

# 3. trains: surrogate gradients reach every parameter (decay and threshold included), compiled
args = types.SimpleNamespace(lr_muon=0.05, lr_embed=0.006, lr_scalar=0.02, wd=0.0, ce_chunk=2048)
muon, adam = train.build_optimizers(m, args)
fwd = torch.compile(m)
for step in range(3):
    ids = torch.randint(0, 32763, (4, T + 1), device="cuda")
    with torch.autocast("cuda", torch.bfloat16):
        loss = m.loss(fwd(ids[:, :-1]), ids[:, 1:])
    loss.backward()
    if step == 2:
        dead = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
        ok(not dead, "every parameter gets gradient (decay, threshold included)", str(dead[:4]))
    muon.step(); adam.step(); m.zero_grad(set_to_none=True)
ok(torch.isfinite(loss).item() and "neurons firing" in m.depth_stats(), "compiled training + firing-rate log",
   m.depth_stats().strip())
