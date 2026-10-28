"""Shared module + bias balancing: shapes, usage, the bias touches routed modules only, the
compiled forward sees bias updates, and old softmax checkpoints still load and route the same."""
import sys
import os
MODEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MODEL)
import torch
from brain import BrainConfig, BrainGPT

torch.manual_seed(0)
ok = lambda c, name, extra="": print(("PASS " if c else "FAIL ") + name + (f"  ({extra})" if extra else ""))
cfg = BrainConfig(n_ranks=32763, seq_len=128, n_layer=6, n_head=3, d_model=384, n_kv_head=1, trunk_layers=2,
                  gate_layers=2, n_modules=4, module_dim=128, module_layers=2, module_heads=2, module_kv_heads=1,
                  top_k=2, n_shared=1, sigmoid_router=True, memory_sub=0, critic=False, balance_bias=0.01)
with torch.device("cuda"):
    m = BrainGPT(cfg)
ok(m.router.weight.shape == (3, 384) and m.route_bias.shape == (3,), "router and bias cover the 3 routed modules")
fwd = torch.compile(m)
x = torch.randint(0, cfg.n_vocab, (2, 128), device="cuda")
with torch.autocast("cuda", torch.bfloat16):
    out = fwd(x)
    loss, parts = m.train_loss(out, x, draft_weight=0, critic_weight=0, balance_weight=0)
u = parts["usage"]
ok(u[0].item() == 1.0 and abs(u.sum().item() - 3) < 1e-4, "shared module fires on every token, 2 routed per token",
   f"usage {[round(v, 3) for v in u.tolist()]}")
gates, _, fired = m.route(torch.randn(2, 128, 384, device="cuda"))
ok(torch.allclose(gates[..., 0], torch.ones_like(gates[..., 0])) and torch.allclose(gates[..., 1:].sum(-1),
   torch.ones_like(gates[..., 0])), "shared gate is 1, routed gates sum to 1")
loss.backward()
m.update_balance(u)
ok(m.route_bias.abs().sum().item() > 0, "bias moves after one update", str(m.route_bias.tolist()))
m.route_bias[:] = torch.tensor([1.0, 1.0, -1.0])
with torch.autocast("cuda", torch.bfloat16):
    u2 = m.train_loss(fwd(x), x, draft_weight=0, critic_weight=0, balance_weight=0)[1]["usage"]
ok(u2.tolist() == [1.0, 1.0, 1.0, 0.0], "compiled forward follows the bias", str(u2.tolist()))

# an old checkpoint config (softmax router, balance_bias set but bias never moved) routes as trained
old = BrainConfig(**{**cfg.__dict__, "n_shared": 0, "sigmoid_router": False, "n_modules": 4})
with torch.device("cuda"):
    mo = BrainGPT(old)
h = torch.randn(2, 16, 384, device="cuda")
g, _, f = mo.route(h)
logits = mo.router(__import__("gpt").norm(h)).float()
top = logits.topk(2, -1).indices
p = torch.softmax(logits, -1).gather(-1, top)
ref = torch.zeros_like(logits).scatter(-1, top, p / p.sum(-1, keepdim=True))
ok(torch.allclose(g, ref), "old softmax routing reproduced exactly")
