"""Dead-module revival (brain.py revive)."""
import sys
import os
MODEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MODEL)
import torch
from brain import BrainConfig, BrainGPT

ok = lambda c, name, extra="": print(("PASS " if c else "FAIL ") + name + (f"  ({extra})" if extra else ""))
base = dict(n_ranks=32763, seq_len=128, n_layer=6, n_head=3, d_model=384, n_kv_head=1, trunk_layers=2, gate_layers=2,
            n_modules=9, module_dim=128, module_layers=2, module_heads=2, module_kv_heads=1, top_k=2, n_shared=1,
            memory_sub=0, critic=False, module_mode="routed_attn", revive=0.1)
with torch.device("cuda"):
    m = BrainGPT(BrainConfig(**base))
ok(m.route_bias.shape == (8,), "revival keeps a bias over the 8 routed modules")
fair = 2 / 8
usage = torch.tensor([1.0] + [fair] * 6 + [0.0, 0.02 * fair], device="cuda")  # shared, 6 healthy, 2 dead
for _ in range(10):
    m.update_after_step({"usage": usage})
ok(abs(m.route_bias[6].item() - 1.0) < 1e-5 and abs(m.route_bias[7].item() - 1.0) < 1e-5 and m.route_bias[:6].abs().max() == 0,
   "dead modules get +0.1 per step, healthy ones nothing", str([round(v, 2) for v in m.route_bias.tolist()]))
m.route_bias[:] = 2.0
for _ in range(20):
    m.update_after_step({"usage": torch.tensor([1.0] + [fair] * 8, device="cuda")})
ok(abs(m.route_bias[0].item() - 2.0 * 0.95 ** 20) < 1e-4, "the boost fades once a module carries its share",
   f"{m.route_bias[0].item():.3f}")
# the bias steers routing in the compiled forward
m.route_bias[:] = 0; m.route_bias[7] = 50.0
x = torch.randint(0, 32763, (2, 128), device="cuda")
fwd = torch.compile(m)
with torch.autocast("cuda", torch.bfloat16):
    out = fwd(x)
    parts = m.train_loss(out, x, draft_weight=0, critic_weight=0)[1]
ok(parts["usage"][8].item() == 1.0, "a large bias routes every token to the revived module", f"{parts['usage'][8].item():.2f}")
# old configs (no revive, no balance_bias) have no bias at all
with torch.device("cuda"):
    m0 = BrainGPT(BrainConfig(**{**base, "revive": 0.0}))
ok(not hasattr(m0, "route_bias"), "revive off: no router bias, routing unchanged")
