"""Correctness checks for model/brain.py (tiny config, no compile)."""
import sys, types
import os
MODEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MODEL)
import torch
from brain import BrainConfig, BrainGPT
import train

torch.manual_seed(0)
ok = lambda c, name, extra="": print(("PASS " if c else "FAIL ") + name + (f"  ({extra})" if extra else ""))

cfg = BrainConfig(n_ranks=32763, seq_len=128, n_layer=6, n_head=3, d_model=384, n_kv_head=1, value_residual=True,
                  unet=True, trunk_layers=2, gate_layers=2, n_modules=4, module_dim=128, module_layers=2,
                  module_heads=2, module_kv_heads=1, top_k=2, memory_sub=32, memory_dim=64, memory_k=8)
with torch.device("cuda"):
    m = BrainGPT(cfg)
args = types.SimpleNamespace(lr_muon=0.05, lr_embed=0.006, lr_scalar=0.02, wd=0.0, ce_chunk=2048, draft_weight=0.1,
                             critic_weight=0.1, balance_weight=0.01, aux_stride=8)
muon, adam = train.build_optimizers(m, args)  # asserts every parameter is in exactly one group
ok(True, "optimizer groups cover every parameter once", f"{m.num_params() / 1e6:.2f}M params")

x = torch.randint(0, cfg.n_vocab, (2, 128), device="cuda")
with torch.autocast("cuda", torch.bfloat16):
    out = m(x)
ok(isinstance(out, tuple) and out[0].shape == (2, 128, 384) and out[2].shape == (2, 128) and out[4].shape == (4,),
   "training forward shapes")
loss, parts = train.step_loss(m, out, x, args, offset=3)
ok(torch.isfinite(loss).item() and set(parts) >= {"main", "draft", "critic_mse", "balance", "usage"},
   "train loss + parts", ", ".join(f"{k}={v.item():.3f}" for k, v in parts.items() if v.dim() == 0))
ok(abs(parts["usage"].sum().item() - cfg.top_k) < 1e-4, "each token fires exactly top_k modules",
   f"usage {parts['usage'].tolist()}")
loss.backward()
no_grad = [n for n, p in m.named_parameters() if p.grad is None]
# at init the zero-initialised output projections block gradients to what sits behind them; one step fixes that
print("     params without grad at step 0:", no_grad[:8], "..." if len(no_grad) > 8 else "")
for _ in range(2):  # module inner layers sit behind two zero-initialised projections: grads reach them on step 3
    muon.step(); adam.step(); m.zero_grad(set_to_none=True)
    with torch.autocast("cuda", torch.bfloat16):
        out = m(x)
    train.step_loss(m, out, x, args)[0].backward()
no_grad = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
ok(not no_grad, "every parameter gets gradient by step 3", str(no_grad))
muon.step(); adam.step()
ok(all(torch.isfinite(p).all() for p in m.parameters()), "3-D Muon update keeps weights finite")

m.eval()
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    for t in (5, 64, 127):
        x2 = x.clone()
        x2[:, t] = (x2[:, t] + 1) % cfg.n_vocab
        a, b = m(x).float(), m(x2).float()
        before = (a[:, :t] - b[:, :t]).abs().max().item() if t else 0.0
        after = (a[:, t:] - b[:, t:]).abs().max().item()
        ok(before == 0.0 and after > 0, f"causal: changing token {t} leaves positions <{t} unchanged",
           f"max change before {before:.2e}, at/after {after:.2e}")
m.train()

# the auxiliary losses must not read the future either: the draft/critic at position t only see tokens <= t
with torch.autocast("cuda", torch.bfloat16):
    o1, o2 = m(x), m(x2)
same = all((p[:, :127] - q[:, :127]).abs().max().item() == 0 for p, q in ((o1[1].float(), o2[1].float()),
                                                                           (o1[2], o2[2])))
ok(same, "draft and critic estimate are causal too")
