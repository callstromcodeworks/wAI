"""The judge (brain.py judge): one judging module that allocates routed modules per token."""
import sys
import os
MODEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MODEL)
import torch
from brain import BrainConfig, BrainGPT

ok = lambda c, name, extra="": print(("PASS " if c else "FAIL ") + name + (f"  ({extra})" if extra else ""))
base = dict(n_ranks=32763, seq_len=128, n_layer=6, n_head=3, d_model=384, n_kv_head=1, trunk_layers=2, gate_layers=2,
            n_modules=9, module_dim=128, module_layers=2, module_heads=2, module_kv_heads=1, top_k=2, n_shared=1,
            memory_sub=0, critic=False, module_mode="routed_attn", capacity=2.0, judge=2)
torch.manual_seed(0)
with torch.device("cuda"):
    m = BrainGPT(BrainConfig(**base))
x = torch.randint(0, 32763, (2, 128), device="cuda")

with torch.autocast("cuda", torch.bfloat16):
    out = m(x)
    loss, parts = m.train_loss(out, x, draft_weight=0, critic_weight=0.1)
need = ("critic_nll", "critic_corr", "judge_q", "judge_trust")
ok(all(k in parts for k in need), "judge parts logged", ", ".join(k for k in need if k in parts))
ok(abs(parts["usage"][1:].sum().item() - 2) < 1e-4, "untrusted: every token fires exactly top_k routed modules",
   f"{parts['usage'][1:].sum().item():.3f}")

# the judge's loss must not reach the shared module (its head reads detached features)
m.zero_grad()
with torch.autocast("cuda", torch.bfloat16):
    err, logvar = m(x)[2], m(x)[5]
    jl = (err.float() - 1.0).square().mean() + logvar.float().mean()
jl.backward()
g_shared = sum(p.grad.abs().sum().item() for p in m.shared.parameters() if p.grad is not None)
g_head = sum(p.grad.abs().sum().item() for p in m.judge_head.parameters() if p.grad is not None)
ok(g_shared == 0 and g_head > 0, "judge loss trains its head and nothing else", f"shared {g_shared:.2e}, head {g_head:.2e}")

# trusted, with cut-offs at its own estimate's thirds: modules per token still average top_k
m.trust.fill_(1.0)
with torch.no_grad():
    m.judge_head[2].weight.normal_(std=0.5)
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    err = m(x)[2]
    m.judge_q.copy_(torch.quantile(err.float().flatten(), torch.tensor([1 / 3, 2 / 3], device="cuda")))
    usage = m(x)[4]
ok(abs(usage[1:].sum().item() - 2) < 0.05, "trusted: modules per token average top_k", f"{usage[1:].sum().item():.3f}")

# routing directly: low estimate -> top_k - 1 modules, high -> top_k + 1
m.judge_q.copy_(torch.tensor([-1.0, 1.0]))
with torch.no_grad():
    judged = torch.cat([torch.full((2, 64), -10.0, device="cuda"), torch.full((2, 64), 10.0, device="cuda")], 1)
    g, _, fired, _ = m.route(torch.randn(2, 128, 384, device="cuda"), judged)
k = fired[..., 1:].sum(-1)
ok((k[:, :64] == 1).all().item() and (k[:, 64:] == 3).all().item(), "easy tokens fire top_k - 1, hard tokens top_k + 1",
   f"{k[0, 0].item():.0f} / {k[0, -1].item():.0f}")
ok((g[..., 1:].sum(-1) - 1).abs().max().item() < 1e-5, "gates over a token's chosen modules sum to 1")

# causality with allocation on (cut-offs are running averages, not this batch's ranks)
m.judge_q.copy_(torch.quantile(err.float().flatten(), torch.tensor([1 / 3, 2 / 3], device="cuda")))
m.eval()
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    a = m(x).float()
    for t in (5, 64, 127):
        x2 = x.clone()
        x2[:, t] = (x2[:, t] + 1) % 32763
        b = m(x2).float()
        before = (a[:, :t] - b[:, :t]).abs().max().item()
        ok(before == 0.0, f"causal with allocation: changing token {t} leaves earlier positions unchanged", f"{before:.2e}")

# compiled forward follows trust
m.train()
fwd = torch.compile(m)
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    m.trust.fill_(0.0)
    u0 = fwd(x)[4][1:].clone()
    m.trust.fill_(1.0)
    u1 = fwd(x)[4][1:].clone()
ok((u0 - u1).abs().max().item() > 0, "compiled forward follows trust", f"max usage change {(u0 - u1).abs().max().item():.3f}")

# trust is earned from the track record; cut-offs follow the estimate's thirds
with torch.device("cuda"):
    m2 = BrainGPT(BrainConfig(**base))
for i in range(150):
    m2.update_after_step({"critic_corr": torch.tensor(0.6, device="cuda"), "critic_logvar": torch.tensor(0.0, device="cuda"),
                          "judge_q": torch.tensor([1.0, 2.0], device="cuda"), "usage": torch.full((9,), 0.25, device="cuda")})
ok(m2.trust.item() == 1.0 and abs(m2.judge_q[0].item() - 1.0) < 1e-5, "trust earned after observing; cut-offs tracked",
   f"trust {m2.trust.item():.2f}, cut-offs {[round(v, 2) for v in m2.judge_q.tolist()]}")

# with the surprise input: still causal, still no gradient into the shared module, and surprise is real
with torch.device("cuda"):
    ms = BrainGPT(BrainConfig(**{**base, "judge_surprise": True}))
ms.trust.fill_(1.0)
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    ms.judge_head[2].weight.normal_(std=0.5)
    err = ms(x)[2]
    ms.judge_q.copy_(torch.quantile(err.float().flatten(), torch.tensor([1 / 3, 2 / 3], device="cuda")))
ms.zero_grad()
with torch.autocast("cuda", torch.bfloat16):
    o = ms(x)
    ((o[2].float() - 1).square().mean() + o[5].float().mean()).backward()
g_shared = sum(p.grad.abs().sum().item() for p in ms.shared.parameters() if p.grad is not None)
ok(g_shared == 0, "surprise: judge loss still trains nothing outside its head", f"{g_shared:.2e}")
with torch.no_grad():
    sp = ms.surprise(torch.randn(2, 128, 384, device="cuda"), x)
ok(sp[:, 0].abs().max().item() == 0 and sp[:, 1:].min().item() > 0, "surprise: 0 at the first token, positive after",
   f"mean {sp[:, 1:].mean().item() * 10:.2f} nats")
ms.eval()
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    a = ms(x).float()
    worst = 0.0
    for t in (5, 64, 127):
        x2 = x.clone()
        x2[:, t] = (x2[:, t] + 1) % 32763
        worst = max(worst, (a[:, :t] - ms(x2).float()[:, :t]).abs().max().item())
ok(worst == 0.0, "surprise: causal with allocation (changing token t leaves earlier positions unchanged)", f"{worst:.2e}")
