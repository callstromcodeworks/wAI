"""Checks for the critic that earns its say (brain.py critic_gate)."""
import sys
import os
MODEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MODEL)
import torch
from brain import BrainConfig, BrainGPT

torch.manual_seed(0)
ok = lambda c, name, extra="": print(("PASS " if c else "FAIL ") + name + (f"  ({extra})" if extra else ""))
base = dict(n_ranks=32763, seq_len=128, n_layer=6, n_head=3, d_model=384, n_kv_head=1, trunk_layers=2, gate_layers=2,
            n_modules=4, module_dim=128, module_layers=2, module_heads=2, module_kv_heads=1, top_k=2, n_shared=1,
            memory_sub=0, critic=True, critic_detach=True)
x = torch.randint(0, 32763, (2, 128), device="cuda")


def grads(m):
    m.zero_grad(set_to_none=True)
    with torch.autocast("cuda", torch.bfloat16):
        loss, parts = m.train_loss(m(x), x, draft_weight=0.0, critic_weight=0.1, balance_weight=0.01)
    loss.backward()
    g = lambda p: 0.0 if p.grad is None else p.grad.abs().sum().item()
    return g(m.critic.w_out), g(m.err_to_resid), g(m.err_head.weight), parts


for gate in (1, 2):
    with torch.device("cuda"):
        m = BrainGPT(BrainConfig(**base, critic_gate=gate))
    ok(m.err_head.out_features == (2 if gate == 2 else 1), f"gate {gate}: error head outputs", str(m.err_head.out_features))
    w, r, h, parts = grads(m)
    ok(w == 0 and r == 0 and h > 0, f"gate {gate}: while observing, the main loss gives the correction path no gradient "
       f"and the critic still learns", f"w_out {w:.2e}, err_to_resid {r:.2e}, err_head {h:.2e}")
    ok({"critic_corr", "critic_say"} <= set(parts) and ("critic_nll" in parts) == (gate == 2), f"gate {gate}: logged parts",
       ", ".join(sorted(parts)))
    m.trust.fill_(1.0)
    w, r, h, _ = grads(m)
    ok(w > 0 and r > 0, f"gate {gate}: with trust, the correction path learns", f"w_out {w:.2e}, err_to_resid {r:.2e}")

# trust follows the track record, held at 0 while observing
with torch.device("cuda"):
    m = BrainGPT(BrainConfig(**base, critic_gate=2, gate_observe_steps=100))
trace = {}
for step in range(1, 301):
    m.update_after_step({"critic_corr": torch.tensor(0.45, device="cuda"), "critic_logvar": torch.tensor(1.0, device="cuda")})
    trace[step] = m.trust.item()
ok(trace[100] == 0.0 and abs(trace[101] - 0.5) < 1e-5 and abs(trace[300] - 0.5) < 1e-5,
   "trust: 0 for 100 observing steps, then (corr 0.45 - 0.3) / 0.3 = 0.5", f"{trace[100]}, {trace[101]:.3f}, {trace[300]:.3f}")
for _ in range(400):
    m.update_after_step({"critic_corr": torch.tensor(0.9, device="cuda"), "critic_logvar": torch.tensor(1.0, device="cuda")})
ok(abs(m.trust.item() - 1.0) < 1e-6, "trust caps at 1 for a strong record", f"{m.trust.item():.3f}")
for _ in range(400):
    m.update_after_step({"critic_corr": torch.tensor(0.1, device="cuda"), "critic_logvar": torch.tensor(1.0, device="cuda")})
ok(m.trust.item() == 0.0, "trust falls back to 0 when the record turns bad", f"{m.trust.item():.3f}")

# per-token confidence: surer than usual -> more say
lv = torch.tensor([[0.0, 1.0, 2.0]], device="cuda")
m.trust.fill_(1.0); m.logvar_ref.fill_(1.0)
say = m.critic_say(lv).flatten().tolist()
ok(say[0] > 0.5 and abs(say[1] - 0.5) < 1e-6 and say[2] < 0.5, "confidence: surer than usual > 0.5 > less sure",
   ", ".join(f"{s:.3f}" for s in say))

# the compiled forward sees trust changes, and gating stays causal
with torch.device("cuda"):
    m = BrainGPT(BrainConfig(**base, critic_gate=2))
with torch.no_grad():
    m.err_to_resid.normal_(0, 0.5); m.critic.w_out.normal_(0, 0.02); m.err_head.weight.normal_(0, 0.02)
m.eval()
fwd = torch.compile(m)
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    a = fwd(x).float()
    m.trust.fill_(1.0); m.logvar_ref.fill_(0.5)
    b = fwd(x).float()
    ok((a - b).abs().max().item() > 0, "compiled forward follows trust", f"max change {(a - b).abs().max().item():.3f}")
    x2 = x.clone(); x2[:, 64] = (x2[:, 64] + 1) % 32763
    c = fwd(x2).float()
    ok((b[:, :64] - c[:, :64]).abs().max().item() == 0, "gated critic is causal")

# old checkpoints load and run unchanged
CKPT = os.path.join(MODEL, "runs", "sweep_b32k_critic2", "ckpt.pt")  # an existing run, if this machine has it
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    with torch.device("cuda"):
        old = BrainGPT(BrainConfig(**ck["config"]))
    old.load_state_dict(ck["model"])  # strict
    old.eval()
    r = torch.randint(0, 32763, (1, 1025), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        loss = old.token_losses(old(r[:, :-1]), r[:, 1:]).mean().item()
    ok(old.cfg.critic_gate == 0 and loss == loss, "round-3 critic2 checkpoint loads strictly and runs", f"loss {loss:.3f}")
else:
    print("SKIP old checkpoint check: no sweep_b32k_critic2 here")
