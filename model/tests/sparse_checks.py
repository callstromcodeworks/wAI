"""Checks for the sparse module modes (brain.py module_mode routed_attn / ffn)."""
import sys, types
import os
MODEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MODEL)
import torch
import torch.nn.functional as F
from brain import BrainConfig, BrainGPT, rope_at
from gpt import norm
import train

torch.manual_seed(0)
ok = lambda c, name, extra="": print(("PASS " if c else "FAIL ") + name + (f"  ({extra})" if extra else ""))
base = dict(n_ranks=32763, seq_len=64, n_layer=6, n_head=3, d_model=384, n_kv_head=1, trunk_layers=2, gate_layers=2,
            n_modules=5, module_dim=128, module_layers=2, module_heads=2, module_kv_heads=1, top_k=2, n_shared=1,
            memory_sub=0, critic=False)
args = types.SimpleNamespace(lr_muon=0.05, lr_embed=0.006, lr_scalar=0.02, wd=0.0, ce_chunk=2048, draft_weight=0.0,
                             critic_weight=0.0, balance_weight=0.01, aux_stride=8)
x = torch.randint(0, 32763, (3, 64), device="cuda")


def randomize(m):  # zero-initialised outputs would make every comparison trivially equal
    with torch.no_grad():
        for p in list(m.bank.parameters()) + (list(m.shared.parameters()) if m.shared is not None else []):
            p.normal_(0, 0.05)


for mode in ("routed_attn", "ffn"):
    with torch.device("cuda"):
        m = BrainGPT(BrainConfig(**base, module_mode=mode))
    R = 4
    ok(m.bank.E == R and m.shared.E == 1 and m.bank.attention == (mode == "routed_attn"), f"{mode}: banks built",
       m.describe())
    muon, adam = train.build_optimizers(m, args)  # asserts every parameter is in exactly one group
    for step in range(3):
        m.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16):
            out = m(x)
            loss, parts = train.step_loss(m, out, x, args)
        loss.backward()
        if step < 2:
            muon.step(); adam.step()
    no_grad = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    ok(not no_grad, f"{mode}: every parameter gets gradient by step 3", str(no_grad[:5]))
    ok("dropped" in parts and 0 <= parts["dropped"].item() <= 1, f"{mode}: dropped share logged",
       f"{parts['dropped'].item():.3f}")

    # the routed output equals a direct per-module computation on exactly that module's tokens
    randomize(m)
    m.eval()
    with torch.no_grad():
        h = torch.randn(3, 64, 384, device="cuda")
        gates, _, fired, _ = m.route(h)
        g, f = gates[..., 1:], fired[..., 1:]
        cos, sin = m.rope_cos_m[:64], m.rope_sin_m[:64]
        got, dropped = m.routed_bank(h, g, f, cos, sin)
        C = min(64, -(-int(m.cfg.capacity * 64 * 2) // R))
        ref = torch.zeros_like(h)
        bank = m.bank
        for b in range(3):
            for e in range(R):
                toks = f[b, :, e].nonzero().flatten()[:C]  # its first C tokens, in time order
                if len(toks) == 0:
                    continue
                xe = h[b, toks]                                                # [n, D]
                he = norm(xe) @ bank.w_in.view(R, 128, 384)[e].t()             # [n, de]
                for l in range(bank.L):
                    if bank.attention:
                        q, k, v = (norm(he) @ bank.qkv[l][e].t()).split(bank.qkv_rows, -1)
                        n = len(toks)
                        q = rope_at(norm(q.view(1, n, 2, 64)), cos[toks][None], sin[toks][None]).transpose(1, 2)
                        k = rope_at(norm(k.view(1, n, 1, 64)), cos[toks][None], sin[toks][None]).transpose(1, 2)
                        v = v.view(1, n, 1, 64).transpose(1, 2)
                        y = F.scaled_dot_product_attention(q, k, v, is_causal=n > 1, scale=bank.attn_scale, enable_gqa=True)
                        he = he + y.transpose(1, 2).reshape(n, 128) @ bank.proj[l][e].t()
                    he = he + F.relu(norm(he) @ bank.fc[l][e].t()).square() @ bank.fc2[l][e].t()
                ref[b, toks] += (he @ bank.w_out.view(R, 128, 384)[e]) * g[b, toks, e][:, None]
        err = (got.float() - ref).abs().max().item()
        ok(err < 1e-3, f"{mode}: routed output = each module run on exactly its own tokens", f"max diff {err:.2e}, "
           f"dropped {dropped.item():.3f}")

        # causal, including which tokens get a slot
        m.cfg.capacity = 0.6  # force drops so capacity decisions are tested too
        with torch.autocast("cuda", torch.bfloat16):
            a = m(x).float()
        for t in (10, 40):
            x2 = x.clone(); x2[:, t] = (x2[:, t] + 1) % 32763
            with torch.autocast("cuda", torch.bfloat16):
                b2 = m(x2).float()
            ok((a[:, :t] - b2[:, :t]).abs().max().item() == 0, f"{mode}: causal at token {t} with capacity drops")
        _, d = m.routed_bank(h, g, f, cos, sin)
        ok(d.item() > 0, f"{mode}: low capacity drops some firings", f"{d.item():.3f}")
        m.cfg.capacity = 1.25

    # compiled training forward
    m.train()
    fwd = torch.compile(m)
    with torch.autocast("cuda", torch.bfloat16):
        loss = m.train_loss(fwd(x), x, draft_weight=0, critic_weight=0)[0]
    ok(torch.isfinite(loss).item(), f"{mode}: compiled forward runs", f"loss {loss.item():.3f}")

# dense mode and old checkpoints unchanged
CKPT = os.path.join(MODEL, "runs", "sweep_b32k_bank3_t100m", "ckpt.pt")  # an existing run, if this machine has it
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location="cpu",
                    weights_only=False)
    with torch.device("cuda"):
        old = BrainGPT(BrainConfig(**ck["config"]))
    old.load_state_dict(ck["model"])
    old.eval()
    r = torch.randint(0, 32763, (1, 1025), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        loss = old.token_losses(old(r[:, :-1]), r[:, 1:]).mean().item()
    ok(old.cfg.module_mode == "dense" and loss == loss, "bank3 checkpoint loads strictly and runs", f"loss {loss:.3f}")
else:
    print("SKIP old checkpoint check: no sweep_b32k_bank3_t100m here")
