"""The memory switches (train.py --grad_ckpt, --bf16_state): checkpointed blocks give the same gradients as kept
activations (plain GPT with shared K/V, value residual and U-net skips, which pass state between blocks; a brain model
with a module bank; per-block compile), bf16 gradient buffers sum the micro-batches, Muon runs on them with bf16
momentum and survives a save/load, and per-block compile with bf16 state lowers peak memory."""
import os
import sys
import types

MODEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MODEL)
import torch
from torch.nn.attention import sdpa_kernel

import train
from brain import BrainConfig, BrainGPT
from gpt import GPT, GPTConfig
from muon import Muon

ok = lambda c, name, extra="": print(("PASS " if c else "FAIL ") + name + (f"  ({extra})" if extra else ""))
args = types.SimpleNamespace(lr_muon=0.05, lr_embed=0.006, lr_scalar=0.02, wd=0.0, ce_chunk=2048, draft_weight=0.1,
                             critic_weight=0.0, balance_weight=0.01, aux_stride=8, grad_ckpt=1, bf16_state=0)
x = torch.randint(0, 32763, (2, 64), device="cuda")


def randomize(m):  # zero-initialised output projections would make the comparison trivially equal
    torch.manual_seed(1)
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() >= 2:
                p.normal_(0, 0.03)


def grads(m, fwd, ckpt):
    m.checkpoint_blocks = ckpt
    m.zero_grad(set_to_none=True)
    with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(train.SDPA_BACKENDS, set_priority=True):
        out = fwd(x)
    loss = train.step_loss(m, out, x, args)[0]
    train.backward(loss, args)
    return {n: p.grad.detach().float().clone() for n, p in m.named_parameters() if p.grad is not None}


def worst(a, b):
    assert a.keys() == b.keys(), "different parameters got gradients"
    return max(((a[k] - b[k]).abs().max() / b[k].abs().max().clamp(min=1e-12)).item() for k in a)


# 1. plain GPT, state carried between blocks through ctx (value residual's first values, K/V shared by 2 layers)
torch.manual_seed(0)
with torch.device("cuda"):
    g = GPT(GPTConfig(n_ranks=32763, seq_len=64, n_layer=6, n_head=4, d_model=256, n_kv_head=2, kv_share=2,
                      value_residual=True, unet=True))
randomize(g)
kept, ckpt = grads(g, g, False), grads(g, g, True)
w = worst(ckpt, kept)
ok(w < 1e-6, "GPT (shared K/V, value residual, U-net): checkpointed gradients = kept", f"worst rel diff {w:.1e}")

# 2. brain model with a sparse module bank (only the main blocks are checkpointed)
with torch.device("cuda"):
    b = BrainGPT(BrainConfig(n_ranks=32763, seq_len=64, n_layer=6, n_head=4, d_model=256, n_kv_head=1, trunk_layers=2,
                             gate_layers=2, n_modules=5, module_dim=128, module_layers=2, module_heads=2,
                             module_kv_heads=1, top_k=2, n_shared=1, memory_sub=0, critic=False,
                             module_mode="routed_attn"))
randomize(b)
kept, ckpt = grads(b, b, False), grads(b, b, True)
w = worst(ckpt, kept)
ok(w < 1e-6, "BrainGPT (sparse bank): checkpointed gradients = kept", f"worst rel diff {w:.1e}")

# 3. per-block compile (what train.py uses with either switch): checkpointed = kept, and close to eager
eager = grads(g, g, False)
g.compile_regions()
kept, ckpt = grads(g, g, False), grads(g, g, True)
w, w_eager = worst(ckpt, kept), worst(kept, eager)
ok(w < 1e-6 and w_eager < 0.05, "per-block compile: checkpointed = kept, both near eager",
   f"ckpt vs kept {w:.1e}, compiled vs eager {w_eager:.1e}")
b.compile_regions()
kept = grads(b, b, False)
ok(all(torch.isfinite(t).all() for t in kept.values()), "BrainGPT per-block compile runs (bank, routing compiled)")

# 4. bf16 gradient buffers: two micro-batches summed, p.grad freed, Muon with bf16 momentum, save/load
torch.manual_seed(0)
with torch.device("cuda"):
    m = GPT(GPTConfig(n_ranks=32763, seq_len=64, n_layer=4, n_head=4, d_model=256))
randomize(m)
mats = m.matrix_params()
x2 = torch.randint(0, 32763, (2, 64), device="cuda")
ref = {}
for xi in (x, x2):  # fp32 reference sum
    with torch.autocast("cuda", torch.bfloat16):
        m.loss(m(xi), xi).backward()
ref = [p.grad.float().clone() for p in mats]
m.zero_grad(set_to_none=True)
lowp = train.Bf16Grads(mats)
for xi in (x, x2):
    with torch.autocast("cuda", torch.bfloat16):
        m.loss(m(xi), xi).backward()
freed = all(p.grad is None for p in mats)
err = max(((lowp.get(p).float() - r).norm() / r.norm()).item() for p, r in zip(mats, ref))
ok(freed and err < 1e-2, "bf16 buffers hold the micro-batch sum; fp32 gradients freed", f"worst rel error {err:.1e}")

opt32 = Muon(mats, lr=0.02)
opt16 = Muon(mats, lr=0.02, state_dtype=torch.bfloat16)
opt16.grad_of = lowp.get
before = [p.detach().clone() for p in mats]
for p, r in zip(mats, ref):
    p.grad = r
opt32.step()
d32 = [p.detach() - q for p, q in zip(mats, before)]
with torch.no_grad():
    for p, q in zip(mats, before):
        p.copy_(q)
        p.grad = None
opt16.step()
d16 = [p.detach() - q for p, q in zip(mats, before)]
cos = min(torch.nn.functional.cosine_similarity(a.flatten(), c.flatten(), 0).item() for a, c in zip(d16, d32))
dtypes = {opt16.state[p]["momentum_buffer"].dtype for p in mats}
ok(dtypes == {torch.bfloat16} and cos > 0.99, "Muon on bf16 buffers: bf16 momentum, update matches fp32",
   f"min cosine {cos:.4f}")
sd = opt16.state_dict()
opt16b = Muon(mats, lr=0.02, state_dtype=torch.bfloat16)
opt16b.load_state_dict(sd)  # torch casts state to the parameter dtype (fp32) on load
opt16b.grad_of = lowp.get
opt16b.step()
ok({opt16b.state[p]["momentum_buffer"].dtype for p in mats} == {torch.bfloat16}, "bf16 momentum survives save/load")
lowp.clear()
ok(lowp.get(mats[0]) is None, "cleared buffers read as no gradient")

# 5. peak memory while accumulating: per-block compile + bf16 buffers vs the default whole-model compile
def peak(lowp_on):
    torch._dynamo.reset()
    torch.manual_seed(0)
    with torch.device("cuda"):
        m = GPT(GPTConfig(n_ranks=32763, seq_len=64, n_layer=4, n_head=6, d_model=768))
    buf = None
    if lowp_on:
        buf = train.Bf16Grads(m.matrix_params())
        m.compile_regions()
        f = m
    else:
        f = torch.compile(m, dynamic=False)
    xs = torch.randint(0, 32763, (1, 64), device="cuda")

    def micro():
        with torch.autocast("cuda", torch.bfloat16):
            m.loss(f(xs), xs).backward()
    micro()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    micro()  # second micro-batch: gradients already held
    torch.cuda.synchronize()
    out = torch.cuda.max_memory_allocated()
    del m, f, buf
    torch.cuda.empty_cache()
    return out


p32, p16 = peak(False), peak(True)
ok(p16 < p32, "per-block compile + bf16 state lowers peak memory", f"{p32 / 2**20:.0f} -> {p16 / 2**20:.0f} MiB")
