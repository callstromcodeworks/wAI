"""Checks for memory.py (concept attention, fast weights) inside gpt.py."""
import sys, types
import os
MODEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MODEL)
import torch
import torch.nn.functional as F
from gpt import GPT, GPTConfig, norm, rope_tables
from memory import ConceptAttention, FastWeights
import train

torch.manual_seed(0)
# FlexAttention's Triton kernel multiplies fp32 in TF32 on Hopper (~1e-3 relative): 1e-2 still catches any
# masking or selection error, which shows up as O(1) differences (training itself runs in bf16)
TOL = 1e-2
ok = lambda c, name, extra="": print(("PASS " if c else "FAIL ") + name + (f"  ({extra})" if extra else ""))
T, b, W = 1024, 64, 256
base = dict(n_ranks=32763, seq_len=T, n_layer=4, n_head=4, d_model=256, n_kv_head=2, window=W, concept_block=b)
cfgc = GPTConfig(**base, attn_mode="concept", concept_topk=2)
B, H, Hkv, D = 2, 4, 2, 64
q = norm(torch.randn(B, H, T, D, device="cuda"))
k = norm(torch.randn(B, Hkv, T, D, device="cuda"))
v = torch.randn(B, Hkv, T, D, device="cuda")
x = torch.randn(B, T, 256, device="cuda")
scale = 0.12
t = torch.arange(T, device="cuda")
blk = t // b
causal = t[None, :] <= t[:, None]

# 1. window mode = dense attention over own block + W/b previous blocks
with torch.device("cuda"):
    mw = ConceptAttention(GPTConfig(**base, attn_mode="window"))
y = mw(q, k, v, x, scale)
local = causal & (blk[None, :] >= blk[:, None] - W // b)
ref = F.scaled_dot_product_attention(q, k, v, attn_mask=local, scale=scale, enable_gqa=True)
ok((y - ref).abs().max().item() < TOL, "window = dense reference", f"max diff {(y - ref).abs().max().item():.1e}")

# 2. concept mode = dense reference with the same pooling, selection and gates
with torch.device("cuda"):
    mc = ConceptAttention(cfgc)
with torch.no_grad():
    mc.pool.normal_(0, 0.5); mc.gate.weight.normal_(0, 0.05)
y = mc(q, k, v, x, scale)
nq = T // b
w = torch.softmax((k.view(B, Hkv, nq, b, D) * mc.pool.view(1, Hkv, 1, 1, D)).sum(-1), -1)
kc = norm((w[..., None] * k.view(B, Hkv, nq, b, D)).sum(3)); vc = (w[..., None] * v.view(B, Hkv, nq, b, D)).sum(3)
kc = torch.cat([kc.new_zeros(B, Hkv, 1, D), kc], 2).repeat_interleave(2, 1)
vc = torch.cat([vc.new_zeros(B, Hkv, 1, D), vc], 2).repeat_interleave(2, 1)
seen = torch.cat([torch.ones(T, 1, dtype=torch.bool, device="cuda"), (torch.arange(nq, device="cuda") + 1)[None] * b <= t[:, None]], 1)
p = torch.softmax((q @ kc.transpose(-1, -2) * scale).masked_fill(~seen, float("-inf")), -1)
y_cmp = p @ vc
score = p[..., 1:].reshape(B, H, nq, b, nq).sum((1, 3))
score = torch.cat([torch.full_like(score[:, :1], float("-inf")), score[:, :-1]], 1)
mask = local[None].repeat(B, 1, 1)
for bi in range(B):
    for i in range(nq):
        elig = [j for j in range(nq) if j < i - W // b]
        if elig:
            s = score[bi, i, elig]
            for jj in s.topk(min(2, len(elig))).indices.tolist():
                j = elig[jj]
                mask[bi, i * b:(i + 1) * b, j * b:(j + 1) * b] = True
mask &= causal
y_loc = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None], scale=scale, enable_gqa=True)
g = torch.sigmoid(mc.gate(x)).view(B, T, 2, H).permute(2, 0, 3, 1)[..., None]
ref = g[0] * y_loc + g[1] * y_cmp
ok((y - ref).abs().max().item() < TOL, "concept = dense reference (pooling, selection, expansion, gates)",
   f"max diff {(y - ref).abs().max().item():.1e}")

# 3. fast weights: chunked parallel form = step-by-step Hebbian recurrence
with torch.device("cuda"):
    fw = FastWeights(GPTConfig(**base, fast_heads=2, fast_dim=32, fast_chunk=128))
with torch.no_grad():
    fw.proj.weight.normal_(0, 0.05); fw.write.weight.normal_(0, 0.05)
xs = torch.randn(1, 512, 256, device="cuda")
y = fw(xs)
qq, kk, vv = fw.qkv(xs).view(1, 512, 3, 2, 32).permute(2, 0, 3, 1, 4)
qq, kk = F.normalize(qq, dim=-1), F.normalize(kk, dim=-1)
vv = vv * torch.sigmoid(fw.write(xs)).transpose(1, 2)[..., None]
lam = torch.sigmoid(fw.decay)[None, :, None, None]
M = torch.zeros(1, 2, 32, 32, device="cuda"); outs = []
for s_ in range(512):
    M = lam * M + kk[:, :, s_, :, None] * vv[:, :, s_, None, :]
    outs.append((qq[:, :, s_, None, :] @ M)[:, :, 0])
ref = fw.proj(torch.stack(outs, 2).transpose(1, 2).reshape(1, 512, 64))
ok((y - ref).abs().max().item() < 1e-3, "fast weights: chunked = Hebbian recurrence", f"max diff {(y - ref).abs().max().item():.1e}")

# 4. whole model: causal, every parameter learns, compiles
args = types.SimpleNamespace(lr_muon=0.05, lr_embed=0.006, lr_scalar=0.02, wd=0.0, ce_chunk=2048)
for mode, extra in (("window", {}), ("concept", {}), ("concept", {"fast_heads": 2, "fast_dim": 32, "fast_chunk": 128})):
    cfg = GPTConfig(**{**base, "attn_mode": mode, "concept_topk": 2, **extra})
    with torch.device("cuda"):
        m = GPT(cfg)
    muon, adam = train.build_optimizers(m, args)
    ids = torch.randint(0, 32763, (2, T + 1), device="cuda")
    for step in range(2):
        m.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16):
            loss = m.loss(m(ids[:, :-1]), ids[:, 1:])
        loss.backward()
        if step == 0:
            muon.step(); adam.step()
    dead = [n for n, p_ in m.named_parameters() if p_.grad is None or p_.grad.abs().sum() == 0]
    name = f"{mode}" + (" + fast weights" if extra else "")
    ok(not dead, f"{name}: every parameter gets gradient", str(dead[:4]))
    m.eval()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        a = m(ids[:, :-1]).float()
        for pos in (100, 700):
            ids2 = ids[:, :-1].clone(); ids2[:, pos] = (ids2[:, pos] + 1) % 32763
            bb = m(ids2).float()
            ok((a[:, :pos] - bb[:, :pos]).abs().max().item() == 0, f"{name}: causal at token {pos}")
    m.train()
    fwd = torch.compile(m)
    with torch.autocast("cuda", torch.bfloat16):
        l2 = m.loss(fwd(ids[:, :-1]), ids[:, 1:])
    ok(torch.isfinite(l2).item(), f"{name}: compiled forward", f"loss {l2.item():.3f}")

# 5. existing checkpoints load unchanged
CKPT = os.path.join(MODEL, "runs", "final_ec32k_gqa", "ckpt.pt")  # an existing run, if this machine has it
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    with torch.device("cuda"):
        old = GPT(GPTConfig(**ck["config"]))
    old.load_state_dict(ck["model"])
    ok(old.cfg.attn_mode == "full" and old.blocks[0].fast is None, "final 12-hour model loads strictly, full attention")
else:
    print("SKIP old checkpoint check: no final_ec32k_gqa here")

# 6. position-free concepts (concept_rope=False)
with torch.device("cuda"):
    mc2 = ConceptAttention(cfgc)
with torch.no_grad():
    mc2.pool.normal_(0, 0.5); mc2.gate.weight.normal_(0, 0.05)
same = (mc2(q, k, v, x, scale) - mc2(q, k, v, x, scale, q, k)).abs().max().item()
ok(same == 0, "passing the same q/k as position-free inputs changes nothing (plumbing)")
cfgn = GPTConfig(**{**base, "attn_mode": "concept", "concept_topk": 2, "concept_rope": False})
with torch.device("cuda"):
    mn = GPT(cfgn)
ids = torch.randint(0, 32763, (2, T + 1), device="cuda")
with torch.autocast("cuda", torch.bfloat16):
    mn.loss(mn(ids[:, :-1]), ids[:, 1:]).backward()
dead = [n for n, p_ in mn.named_parameters() if p_.grad is None]
ok(not dead, "position-free concepts: every parameter gets gradient", str(dead[:3]))
mn.eval()
with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
    a = mn(ids[:, :-1]).float()
    ids2 = ids[:, :-1].clone(); ids2[:, 600] = (ids2[:, 600] + 1) % 32763
    ok((a[:, :600] - mn(ids2).float()[:, :600]).abs().max().item() == 0, "position-free concepts: causal at token 600")
# a concept's key doesn't depend on where its block sits: shifting the positions leaves kc unchanged
blk_attn = mn.blocks[0].attn
with torch.no_grad():
    h = norm(torch.randn(1, T, 256, device="cuda"))
    ctx1 = {"cache": None, "v0": None, "kv": None, "rope_mla": None}
    ctx2 = {"cache": None, "v0": None, "kv": None, "rope_mla": None}
    blk_attn(h, mn.rope_cos[:T], mn.rope_sin[:T], ctx1)
    shifted_cos, shifted_sin = rope_tables(2 * T, 64, 10000.0, device="cuda")
    blk_attn(h, shifted_cos[T:], shifted_sin[T:], ctx2)
    ok(torch.equal(ctx1["k_nope"], ctx2["k_nope"]) and not torch.equal(ctx1["kv"][0], ctx2["kv"][0]),
       "concept keys are position-free while window keys keep their positions")
