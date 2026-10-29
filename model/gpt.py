"""GPT for ec200k-family vocabularies, built to trade weight size against KV cache.

Memory at inference is weights plus KV cache, and this model exposes a knob for each:

- Vocabulary (n_ranks): the embedding is vocab x d_model. A rank prefix of ec200k is an exact
  smaller BPE vocabulary with fewer weights but more tokens per byte of text, so more KV cache
  and more forward passes per byte.
- n_kv_head (GQA/MQA): KV cache scales with n_kv_head; queries keep all n_head heads.
- kv_share (cross-layer attention): consecutive groups of kv_share layers share one layer's
  K and V, so the cache scales with n_layer / kv_share.
- mla_rank (multi-head latent attention, DeepSeek-V2): each layer caches one latent of
  mla_rank values plus a shared mla_rope_dim RoPE key per token, and K and V for all heads are
  up-projected from it. The cache scales with mla_rank + mla_rope_dim instead of 2 x heads x head_dim.

Architecture: pre-norm blocks with RMSNorm (no learned gains), RoPE, QK-norm, ReLU^2 MLP,
tied input/output embeddings, soft-capped logits. Optional, both from small-model speedruns:
- value_residual: each K/V layer's values are mixed with the first layer's values using two
  learned scalars, which gives deep layers a direct path to token-level information.
- unet: the outputs of the first half of the blocks are added (learned scalar weights) to
  the inputs of the mirrored blocks in the second half.

The vocabulary is padded to a multiple of 64 for tensor-core GEMMs; padding ids never appear
as targets and are masked in generation. forward() returns final hidden states; the loss
runs through fused_ce, which never materializes the [tokens, vocab] logits.

KV caches: every attention layer that makes K/V calls cache.extend(kv_index, q, k, v), which
returns what to attend over. Training passes no cache; generation, CUDA-graph decoding
(fastdecode.py) and KV-trimming experiments (cache_lab.py) each supply their own.
"""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from fused_ce import linear_cross_entropy, per_token_loss
from tokenizer import FULL_RANKS, SPECIAL_NAMES
from memory import ConceptAttention, FastWeights


@dataclass
class GPTConfig:
    n_ranks: int = FULL_RANKS  # regular tokens; the specials follow them
    seq_len: int = 1024
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 0  # 0 = n_head (plain multi-head attention)
    kv_share: int = 1   # layers per shared K/V group (1 = every layer has its own)
    d_model: int = 768
    mlp_ratio: int = 4
    rope_base: float = 10000.0
    logit_softcap: float = 15.0
    attn_scale: float = 0.12  # q and k are RMS-normalized, so this sets attention sharpness
    value_residual: bool = False
    unet: bool = False
    mla_rank: int = 0      # 0 = standard attention
    mla_rope_dim: int = 32
    # long-context memory (memory.py, PLAN_concept_memory.md): "full" attention over every past token,
    # "window" = only the last `window` tokens, "concept" = window + one learned summary per older
    # block of concept_block tokens + the concept_topk older blocks expanded back to full detail
    attn_mode: str = "full"
    window: int = 512
    concept_block: int = 64
    concept_topk: int = 4
    fast_heads: int = 0    # Hebbian fast-weight memory heads per block (0 = none); reset every sequence
    fast_dim: int = 64
    fast_chunk: int = 256
    # signal / no-signal routing through depth: every depth_every-th block (not the first) has a router;
    # a token is processed by that block only if its router fires, else it passes on the residual. A
    # threshold, adjusted after each step, keeps about depth_capacity of the tokens firing (0 = off)
    depth_every: int = 0
    depth_capacity: float = 0.25
    depth_step: float = 0.2      # threshold control gain (0.05 let blocks go silent for 100+ steps)

    def __post_init__(self):
        self.n_kv_head = self.n_kv_head or self.n_head
        assert self.d_model % self.n_head == 0 and self.n_head % self.n_kv_head == 0
        assert self.attn_mode in ("full", "window", "concept")
        assert not self.depth_every or (self.attn_mode == "full" and self.kv_share == 1 and not self.mla_rank
                                         and not self.fast_heads), "depth routing needs plain full attention"
        assert self.attn_mode == "full" or (not self.mla_rank and self.window % self.concept_block == 0)
        if self.mla_rank:
            assert self.kv_share == 1 and self.n_kv_head == self.n_head, "MLA replaces GQA and K/V sharing"
            assert not self.value_residual, "MLA rebuilds past values from latents, so they can't be mixed"

    @property
    def head_dim(self):
        return self.d_model // self.n_head

    @property
    def n_vocab(self):  # real ids
        return self.n_ranks + len(SPECIAL_NAMES)

    @property
    def vocab_size(self):  # embedding rows, padded to a multiple of 64
        return -(-self.n_vocab // 64) * 64

    @property
    def n_kv_layers(self):
        return -(-self.n_layer // self.kv_share)

    @property
    def cache_layout(self):
        """(heads, values per head) stored per token in each K/V layer's cache."""
        return (1, self.mla_rank + self.mla_rope_dim) if self.mla_rank else (self.n_kv_head, self.head_dim)


PRESETS = {
    "tiny": dict(n_layer=6, n_head=3, d_model=384),     # smoke tests
    "small": dict(n_layer=12, n_head=6, d_model=768),   # GPT-2 small shape
    "medium": dict(n_layer=16, n_head=8, d_model=1024),
    "deep": dict(n_layer=24, n_head=8, d_model=1024),
}


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def rope_tables(seq_len, head_dim, base, device=None):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32, device=device), inv_freq)
    return freqs.cos(), freqs.sin()  # [T, head_dim / 2]


def apply_rope(x, cos, sin):
    # x: [B, T, H, D]; rotate the two halves of the head dimension. cos/sin: [T, D/2] shared, or [B, T, D/2]
    # when each sequence has its own positions (tokens routed into a block, see Block)
    x1, x2 = x.float().chunk(2, dim=-1)
    cos, sin = (cos[None, :, None, :], sin[None, :, None, :]) if cos.dim() == 2 else (cos[:, :, None, :], sin[:, :, None, :])
    return torch.cat((x1 * cos + x2 * sin, x2 * cos - x1 * sin), dim=-1).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg, layer):
        super().__init__()
        self.cfg = cfg
        self.makes_kv = layer % cfg.kv_share == 0
        self.kv_index = layer // cfg.kv_share
        self.q_dim, self.kv_dim = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
        # One fused projection. Muon orthogonalizes its q, k and v row blocks separately.
        self.qkv = nn.Linear(cfg.d_model, sum(self.row_splits()), bias=False)
        if cfg.mla_rank:  # latent -> per-head K (non-RoPE part) and V
            self.kv_up = nn.Linear(cfg.mla_rank, cfg.n_head * (cfg.head_dim - cfg.mla_rope_dim) + self.q_dim,
                                   bias=False)
        self.proj = nn.Linear(self.q_dim, cfg.d_model, bias=False)
        self.v_mix = nn.Parameter(torch.tensor([0.5, 0.5])) if cfg.value_residual and self.makes_kv and layer else None
        # Per-layer hook applied to K and V after RoPE; analyze.py uses it to simulate cache quantization.
        self.kv_transform = None
        self.memory = ConceptAttention(cfg) if cfg.attn_mode != "full" else None

    def row_splits(self):
        c = self.cfg
        if c.mla_rank:
            return [self.q_dim, c.mla_rank, c.mla_rope_dim]
        return [self.q_dim, self.kv_dim, self.kv_dim] if self.makes_kv else [self.q_dim]

    def mla_forward(self, x, rope, ctx):
        c = self.cfg
        B, T, _ = x.shape
        H, D, R = c.n_head, c.head_dim, c.mla_rope_dim
        q, lat, kr = self.qkv(x).split(self.row_splits(), dim=-1)
        cos_r, sin_r = rope
        kr = apply_rope(kr.view(B, T, 1, R), cos_r, sin_r)
        latent = torch.cat([norm(lat), kr.view(B, T, R).type_as(lat)], dim=-1)[:, None]  # [B, 1, T, rank + R]
        if self.kv_transform is not None:
            latent = self.kv_transform(latent)
        q = q.view(B, T, H, D)
        q = torch.cat([q[..., :D - R], apply_rope(q[..., D - R:], cos_r, sin_r)], dim=-1)
        q = norm(q).transpose(1, 2)
        if ctx["cache"] is None:
            lat_all, mask, causal = latent, None, True
        else:
            lat_all, _, mask, causal = ctx["cache"].extend(self.kv_index, q, latent, latent)
        S = lat_all.size(2)
        lat_all = lat_all[:, 0]
        k_nope, v = self.kv_up(lat_all[..., :c.mla_rank]).split([H * (D - R), H * D], dim=-1)
        k_rope = lat_all[..., None, c.mla_rank:].expand(B, S, H, R).type_as(k_nope)
        k = torch.cat([k_nope.view(B, S, H, D - R), k_rope], dim=-1)
        k, v = norm(k).transpose(1, 2), v.view(B, S, H, D).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal and mask is None and T > 1,
                                           scale=c.attn_scale)
        return self.proj(y.transpose(1, 2).reshape(B, T, self.q_dim))

    def forward(self, x, cos, sin, ctx):
        c = self.cfg
        if c.mla_rank:
            return self.mla_forward(x, ctx["rope_mla"], ctx)
        B, T, _ = x.shape
        out = self.qkv(x)
        if self.makes_kv:
            q, k, v = out.split(self.row_splits(), dim=-1)
            k = apply_rope(norm(k.view(B, T, c.n_kv_head, c.head_dim)), cos, sin).transpose(1, 2)
            v = v.view(B, T, c.n_kv_head, c.head_dim).transpose(1, 2)
            if c.value_residual:
                if ctx["v0"] is None:
                    ctx["v0"] = v
                else:
                    v = self.v_mix[0] * v + self.v_mix[1] * ctx["v0"]
            if self.kv_transform is not None:
                k, v = self.kv_transform(k), self.kv_transform(v)
        else:
            q = out
        q = apply_rope(norm(q.view(B, T, c.n_head, c.head_dim)), cos, sin).transpose(1, 2)
        if self.makes_kv:
            # (keys, values, attention mask or None, causal?) shared with the layers that reuse them
            ctx["kv"] = (k, v, None, True) if ctx["cache"] is None else ctx["cache"].extend(self.kv_index, q, k, v)
        K, V, mask, causal = ctx["kv"]
        if self.memory is not None:
            assert ctx["cache"] is None, "window/concept attention has no incremental decoding yet"
            y = self.memory(q, K, V, x, c.attn_scale)
        else:
            y = F.scaled_dot_product_attention(q, K, V, attn_mask=mask, is_causal=causal and mask is None and T > 1,
                                               scale=c.attn_scale, enable_gqa=c.n_kv_head != c.n_head)
        return self.proj(y.transpose(1, 2).reshape(B, T, self.q_dim))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.mlp_ratio * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.mlp_ratio * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        return self.proj(F.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, cfg, layer):
        super().__init__()
        self.cfg = cfg
        self.attn = Attention(cfg, layer)
        self.fast = FastWeights(cfg) if cfg.fast_heads else None
        self.mlp = MLP(cfg)
        self.router = None
        if cfg.depth_every and layer > 0 and layer % cfg.depth_every == cfg.depth_every - 1:
            self.router = nn.Linear(cfg.d_model, 1, bias=False)
            self.register_buffer("depth_bias", torch.zeros(()))   # threshold, set by GPT.update_after_step
            self.register_buffer("wanted", torch.zeros(()))       # share of tokens whose router fired
            self.register_buffer("fired", torch.zeros(()))        # ... and that fit the block's capacity

    def routed_forward(self, x, cos, sin, ctx):
        """Only the tokens whose router fires run through this block (in time order, up to capacity);
        attention inside sees just those tokens, at their real positions. The block's change is scaled by
        the router's sigmoid so the router learns which tokens are worth processing. Whether a token gets
        processed depends only on it and earlier tokens, so this stays causal."""
        B, T, D = x.shape
        C = max(1, math.ceil(self.cfg.depth_capacity * T))
        # scale-free score: the router's weight is normalised, so its output can't outgrow the threshold
        # (raw router logits grew faster than the threshold could follow and blocks went quiet)
        w = self.router.weight.float()
        s = (norm(x).float() @ (w / (w.norm() + 1e-6)).t()).squeeze(-1) * 4.0  # [B, T], about -4..4
        want = (s + self.depth_bias) > 0
        slot = want.long().cumsum(1) - 1
        keep = want & (slot < C)
        dump = torch.where(keep, slot, torch.full_like(slot, C))
        idx = torch.full((B, C + 1), T, device=x.device, dtype=torch.long)
        idx.scatter_(1, dump, torch.arange(T, device=x.device).expand(B, T))
        idx = idx[:, :C]                                                     # [B, C] token in each slot (T = empty)
        b = torch.arange(B, device=x.device)[:, None]
        xs = torch.cat([x, x.new_zeros(B, 1, D)], 1)[b, idx]                 # [B, C, D]
        pos = idx.clamp(max=T - 1)
        v0 = ctx["v0"]
        if v0 is not None:  # value residual: the first layer's values at the same tokens
            v0p = torch.cat([v0, v0.new_zeros(*v0.shape[:2], 1, v0.size(-1))], 2)
            ctx["v0"] = v0p.gather(2, idx[:, None, :, None].expand(B, v0.size(1), C, v0.size(-1)))
        y = xs + self.attn(norm(xs), cos[pos], sin[pos], ctx)
        y = y + self.mlp(norm(y))
        ctx["v0"] = v0
        g = torch.cat([torch.sigmoid(s), s.new_zeros(B, 1)], 1)[b, idx]     # router weight per slot
        out = torch.cat([x, x.new_zeros(B, 1, D)], 1)
        out.scatter_add_(1, idx[..., None].expand(B, C, D), ((y - xs).float() * g[..., None]).to(x.dtype))
        self.wanted.copy_(want.float().mean().detach())
        self.fired.copy_(keep.float().mean().detach())
        return out[:, :T]

    def forward(self, x, cos, sin, ctx):
        if self.router is not None and ctx["cache"] is None:
            return self.routed_forward(x, cos, sin, ctx)
        h = norm(x)
        x = x + self.attn(h, cos, sin, ctx)
        if self.fast is not None:  # Hebbian memory reads the same input, in parallel with attention
            assert ctx["cache"] is None, "fast weights have no incremental decoding yet"
            x = x + self.fast(h)
        return x + self.mlp(norm(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.n_skip = cfg.n_layer // 2 if cfg.unet else 0
        self.skip_w = nn.Parameter(torch.ones(self.n_skip)) if self.n_skip else None
        cos, sin = rope_tables(cfg.seq_len, cfg.head_dim, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        if cfg.mla_rank:
            cos, sin = rope_tables(cfg.seq_len, cfg.mla_rope_dim, cfg.rope_base)
            self.register_buffer("rope_cos_mla", cos, persistent=False)
            self.register_buffer("rope_sin_mla", sin, persistent=False)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.normal_(self.wte.weight, std=0.02)
        for name, p in self.blocks.named_parameters():
            if p.dim() < 2:
                continue  # learned scalars keep their constructor values
            if name.endswith("proj.weight"):
                nn.init.zeros_(p)  # each block starts as the identity
            else:
                bound = math.sqrt(3) * 0.5 / math.sqrt(p.shape[1])
                nn.init.uniform_(p, -bound, bound)

    def forward(self, idx, cache=None, pos=0):
        """Final hidden states, [B, T, d_model] in bf16. pos (int or [1] tensor) is the first position."""
        T = idx.size(1)
        if isinstance(pos, torch.Tensor):
            p = pos + torch.arange(T, device=idx.device)
            take = lambda t: t.index_select(0, p)
        else:
            take = lambda t: t[pos:pos + T]
        cos, sin = take(self.rope_cos), take(self.rope_sin)
        rope_mla = (take(self.rope_cos_mla), take(self.rope_sin_mla)) if self.cfg.mla_rank else None
        ctx = {"cache": cache, "v0": None, "kv": None, "rope_mla": rope_mla}  # per-forward state shared by the layers
        x = norm(self.wte(idx)).bfloat16()
        skips = []
        L = len(self.blocks)
        for i, block in enumerate(self.blocks):
            if i >= L - self.n_skip:
                x = x + self.skip_w[L - 1 - i] * skips.pop()
            x = block(x, cos, sin, ctx)
            if i < self.n_skip:
                skips.append(x)
        return norm(x).bfloat16()

    def loss(self, hidden, targets, chunk=2048):
        return linear_cross_entropy(hidden.flatten(0, 1), self.wte.weight, targets.flatten(),
                                    self.cfg.logit_softcap, chunk)

    def token_losses(self, hidden, targets, chunk=2048):
        return per_token_loss(hidden.flatten(0, 1), self.wte.weight, targets.flatten(),
                              self.cfg.logit_softcap, chunk).view_as(targets)

    def logits(self, hidden):
        """Soft-capped fp32 logits over the real vocabulary (padding masked to -inf)."""
        logits = (hidden.to(self.wte.weight.dtype) @ self.wte.weight.t()).float()
        cap = self.cfg.logit_softcap
        if cap:
            logits = cap * torch.tanh(logits / cap)
        logits[..., self.cfg.n_vocab:] = float("-inf")
        return logits

    def muon_splits(self):
        """Row blocks of each hidden matrix that Muon should orthogonalize separately."""
        c = self.cfg
        splits = {b.attn.qkv.weight: b.attn.row_splits() for b in self.blocks}
        splits.update({b.fast.qkv.weight: b.fast.row_splits() for b in self.blocks if b.fast is not None})
        if c.mla_rank:
            for b in self.blocks:
                splits[b.attn.kv_up.weight] = [c.n_head * (c.head_dim - c.mla_rope_dim), c.n_head * c.head_dim]
        return splits

    def matrix_params(self):
        return [p for p in self.blocks.parameters() if p.dim() == 2]

    @torch.no_grad()
    def update_after_step(self, parts=None):
        """Depth routing: nudge each routed block's threshold so about depth_capacity of tokens fire."""
        c = self.cfg
        for b in self.blocks:
            if b.router is not None:
                b.depth_bias += c.depth_step * (c.depth_capacity - b.wanted) / c.depth_capacity

    def depth_stats(self):
        fired = [b.fired.item() for b in self.blocks if b.router is not None]
        return f" | depth routing: {min(fired):.0%}-{max(fired):.0%} of tokens processed" if fired else ""

    def scalar_params(self):
        return [p for p in self.parameters() if p.dim() < 2]

    # ---- accounting --------------------------------------------------------

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def embedding_params(self):
        return self.wte.weight.numel()

    def kv_bytes_per_token(self, bytes_per_value=2):
        c = self.cfg
        heads, dim = c.cache_layout
        per_layer = heads * dim if c.mla_rank else 2 * heads * dim  # MLA caches one latent, not K and V
        return c.n_kv_layers * per_layer * bytes_per_value

    def flops_per_token(self):
        """Training FLOPs per token: 6 x matmul params (incl. the LM head) + causal attention."""
        c = self.cfg
        matmul = sum(p.numel() for p in self.matrix_params()) + c.vocab_size * c.d_model
        for b in self.blocks:  # a routed block only computes on its share of the tokens
            if b.router is not None:
                matmul -= (1 - c.depth_capacity) * sum(p.numel() for p in b.parameters() if p.dim() == 2)
        keys = c.seq_len  # keys each query attends to
        if c.attn_mode != "full":
            keys = c.window + c.concept_block
            if c.attn_mode == "concept":
                keys += c.concept_topk * c.concept_block + c.seq_len // c.concept_block
        fast = c.n_layer * c.fast_heads * c.fast_dim * (c.fast_chunk + c.fast_dim) if c.fast_heads else 0
        return 6 * matmul + 6 * c.n_layer * keys * c.n_head * c.head_dim + 6 * fast

    def config_dict(self):
        return asdict(self.cfg)


class GrowingCache:
    """KV cache for incremental generation: preallocated buffers, a prompt prefill at
    position 0, then any number of tokens at a time. Call advance(n) after each forward."""

    def __init__(self, model, batch, max_len, dtype=torch.bfloat16):
        c = model.cfg
        shape = (batch, c.cache_layout[0], max_len, c.cache_layout[1])
        dev = model.wte.weight.device
        self.k = [torch.zeros(shape, dtype=dtype, device=dev) for _ in range(c.n_kv_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=dev) for _ in range(c.n_kv_layers)]
        self.pos = 0

    def extend(self, i, q, k, v):
        B, _, T, _ = k.shape
        self.k[i][:B, :, self.pos:self.pos + T] = k
        self.v[i][:B, :, self.pos:self.pos + T] = v
        end = self.pos + T
        if T > 1 and self.pos > 0:  # several new tokens after a prefix: causal mask offset by pos
            mask = torch.ones(T, end, dtype=torch.bool, device=k.device).tril(self.pos)
            return self.k[i][:B, :, :end], self.v[i][:B, :, :end], mask, False
        return self.k[i][:B, :, :end], self.v[i][:B, :, :end], None, T > 1

    def advance(self, n):
        self.pos += n
