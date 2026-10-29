"""Long-context memory for gpt.py: concept memory and fast weights. See PLAN_concept_memory.md.

Concept memory (attn_mode "window" / "concept"): instead of attending to every past token, each
token reads
  1. recent text in full: its own block of concept_block tokens plus the window // concept_block
     blocks before it (a sliding window);
  2. older text as concepts: every completed block is pooled into one key/value (a learned
     attention pooling per head), so T tokens of history become T / concept_block slots;
  3. detail on demand (the hippocampus step): each block of queries picks the concept_topk older
     blocks that the block before it attended to most in (2) (its own queries would let early
     tokens' choice depend on later ones), and reads their original tokens in full alongside the
     window.
(1) + (3) run as one block-sparse FlexAttention call (nothing outside those blocks is computed);
(2) is a small dense attention over the concepts. Per-head gates from the token mix the two.
"window" is (1) alone, the baseline. The choice in (3) is not differentiable (as in DeepSeek's
Native Sparse Attention); the concepts learn through (2).

Fast weights (fast_heads > 0): a Hebbian associative memory, the trainable form of learning
during use. Each head keeps a matrix M that every token writes into, M <- decay * M + w_t k_t v_t^T
(strengthen the link between what this token's key and value), and reads from, y_t = q_t M. The
write strength w_t is chosen per token, the decay is learned per head (each head forgets at its own
rate). M starts empty for every sequence: the model adapts within one conversation and keeps
nothing afterwards, so no text can permanently rewrite it. Computed chunk by chunk in parallel.

Both are for training and full-sequence scoring (cache=None); incremental decoding is later work.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention

_flex = torch.compile(flex_attention, dynamic=False)


def _causal(b, h, q_idx, kv_idx):
    return kv_idx <= q_idx


class ConceptAttention(nn.Module):
    """Window + concepts + expanded blocks over already-projected q [B, H, T, D], k, v [B, Hkv, T, D]."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.b = cfg.concept_block
        self.wb = cfg.window // cfg.concept_block  # previous blocks in the window
        self.topk = cfg.concept_topk if cfg.attn_mode == "concept" else 0
        self.concepts = cfg.attn_mode == "concept"
        # kernel tiles no bigger than the sparse blocks: on Hopper FlexAttention defaults to 128-row query
        # tiles, which a 64-token block can't hold
        t = min(64, self.b)
        self.tiles = {"BLOCK_M": t, "BLOCK_N": t,
                      "bwd_BLOCK_M1": t, "bwd_BLOCK_N1": t, "bwd_BLOCK_M2": t, "bwd_BLOCK_N2": t}
        H, Hkv, D = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        if self.concepts:
            self.pool = nn.Parameter(torch.zeros(Hkv * D))  # pooling query per kv head (0 = mean to start); 1-D: AdamW
            self.gate = nn.Linear(cfg.d_model, 2 * H)       # per-head weights of local and concept branches
            with torch.no_grad():
                self.gate.weight.zero_()
                self.gate.bias.copy_(torch.cat([torch.full((H,), 3.0), torch.zeros(H)]))  # ~0.95 local, 0.5 concepts

    @torch.compiler.disable  # built eagerly: Inductor fails to compile BlockMask construction (torch 2.14)
    def block_mask(self, B, T, device, selected=None, n_valid=None):
        """kv block lists per query block: the window blocks (own block first), then selected blocks.
        Window-only masks never change for a given shape, so they're cached."""
        if selected is None:
            key = (B, T, str(device))
            if getattr(self, "_cached", (None,))[0] != key:
                self._cached = (key, self._build(B, T, device, None, None))
            return self._cached[1]
        return self._build(B, T, device, selected, n_valid)

    def _build(self, B, T, device, selected, n_valid):
        nq = T // self.b
        i = torch.arange(nq, device=device)
        local = (i[:, None] - torch.arange(self.wb + 1, device=i.device)[None, :]).clamp(min=0)  # [nq, wb+1]
        n_local = torch.minimum(i, torch.tensor(self.wb, device=i.device)) + 1
        idx = local[None].expand(B, -1, -1)
        num = n_local[None].expand(B, -1)
        if selected is not None:
            idx = torch.cat([idx, selected], -1)
            num = num + n_valid
        assert idx.size(-1) <= nq, "window + expanded blocks exceed the sequence"
        idx = F.pad(idx, (0, nq - idx.size(-1)))  # FlexAttention wants one slot per kv block; entries past num are unused
        return BlockMask.from_kv_blocks(num[:, None].int().contiguous(), idx[:, None].int().contiguous(),
                                        BLOCK_SIZE=self.b, mask_mod=_causal)

    def forward(self, q, k, v, x, scale):
        B, H, T, D = q.shape
        Hkv, b = k.size(1), self.b
        assert T % b == 0, f"sequence length {T} must be a multiple of concept_block {b}"
        nq = T // b
        gqa = Hkv != H
        if not self.concepts:
            return _flex(q, k, v, block_mask=self.block_mask(B, T, q.device), scale=scale, enable_gqa=gqa,
                         kernel_options=self.tiles)

        # (2) concepts: pool each block's keys and values, attend to completed blocks (plus a null slot)
        kb, vb = k.view(B, Hkv, nq, b, D), v.view(B, Hkv, nq, b, D)
        w = torch.softmax((kb.float() * self.pool.view(1, Hkv, 1, 1, D)).sum(-1), dim=-1)  # [B, Hkv, nq, b]
        kc = F.rms_norm((w[..., None] * kb.float()).sum(3), (D,))
        vc = (w[..., None] * vb.float()).sum(3)
        null = kc.new_zeros(B, Hkv, 1, D)
        kc, vc = torch.cat([null, kc], 2), torch.cat([null, vc], 2)                         # [B, Hkv, nq+1, D]
        rep = H // Hkv
        kc, vc = kc.repeat_interleave(rep, 1), vc.repeat_interleave(rep, 1)
        t = torch.arange(T, device=q.device)
        seen = torch.cat([torch.ones(T, 1, dtype=torch.bool, device=q.device),
                          (torch.arange(nq, device=q.device)[None, :] + 1) * b <= t[:, None]], 1)  # block done before t
        s = (q.float() @ kc.transpose(-1, -2)) * scale
        p = torch.softmax(s.masked_fill(~seen, float("-inf")), dim=-1)                     # [B, H, T, nq+1]
        y_cmp = (p @ vc).to(q.dtype)

        # (3) per query block, the topk older blocks it attended to most, outside its window
        selected = n_valid = None
        if self.topk:
            with torch.no_grad():
                score = p[..., 1:].view(B, H, nq, b, nq).sum((1, 3))                        # [B, nq, nq]
                # each query block chooses by what the block before it attended to: summing its own
                # queries would let a token's choice depend on later tokens in the same block
                score = torch.cat([torch.full_like(score[:, :1], float("-inf")), score[:, :-1]], 1)
                qi = torch.arange(nq, device=q.device)
                eligible = qi[None, :] < (qi[:, None] - self.wb)                            # wholly before the window
                score = score.masked_fill(~eligible, float("-inf"))
                k_sel = min(self.topk, nq)
                selected = score.topk(k_sel, dim=-1).indices                                # [B, nq, k]
                n_valid = (qi - self.wb).clamp(0, k_sel)[None].expand(B, -1)
        y_loc = _flex(q, k, v, block_mask=self.block_mask(B, T, q.device, selected, n_valid), scale=scale, enable_gqa=gqa,
                      kernel_options=self.tiles)

        g = torch.sigmoid(self.gate(x).float()).view(B, T, 2, H).permute(2, 0, 3, 1)[..., None]  # [2, B, H, T, 1]
        return (g[0] * y_loc.float() + g[1] * y_cmp.float()).to(q.dtype)


class FastWeights(nn.Module):
    """Hebbian fast-weight memory: per head, M <- decay * M + w_t k_t v_t^T, read y_t = q_t M."""

    def __init__(self, cfg):
        super().__init__()
        self.H, self.d, self.chunk = cfg.fast_heads, cfg.fast_dim, cfg.fast_chunk
        self.qkv = nn.Linear(cfg.d_model, 3 * self.H * self.d, bias=False)
        self.write = nn.Linear(cfg.d_model, self.H)
        self.decay = nn.Parameter(torch.linspace(3.0, 8.0, self.H))  # sigmoid -> forgetting horizons ~20 to ~3000 tokens
        self.proj = nn.Linear(self.H * self.d, cfg.d_model, bias=False)  # zero-initialised by GPT.reset_parameters
        with torch.no_grad():
            self.write.weight.zero_()
            self.write.bias.fill_(-2.0)  # write strength ~0.12 to start

    def row_splits(self):
        return [self.H * self.d] * 3

    def forward(self, x):
        B, T, _ = x.shape
        H, d, C = self.H, self.d, min(self.chunk, T)
        assert T % C == 0, f"sequence length {T} must be a multiple of fast_chunk {C}"
        q, k, v = self.qkv(x).float().view(B, T, 3, H, d).permute(2, 0, 3, 1, 4)  # each [B, H, T, d]
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        v = v * torch.sigmoid(self.write(x).float()).transpose(1, 2)[..., None]   # per-token write strength
        a = F.logsigmoid(self.decay)[None, :, None, None]                         # log decay per head
        pos = torch.arange(C, device=x.device, dtype=torch.float32)
        rel = pos[:, None] - pos[None, :]
        intra_decay = torch.where(rel >= 0, torch.exp(a * rel.clamp(min=0)), torch.zeros_like(rel))  # [1, H, C, C]
        read_decay = torch.exp(a * (pos + 1)[:, None])                            # state decays into chunk position
        write_decay = torch.exp(a * (C - 1 - pos)[:, None])                       # token's write, as seen at chunk end
        M = q.new_zeros(B, H, d, d)
        out = []
        for c0 in range(0, T, C):
            qc, kc, vc = q[:, :, c0:c0 + C], k[:, :, c0:c0 + C], v[:, :, c0:c0 + C]
            y = ((qc @ kc.transpose(-1, -2)) * intra_decay) @ vc + (qc * read_decay) @ M
            M = torch.exp(a * C) * M + (kc * write_decay).transpose(-1, -2) @ vc
            out.append(y)
        y = torch.cat(out, 2).transpose(1, 2).reshape(B, T, H * d)
        return self.proj(y.to(x.dtype))
