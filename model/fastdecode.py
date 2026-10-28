"""Fast decoding: bf16 weights, a static KV cache, and a CUDA-graphed decode step.

Eager decoding launches ~15 kernels per layer per token, and on Windows (WDDM) the launch
overhead alone outweighs the math for a model this size. Here one decode step is compiled
with CUDA graphs (torch.compile mode="reduce-overhead") over a fixed-size cache, so each token
costs about what the hardware needs: reading the weights once plus the KV cache so far. Those
are exactly the two quantities the weight-size vs cache trade-off controls.

    dec = StaticDecoder(model, kv_window=1024)
    new_ids = dec.generate(prompt_ids, 256)            # greedy

Streaming with bounded memory: the KV cache is a ring buffer of kv_window slots, so once a
generation outgrows it the oldest tokens drop out of attention. With RoPE and QK-norm the
attention scores depend only on relative positions, so generation can run past the trained
context as long as the window stays within it. (cache_lab.py measured plain truncation as
good as attention-score selection or attention sinks for this model.)

With neural_cache (the dict cache_lab.py --save_gate writes), each step also looks up the
current final hidden state among the last nc_len ones and mixes in the tokens that followed
the best matches, gated per token. That brings back long-range copying the trimmed KV cache
loses, for nc_dim values per token instead of a full layer stack of K and V.
"""
import copy
import math
import time

import torch
import torch.nn.functional as F

from gpt import rope_tables


class StaticCache:
    """Ring buffer of `window` K/V slots per layer. A prompt is prefilled eagerly at position 0;
    each decode step writes one token at slot pos % window. Every step has identical shapes,
    so it can replay as a CUDA graph."""

    def __init__(self, model, batch, window, dtype=torch.bfloat16):
        c = model.cfg
        shape = (batch, c.cache_layout[0], window, c.cache_layout[1])
        self.k = [torch.zeros(shape, dtype=dtype, device="cuda") for _ in range(c.n_kv_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device="cuda") for _ in range(c.n_kv_layers)]
        for t in self.k + self.v:
            torch._dynamo.mark_static_address(t)
        self.window = window
        self.slots = torch.arange(window, device="cuda")
        self.pos = None  # [1] tensor during decode steps

    def extend(self, i, q, k, v):
        T = k.size(2)
        if T > 1:  # prefill at position 0
            self.k[i][:, :, :T] = k
            self.v[i][:, :, :T] = v
            return self.k[i][:, :, :T], self.v[i][:, :, :T], None, True
        slot = self.pos % self.window
        self.k[i].index_copy_(2, slot, k)
        self.v[i].index_copy_(2, slot, v)
        # slots written so far; once the ring has wrapped, all of them are inside the window
        return self.k[i], self.v[i], (self.slots <= self.pos).view(1, 1, 1, -1), False


class StaticDecoder:
    def __init__(self, model, kv_window=None, batch=1, compile=True, neural_cache=None, nc_len=4096,
                 max_positions=1 << 16):
        c = model.cfg
        self.cfg, self.batch = c, batch
        self.window = kv_window or c.seq_len
        assert self.window <= c.seq_len, "the KV window must fit in the trained context"
        # bf16 copy of the weights (2 bytes read per parameter per token). RoPE tables stay fp32
        # and are extended so a stream can run past seq_len.
        self.m = copy.deepcopy(model).bfloat16().eval()
        self.m.rope_cos, self.m.rope_sin = rope_tables(max_positions, c.head_dim, c.rope_base, "cuda")
        if c.mla_rank:
            self.m.rope_cos_mla, self.m.rope_sin_mla = rope_tables(max_positions, c.mla_rope_dim, c.rope_base,
                                                                   "cuda")
        self.max_positions = max_positions
        self.cache = StaticCache(self.m, batch, self.window)
        self.nc = neural_cache
        if self.nc:
            dim = self.nc["nc_dim"] or c.d_model
            self.nc_proj = None
            if self.nc["nc_dim"]:  # the same random projection cache_lab used
                g = torch.Generator(device="cuda").manual_seed(0)
                self.nc_proj = torch.randn(c.d_model, dim, device="cuda", generator=g) / math.sqrt(dim)
            self.nc_len = nc_len
            self.nc_h = torch.zeros(batch, nc_len, dim, dtype=torch.bfloat16, device="cuda")
            self.nc_next = torch.zeros(batch, nc_len, dtype=torch.long, device="cuda")
            self.nc_pos = torch.full((nc_len,), -1, dtype=torch.long, device="cuda")  # position in each slot
            for t in (self.nc_h, self.nc_next, self.nc_pos):
                torch._dynamo.mark_static_address(t)
        self.step = (torch.compile(self._step, mode="reduce-overhead", fullgraph=True, dynamic=False)
                     if compile else self._step)

    # ---- neural cache ------------------------------------------------------

    def _nc_key(self, h):
        return F.normalize(h.float() if self.nc_proj is None else h.float() @ self.nc_proj, dim=-1)

    def _nc_probs(self, logits, h, pos):
        """Mix the model's distribution with the neural cache's for the token after position pos.
        h: [B, d] final hidden state at pos; stored entries before pos know their next token."""
        sim = torch.einsum("bld,bd->bl", self.nc_h.float(), self._nc_key(h))
        valid = (self.nc_pos >= 0) & (self.nc_pos < pos)
        sim = sim.masked_fill(~valid, float("-inf"))
        maxcos = sim.amax(-1).clamp(min=-1)
        p = torch.softmax(sim * self.nc["theta"], -1).nan_to_num(0.0)
        p20 = torch.softmax(sim * 20.0, -1).nan_to_num(0.0)
        top_tok = self.nc_next.gather(1, p20.argmax(-1, keepdim=True))
        top = (p20 * (self.nc_next == top_tok)).sum(-1)
        a, b, c = self.nc["gate"]
        g = (torch.sigmoid(a * maxcos + b * top + c) * valid.any().float())[:, None]
        p_cache = torch.zeros_like(logits).scatter_add_(1, self.nc_next, p)
        return (1 - g) * torch.softmax(logits, -1) + g * p_cache

    # ---- decoding ----------------------------------------------------------

    @torch.no_grad()
    def _step(self, tok, pos):
        """tok: [B, 1] ids at position pos ([1] tensor). Returns fp32 logits [B, vocab], or
        log-probabilities when the neural cache is on."""
        self.cache.pos = pos
        h = self.m(tok, self.cache, pos)[:, -1]
        logits = self.m.logits(h)
        if not self.nc:
            return logits
        self.nc_next.index_copy_(1, (pos - 1) % self.nc_len, tok)  # tok followed position pos - 1
        probs = self._nc_probs(logits, h, pos)
        slot = pos % self.nc_len
        self.nc_h.index_copy_(1, slot, self._nc_key(h)[:, None].to(self.nc_h.dtype))
        self.nc_pos.index_copy_(0, slot, pos)
        return probs.clamp(min=1e-30).log()

    @torch.no_grad()
    def prefill(self, idx):
        """Fill the caches from a prompt ([B, T], T <= kv_window); returns next-token logits."""
        T = idx.size(1)
        h = self.m(idx, self.cache, 0)
        logits = self.m.logits(h[:, -1])
        if not self.nc:
            return logits
        self.nc_pos.fill_(-1)
        n = min(T, self.nc_len)
        positions = torch.arange(T - n, T, device="cuda")
        slots = positions % self.nc_len
        self.nc_h.index_copy_(1, slots, self._nc_key(h[:, T - n:]).to(self.nc_h.dtype))
        self.nc_pos.index_copy_(0, slots, positions)
        self.nc_next.index_copy_(1, slots[:-1], idx[:, T - n + 1:])  # position p was followed by idx[p + 1]
        last = torch.full((1,), T - 1, dtype=torch.long, device="cuda")
        return self._nc_probs(logits, h[:, -1], last).clamp(min=1e-30).log()

    @torch.no_grad()
    def generate(self, prompt_ids, n_new, temperature=0.0, top_k=0, stop_id=None, on_token=None):
        """Decode up to n_new tokens after prompt_ids ([T] or [B, T]). With on_token (batch 1),
        each token is passed to it as it is produced and stop_id ends generation; otherwise
        there is no per-token host sync."""
        idx = torch.as_tensor(prompt_ids, dtype=torch.long, device="cuda")
        idx = idx.expand(self.batch, -1) if idx.dim() == 1 else idx
        idx = idx[:, -self.window:]  # the prompt must fit the KV window
        T = idx.size(1)
        assert T + n_new <= self.max_positions
        logits = self.prefill(idx)
        out = torch.empty(self.batch, n_new, dtype=torch.long, device="cuda")
        for i in range(n_new):
            if temperature > 0:
                logits = logits / temperature
                if top_k:
                    logits = logits.masked_fill(logits < torch.topk(logits, top_k).values[:, -1:], float("-inf"))
                tok = torch.multinomial(torch.softmax(logits, -1), 1)
            else:
                tok = logits.argmax(-1, keepdim=True)
            out[:, i:i + 1] = tok
            if on_token is not None:
                t = int(tok[0, 0])
                if t == stop_id:
                    return out[:, :i]
                on_token(t)
            logits = self.step(tok, torch.full((1,), T + i, dtype=torch.long, device="cuda")).clone()
        return out

    def benchmark(self, prompt_len=256, n_new=256):
        """Steady-state decode speed in tokens/s (all rows) after a warm-up pass."""
        prompt = torch.randint(0, self.cfg.n_ranks, (self.batch, min(prompt_len, self.window)), device="cuda")
        self.generate(prompt, 8)  # compile + capture the graph
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.generate(prompt, n_new)
        torch.cuda.synchronize()
        return self.batch * n_new / (time.perf_counter() - t0)
