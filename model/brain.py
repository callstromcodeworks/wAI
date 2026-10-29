"""BrainGPT: specialist modules that fire per token, an associative memory, and a critic.

The main stack is the GPT's blocks (same attention, value residuals, U-net skips, FP8). Three
kinds of component are wired into it, and every one passes vectors into the shared residual
stream (no text between parts):

    tokens -> trunk blocks ------------------------------+
                 |-> module bank: E small transformers   |  each token: top-k modules fire,
                 |   (one batched GEMM per layer)        |  the rest stay silent
                 |-> memory: product-key lookup          |  16 of 16k slots read per token
                 +---------------------------------------+-> integrator blocks -> draft
                                                                    |-> critic: small transformer
                                                                    |   reads the draft, predicts
                                                                    |   how wrong it will be, and
                                                                    |   writes a correction
                                                         final gate blocks <-+ -> output

- Module bank: nothing is assigned. A router picks the top-k modules per token, and load
  balancing stops one module from taking every token, so specialisation comes from training.
  Balancing is either an auxiliary loss, or (balance_bias > 0) a per-module bias used only to
  choose modules: after each step it is nudged up for modules that fired less than their share
  and down for the rest, so balancing adds no gradient of its own. sigmoid_router makes router
  scores bounded (0..1, as in DeepSeek-V3) so a small bias can outweigh them; on raw logits the
  router outgrew the bias and collapsed onto two modules. Even with bounded scores, training
  pushed one module to fire on nearly every token, and evicting it with the bias handed that
  role from module to module and hurt the loss. n_shared gives that role a module of its own
  (fires on every token, as DeepSeek's shared experts), and the rest are routed and balanced. Every module processes every token (its attention needs the whole context, as a
  brain region keeps receiving input), but only the chosen ones write to the stream. The E
  modules have identical shapes, so each of their layers runs as one batched matmul.
- Memory: a product-key memory layer (two sets of 128 sub-keys address 16,384 value slots).
  The query picks 16 slots; their values are mixed by softmax weight, gated per token.
- Critic: reads the residual stream before the final gate (the draft), trained with an extra
  loss to predict the draft's next-token loss. With critic_detach it reads a detached copy, so
  that loss trains the critic without reshaping the main stack (round-2 runs showed an attached
  critic costing ~3% per token). Its correction and its error estimate both
  feed the final gate blocks, which produce the output. The draft also gets a small
  next-token loss of its own, so it is a real draft.
  With critic_gate the critic earns that say instead of having it from step 0: it starts out
  only observing (its output scaled by 0), and its trust grows with its track record, the
  correlation between its error estimates and the draft's real errors. critic_gate 2 also has
  it predict how unsure it is per token (Gaussian NLL), and it acts most where it is surer.

forward() returns the final hidden states in eval mode, and (hidden, draft, error estimate,
balance loss, share of tokens each module fired on, the estimate's log-variance or None, the
share of routed firings dropped for capacity or None) in training; train_loss() turns the latter into one loss plus its parts.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from fused_ce import linear_cross_entropy, per_token_loss
from gpt import GPT, GPTConfig, apply_rope, norm, rope_tables


@dataclass
class BrainConfig(GPTConfig):
    arch: str = "brain"
    trunk_layers: int = 3     # main blocks before the module bank and memory read the stream
    gate_layers: int = 2      # main blocks after the critic (the final gate)
    n_modules: int = 8        # 0 = no module bank
    module_dim: int = 256
    module_layers: int = 2
    module_heads: int = 4
    module_kv_heads: int = 2
    top_k: int = 2            # routed modules that fire per token; 0 = all, softmax-weighted (dense)
    n_shared: int = 0         # modules (the first n) that fire on every token; the rest are routed
    sigmoid_router: bool = False  # router scores are sigmoids (bounded, for bias balancing), not softmax
    memory_sub: int = 128     # sub-keys per half: memory_sub ** 2 slots; 0 = no memory
    memory_dim: int = 256     # query size
    memory_k: int = 16        # slots read per token
    # how the modules compute: "dense" = every module on every token, only the routed ones write;
    # "routed_attn" = each routed module runs only on its tokens, attending over just those (in order);
    # "ffn" = routed modules are feed-forward only, on their tokens (shared modules stay dense)
    module_mode: str = "dense"
    capacity: float = 1.25    # sparse modes: slots per module per sequence = capacity x its fair share
    critic: bool = True
    critic_detach: bool = False  # critic observes the draft without sending gradients into it
    # the critic earns its say: 0 = full say from the start; 1 = scaled by its track record (trust);
    # 2 = trust x its per-token confidence (it also predicts how unsure it is about each token)
    critic_gate: int = 0
    gate_lo: float = 0.3         # trust is 0 while the critic's error correlation is below this...
    gate_hi: float = 0.6         # ...and 1 at or above this
    gate_observe_steps: int = 100  # optimizer steps of observing only, whatever the record
    balance_bias: float = 0.0    # router-bias balancing step per optimizer step (0 = use the loss)
    # dead-module revival: a routed module firing on under 10% of its fair share gets this much more
    # router bias each step until it carries load; the boost fades once it has half its share (0 = off)
    revive: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        assert 0 < self.trunk_layers and self.trunk_layers + self.gate_layers <= self.n_layer
        assert self.module_dim % self.module_heads == 0 and self.module_heads % self.module_kv_heads == 0
        assert not self.n_modules or self.top_k <= self.n_modules - self.n_shared
        assert not self.critic_gate or (self.critic and self.critic_detach), "a gated critic must observe detached"
        assert self.module_mode in ("dense", "routed_attn", "ffn")
        assert self.module_mode == "dense" or (self.n_modules and self.top_k), "sparse modes route top-k"
        assert self.gate_lo < self.gate_hi


def rope_at(x, cos, sin):
    """apply_rope with a position per row: x [G, T, H, hd], cos/sin [G, T, hd/2]."""
    x1, x2 = x.float().chunk(2, dim=-1)
    cos, sin = cos[:, :, None, :], sin[:, :, None, :]
    return torch.cat((x1 * cos + x2 * sin, x2 * cos - x1 * sin), dim=-1).type_as(x)


class ModuleBank(nn.Module):
    """E small causal transformers with identical shapes, stored stacked and run batched.

    Input and output projections are single GEMMs over all modules ([E*de, D] matrices whose
    row blocks are the modules); each layer's weights are [E, out, in] and run as torch.bmm.
    forward(x, gates) returns (sum_e gates[..., e] * module_e(x), last hidden [E, N, de]).
    forward_routed(xd, ...) runs each module on its own tokens only (the sparse modes).
    attention=False drops the attention sublayers: feed-forward modules.
    """

    def __init__(self, cfg, E, attention=True):
        super().__init__()
        self.E, self.de, self.L = E, cfg.module_dim, cfg.module_layers
        self.H, self.Hkv = cfg.module_heads, cfg.module_kv_heads
        self.hd = self.de // self.H
        self.attention = attention
        D, de, hd = cfg.d_model, self.de, self.hd
        self.qkv_rows = [self.H * hd, self.Hkv * hd, self.Hkv * hd]
        self.w_in = nn.Parameter(torch.empty(E * de, D))
        self.w_out = nn.Parameter(torch.empty(E * de, D))
        n_attn = self.L if attention else 0
        self.qkv = nn.ParameterList(nn.Parameter(torch.empty(E, sum(self.qkv_rows), de)) for _ in range(n_attn))
        self.proj = nn.ParameterList(nn.Parameter(torch.empty(E, de, self.H * hd)) for _ in range(n_attn))
        self.fc = nn.ParameterList(nn.Parameter(torch.empty(E, 4 * de, de)) for _ in range(self.L))
        self.fc2 = nn.ParameterList(nn.Parameter(torch.empty(E, de, 4 * de)) for _ in range(self.L))
        self.attn_scale = cfg.attn_scale

    @torch.no_grad()
    def reset_parameters(self):
        def uniform(p, fan_in):
            b = math.sqrt(3) * 0.5 / math.sqrt(fan_in)
            nn.init.uniform_(p, -b, b)
        uniform(self.w_in, self.w_in.shape[1])
        nn.init.zeros_(self.w_out)  # the bank starts silent
        for l in range(self.L):
            uniform(self.fc[l], self.de)
            nn.init.zeros_(self.fc2[l])
        for l in range(len(self.qkv)):
            uniform(self.qkv[l], self.de)
            nn.init.zeros_(self.proj[l])

    def muon_splits(self):
        s = {self.w_in: [self.de] * self.E, self.w_out: [self.de] * self.E}
        s.update({p: self.qkv_rows for p in self.qkv})
        return s

    def layers(self, h, groups, T, cos, sin):
        """The modules' blocks on h [E, groups*T, de]: attention runs within each of the E*groups
        sequences of length T (rope tables [T, hd/2] shared, or [E*groups, T, hd/2] per sequence)."""
        E, H, Hkv, hd = self.E, self.H, self.Hkv, self.hd
        rope = (apply_rope if cos.dim() == 2 else rope_at) if self.attention else None
        for l in range(self.L):
            if self.attention:
                q, k, v = torch.bmm(norm(h), self.qkv[l].transpose(1, 2)).split(self.qkv_rows, dim=-1)
                q = rope(norm(q.reshape(E * groups, T, H, hd)), cos, sin).transpose(1, 2)
                k = rope(norm(k.reshape(E * groups, T, Hkv, hd)), cos, sin).transpose(1, 2)
                v = v.reshape(E * groups, T, Hkv, hd).transpose(1, 2)
                y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1, scale=self.attn_scale,
                                                   enable_gqa=Hkv != H)
                h = h + torch.bmm(y.transpose(1, 2).reshape(E, groups * T, H * hd), self.proj[l].transpose(1, 2))
            h = h + torch.bmm(F.relu(torch.bmm(norm(h), self.fc[l].transpose(1, 2))).square(),
                              self.fc2[l].transpose(1, 2))
        return h

    def forward(self, x, cos, sin, gates=None):
        B, T, D = x.shape
        E, de = self.E, self.de
        N = B * T
        h = (norm(x).reshape(N, D) @ self.w_in.t()).view(N, E, de).transpose(0, 1).contiguous()  # [E, N, de]
        h = self.layers(h, B, T, cos, sin)
        mixed = h if gates is None else h * gates.reshape(N, E).t()[..., None].to(h.dtype)
        out = mixed.transpose(0, 1).reshape(N, E * de) @ self.w_out  # gate, then one GEMM over all modules
        return out.view(B, T, D), h

    def forward_routed(self, xd, groups, C, cos=None, sin=None):
        """Each module on its own tokens: xd [E, groups*C, D] holds module e's C slots per sequence
        (in time order, zero rows as padding after them). Returns [E, groups*C, D]."""
        E, de, D = self.E, self.de, xd.size(-1)
        h = torch.bmm(norm(xd), self.w_in.view(E, de, D).transpose(1, 2))
        h = self.layers(h, groups, C, cos, sin)
        return torch.bmm(h, self.w_out.view(E, de, D))


class ProductKeyMemory(nn.Module):
    """Associative memory: memory_sub ** 2 value slots addressed by two sets of sub-keys."""

    def __init__(self, cfg):
        super().__init__()
        self.n, self.k, self.dk = cfg.memory_sub, cfg.memory_k, cfg.memory_dim
        self.query = nn.Linear(cfg.d_model, self.dk, bias=False)
        self.keys = nn.Parameter(torch.empty(2, self.n, self.dk // 2))
        self.values = nn.Parameter(torch.empty(self.n * self.n, cfg.d_model))
        self.gate = nn.Linear(cfg.d_model, 1)

    @torch.no_grad()
    def reset_parameters(self):
        b = math.sqrt(3) * 0.5 / math.sqrt(self.query.in_features)
        nn.init.uniform_(self.query.weight, -b, b)
        nn.init.normal_(self.keys, std=1 / math.sqrt(self.dk // 2))
        nn.init.normal_(self.values, std=0.02)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)  # starts mostly closed

    def forward(self, x):
        B, T, D = x.shape
        xn = norm(x).reshape(B * T, D)
        q = norm(self.query(xn).float()).view(-1, 2, self.dk // 2)
        s1 = q[:, 0] @ self.keys[0].float().t()  # [N, n]
        s2 = q[:, 1] @ self.keys[1].float().t()
        v1, i1 = s1.topk(self.k, dim=-1)
        v2, i2 = s2.topk(self.k, dim=-1)
        scores = (v1[:, :, None] + v2[:, None, :]).flatten(1)  # [N, k*k] candidate slots
        best, idx = scores.topk(self.k, dim=-1)
        slot = i1.gather(1, idx // self.k) * self.n + i2.gather(1, idx % self.k)
        w = torch.softmax(best, dim=-1)
        out = F.embedding_bag(slot, self.values, per_sample_weights=w.to(self.values.dtype), mode="sum")
        return (torch.sigmoid(self.gate(xn).float()) * out.float()).to(x.dtype).view(B, T, D)


class BrainGPT(GPT):
    def __init__(self, cfg: BrainConfig):
        super().__init__(cfg)  # embedding, main blocks, U-net weights, RoPE tables
        mhd = cfg.module_dim // cfg.module_heads
        cos, sin = rope_tables(cfg.seq_len, mhd, cfg.rope_base)
        self.register_buffer("rope_cos_m", cos, persistent=False)
        self.register_buffer("rope_sin_m", sin, persistent=False)
        n_routed = cfg.n_modules - cfg.n_shared
        self.shared = None
        if cfg.module_mode == "dense":
            self.bank = ModuleBank(cfg, cfg.n_modules) if cfg.n_modules else None
        else:  # sparse: the bank holds the routed modules only; shared modules run densely beside it
            attention = cfg.module_mode == "routed_attn"
            self.bank = ModuleBank(cfg, n_routed, attention=attention)
            if cfg.n_shared:
                self.shared = ModuleBank(cfg, cfg.n_shared, attention=attention)
        self.router = nn.Linear(cfg.d_model, n_routed, bias=False) if cfg.n_modules else None
        if cfg.n_modules and (cfg.balance_bias or cfg.revive):
            self.register_buffer("route_bias", torch.zeros(n_routed))  # used only to choose modules
        self.memory = ProductKeyMemory(cfg) if cfg.memory_sub else None
        if cfg.critic:
            self.critic = ModuleBank(cfg, 1)
            # error estimate, plus (critic_gate 2) the log-variance of that estimate: how unsure it is
            self.err_head = nn.Linear(cfg.module_dim, 2 if cfg.critic_gate == 2 else 1)
            self.err_to_resid = nn.Parameter(torch.zeros(cfg.d_model))  # error estimate -> final gate
            if cfg.critic_gate:  # the earned say: not learned, updated after each step (update_after_step)
                for name in ("trust", "corr_ema", "logvar_ref", "gate_steps"):
                    self.register_buffer(name, torch.zeros(()))
        else:
            self.critic = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        super().reset_parameters()
        if getattr(self, "bank", None) is not None:
            self.bank.reset_parameters()
            nn.init.normal_(self.router.weight, std=0.02)
        if getattr(self, "shared", None) is not None:
            self.shared.reset_parameters()
        if getattr(self, "memory", None) is not None:
            self.memory.reset_parameters()
        if getattr(self, "critic", None) is not None:
            self.critic.reset_parameters()
            nn.init.zeros_(self.err_head.weight)
            nn.init.constant_(self.err_head.bias[0], 3.0)  # about the average loss early in training
            if self.err_head.bias.numel() > 1:
                nn.init.constant_(self.err_head.bias[1], 2.0)  # unsure to begin with (std ~2.7 nats)
            nn.init.zeros_(self.err_to_resid)

    def route(self, x):
        """Per-token gates [B, T, E] (shared modules at 1, top-k routed renormalized), the
        load-balancing loss over the routed modules, and which modules fired."""
        c = self.cfg
        logits = self.router(norm(x)).float()
        if c.sigmoid_router:
            scores = torch.sigmoid(logits)
            probs = scores / scores.sum(-1, keepdim=True)
        else:
            scores, probs = logits, torch.softmax(logits, dim=-1)
        n_routed = c.n_modules - c.n_shared
        if c.top_k:
            choose = scores + self.route_bias if (c.balance_bias or c.revive) else scores
            top_i = choose.topk(c.top_k, dim=-1).indices
            top_v = probs.gather(-1, top_i)
            gates = torch.zeros_like(probs).scatter(-1, top_i, top_v / top_v.sum(-1, keepdim=True))
            fired = torch.zeros_like(probs).scatter(-1, top_i, 1.0)
        else:
            gates, fired = probs, torch.ones_like(probs)
        frac = fired.flatten(0, 1).mean(0) / max(1, c.top_k or n_routed)  # share of firings per module
        balance = n_routed * (frac * probs.flatten(0, 1).mean(0)).sum()
        if c.n_shared:
            ones = gates.new_ones(*gates.shape[:-1], c.n_shared)
            gates, fired = torch.cat([ones, gates], -1), torch.cat([ones, fired], -1)
        return gates, balance, fired

    @torch.no_grad()
    def update_balance(self, usage):
        """After an optimizer step: raise the bias of modules that fired on less than their share."""
        c = self.cfg
        if c.n_modules and c.balance_bias and c.top_k:
            routed = usage[c.n_shared:].to(self.route_bias)
            self.route_bias += c.balance_bias * torch.sign(c.top_k / (c.n_modules - c.n_shared) - routed)
        if c.n_modules and c.revive and c.top_k:
            routed = usage[c.n_shared:].to(self.route_bias)
            fair = c.top_k / (c.n_modules - c.n_shared)
            self.route_bias += c.revive * (routed < 0.1 * fair).float()                     # quiet: push it in
            self.route_bias *= torch.where(routed >= 0.5 * fair, 0.95, 1.0)                # carrying load: fade

    @torch.no_grad()
    def update_after_step(self, parts):
        """After an optimizer step, from the step's averaged loss parts: router-bias balancing, and
        the critic's trust. Trust follows its track record: an EMA of how well its error estimates
        correlated with the draft's real errors, mapped from gate_lo..gate_hi to 0..1, and held at 0
        for the first gate_observe_steps steps. All tensor ops, so no GPU sync."""
        c = self.cfg
        super().update_after_step(parts)  # depth routing thresholds, if any
        if "usage" in parts:
            self.update_balance(parts["usage"])
        if not c.critic_gate or "critic_corr" not in parts:
            return
        first = self.gate_steps == 0
        corr = parts["critic_corr"].float()
        self.corr_ema.copy_(torch.where(first, corr, 0.98 * self.corr_ema + 0.02 * corr))
        if "critic_logvar" in parts:
            lv = parts["critic_logvar"].float()
            self.logvar_ref.copy_(torch.where(first, lv, 0.98 * self.logvar_ref + 0.02 * lv))
        self.gate_steps += 1
        earned = ((self.corr_ema - c.gate_lo) / (c.gate_hi - c.gate_lo)).clamp(0, 1)
        self.trust.copy_(torch.where(self.gate_steps > c.gate_observe_steps, earned, torch.zeros_like(earned)))

    def critic_say(self, logvar):
        """How much of the critic's output reaches the gate: trust, times (critic_gate 2) a per-token
        confidence that is above 0.5 where it is surer than usual. Detached: the main loss can't
        game it. Returns a [B, T, 1] float tensor (or the scalar trust)."""
        say = self.trust
        if logvar is not None:
            say = say * torch.sigmoid(2.0 * (self.logvar_ref - logvar.detach()))[..., None]
        return say

    def routed_bank(self, x, gates, fired, cos, sin):
        """Sparse modes: each routed module computes only on the tokens routed to it.

        Per sequence, module e gets C = capacity x (T * top_k / routed) slots, filled with its tokens
        in time order (whether a token gets a slot depends only on earlier tokens, so it stays
        causal; tokens past capacity skip that module). The slots run through the module as one
        batched sequence (attention, in routed_attn mode, sees only the module's own tokens, at
        their original positions), and the outputs, scaled by the gate, are added back at their
        tokens. Returns (bank output [B, T, D], share of routed firings dropped for capacity)."""
        c = self.cfg
        B, T, D = x.shape
        R = c.n_modules - c.n_shared
        C = min(T, math.ceil(c.capacity * T * c.top_k / R))
        fired = fired.bool()
        slot = fired.long().cumsum(1) - 1                                     # [B, T, R] slot within its module
        keep = fired & (slot < C)
        dump = torch.where(keep, slot, torch.full_like(slot, C))              # dropped/unrouted -> spare slot C
        t_of = torch.arange(T, device=x.device).view(1, T, 1).expand(B, T, R)
        idx = torch.full((B, R, C + 1), T, device=x.device, dtype=torch.long)  # T = padding (a zero row)
        idx.scatter_(2, dump.transpose(1, 2), t_of.transpose(1, 2))
        idx = idx[..., :C]                                                    # [B, R, C] token in each slot
        b = torch.arange(B, device=x.device).view(B, 1, 1)
        xd = torch.cat([x, x.new_zeros(B, 1, D)], 1)[b, idx]                  # [B, R, C, D]
        xd = xd.transpose(0, 1).reshape(R, B * C, D)
        if self.bank.attention:
            pos = idx.clamp(max=T - 1).transpose(0, 1).reshape(R * B, C)      # each slot's original position
            yd = self.bank.forward_routed(xd, B, C, cos[pos], sin[pos])
        else:
            yd = self.bank.forward_routed(xd, B, C)
        g = torch.cat([gates, gates.new_zeros(B, 1, R)], 1)                   # gate of each slot's token
        g = g[b, idx, torch.arange(R, device=x.device).view(1, R, 1)]          # [B, R, C]
        yd = yd.view(R, B, C, D).transpose(0, 1) * g[..., None].to(yd.dtype)
        out = x.new_zeros(B, T + 1, D)
        out.scatter_add_(1, idx.reshape(B, R * C, 1).expand(-1, -1, D), yd.reshape(B, R * C, D).to(x.dtype))
        dropped = 1 - keep.sum() / fired.sum().clamp(min=1)
        return out[:, :T], dropped.float().detach()

    def forward(self, idx, cache=None, pos=0):
        assert cache is None, "BrainGPT has no incremental decoding yet"
        c = self.cfg
        T = idx.size(1)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        cos_m, sin_m = self.rope_cos_m[:T], self.rope_sin_m[:T]
        ctx = {"cache": None, "v0": None, "kv": None, "rope_mla": None}
        x = norm(self.wte(idx)).bfloat16()
        L = len(self.blocks)
        skips, draft, err, logvar, balance = [], None, None, None, x.new_zeros((), dtype=torch.float32)
        dropped = None
        usage = x.new_zeros(max(1, c.n_modules), dtype=torch.float32)
        for i, block in enumerate(self.blocks):
            if i == c.trunk_layers:  # modules and memory read the trunk's output, in parallel
                add = 0
                if self.bank is not None:
                    gates, balance, fired = self.route(x)
                    if c.module_mode == "dense":
                        add = add + self.bank(x, cos_m, sin_m, gates)[0]
                    else:
                        S = c.n_shared
                        if self.shared is not None:
                            add = add + self.shared(x, cos_m, sin_m)[0]
                        routed, dropped = self.routed_bank(x, gates[..., S:], fired[..., S:], cos_m, sin_m)
                        add = add + routed
                    usage = fired.detach().flatten(0, 1).mean(0)
                if self.memory is not None:
                    add = add + self.memory(x)
                if getattr(self, "grown", None) is not None:  # modules added later (forgetting.py grow)
                    g = torch.sigmoid(self.grown_gate(norm(x)).float())                  # [B, T, k]
                    add = add + self.grown(x, cos_m, sin_m, g.to(x.dtype))[0]
                x = x + add
            if i == L - c.gate_layers:
                draft = x
                if self.critic is not None:  # examine the draft; correction + error estimate feed the gate
                    corr, hc = self.critic(x.detach() if c.critic_detach else x, cos_m, sin_m)
                    head = self.err_head(norm(hc[0])).float().view(*x.shape[:2], -1)
                    err = head[..., 0]
                    if head.size(-1) > 1:
                        logvar = head[..., 1].clamp(-6, 6)
                    fix = corr + err[..., None].to(x.dtype) * self.err_to_resid.to(x.dtype)
                    if c.critic_gate:  # its say so far: nothing while it only observes
                        fix = fix * self.critic_say(logvar).to(x.dtype)
                    x = x + fix
            if i >= L - self.n_skip:
                x = x + self.skip_w[L - 1 - i] * skips.pop()
            x = block(x, cos, sin, ctx)
            grown_at = getattr(self, "grown_at", None)  # modules added later at several depths (forgetting.py)
            if grown_at is not None and str(i) in grown_at:
                bank_i, gate_i = grown_at[str(i)], self.grown_gates[str(i)]
                x = x + bank_i(x, cos_m, sin_m, torch.sigmoid(gate_i(norm(x)).float()).to(x.dtype))[0]
            if i < self.n_skip:
                skips.append(x)
        out = norm(x).bfloat16()
        if not self.training:
            return out
        return out, norm(draft).bfloat16(), err, balance, usage, logvar, dropped

    def train_loss(self, out, targets, chunk=2048, draft_weight=0.1, critic_weight=0.1, balance_weight=0.01,
                   stride=8, offset=0):
        """Main next-token loss plus the auxiliaries. The draft and critic losses use every
        stride-th position (from offset), which keeps their extra LM-head work to 1/stride."""
        hidden, draft, err, balance, usage, logvar, dropped = out
        main = self.loss(hidden, targets, chunk)
        parts = {"main": main.detach()}
        total = main
        sel = slice(offset, None, stride)
        d_h, d_y = draft[:, sel].flatten(0, 1), targets[:, sel].flatten()
        if draft_weight:
            d_loss = linear_cross_entropy(d_h, self.wte.weight, d_y, self.cfg.logit_softcap, chunk)
            total = total + draft_weight * d_loss
            parts["draft"] = d_loss.detach()
        if err is not None and critic_weight:
            target = per_token_loss(d_h, self.wte.weight, d_y, self.cfg.logit_softcap, chunk)
            pred = err[:, sel].flatten()
            if logvar is not None:  # Gaussian NLL: it learns both the error and how sure it can be
                lv = logvar[:, sel].flatten()
                c_loss = 0.5 * ((target - pred) ** 2 * torch.exp(-lv) + lv).mean()
                parts["critic_nll"] = c_loss.detach()
                parts["critic_logvar"] = lv.detach().mean()
            else:
                c_loss = F.mse_loss(pred, target)
                parts["critic_mse"] = c_loss.detach()
            total = total + critic_weight * c_loss
            parts["critic_corr"] = torch.corrcoef(torch.stack([pred.detach(), target]))[0, 1].nan_to_num(0.0)
            if self.cfg.critic_gate:
                parts["critic_say"] = self.critic_say(None if logvar is None else logvar[:, sel]).mean()
        if self.bank is not None:
            parts["usage"] = usage  # [E] share of tokens each module fired on; the bias balancer reads it
            if dropped is not None:
                parts["dropped"] = dropped  # sparse modes: share of routed firings past a module's capacity
            if balance_weight:
                total = total + balance_weight * balance
                parts["balance"] = balance.detach()
        return total, parts

    # ---- optimizer groups and accounting -----------------------------------

    def muon_splits(self):
        s = super().muon_splits()
        for bank in (self.bank, self.shared, self.critic):
            if bank is not None:
                s.update(bank.muon_splits())
        return s

    def matrix_params(self):
        """Hidden matrices for Muon: main blocks, the module bank, the critic, the memory query."""
        ps = super().matrix_params()
        for bank in (self.bank, self.shared, self.critic):
            if bank is not None:
                ps += list(bank.parameters())
        if self.memory is not None:
            ps.append(self.memory.query.weight)
        return ps

    def embedding_like_params(self):
        """AdamW at the embedding LR: tables read sparsely, and the small heads."""
        ps = [self.wte.weight]
        if self.memory is not None:
            ps += [self.memory.values, self.memory.keys, self.memory.gate.weight]
        if self.router is not None:
            ps.append(self.router.weight)
        if self.critic is not None:
            ps.append(self.err_head.weight)
        return ps

    def scalar_params(self):
        return [p for p in self.parameters() if p.dim() < 2]

    def modules_per_token(self):
        """Modules (plus the critic) that compute on each token: all of them in dense mode; the
        shared ones plus top_k routed (at most) in the sparse modes."""
        c = self.cfg
        active = c.n_modules if c.module_mode == "dense" else c.n_shared + c.top_k
        return active + int(c.critic)

    def kv_bytes_per_token(self, bytes_per_value=2):
        c = self.cfg
        per_module = c.module_layers * 2 * c.module_kv_heads * (c.module_dim // c.module_heads)
        with_attn = self.modules_per_token() if c.module_mode != "ffn" else c.critic  # ffn modules keep no KV
        return super().kv_bytes_per_token(bytes_per_value) + with_attn * per_module * bytes_per_value

    def flops_per_token(self):
        c = self.cfg
        banks = [b for b in (self.bank, self.shared, self.critic) if b is not None]
        bank_params = sum(p.numel() for b in banks for p in b.parameters())
        per_module = bank_params / max(1, sum(b.E for b in banks))
        dense = sum(p.numel() for p in self.matrix_params()) - bank_params + c.vocab_size * c.d_model
        dense += self.modules_per_token() * per_module
        memory = (2 * c.memory_sub * c.memory_dim // 2 + c.memory_k * c.d_model) if self.memory is not None else 0
        attn = c.n_layer * c.seq_len * c.n_head * c.head_dim
        span = c.seq_len if c.module_mode == "dense" else c.seq_len * c.top_k * c.capacity / (c.n_modules - c.n_shared)
        if c.module_mode != "ffn":
            attn += self.modules_per_token() * c.module_layers * span * c.module_dim
        return 6 * (dense + memory) + 6 * attn

    def activation_factor(self):
        """Activation memory relative to a GPT of the same main stack (for picking the micro-batch)."""
        c = self.cfg
        active = self.modules_per_token() if c.module_mode == "dense" else c.n_shared + c.top_k * c.capacity + c.critic
        extra = active * c.module_layers * c.module_dim * 1.6
        return 1 + extra / (c.n_layer * c.d_model)

    def describe(self):
        c = self.cfg
        bank = f"bank {c.n_modules}x{c.module_dim}x{c.module_layers}L " + (f"{c.n_shared} shared + " if c.n_shared else "") \
            + (f"top{c.top_k}" if c.top_k else "dense") if c.n_modules else "no bank"
        if c.n_modules and c.module_mode != "dense":
            bank += {"routed_attn": " sparse (attention over own tokens)", "ffn": " sparse feed-forward"}[c.module_mode]
        mem = f"mem {c.memory_sub ** 2 // 1024}k/{c.memory_k}" if c.memory_sub else "no mem"
        critic = "no critic" if not c.critic else "critic" + (" detached" if c.critic_detach else "") + \
            {0: "", 1: " gated by trust", 2: " gated by trust x confidence"}[c.critic_gate]
        return f"brain trunk{c.trunk_layers} gate{c.gate_layers} {bank} {mem} {critic}"
