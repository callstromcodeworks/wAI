"""Pretrain the GPT on fasttok-tokenized shards (see prepare_data.py).

    python model/train.py --vocab ec64k --preset small --hours 8 --run small64k
    python model/train.py --vocab ec32k --preset small --n_kv_head 2 --kv_share 2 --hours 0.5
    python model/train.py --resume runs/small64k/ckpt.pt
    python model/train.py --vocab ec32k --arch brain --n_layer 8 --hours 0.33   # brain.py

Throughput choices, each measured on an RTX 4070 (see bench.py and README.md):
- torch.compile on the model forward (fused norms, RoPE, ReLU^2, residual adds).
- cuDNN attention, pinned explicitly. The Windows PyTorch build has no FlashAttention, and
  cuDNN was 1.56x faster than the memory-efficient kernel SDPA would otherwise choose.
- Fused LM head + cross-entropy (fused_ce.py). Vocabulary logits are never materialized.
- bf16 autocast, fused AdamW, and Muon with batched, compiled Newton-Schulz.
- Data is memmapped and assembled by a background thread into pinned memory, then copied
  asynchronously. No host sync happens inside a step except at log intervals.
- The micro-batch is the largest that fits, checked with a real compiled step.
- A VRAM cap: on Windows, CUDA silently spills to system RAM when VRAM runs out, which runs
  10x slower without an error. Capping the allocator turns that into a clean OOM instead.
"""
import argparse
import base64
import glob
import json
import math
import os
import queue
import threading
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from brain import BrainConfig, BrainGPT
from gpt import GPT, GPTConfig, PRESETS
from muon import Muon
from prepare_data import load_shard
from tokenizer import DEFAULT_VOCAB, VOCABS

HERE = os.path.dirname(os.path.abspath(__file__))
SDPA_BACKENDS = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                 SDPBackend.MATH]


# ----------------------------------------------------------------------------- data

class SourceStream:
    """Contiguous (T+1)-token rows from one source's train shards. Each epoch visits the
    shards in a new random order, starting at a random offset."""

    def __init__(self, files, T, seed, n_ranks):
        if not files:
            raise FileNotFoundError("no train shards found; run prepare_data.py for this vocabulary")
        self.files, self.T, self.n_ranks = files, T, n_ranks
        self.rng = np.random.default_rng(seed)
        self.epoch, self.order, self.idx, self.pos = 0, [], 0, 0
        self._next_shard()

    def _next_shard(self):
        if self.idx >= len(self.order):
            self.order = self.rng.permutation(len(self.files)).tolist()
            self.idx = 0
            self.epoch += 1
        self.shard, n_ranks = load_shard(self.files[self.order[self.idx]])
        if n_ranks != self.n_ranks:
            raise ValueError(f"{self.files[self.order[self.idx]]} was tokenized with {n_ranks} ranks, not {self.n_ranks}")
        self.idx += 1
        self.pos = int(self.rng.integers(0, self.T))

    def take(self, n):
        out = np.empty((n, self.T + 1), dtype=np.int32)
        for r in range(n):
            if self.pos + self.T + 1 > len(self.shard):
                self._next_shard()
            out[r] = self.shard[self.pos:self.pos + self.T + 1]
            self.pos += self.T
        return out

    def state_dict(self):
        return dict(epoch=self.epoch, order=self.order, idx=self.idx, pos=self.pos, rng=self.rng.bit_generator.state)

    def load_state_dict(self, s):
        self.epoch, self.order, self.idx = s["epoch"], s["order"], s["idx"] - 1
        self.rng.bit_generator.state = s["rng"]
        self._next_shard()
        self.pos = s["pos"]


class MixLoader:
    """Micro-batches of B rows. Each row is drawn from a source with probability given by the mix."""

    def __init__(self, data_dir, mix, B, T, seed, n_ranks):
        self.names = list(mix)
        self.p = np.array([mix[n] for n in self.names], dtype=np.float64)
        self.p /= self.p.sum()
        self.streams = [SourceStream(sorted(glob.glob(os.path.join(data_dir, n, "train_*.bin"))), T, seed + i, n_ranks)
                        for i, n in enumerate(self.names)]
        self.B = B
        self.rng = np.random.default_rng(seed)

    def next(self):
        counts = self.rng.multinomial(self.B, self.p)
        return np.concatenate([s.take(c) for s, c in zip(self.streams, counts) if c])

    def state_dict(self):
        return dict(rng=self.rng.bit_generator.state, streams=[s.state_dict() for s in self.streams])

    def load_state_dict(self, s):
        self.rng.bit_generator.state = s["rng"]
        for stream, st in zip(self.streams, s["streams"]):
            stream.load_state_dict(st)


class Prefetcher:
    """Builds batches on a background thread into pinned memory; next() returns a GPU tensor.

    The saved loader state runs up to `depth` batches ahead of training, so a resumed run
    skips at most that many batches.
    """

    def __init__(self, loader, depth=6):
        self.loader = loader
        self.q = queue.Queue(depth)
        self.lock = threading.Lock()
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self):
        while True:
            with self.lock:
                rows = self.loader.next()
            self.q.put(torch.from_numpy(rows).pin_memory())

    def next(self):
        return self.q.get().cuda(non_blocking=True)

    def state_dict(self):
        with self.lock:
            return self.loader.state_dict()


def token_byte_lengths(n_ranks, size, vocab_path=DEFAULT_VOCAB):
    """UTF-8 byte length of each token id (0 for special and padding ids), for bits-per-byte."""
    lengths = np.zeros(size, dtype=np.int32)
    with open(vocab_path, "rb") as f:
        for line in f:
            tok, rank = line.split()
            if int(rank) < n_ranks:
                lengths[int(rank)] = len(base64.b64decode(tok))
    return lengths


# ----------------------------------------------------------------------------- helpers

def keep_awake():
    """Ask Windows not to idle-sleep while this process runs; released automatically on exit."""
    if os.name == "nt":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def lr_multiplier(step, total, warmup, cooldown_frac):
    """Warmup-stable-decay: linear warmup, flat, then linear decay to 0 over the last cooldown_frac."""
    if step < warmup:
        return (step + 1) / warmup
    decay_start = int(total * (1 - cooldown_frac))
    if step < decay_start:
        return 1.0
    return max(0.0, (total - step) / max(1, total - decay_start))


def measure_peak_tflops():
    """Sustained bf16 GEMM rate: ~1 s of warm-up so the clocks settle (a 0.2 s burst read as low
    as 28 TFLOPS before boost), then ~2 s of back-to-back matmuls timed with CUDA events."""
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    for _ in range(40):
        a @ a
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(80):
        a @ a
    end.record()
    torch.cuda.synchronize()
    return 2 * 8192**3 * 80 / (start.elapsed_time(end) / 1e3) / 1e12


@torch.no_grad()
def evaluate(model, fwd, data_dir, names, T, B, eval_bytes, token_bytes):
    """Validation loss (nats/token) and bits per byte for each source.

    Each source is scored on (about) its first eval_bytes of validation text, whatever the
    vocabulary, so bits per byte compares models with different tokenizers on the same text.
    """
    out = {}
    tb = token_bytes.cpu().numpy()
    for name in names:
        val, _ = load_shard(os.path.join(data_dir, name, "val_000000.bin"))
        bytes_per_token = tb[val[:1_000_000]].mean()
        n_rows = math.ceil(eval_bytes / (bytes_per_token * T) / B) * B  # whole batches: no recompile
        n_rows = min(n_rows, (len(val) - 1) // T // B * B)
        loss_sum = torch.zeros((), device="cuda", dtype=torch.float64)
        byte_sum = torch.zeros((), device="cuda", dtype=torch.int64)
        for i in range(0, n_rows, B):
            rows = np.stack([val[r * T:r * T + T + 1] for r in range(i, i + B)]).astype(np.int32)
            rows = torch.from_numpy(rows).cuda()
            x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
            with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPA_BACKENDS, set_priority=True):
                h = fwd(x)
            loss_sum += model.token_losses(h, y).sum(dtype=torch.float64)
            byte_sum += token_bytes[y].sum()
        loss = loss_sum.item() / (n_rows * T)
        out[name] = dict(loss=loss, bpb=loss_sum.item() / math.log(2) / byte_sum.item())
    return out


def embedding_like(model):
    return model.embedding_like_params() if hasattr(model, "embedding_like_params") else [model.wte.weight]


def build_optimizers(model, args):
    """Muon for the hidden matrices; AdamW for the embedding (and other lookup tables) and scalars."""
    groups_params = [model.matrix_params(), embedding_like(model), model.scalar_params()]
    seen = [id(p) for g in groups_params for p in g]
    assert len(seen) == len(set(seen)) == len(list(model.parameters())), "every parameter in exactly one group"
    muon = Muon(model.matrix_params(), lr=args.lr_muon, momentum=0.95, weight_decay=args.wd,
                splits=model.muon_splits())
    groups = [dict(params=embedding_like(model), lr=args.lr_embed, base_lr=args.lr_embed)]
    if model.scalar_params():
        groups.append(dict(params=model.scalar_params(), lr=args.lr_scalar, base_lr=args.lr_scalar))
    adam = torch.optim.AdamW(groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0, fused=True)
    return muon, adam


def optimizer_state_bytes(model):
    adam = sum(p.numel() for p in embedding_like(model)) + sum(p.numel() for p in model.scalar_params())
    return adam * 8 + (model.num_params() - adam) * 4  # AdamW m+v; Muon momentum on the matrices


def step_loss(model, out, y, args, offset=0):
    """(loss, parts): the brain's main + auxiliary losses, or the plain next-token loss."""
    if hasattr(model, "train_loss"):
        return model.train_loss(out, y, args.ce_chunk, args.draft_weight, args.critic_weight, args.balance_weight,
                                args.aux_stride, offset)
    return model.loss(out, y, args.ce_chunk), {}


def pick_micro_batch(model, fwd, seq_len, ce_chunk, batch_seqs, args):
    """Largest power-of-two micro-batch dividing batch_seqs that fits in memory for training.

    Starts from an estimate (activations measured at ~34 bytes per layer x d_model x token)
    and confirms with a real compiled step on random tokens, stepping down on OOM. The trial
    reserves what a real step has on top: optimizer state, the fp32 gradients that persist
    across gradient-accumulation micro-steps, and a margin for allocator fragmentation.
    """
    c = model.cfg
    budget = torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    if torch.cuda.get_per_process_memory_fraction() < 1.0:
        budget = min(budget, torch.cuda.get_per_process_memory_fraction() * torch.cuda.mem_get_info()[1]
                     - torch.cuda.memory_allocated())
    reserve_bytes = optimizer_state_bytes(model) + model.num_params() * 4 + 384 * 2**20
    fixed = model.num_params() * 4 + reserve_bytes + ce_chunk * c.vocab_size * 2 + c.vocab_size * c.d_model * 6
    per_seq = 36 * c.n_layer * c.d_model * seq_len
    if hasattr(model, "activation_factor"):
        per_seq = int(per_seq * model.activation_factor())
    estimate = max(1, int((budget - fixed) / per_seq))
    for b in (64, 32, 16, 8, 4, 2, 1):
        if b > estimate or batch_seqs % b:
            continue
        reserve = None
        try:
            reserve = torch.empty(reserve_bytes, dtype=torch.uint8, device="cuda")
            x = torch.randint(0, c.n_vocab, (b, seq_len), device="cuda", dtype=torch.int32)
            with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPA_BACKENDS, set_priority=True):
                h = fwd(x)
            step_loss(model, h, x, args)[0].backward()
            torch.cuda.synchronize()
            return b
        except torch.OutOfMemoryError:
            pass
        finally:
            del reserve
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    raise SystemExit("model does not fit even at micro-batch 1")


# ----------------------------------------------------------------------------- main

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab", default="ec64k", choices=sorted(VOCABS))
    ap.add_argument("--preset", default="small", choices=sorted(PRESETS))
    ap.add_argument("--n_layer", type=int, help="override the preset")
    ap.add_argument("--n_head", type=int, help="override the preset")
    ap.add_argument("--d_model", type=int, help="override the preset")
    ap.add_argument("--n_kv_head", type=int, default=0, help="K/V heads (GQA); 0 = n_head")
    ap.add_argument("--kv_share", type=int, default=1, help="layers sharing one K/V (cross-layer attention)")
    ap.add_argument("--value_residual", type=int, default=0, help="mix each layer's values with the first layer's")
    ap.add_argument("--unet", type=int, default=0, help="U-net skips from the first half of blocks to the second")
    ap.add_argument("--mla_rank", type=int, default=0, help="multi-head latent attention: latent size (0 = off)")
    ap.add_argument("--mla_rope_dim", type=int, default=32, help="RoPE key dims cached alongside the MLA latent")
    ap.add_argument("--attn_mode", default="full", choices=["full", "window", "concept"],
                    help="memory.py: full attention, a sliding window, or window + concepts + expanded blocks")
    ap.add_argument("--window", type=int, default=512, help="window/concept: recent tokens read in full")
    ap.add_argument("--concept_block", type=int, default=64, help="concept: tokens summarised per concept")
    ap.add_argument("--concept_topk", type=int, default=4, help="concept: older blocks expanded per query block (0 = none)")
    ap.add_argument("--concept_rope", type=int, default=1, help="concept: 0 = position-free concepts (gist without timestamp)")
    ap.add_argument("--fast_heads", type=int, default=0, help="Hebbian fast-weight heads per block (0 = none)")
    ap.add_argument("--fast_dim", type=int, default=64, help="fast-weight key/value size per head")
    ap.add_argument("--depth_every", type=int, default=0, help="route tokens through every n-th block (0 = off)")
    ap.add_argument("--depth_capacity", type=float, default=0.25, help="share of tokens a routed block processes")
    ap.add_argument("--mlp_mode", default="relu2", choices=["relu2", "spiking"],
                    help="spiking: MLP neurons integrate over recent tokens and fire binary spikes past a threshold")
    ap.add_argument("--spike_window", type=int, default=8, help="spiking: tokens a neuron integrates over")
    ap.add_argument("--spike_levels", type=int, default=1, help="spiking: spikes per token (1 = binary, 3 = 2-bit rate code)")
    ap.add_argument("--mlp_ratio", type=int, default=0, help="MLP hidden size / d_model (0 = the preset's, 4)")
    ap.add_argument("--depth_mode", default="block", choices=["block", "mlp"],
                    help="block: unrouted tokens skip the block; mlp: all get attention, routed tokens get the MLP")
    ap.add_argument("--arch", default="gpt", choices=["gpt", "brain"], help="brain = brain.py (modules, memory, critic)")
    ap.add_argument("--trunk_layers", type=int, default=3, help="brain: main blocks before the modules and memory")
    ap.add_argument("--gate_layers", type=int, default=2, help="brain: main blocks after the critic")
    ap.add_argument("--n_modules", type=int, default=8, help="brain: modules in the bank (0 = none)")
    ap.add_argument("--module_dim", type=int, default=256)
    ap.add_argument("--module_layers", type=int, default=2)
    ap.add_argument("--module_heads", type=int, default=4)
    ap.add_argument("--module_kv_heads", type=int, default=2)
    ap.add_argument("--top_k", type=int, default=2, help="brain: routed modules firing per token (0 = all, dense)")
    ap.add_argument("--n_shared", type=int, default=0, help="brain: modules that fire on every token (not routed)")
    ap.add_argument("--module_mode", default="dense", choices=["dense", "routed_attn", "ffn"],
                    help="brain: dense = all modules compute on every token; routed_attn = routed modules compute and "
                         "attend only over their own tokens; ffn = routed modules are feed-forward, on their tokens")
    ap.add_argument("--capacity", type=float, default=1.25, help="brain sparse modes: slots per module vs its fair share")
    ap.add_argument("--revive", type=float, default=0.0, help="brain: router bias step for dead modules (0 = off)")
    ap.add_argument("--judge", type=int, default=0,
                    help="brain: one judging module (1 judge only, 2 its estimate allocates modules per token, 3 random)")
    ap.add_argument("--router_z", type=float, default=0.0, help="brain: router z-loss weight (bounds router logits)")
    ap.add_argument("--sigmoid_router", type=int, default=0, help="brain: bounded sigmoid router scores (for --balance_bias)")
    ap.add_argument("--memory_sub", type=int, default=128, help="brain: memory sub-keys per half (slots = sub^2; 0 = none)")
    ap.add_argument("--memory_k", type=int, default=16, help="brain: memory slots read per token")
    ap.add_argument("--critic", type=int, default=1, help="brain: critic + final gate (0 = off)")
    ap.add_argument("--critic_detach", type=int, default=0, help="brain: critic reads a detached draft")
    ap.add_argument("--critic_gate", type=int, default=0,
                    help="brain: critic earns its say (0 = full say; 1 = by track record; 2 = track record x per-token confidence)")
    ap.add_argument("--gate_lo", type=float, default=0.3, help="brain: critic trust is 0 below this error correlation")
    ap.add_argument("--gate_hi", type=float, default=0.6, help="brain: critic trust is 1 at this error correlation")
    ap.add_argument("--gate_observe_steps", type=int, default=100, help="brain: steps the critic only observes")
    ap.add_argument("--balance_bias", type=float, default=0.0,
                    help="brain: router-bias balancing step (0 = balance with the loss instead)")
    ap.add_argument("--draft_weight", type=float, default=0.1, help="brain: next-token loss weight on the draft")
    ap.add_argument("--critic_weight", type=float, default=0.1, help="brain: weight of the critic's error-prediction loss")
    ap.add_argument("--balance_weight", type=float, default=0.01, help="brain: module load-balancing loss weight")
    ap.add_argument("--aux_stride", type=int, default=8, help="brain: draft/critic losses on every n-th position")
    ap.add_argument("--run", default=None, help="run name; output goes to model/runs/<run>")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--mix", default="fineweb-edu=0.6,code=0.4", help="source=weight,... (rows per batch)")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--micro_batch", type=int, default=0, help="sequences per forward pass; 0 = largest that fits")
    ap.add_argument("--batch_tokens", type=int, default=131072, help="tokens per optimizer step")
    ap.add_argument("--tokens", type=float, default=None, help="total training tokens")
    ap.add_argument("--hours", type=float, default=None, help="train for this long (sets the step count from measured speed)")
    ap.add_argument("--warmup_steps", type=int, default=30)
    ap.add_argument("--cooldown_frac", type=float, default=0.4)
    ap.add_argument("--lr_muon", type=float, default=0.05)
    ap.add_argument("--lr_embed", type=float, default=0.006)
    ap.add_argument("--lr_scalar", type=float, default=0.02, help="AdamW LR for learned scalars")
    ap.add_argument("--wd", type=float, default=0.0, help="decoupled weight decay on Muon params")
    ap.add_argument("--ce_chunk", type=int, default=2048, help="tokens per fused LM-head chunk")
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--fp8", type=int, default=0, help="FP8 GEMMs for the transformer's linear layers (fp8.py)")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_bytes", type=int, default=4_000_000, help="validation text per source (bytes)")
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--stop_at", type=int, default=0, help="stop (eval + save) at this step, keeping the full schedule")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--mem_guard_mb", type=int, default=256, help="VRAM to leave free (0 disables the cap)")
    ap.add_argument("--vram_gb", type=float, default=0,
                    help="cap this run's VRAM (several runs sharing one GPU; 0 = all free memory minus the guard)")
    ap.add_argument("--keep_awake", type=int, default=1, help="keep Windows from idle-sleeping during training")
    return ap.parse_args()


def main():
    args = parse_args()
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved = ckpt["args"]
        for k in ("vocab", "preset", "run", "data", "mix", "seq_len", "micro_batch", "batch_tokens", "warmup_steps",
                  "cooldown_frac", "lr_muon", "lr_embed", "lr_scalar", "wd", "seed", "ce_chunk", "fp8", "arch",
                  "draft_weight", "critic_weight", "balance_weight", "aux_stride"):
            if k in saved:
                setattr(args, k, saved[k])
    args.run = args.run or f"{args.preset}_{args.vocab}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(HERE, "runs", args.run)
    os.makedirs(out_dir, exist_ok=True)

    if args.keep_awake:
        keep_awake()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # bf16 split-K reductions cost up to ~1% relative error in long-K GEMMs (the weight
    # gradients reduce over every token in the micro-batch). Keep reductions in fp32.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if args.vram_gb:  # a share of a GPU that other runs use too
        total = torch.cuda.mem_get_info()[1]
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_gb * 2**30 / total))
    elif args.mem_guard_mb:
        free, total = torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction(max(0.1, (free - args.mem_guard_mb * 2**20) / total))

    if ckpt:
        cfg = (BrainConfig if ckpt["config"].get("arch") == "brain" else GPTConfig)(**ckpt["config"])
    else:
        arch = dict(PRESETS[args.preset])
        arch.update({k: getattr(args, k) for k in ("n_layer", "n_head", "d_model", "mlp_ratio") if getattr(args, k)})
        common = dict(n_ranks=VOCABS[args.vocab], seq_len=args.seq_len, n_kv_head=args.n_kv_head,
                      kv_share=args.kv_share, value_residual=bool(args.value_residual), unet=bool(args.unet),
                      mla_rank=args.mla_rank, mla_rope_dim=args.mla_rope_dim, attn_mode=args.attn_mode,
                      window=args.window, concept_block=args.concept_block, concept_topk=args.concept_topk, concept_rope=bool(args.concept_rope),
                      fast_heads=args.fast_heads, fast_dim=args.fast_dim, depth_every=args.depth_every,
                      depth_capacity=args.depth_capacity, depth_mode=args.depth_mode, mlp_mode=args.mlp_mode,
                      spike_window=args.spike_window, spike_levels=args.spike_levels, **arch)
        if args.arch == "brain":
            cfg = BrainConfig(**common, **{k: getattr(args, k) for k in (
                "trunk_layers", "gate_layers", "n_modules", "module_dim", "module_layers", "module_heads",
                "module_kv_heads", "top_k", "n_shared", "memory_sub", "memory_k", "balance_bias", "critic_gate",
                "gate_lo", "gate_hi", "gate_observe_steps", "module_mode", "capacity", "revive", "router_z", "judge")}, critic=bool(args.critic),
                critic_detach=bool(args.critic_detach), sigmoid_router=bool(args.sigmoid_router))
        else:
            cfg = GPTConfig(**common)
    with torch.device("cuda"):
        model = BrainGPT(cfg) if isinstance(cfg, BrainConfig) else GPT(cfg)
    muon, adam = build_optimizers(model, args)
    if args.fp8:
        import fp8
        print(f"fp8: {fp8.enable(model)} linear layers")
    fwd = torch.compile(model, dynamic=False) if args.compile else model
    data_dir = os.path.join(args.data, args.vocab)

    if not args.micro_batch:
        args.micro_batch = pick_micro_batch(model, fwd, args.seq_len, args.ce_chunk,
                                            max(1, args.batch_tokens // args.seq_len), args)
    accum = max(1, round(args.batch_tokens / (args.micro_batch * args.seq_len)))
    tokens_per_step = accum * args.micro_batch * args.seq_len
    mix = {k: float(v) for k, v in (kv.split("=") for kv in args.mix.split(","))}
    loader = MixLoader(data_dir, mix, args.micro_batch, args.seq_len, args.seed, cfg.n_ranks)
    step, tokens_seen, total_steps = 0, 0, None
    if args.tokens:
        total_steps = math.ceil(args.tokens / tokens_per_step)
    if ckpt:
        model.load_state_dict(ckpt["model"])
        muon.load_state_dict(ckpt["muon"])
        adam.load_state_dict(ckpt["adam"])
        loader.load_state_dict(ckpt["loader"])
        step, tokens_seen, total_steps = ckpt["step"], ckpt["tokens_seen"], ckpt["total_steps"]
        print(f"resumed {args.resume} at step {step}")
    if total_steps is None and not args.hours:
        raise SystemExit("give --tokens or --hours")
    prefetch = Prefetcher(loader)

    token_bytes = torch.from_numpy(token_byte_lengths(cfg.n_ranks, cfg.vocab_size)).cuda()
    peak = measure_peak_tflops()
    fpt = model.flops_per_token()
    n_params, n_emb = model.num_params(), model.embedding_params()
    extras = "".join(f" +{k}" for k in ("value_residual", "unet") if getattr(cfg, k))
    extras += f" +mla{cfg.mla_rank}r{cfg.mla_rope_dim}" if cfg.mla_rank else ""
    extras += (f" +{cfg.attn_mode}{cfg.window}" + (f" concepts/{cfg.concept_block} top{cfg.concept_topk}" if cfg.attn_mode == "concept" else "")) if cfg.attn_mode != "full" else ""
    extras += f" +fast{cfg.fast_heads}x{cfg.fast_dim}" if cfg.fast_heads else ""
    extras += f" | {model.describe()}" if hasattr(model, "describe") else ""
    print(f"model: {args.vocab} vocab, L{cfg.n_layer} d{cfg.d_model} h{cfg.n_head} kv{cfg.n_kv_head} "
          f"share{cfg.kv_share}{extras} | {n_params / 1e6:.1f}M params ({n_emb / 1e6:.1f}M embedding) = "
          f"{n_params * 2 / 2**20:.0f} MiB bf16 | KV cache {model.kv_bytes_per_token() / 1024:.1f} KiB/token | "
          f"{fpt / 1e9:.2f} GFLOP/token")
    print(f"batch: {args.micro_batch} x {args.seq_len} tokens x {accum} accum = {tokens_per_step:,} tokens/step; "
          f"measured bf16 GEMM sustained {peak:.1f} TFLOPS", flush=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(dict(args=vars(args), model=model.config_dict(), params=n_params, embedding_params=n_emb,
                       kv_bytes_per_token=model.kv_bytes_per_token(), flops_per_token=fpt), f, indent=2)
    log = open(os.path.join(out_dir, "log.jsonl"), "a")

    def save():
        state = dict(model=model.state_dict(), muon=muon.state_dict(), adam=adam.state_dict(),
                     loader=prefetch.state_dict(), step=step, tokens_seen=tokens_seen, total_steps=total_steps,
                     args=vars(args), config=model.config_dict())
        tmp = os.path.join(out_dir, "ckpt.pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, os.path.join(out_dir, "ckpt.pt"))

    def run_eval():
        model.eval()
        res = evaluate(model, fwd, data_dir, loader.names, args.seq_len, args.micro_batch, args.eval_bytes,
                       token_bytes)
        model.train()
        mixed_bpb = sum(mix[n] * res[n]["bpb"] for n in res) / sum(mix.values())
        print(f"eval step {step}: " + "  ".join(f"{n} loss {r['loss']:.4f} bpb {r['bpb']:.4f}" for n, r in res.items())
              + f"  | mix bpb {mixed_bpb:.4f}", flush=True)
        log.write(json.dumps(dict(step=step, tokens=tokens_seen, train_seconds=round(train_time, 1), eval=res,
                                  mix_bpb=mixed_bpb)) + "\n")
        log.flush()

    loss_acc = torch.zeros((), device="cuda")
    parts_acc = {}  # brain: auxiliary losses and module usage, summed between log lines
    t_log, steps_since_log, timed_steps, timed_time = time.perf_counter(), 0, 0, 0.0
    train_start = time.perf_counter()
    train_time = 0.0  # wall time spent in training steps (excludes eval and checkpoints)
    bad_steps = 0  # consecutive skipped steps (non-finite loss or gradient)
    params = [p for p in model.parameters() if p.requires_grad]
    while (total_steps is None or step < total_steps) and not (args.stop_at and step >= args.stop_at):
        t_step = time.perf_counter()
        mult = lr_multiplier(step, total_steps or 10**9, args.warmup_steps, args.cooldown_frac)
        for g in muon.param_groups:
            g["lr"] = args.lr_muon * mult
            g["momentum"] = 0.85 + 0.10 * min(1.0, step / 300)
        for g in adam.param_groups:
            g["lr"] = g["base_lr"] * mult

        step_parts = {}  # this step's loss parts, averaged over micro-batches (brain: balancer, critic trust)
        step_main = torch.zeros((), device="cuda")
        for micro in range(accum):
            rows = prefetch.next()
            x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
            with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPA_BACKENDS, set_priority=True):
                h = fwd(x)
            loss, parts = step_loss(model, h, y, args, offset=(step * accum + micro) % args.aux_stride)
            loss = loss / accum
            loss.backward()
            step_main += parts.get("main", loss.detach() * accum) / accum
            for k, v in parts.items():
                step_parts[k] = step_parts.get(k, 0) + v / accum
        # guard: a non-finite loss or gradient skips the step (weights untouched); a run that stays broken
        # stops without saving, so its last checkpoint is still good
        gnorm = torch.stack(torch._foreach_norm([p.grad for p in params if p.grad is not None])).float().sum()
        if not torch.isfinite(step_main + gnorm).item():
            model.zero_grad(set_to_none=True)
            bad_steps += 1
            print(f"step {step}: non-finite loss or gradient, skipped ({bad_steps} in a row)", flush=True)
            if bad_steps >= 20:
                print("failed: training diverged (20 non-finite steps in a row); last checkpoint kept", flush=True)
                raise SystemExit(3)
            continue
        bad_steps = 0
        loss_acc += step_main
        step_parts["grad_norm"] = gnorm  # sum of per-tensor gradient norms: watch for growth before trouble
        for k, v in step_parts.items():
            parts_acc[k] = parts_acc.get(k, 0) + v
        muon.step()
        adam.step()
        model.zero_grad(set_to_none=True)
        if hasattr(model, "update_after_step"):
            model.update_after_step(step_parts)
        step += 1
        tokens_seen += tokens_per_step
        steps_since_log += 1

        if step == 1:
            torch.cuda.synchronize()
            print(f"first step (includes compile): {time.perf_counter() - train_start:.1f}s", flush=True)
            t_log, steps_since_log = time.perf_counter(), 0
            loss_acc.zero_()
            parts_acc.clear()
            continue
        if total_steps is None:  # --hours: time a few steady-state steps, then fix the schedule
            torch.cuda.synchronize()
            if step > 3:
                timed_steps += 1
                timed_time += time.perf_counter() - t_step
            if timed_steps == 5:
                sec_per_step = timed_time / timed_steps * 1.03  # leave ~3% for evals and checkpoints
                total_steps = step + int((args.hours * 3600 - (time.perf_counter() - train_start)) / sec_per_step)
                print(f"--hours {args.hours}: {sec_per_step:.2f} s/step -> {total_steps} steps "
                      f"({total_steps * tokens_per_step / 1e9:.3f}B tokens)", flush=True)
        train_time += time.perf_counter() - t_step

        if step % args.log_every == 0:
            loss_val = loss_acc.item() / steps_since_log
            dt = time.perf_counter() - t_log
            tps = steps_since_log * tokens_per_step / dt
            mem = torch.cuda.max_memory_allocated() / 2**30
            aux = {k: (v / steps_since_log).tolist() for k, v in parts_acc.items() if k != "main"}
            aux_txt = "".join(f" | {k} {v:.3f}" for k, v in aux.items() if not isinstance(v, list))  # lists: json only
            if "usage" in aux:
                aux_txt += f" | modules fire {min(aux['usage']):.0%}-{max(aux['usage']):.0%} of tokens"
            if hasattr(model, "depth_stats"):
                aux_txt += model.depth_stats()
            print(f"step {step}/{total_steps} | loss {loss_val:.4f} | lr x{mult:.3f} | {tps:,.0f} tok/s | "
                  f"MFU {fpt * tps / 1e12 / peak:.1%} | {dt / steps_since_log:.2f} s/step | mem {mem:.1f} GiB | "
                  f"{tokens_seen / 1e9:.3f}B tokens{aux_txt}", flush=True)
            log.write(json.dumps(dict(step=step, tokens=tokens_seen, loss=loss_val, tok_s=tps, lr_mult=mult,
                                      mfu=fpt * tps / 1e12 / peak, **aux)) + "\n")
            log.flush()
            loss_acc.zero_()
            parts_acc.clear()
            t_log, steps_since_log = time.perf_counter(), 0
        if step % args.eval_every == 0 or step == total_steps:
            run_eval()
            t_log = time.perf_counter()
        if step % args.save_every == 0 or step == total_steps or step == args.stop_at:
            save()
            t_log = time.perf_counter()
        if step == args.stop_at:
            if step % args.eval_every:
                run_eval()
            print(f"stopped at step {step} (--stop_at)", flush=True)
            break
    print(f"done: {step} steps, {tokens_seen / 1e9:.3f}B tokens in {(time.perf_counter() - train_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
