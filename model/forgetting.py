"""Learning without forgetting: teach a pretrained model a new domain, measure what it loses.

    python model/forgetting.py prepare --docs docs.jsonl    # documents -> data/ec32k/crawl/*.npy
    python model/forgetting.py run runs/sweep_b32k_bank2/ckpt.pt --train bank --name bank2_bank
    python model/forgetting.py summary

The new domain is a set of documentation pages (the Python docs and the Rust book), one JSON
object per line with a "text" field, or the ready-made data/ec32k/crawl/{train,val}.npy copied in,
split by page into train and held-out validation. Fine-tuning runs a fixed number of steps on
it, and at intervals the model is scored on held-out docs (what it learned), English web text
and general code (what it kept). All scores are bits per byte on the same text.

--replay F mixes that share of general English/code (60/40, from the pretraining shards) into
every fine-tuning batch, the standard way to limit forgetting, as a reference point.

--train picks which weights change: all (everything but the shared embedding), bank (the
module bank and its router only), blocks:I,J (main blocks I..J only, a parameter-matched
partial update for a model without a bank). The question is whether updating only the
modules learns the new domain while forgetting less than updating everything, or than
updating a comparable slice of an ordinary model.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.attention import sdpa_kernel

from brain import BrainConfig, BrainGPT
from gpt import GPT, GPTConfig
from muon import Muon
from prepare_data import load_shard
from tokenizer import VOCABS, Tokenizer
from train import SDPA_BACKENDS, token_byte_lengths

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, "runs", "forgetting")
T = 1024


def prepare(path, vocab="ec32k", val_frac=0.1, seed=0):
    docs = [json.loads(line)["text"] for line in open(path, encoding="utf-8") if line.strip()]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(docs))
    n_val = max(1, int(len(docs) * val_frac))
    tok = Tokenizer(VOCABS[vocab])
    out = os.path.join(HERE, "data", vocab, "crawl")
    os.makedirs(out, exist_ok=True)
    for split, idx in (("val", order[:n_val]), ("train", order[n_val:])):
        ids = np.concatenate(tok.encode_batch([docs[i] for i in idx], prepend_eot=True)).astype(np.uint16)
        np.save(os.path.join(out, f"{split}.npy"), ids)
        print(f"{split}: {len(idx)} pages, {len(ids):,} tokens")


def distill_term(model, hidden, r, picks, pool):
    """Sleep distillation (kNN distillation): on pool rows the target is (1 - beta) one-hot + beta memory's
    distribution over the next token (its k neighbours' continuations with their weights), beta per pool row.
    hidden [B, T, d], r [B, T + 1]; picks = [(row in batch, pool row)]; pool: w [P, T, k], v [P, T, k], beta [P].
    Returns the change to the batch's mean next-token loss (add it to the plain loss)."""
    b = torch.tensor([p[0] for p in picks], device=hidden.device)
    j = torch.tensor([p[1] for p in picks], device=hidden.device)
    lp = torch.log_softmax(model.logits(hidden[b].float()), -1)  # [n, T, vocab]
    soft = -(pool["w"][j].float() * lp.gather(-1, pool["v"][j].long())).sum(-1)
    hard = -lp.gather(-1, r[b, 1:, None])[..., 0]
    return (pool["beta"][j, None] * (soft - hard)).sum() / r[:, 1:].numel()


def load_model(path):
    if not path.endswith(".pt"):  # an exported run (safetensors + JSON), e.g. one trained on a rented GPU
        import safe_checkpoint
        model, meta = safe_checkpoint.load_model(path)
        return model, meta
    ck = torch.load(path, map_location="cpu", weights_only=False)
    is_brain = ck["config"].get("arch") == "brain"
    cfg = (BrainConfig if is_brain else GPTConfig)(**ck["config"])
    with torch.device("cuda"):
        model = BrainGPT(cfg) if is_brain else GPT(cfg)
    model.load_state_dict(ck["model"])
    return model, ck


def load_grown(path):
    """(model, meta) from a continual run saved with --save: its base export plus the grown parts, in the order they
    were added (modules at the same depth apply in turn). Nothing is unpickled."""
    from safetensors.torch import load_file
    from brain import ModuleBank
    meta = json.load(open(os.path.join(path, "grown.json")))
    base = meta["base"] if os.path.isabs(meta["base"]) else os.path.join(HERE, meta["base"])
    model, _ = load_model(base)
    model.grown_at, model.grown_gates = nn.ModuleDict(), nn.ModuleDict()
    with torch.device("cuda"):
        for key in meta["keys"]:
            model.grown_at[key] = ModuleBank(model.cfg, meta["k"][key], attention=True)
            model.grown_gates[key] = nn.Linear(model.cfg.d_model, meta["k"][key])
    parts = nn.ModuleDict({"grown_at": model.grown_at, "grown_gates": model.grown_gates})
    parts.load_state_dict(load_file(os.path.join(path, "grown.safetensors"), device="cuda"))
    model.eval()
    return model, meta


def rows_from(ids, n_rows, offset=0):
    n_rows = min(n_rows, (len(ids) - 1 - offset) // T)
    return np.stack([ids[offset + r * T: offset + r * T + T + 1] for r in range(n_rows)]).astype(np.int64)


@torch.no_grad()
def score(model, rows, token_bytes, batch=8):
    """Bits per byte over the rows."""
    model.eval()
    loss_sum, byte_sum = 0.0, 0
    for i in range(0, len(rows), batch):
        r = torch.from_numpy(rows[i:i + batch]).cuda()
        x, y = r[:, :-1], r[:, 1:]
        with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPA_BACKENDS, set_priority=True):
            h = model(x)
        loss_sum += model.token_losses(h, y).sum(dtype=torch.float64).item()
        byte_sum += token_bytes[y].sum().item()
    model.train()
    return loss_sum / math.log(2) / byte_sum


def grow(model, k, calib=None):
    """Neurogenesis: give a trained model k new modules beside its existing ones. They form a separate,
    additive bank (each module sees every token, with its own sigmoid gate) whose output starts at exactly
    zero, so the model is unchanged at first; only the new modules and their gate learn, while every old
    weight, including the old router, stays frozen. (Growing the existing routed bank disturbed the old
    routing: a new module competing in the top-k displaces a token's second choice.)
    Returns (model, trainable parameters)."""
    from brain import ModuleBank
    cfg = model.cfg
    with torch.device("cuda"):
        model.grown = ModuleBank(cfg, k, attention=True)
        model.grown_gate = nn.Linear(cfg.d_model, k)
    model.grown.reset_parameters()            # w_out starts at zero: no change until it learns
    with torch.no_grad():
        nn.init.normal_(model.grown_gate.weight, std=0.02)
        nn.init.zeros_(model.grown_gate.bias)
    return model, list(model.grown.parameters()) + list(model.grown_gate.parameters())


def grow_deep(model, k, every=2):
    """Neurogenesis at several depths: k new modules after every `every`-th block (as grow(), each set with
    its own gate and an output starting at zero, old weights frozen)."""
    from brain import ModuleBank
    cfg = model.cfg
    layers = [str(i) for i in range(1, cfg.n_layer, every)]
    with torch.device("cuda"):
        model.grown_at = nn.ModuleDict({i: ModuleBank(cfg, k, attention=True) for i in layers})
        model.grown_gates = nn.ModuleDict({i: nn.Linear(cfg.d_model, k) for i in layers})
    with torch.no_grad():
        for i in layers:
            model.grown_at[i].reset_parameters()
            nn.init.normal_(model.grown_gates[i].weight, std=0.02)
            nn.init.zeros_(model.grown_gates[i].bias)
    return model, list(model.grown_at.parameters()) + list(model.grown_gates.parameters())


def trainable(model, spec):
    if spec == "all":
        return [p for n, p in model.named_parameters() if n != "wte.weight"]
    if spec == "bank":
        assert getattr(model, "bank", None) is not None, "this model has no module bank"
        shared = list(model.shared.parameters()) if getattr(model, "shared", None) is not None else []
        return list(model.bank.parameters()) + shared + list(model.router.parameters())
    if spec.startswith("blocks:"):
        i, j = map(int, spec[7:].split(","))
        return [p for b in model.blocks[i:j + 1] for p in b.parameters()]
    raise SystemExit(f"unknown --train {spec}")


def run(a):
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    model, ck = load_model(a.ckpt)
    vocab = ck["args"]["vocab"]
    for p in model.parameters():
        p.requires_grad_(False)
    masks = []
    if a.train.startswith("growdeep:"):
        model, params = grow_deep(model, int(a.train[9:]))
        for p in model.parameters():
            p.requires_grad_(False)
    elif a.train.startswith("grow:"):
        a.grow_k = int(a.train[5:])
        model, params = grow(model, a.grow_k)
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        params = trainable(model, a.train)
    for p in params:
        p.requires_grad_(True)
    ids_train = np.load(os.path.join(HERE, "data", vocab, "crawl", "train.npy"))
    ids_val = np.load(os.path.join(HERE, "data", vocab, "crawl", "val.npy"))
    keep = {}
    for src in ("fineweb-edu", "code"):
        val, _ = load_shard(os.path.join(HERE, "data", vocab, src, "val_000000.bin"))
        keep[src] = rows_from(val, a.keep_tokens // T)
    docs_val = rows_from(ids_val, 10 ** 9)
    replay = {src: load_shard(os.path.join(HERE, "data", vocab, src, "train_000000.bin"))[0]
              for src in ("fineweb-edu", "code")} if a.replay else {}
    n_replay = int(round(a.batch * a.replay))
    token_bytes = torch.from_numpy(token_byte_lengths(VOCABS[vocab], model.cfg.vocab_size)).cuda()

    ids_matrix = {id(p) for p in model.matrix_params()}
    mats = [p for p in params if id(p) in ids_matrix]
    others = [p for p in params if id(p) not in ids_matrix]
    splits = {p: s for p, s in model.muon_splits().items() if id(p) in {id(q) for q in mats}}
    muon = Muon(mats, lr=a.lr_muon, momentum=0.95, splits=splits) if mats else None
    adam = torch.optim.AdamW(others, lr=a.lr_adam, betas=(0.8, 0.95), eps=1e-10) if others else None

    def evaluate(step):
        r = dict(step=step, docs=score(model, docs_val, token_bytes),
                 **{k: score(model, v, token_bytes) for k, v in keep.items()})
        print(f"step {step:4d} | docs {r['docs']:.4f} | english {r['fineweb-edu']:.4f} | code {r['code']:.4f}", flush=True)
        return r

    n_train = sum(p.numel() for p in params)
    print(f"{a.name}: {a.ckpt} | training {a.train} = {n_train / 1e6:.1f}M of {model.num_params() / 1e6:.1f}M params | "
          f"crawl train {len(ids_train):,} tokens, val {docs_val.size:,}")
    curve = [evaluate(0)]
    rng = np.random.default_rng(a.seed)
    model.train()
    t0 = time.time()
    for step in range(1, a.steps + 1):
        warm = min(1.0, step / 10)
        if muon:
            for g in muon.param_groups:
                g["lr"] = a.lr_muon * warm
        if adam:
            for g in adam.param_groups:
                g["lr"] = a.lr_adam * warm
        for _ in range(a.accum):
            starts = rng.integers(0, len(ids_train) - T - 1, size=a.batch - n_replay)
            rows = [ids_train[s:s + T + 1] for s in starts]
            for _ in range(n_replay):
                src = replay["fineweb-edu" if rng.random() < 0.6 else "code"]
                s = rng.integers(0, len(src) - T - 1)
                rows.append(src[s:s + T + 1])
            r = torch.from_numpy(np.stack(rows).astype(np.int64)).cuda()
            x, y = r[:, :-1], r[:, 1:]
            with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPA_BACKENDS, set_priority=True):
                out = model(x)
            if hasattr(model, "train_loss"):
                loss, parts = model.train_loss(out, y, draft_weight=0.0, critic_weight=0.0, balance_weight=0.0)
            else:
                loss = model.loss(out, y)
            if not torch.isfinite(loss):  # never train on, or report, a diverged model
                raise SystemExit(f"non-finite loss at step {step}: run stopped, results not saved")
            (loss / a.accum).backward()
        for opt in (muon, adam):
            if opt:
                opt.step()
        model.zero_grad(set_to_none=True)
        if step in a.eval_at:
            curve.append(evaluate(step))
    res = dict(name=a.name, ckpt=a.ckpt, train=a.train, replay=a.replay, trainable_params=n_train, steps=a.steps,
               tokens=a.steps * a.accum * a.batch * T, seconds=round(time.time() - t0), curve=curve)
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(res, open(os.path.join(OUT_DIR, f"{a.name}.json"), "w"), indent=1)


def lost_at(curve, target):
    """Mean English/code loss increase at the point the run had learned `target` bpb of docs
    (linear interpolation between evaluations); None if it never learned that much."""
    c0 = curve[0]
    prev = (0.0, 0.0)
    for c in curve[1:]:
        learned = c0["docs"] - c["docs"]
        lost = (c["fineweb-edu"] - c0["fineweb-edu"] + c["code"] - c0["code"]) / 2
        if learned >= target:
            f = (target - prev[0]) / (learned - prev[0])
            return prev[1] + f * (lost - prev[1])
        prev = (learned, lost)
    return None


class Adapter(nn.Module):
    """Standard bottleneck adapter (the parameter-efficient baseline): x + up(relu(down(norm(x)))), up at zero.
    Same call signature as a grown ModuleBank, so it plugs into BrainGPT's grown_at slots."""

    def __init__(self, d, r):
        super().__init__()
        self.down, self.up = nn.Linear(d, r, bias=False), nn.Linear(r, d, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x, cos, sin, g):
        h = self.up(torch.relu(self.down(nn.functional.rms_norm(x, (x.size(-1),)))))
        return h * g[..., :1], None


def continual(a):
    """Continued training by neurogenesis (the owner's design, 5 parts):
    1. small, frequent additions: the new material arrives in --stages parts; each grows --per_stage modules;
    2. placement by need (--placement need): new modules go after the block where the new material's loss
       gradient on the residual stream is largest (fixed: spread over blocks 1, 3, 5, ...);
    3. 'do what the neighbours do' (--init copy): a new module starts as a copy of an existing (frozen) bank
       module's internals with its output projection at zero: silent, but with learned features;
    4. consolidation: after each stage, --consolidate_steps of replay only (half general text, half the new
       material seen so far) at a third of the learning rate, like sleep;
    5. each stage's modules are named, separate parts (grown_at keys "depth_stage") that can be removed.
    The base and earlier stages stay frozen: only the current stage's modules (and their gates) learn.
    --stages 1 --placement fixed is the one-shot baseline (per_stage modules spread over blocks 1, 3, 5).
    With --memory_tokens each specialty also enters an episodic memory (episodic.py); --sleep_distill makes sleep
    distil it: on replayed specialty text the target mixes the true next token with memory's prediction.
    With --stream (wai_struct) nothing is labeled: each day the model reads the specialty mixed into familiar text,
    a hippocampal buffer keeps the windows that surprise it, and the night grows and learns from the buffer."""
    from brain import ModuleBank
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    model, ck = load_model(a.ckpt)
    cfg, vocab = model.cfg, ck["args"]["vocab"]
    for p in model.parameters():
        p.requires_grad_(False)
    model.grown_at, model.grown_gates = nn.ModuleDict(), nn.ModuleDict()
    if a.domains:  # a sequence of specialties (build_domains.py): one stage per domain, every domain scored
        names = a.domains.split(",")
        ddir = os.path.join(HERE, "data", vocab, "domains")
        stages = [np.load(os.path.join(ddir, n, "train.npy")) for n in names]
        vals = {n: rows_from(np.load(os.path.join(ddir, n, "val.npy")), 10 ** 9) for n in names}
        a.stages = len(names)
        val_ids = {n: np.load(os.path.join(ddir, n, "val.npy")) for n in names}
    else:
        ids_train = np.load(os.path.join(HERE, "data", vocab, "crawl", "train.npy"))
        stages = np.array_split(ids_train, a.stages)
        vals = {"docs": rows_from(np.load(os.path.join(HERE, "data", vocab, "crawl", "val.npy")), 10 ** 9)}
    keep = {src: rows_from(load_shard(os.path.join(HERE, "data", vocab, src, "val_000000.bin"))[0], a.keep_tokens // T)
            for src in ("fineweb-edu", "code")}
    general = [load_shard(os.path.join(HERE, "data", vocab, src, "train_000000.bin"))[0]
               for src in ("fineweb-edu", "code")]
    token_bytes = torch.from_numpy(token_byte_lengths(VOCABS[vocab], model.cfg.vocab_size)).cuda()
    rng = np.random.default_rng(a.seed)

    @torch.no_grad()
    def recall(n_seq, batch=64):
        """The model's own memories: n_seq sequences of T + 1 tokens sampled from it before anything new is learned.
        Free recall starts from a document boundary; cued recall (--recall_prefix N) continues the first N tokens
        of a real passage, the way a fragment triggers a memory. Brains replay reconstructions, not records."""
        eot = Tokenizer(VOCABS[vocab]).eot
        model.eval()
        gen = torch.Generator(device="cuda").manual_seed(a.seed)
        out = []
        for i in range(0, n_seq, batch):
            n = min(batch, n_seq - i)
            if a.recall_prefix:  # cues: fragments of real old text
                cues = [window(general[0] if len(general) == 1 or rng.random() < 0.6 else general[1])[:a.recall_prefix]
                        for _ in range(n)]
                ids = torch.from_numpy(np.stack(cues).astype(np.int64)).cuda()
            else:
                ids = torch.full((n, 1), eot, dtype=torch.long, device="cuda")
            while ids.size(1) < T + 1:
                with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPA_BACKENDS, set_priority=True):
                    h = model(ids)[:, -1]
                nxt = torch.multinomial(torch.softmax(model.logits(h) / a.recall_temp, -1), 1, generator=gen)
                ids = torch.cat([ids, nxt], 1)
            out.append(ids.cpu().numpy())
            print(f"  recalled {i + len(ids)}/{n_seq} sequences", flush=True)
        model.train()
        return np.concatenate(out).astype(np.uint16).reshape(-1)  # one stream; windows may span sequences


    def evaluate(tag):
        r = dict(tag=tag, **{k: score(model, v, token_bytes) for k, v in vals.items()},
                 **{k: score(model, v, token_bytes) for k, v in keep.items()})
        print(f"{tag:24s} | " + " | ".join(f"{k} {r[k]:.4f}" for k in vals) +
              f" | english {r['fineweb-edu']:.4f} | code {r['code']:.4f}", flush=True)
        return r

    def window(src):
        st = rng.integers(0, len(src) - T - 1)
        return src[st:st + T + 1]

    pools = {}  # sleep priority: replay drawn from tagged windows instead of at random
    picks = []  # sleep distillation: (row in the batch, pool row) for each new row of the last batch

    def batch(new_sources, new_share):
        rows = []
        picks.clear()
        for b in range(a.batch):
            if rng.random() < new_share:
                if "distill" in pools:  # new material comes from the pool that carries memory's predictions
                    j = int(rng.integers(len(pools["distill"]["rows"])))
                    picks.append((b, j))
                    rows.append(pools["distill"]["rows"][j])
                else:
                    rows.append(pools["new"][rng.integers(len(pools["new"]))] if "new" in pools else
                                window(new_sources[rng.integers(len(new_sources))]))
            else:
                rows.append(pools["old"][rng.integers(len(pools["old"]))] if "old" in pools else
                            window(general[0] if len(general) == 1 or rng.random() < 0.6 else general[1]))
        return torch.from_numpy(np.stack(rows).astype(np.int64)).cuda()

    # training steps run compiled (fixed shapes); scoring, placement and recall stay eager. Each stage's new
    # modules change the graph, so it recompiles once per stage (wake and sleep share it)
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
    fwd = torch.compile(model, dynamic=False) if a.compile else model

    def train(params, steps, new_sources, new_share, lr):
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.8, 0.95), eps=1e-10)
        model.train()
        for step in range(1, steps + 1):
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, step / 10)
            for _ in range(a.accum):
                r = batch(new_sources, new_share)
                with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPA_BACKENDS, set_priority=True):
                    out = fwd(r[:, :-1])
                loss = model.train_loss(out, r[:, 1:], draft_weight=0.0, critic_weight=0.0, balance_weight=0.0)[0]
                if picks:
                    loss = loss + distill_term(model, out[0], r, picks, pools["distill"])
                if not torch.isfinite(loss):  # never train on, or report, a diverged model
                    raise SystemExit(f"non-finite loss at step {step}: run stopped, results not saved")
                (loss / a.accum).backward()
            opt.step()
            model.zero_grad(set_to_none=True)

    @torch.no_grad()
    def dissonance(rows, depths):
        """Mean KL(final prediction || prediction read out after block d) per window, for each d: how far the model's
        view at depth d is from its considered view (its internal conflict there). Returns [len(rows), len(depths)]."""
        outs = {}
        hooks = [model.blocks[d].register_forward_hook(lambda m, i, o, d=d: outs.__setitem__(d, o)) for d in depths]
        model.eval()
        res = torch.zeros(len(rows), len(depths))
        for i in range(0, len(rows), 8):
            r = torch.from_numpy(np.stack(rows[i:i + 8]).astype(np.int64)).cuda()
            with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPA_BACKENDS, set_priority=True):
                h = model(r[:, :-1])
            for b in range(len(r)):
                late = torch.log_softmax(model.logits(h[b].float()), -1)
                for j, d in enumerate(depths):
                    early = torch.log_softmax(model.logits(nn.functional.rms_norm(outs[d][b].float(), (h.size(-1),))), -1)
                    res[i + b, j] = (late.exp() * (late - early)).nan_to_num(0.0).sum(-1).mean().item()
        for hk in hooks:
            hk.remove()
        model.train()
        return res

    def tag_for_sleep(seen):
        """Hippocampal tagging before sleep: of many candidate windows, keep the quarter where the model is most
        conflicted (first impression at the routing depth vs final view), for new and for old material alike.
        The pools must be large: tiny ones (64/256 windows) get replayed ~25 times each and memorised."""
        d = [model.cfg.trunk_layers - 1]
        for kind, n in (("new", a.tag_new), ("old", a.tag_old)):
            cand = [window(seen[rng.integers(len(seen))]) if kind == "new" else
                    window(general[0] if len(general) == 1 or rng.random() < 0.6 else general[1]) for _ in range(2 * n)]
            score = dissonance(cand, d)[:, 0]
            pools[kind] = [cand[j] for j in torch.argsort(score, descending=True)[:n].tolist()]
        print(f"  tagged for sleep: {len(pools['new'])} new + {len(pools['old'])} old windows", flush=True)

    def most_conflicted_depth(src, cands, rows=None):
        """Placement by dissonance: the depth where the new material is most conflicted relative to familiar text.
        rows: draw the new material from these windows (label-free arrival: the hippocampal buffer) instead of src."""
        pick = [rows[i] for i in rng.choice(len(rows), min(16, len(rows)), replace=False)] if rows is not None else \
            [window(src) for _ in range(16)]
        new = dissonance(pick, cands).mean(0)
        old = dissonance([window(general[0] if rng.random() < 0.6 else general[1]) for _ in range(16)], cands).mean(0)
        gap = (new - old).tolist()
        return cands[int(np.argmax(gap))], [round(g, 3) for g in gap]

    def neediest_depth(src):
        """Block whose output most needs to change for the new material: mean gradient norm on the stream."""
        outs = []

        def keep_grad(m, i, o):
            o.retain_grad()
            outs.append(o)

        hooks = [b.register_forward_hook(keep_grad) for b in model.blocks]
        model.wte.weight.requires_grad_(True)
        r = torch.from_numpy(np.stack([window(src) for _ in range(a.batch)]).astype(np.int64)).cuda()
        model.train()
        with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPA_BACKENDS, set_priority=True):
            out = model(r[:, :-1])
        model.train_loss(out, r[:, 1:], draft_weight=0.0, critic_weight=0.0, balance_weight=0.0)[0].backward()
        for h in hooks:
            h.remove()
        model.wte.weight.requires_grad_(False)
        need = [o.grad.float().norm(dim=-1).mean().item() for o in outs[:len(model.blocks)]]
        model.zero_grad(set_to_none=True)
        cands = list(range(1, len(model.blocks) - 1))
        crowd = {i: sum(k.split("_")[0] == str(i) for k in model.grown_at) for i in cands}
        if a.placement == "need_spread":  # a niche that already holds new modules is less attractive
            return max(cands, key=lambda i: need[i] / (1 + crowd[i])), need
        return max(cands, key=lambda i: need[i]), need

    def active_modules(src):
        """How often each routed bank module fires on the new material."""
        r = torch.from_numpy(np.stack([window(src) for _ in range(a.batch)]).astype(np.int64)).cuda()
        model.train()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPA_BACKENDS, set_priority=True):
            usage = model(r[:, :-1])[4]
        u = usage.float().cpu().numpy()
        return u[len(u) - model.bank.E:]

    def silence(key, js, on=True):
        """Gate bias far negative: modules js of part key contribute nothing (removal, or pruning)."""
        g = model.grown_gates[key]
        with torch.no_grad():
            for j in js:
                if on:
                    saved[(key, j)] = g.bias[j].item()
                    g.bias[j] = -30.0
                else:
                    g.bias[j] = saved.pop((key, j))

    def new_loss(src, n=4):
        rr = np.random.default_rng(0)  # same windows each call, so ablations compare like with like
        rows = np.stack([src[st:st + T + 1] for st in rr.integers(0, len(src) - T - 1, n * a.batch)])
        return score(model, rows.astype(np.int64), token_bytes)

    def prune(key, k_keep, src):
        """Survival by usefulness: keep the k_keep modules whose removal hurts the new material most."""
        k = model.grown_gates[key].out_features
        base, harm = new_loss(src), []
        for j in range(k):
            silence(key, [j])
            harm.append(new_loss(src) - base)
            silence(key, [j], on=False)
        dead = [int(j) for j in np.argsort(harm)[:k - k_keep]]
        silence(key, dead)
        for j in dead:
            saved.pop((key, j))  # permanent
        return [round(h, 5) for h in harm], dead

    saved = {}

    def add_adapters(stage):
        """Baseline: adapters in every block, together as many parameters as per_stage grown modules."""
        with torch.device("cuda"):
            budget = sum(p.numel() for p in ModuleBank(cfg, a.per_stage, attention=True).parameters())
        r = max(1, round(budget / (2 * cfg.d_model * len(model.blocks))))
        params, keys = [], []
        for b in range(len(model.blocks)):
            key = f"{b}_{stage}"
            with torch.device("cuda"):
                ad, gate = Adapter(cfg.d_model, r), nn.Linear(cfg.d_model, 1)
            with torch.no_grad():
                nn.init.normal_(gate.weight, std=0.02)
                nn.init.zeros_(gate.bias)
            model.grown_at[key], model.grown_gates[key] = ad, gate
            params += list(ad.parameters()) + list(gate.parameters())
            keys.append(key)
        print(f"  adapters r={r} in {len(keys)} blocks: {sum(p.numel() for p in params):,} params "
              f"(grown-module budget {budget:,})", flush=True)
        return params, keys

    def add_modules(depth, stage, k):
        key = f"{depth}_{stage}"
        with torch.device("cuda"):
            bank, gate = ModuleBank(cfg, k, attention=True), nn.Linear(cfg.d_model, k)
        bank.reset_parameters()  # output projection at zero: silent at first
        with torch.no_grad():
            nn.init.normal_(gate.weight, std=0.02)
            nn.init.zeros_(gate.bias)
            if a.init in ("copy", "copy_active"):  # internals copied from existing modules: their features
                src, de = model.bank, cfg.module_dim
                if a.init == "copy_active":  # the modules the new material uses most (repeats allowed)
                    u = active_modules(new)
                    picks = [int(i) for i in np.argsort(-u)[:k]] if k <= src.E else                         [int(i) for i in np.argsort(-u)[np.arange(k) % src.E]]
                else:
                    picks = rng.choice(src.E, size=k, replace=k > src.E)
                for j, i in enumerate(picks):
                    bank.w_in[j * de:(j + 1) * de] = src.w_in[i * de:(i + 1) * de]
                    for pl_new, pl_src in ((bank.qkv, src.qkv), (bank.proj, src.proj), (bank.fc, src.fc),
                                           (bank.fc2, src.fc2)):
                        for t_new, t_src in zip(pl_new, pl_src):
                            t_new[j] = t_src[i]
        model.grown_at[key], model.grown_gates[key] = bank, gate
        return list(bank.parameters()) + list(gate.parameters()), key

    # label-free arrival (--stream): nobody says what is new. By day the model only reads an unlabeled stream and a
    # hippocampal buffer keeps what surprises it; by night it grows, learns from the buffer, then sleeps.
    familiar = {}  # the model's picture of familiar text, redrawn before each day (calibrate)

    @torch.no_grad()
    def read(rows, batch=8):
        """What surprise needs from each window: its bits per byte (--surprise loss) or its mean state after block D,
        normalised (today@D, maha@D, knn@D). Grown modules at D apply after the hook, as in placement's dissonance."""
        kind, _, d = a.surprise.partition("@")
        outs = {}
        hk = model.blocks[int(d)].register_forward_hook(lambda m, i, o: outs.__setitem__("h", o)) if d else None
        model.eval()
        res = []
        for i in range(0, len(rows), batch):
            r = torch.from_numpy(np.stack(rows[i:i + batch]).astype(np.int64)).cuda()
            x, y = r[:, :-1], r[:, 1:]
            with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPA_BACKENDS, set_priority=True):
                h = model(x)
                if kind == "loss":
                    res.append(model.token_losses(h, y).float().sum(1) / token_bytes[y].sum(1) / math.log(2))
                else:
                    res.append(nn.functional.rms_norm(outs["h"].float(), (h.size(-1),)).mean(1))
        if hk is not None:
            hk.remove()
        model.train()
        return torch.cat(res)

    def surprise(f):
        """Per window, how unlike familiar text it is: loss as is; maha = Mahalanobis distance of the state from
        familiar text's (shrunk covariance); knn = cosine distance to the 10 nearest familiar windows."""
        kind = a.surprise.partition("@")[0]
        if kind == "loss":
            return f.cpu().numpy()
        if kind == "maha":
            z = f - familiar["mu"]
            return ((z @ familiar["prec"]) * z).sum(-1).sqrt().cpu().numpy()
        sims = nn.functional.normalize(f, dim=-1) @ nn.functional.normalize(familiar["ref"], dim=-1).T
        return (1 - sims.topk(10, dim=-1).values.mean(-1)).cpu().numpy()

    def today(f, held):
        """today@D: for each of the day's windows, the share of its --surprise_k nearest neighbours (cosine on the
        state, itself excluded) among the day's windows plus familiar text that come from the day: "this keeps
        coming up today, and I don't usually see it". The same share for familiar windows outside the day is the
        null that sets the threshold. (A single window of a new topic is often no stranger than familiar text; what
        sets the topic apart is that it recurs: surprise_probe.py.)"""
        day_, ref, held = (nn.functional.normalize(v, dim=-1) for v in (f, familiar["ref"], held))
        pool = torch.cat([day_, ref])
        from_day = torch.cat([torch.ones(len(day_)), torch.zeros(len(ref))]).cuda()

        def share(x, self_rows):
            out = []
            for i in range(0, len(x), 1024):
                sims = x[i:i + 1024] @ pool.T
                if self_rows:
                    j = torch.arange(len(sims), device=sims.device)
                    sims[j, i + j] = -2.0
                out.append(from_day[sims.topk(a.surprise_k, dim=-1).indices].mean(-1))
            return torch.cat(out).cpu().numpy()
        return share(day_, True), share(held, False)

    def calibrate():
        """Before each day: read --surprise_ref windows of familiar text with the current model. Half fit the
        picture of familiar text, the other half set the threshold: the --surprise_q quantile of their surprise
        (so about 1 - q of familiar text gets kept by mistake)."""
        ref = [window(general[0] if rng.random() < 0.6 else general[1]) for _ in range(a.surprise_ref)]
        f = read(ref)
        fit, held = f[:len(f) // 2], f[len(f) // 2:]
        if a.surprise.startswith("maha"):
            mu = fit.mean(0)
            c = torch.cov((fit - mu).T)
            c = 0.9 * c + 0.1 * c.diagonal().mean() * torch.eye(len(c), device=c.device)
            familiar.update(mu=mu, prec=torch.linalg.inv(c))
        elif a.surprise.startswith(("knn", "today")):
            familiar["ref"] = fit
        if a.surprise.startswith("today"):  # the threshold depends on the day itself: set in day()
            familiar["held"] = held
        else:
            familiar["thr"] = float(np.quantile(surprise(held), a.surprise_q))

    def day(st):
        """One day's stream: all of the specialty's training text as consecutive windows (random start), mixed with
        familiar text so the specialty is --stream_share of the stream, shuffled, unlabeled. The model only reads it;
        windows above the threshold go to the buffer (at most --buffer_max, the most surprising). The labels are
        used only to report how well the buffer caught the new material."""
        ids = stages[st]
        off = int(rng.integers(0, T))
        n_new = (len(ids) - off - 1) // T
        if a.day_windows:
            n_new = min(n_new, int(a.day_windows * a.stream_share))
        n_old = int(round(n_new * (1 - a.stream_share) / a.stream_share))
        rows = [ids[off + i * T: off + i * T + T + 1] for i in range(n_new)] + \
               [window(general[0] if rng.random() < 0.6 else general[1]) for _ in range(n_old)]
        is_new = np.r_[np.ones(n_new, bool), np.zeros(n_old, bool)]
        perm = rng.permutation(len(rows))
        rows, is_new = [rows[i] for i in perm], is_new[perm]
        if a.stream_oracle:  # control: a perfect detector
            keep, thr = is_new.copy(), float("nan")
        else:
            calibrate()
            if a.surprise.startswith("today"):
                s, null = today(read(rows), familiar["held"])
                thr = float(np.quantile(null, a.surprise_q))
            else:
                s, thr = surprise(read(rows)), familiar["thr"]
            keep = s > thr
            if a.buffer_max and keep.sum() > a.buffer_max:  # the hippocampus is small: the most surprising stay
                keep &= s >= np.sort(s[keep])[-a.buffer_max]
        tp = int((keep & is_new).sum())
        stats = dict(read=len(rows), specialty=n_new, kept=int(keep.sum()), threshold=thr,
                     precision=tp / max(1, int(keep.sum())), recall=tp / max(1, n_new),
                     false_keep=int((keep & ~is_new).sum()) / max(1, n_old))
        print(f"  day {st + 1}: read {len(rows)} windows ({n_new} new), kept {stats['kept']} | precision "
              f"{stats['precision']:.3f} recall {stats['recall']:.3f} familiar kept {stats['false_keep']:.3f}", flush=True)
        return [rows[i] for i in np.flatnonzero(keep)], stats, [rows[i] for i in np.flatnonzero(~keep)]

    if a.replay_source == "recall":  # old material comes only from the model's own memories, wake and sleep
        general = [recall(a.recall_seqs)]
    curve = [evaluate("start")]
    mem, mem_curve, distilled = None, [], []
    if a.save and a.adapter:
        raise SystemExit("--save saves grown modules, not adapters")
    if a.sleep_distill and not a.memory_tokens:
        raise SystemExit("--sleep_distill needs --memory_tokens")
    if a.memory_tokens:  # the hippocampus: each specialty is read into episodic memory as it arrives (episodic.py)
        if not a.domains:
            raise SystemExit("--memory_tokens needs --domains")
        from episodic import EpisodicMemory
        mem = EpisodicMemory(model, token_bytes, k=a.memory_k)
        mem_texts = dict(val_ids, **{src: load_shard(os.path.join(HERE, "data", vocab, src, "val_000000.bin"))[0]
                                     for src in ("fineweb-edu", "code")})

        def prepare_distill(st):
            """Before sleep: re-key memory with the woken model, draw --distill_rows windows from the specialties
            seen so far (beyond the text memory read, so a window never finds itself), and store memory's k
            neighbours and weights for every position. The blend (lam, tau) is the best on each specialty's own
            windows, as in scoring; where memory doesn't help, beta = 0 (plain replay)."""
            from episodic import encode
            mem.reencode()
            per = max(1, a.distill_rows // (st + 1))
            rows_, w_, v_, beta_, info = [], [], [], [], {}
            for k in range(st + 1):
                ids = stages[k]
                lo = int(a.memory_tokens) if len(ids) - int(a.memory_tokens) > 2 * T else 0
                rows = np.stack([ids[o:o + T + 1] for o in rng.integers(lo, len(ids) - T - 1, per)])
                parts = []  # (distances, next tokens of the neighbours, true next token, log p_model) per 128 rows
                for c in range(0, per, 128):
                    h, y, lp = encode(model, rows[c:c + 128].astype(np.int64))
                    d, i = mem.recall(h)
                    parts.append((d, mem.vals[i].int(), y, lp))
                    del h, i
                d, v, y, lp = (torch.cat(t) for t in zip(*parts))
                del parts
                match = (v == y[:, None]).float()
                best = (lp.sum().item(), 0.0, 1.0)  # log-likelihood of the model alone
                for tau in (1.0, 3.0, 10.0, 30.0, 100.0):
                    pm = (torch.softmax(-d / tau, -1) * match).sum(-1)
                    for lam in (0.05, 0.1, 0.2, 0.3, 0.5):
                        ll = torch.log((1 - lam) * lp.exp() + lam * pm + 1e-12).sum().item()
                        if ll > best[0]:
                            best = (ll, lam, tau)
                ll, lam, tau = best
                rows_.append(rows)
                w_.append(torch.softmax(-d / tau, -1).half().view(per, T, -1))
                v_.append(v.view(per, T, -1))
                beta_ += [lam * a.sleep_distill] * per
                info[names[k]] = dict(lam=lam, tau=tau, gain=round((ll - lp.sum().item()) / -lp.sum().item(), 5))
                del d, v, y, lp, match
            pools["distill"] = dict(rows=np.concatenate(rows_), w=torch.cat(w_), v=torch.cat(v_),
                                    beta=torch.tensor(beta_, device="cuda"))
            print("  sleep distillation: " + " | ".join(f"{n[:7]} lam {x['lam']} tau {x['tau']} ({x['gain']:+.2%})"
                                                       for n, x in info.items()), flush=True)
            return info

        def mem_evaluate(tag, held=None):
            r = dict(tag=tag, episodes=len(mem), episodes_held=mem.n_held(), held=held)
            for n, ids in mem_texts.items():
                r[n] = mem.score(ids, a.memory_eval_tokens)
            print(f"  memory {tag:10s} {len(mem):,} episodes ({mem.n_held():,} held) | model -> all / held: " + " | ".join(
                f"{n[:7]} {r[n]['model']:.4f}->{r[n]['memory']:.4f}/{r[n]['held']:.4f}" for n in mem_texts), flush=True)
            return r
    fixed = [int(d) for d in a.depths.split(",")] if a.depths else list(range(1, len(model.blocks) - 1, 2))
    seen, layout, pruned = [], [], []
    buffers, stream_stats = [], []  # label-free arrival: each night's buffer (kept for later sleeps), day reports
    if a.stream and (not a.domains or not a.grow or a.adapter or mem is not None or a.sleep_priority != "none"
                     or a.replay_source != "data" or a.overgrow > 1 or a.stages == 1):
        raise SystemExit("--stream needs --domains with grown modules, more than one stage, and none of: adapters, "
                         "memory, sleep priority, recall, overgrow")
    # experience as the only data (--old_pool N): old material is replayed from N windows kept from before instead of
    # the whole corpus; with --old_from_days the windows each day judged familiar join it (reservoir: N kept, each
    # window read so far equally likely to be among them)
    old_pool = [window(general[0] if rng.random() < 0.6 else general[1]) for _ in range(a.old_pool)]
    old_read = len(old_pool)
    gen_stats = []  # replay by reconstruction: what was written each night, and how long it took
    if a.old_generate or a.rebuild_cue:
        if not a.stream or a.old_pool or a.old_from_days:
            raise SystemExit("--old_generate / --rebuild_cue need --stream and no --old_pool / --old_from_days")
        tok = Tokenizer(VOCABS[vocab])
        eot = tok.eot
    if a.old_refresh and not (a.old_pool and a.stream and not a.old_from_days):
        raise SystemExit("--old_refresh needs --stream and --old_pool, without --old_from_days")
    # study until learned (--master_tol, STRUCT_PLAN 4a): a share of each night's buffer is held back as its exam.
    # The buffer stays in study (wake and sleep) on the nights that follow, and is released after a night that
    # improved its exam by less than --master_tol bits per byte. --study_nights K, the control: exactly K nights each
    exam_mode = bool(a.master_tol or a.study_nights)
    study = []  # per buffer: day, specialty (for reports only), study rows, exam rows, exam bpb after each night
    if exam_mode and (not a.stream or a.rebuild_cue or (a.master_tol and a.study_nights)):
        raise SystemExit("--master_tol / --study_nights need --stream, no --rebuild_cue, and only one of the two")

    def studying():
        return [r for s in study if s["released"] is None for r in s["rows"]]
    reviewed = []  # --review_steps: nights with a review of earlier buffers, and its size
    if a.review_steps and not exam_mode:
        raise SystemExit("--review_steps needs --master_tol or --study_nights")

    def experience(rows):
        nonlocal old_read
        for r in rows:
            old_read += 1
            if len(old_pool) < a.old_pool:
                old_pool.append(r)
            elif (j := int(rng.integers(old_read))) < a.old_pool:
                old_pool[j] = r
    for st in range(a.stages):
        new = stages[st]
        seen.append(new)
        buf = None
        if a.stream:  # day: read the unlabeled stream, keep what surprises
            buf, stats, rest = day(st)
            stream_stats.append(stats)
            if a.old_pool and a.old_from_days:
                experience(rest)
            if len(buf) < a.buffer_min:  # nothing much surprised it: no growth, nothing to learn tonight
                print(f"  night {st + 1}: buffer below {a.buffer_min} windows, no growth", flush=True)
                curve.append(evaluate(f"day {st + 1} (quiet)"))
                continue
            buffers.append(buf)
            if exam_mode:  # hold back the exam; the pre-test is the exam before anything is learned from it
                perm = rng.permutation(len(buf))
                n_exam = max(8, int(round(len(buf) * a.exam_share)))
                exam = np.stack([buf[i] for i in perm[:n_exam]]).astype(np.int64)
                buf = buffers[-1] = [buf[i] for i in perm[n_exam:]]
                study.append(dict(day=st + 1, name=names[st], rows=buf, exam=exam,
                                  u=[score(model, exam, token_bytes)], nights=0, released=None))
        rebuilt = None
        if a.old_refresh:  # control for --old_generate: as many old windows, fresh from the corpus every night
            old_pool[:] = [window(general[0] if rng.random() < 0.6 else general[1]) for _ in range(a.old_pool)]
        if a.old_generate or (a.rebuild_cue and len(buffers) > 1):  # replay by reconstruction (STRUCT_PLAN 3b)
            from reconstruct import generate
            t0, info = time.time(), dict(stage=st + 1)
            if a.old_generate:  # tonight's old material is written by the model as it is now, from document starts
                old_pool[:] = list(generate(model, a.old_generate, T + 1, eot, a.gen_batch, seed=a.seed * 1000 + st))
                info["old"] = a.old_generate
                print(f"  wrote {a.old_generate} old windows ({time.time() - t0:.0f} s), e.g. "
                      f"{tok.decode(old_pool[0][1:].tolist())[:160]!r}", flush=True)
            if a.rebuild_cue and len(buffers) > 1:  # earlier nights survive as cues; the model fills in the rest
                cues = [r for b in buffers[:-1] for r in b]
                pick = rng.choice(len(cues), size=min(a.rebuild_n, len(cues)), replace=False)
                t1 = time.time()
                rebuilt = list(generate(model, len(pick), T + 1, eot, a.gen_batch,
                                        cues=np.stack([cues[j] for j in pick]), seed=a.seed * 1000 + 500 + st))
                info["rebuilt"] = len(pick)
                print(f"  rebuilt {len(pick)} of {len(cues)} earlier windows from {a.rebuild_cue}-token cues "
                      f"({time.time() - t1:.0f} s)", flush=True)
            info["seconds"] = round(time.time() - t0)
            gen_stats.append(info)
        if mem is not None:  # instant: read the new text once, score everything before any training
            mem.arrive(names[st], new, a.memory_tokens)
            mem_curve.append(mem_evaluate(f"arrive {st + 1}"))
        if not a.grow:  # standard baseline: fine-tune the whole model, same schedule (learn, then sleep)
            params, keys = [p for p in model.parameters()], []
        elif a.adapter:  # standard baseline: new adapters each stage, same parameter budget and schedule
            params, keys = add_adapters(st)
        elif a.stages == 1 and a.placement == "fixed":  # one-shot: per_stage modules spread over 3 depths
            params, keys = [], []
            for d in fixed[:3]:
                p_, k_ = add_modules(d, 0, a.per_stage // 3)
                params += p_
                keys.append(k_)
        else:
            if a.multi_depth:  # this stage grows per_stage modules at every listed depth at once
                params, keys = [], []
                depths = fixed
                if a.placement == "dissonance":  # ... or at the most conflicted depths for this material
                    cands = list(range(1, len(model.blocks) - 1))
                    gap = most_conflicted_depth(new, cands, rows=buf)[1]
                    depths = sorted(cands[j] for j in np.argsort(gap)[::-1][:len(fixed)])
                    print(f"  multi-depth by dissonance: depths {depths} (gap {gap})", flush=True)
                for d in depths:
                    p_, k_ = add_modules(d, st, a.per_stage)
                    params += p_
                    keys.append(k_)
            else:
                if a.placement == "dissonance":
                    depth, gap = most_conflicted_depth(new, list(range(1, len(model.blocks) - 1)), rows=buf)
                    print(f"  placement by dissonance: depth {depth} (new - familiar KL by depth {gap})", flush=True)
                elif a.placement in ("need", "need_spread"):
                    depth = neediest_depth(new)[0]
                else:
                    depth = fixed[st % len(fixed)]
                params, key = add_modules(depth, st, a.per_stage * a.overgrow)
                keys = [key]
        for p in params:
            p.requires_grad_(True)
        if keys:
            layout.append(keys)
        lr = a.lr_adam if a.grow else a.lr_full
        if buf is not None:  # night, first part: learn from today's buffer (with the usual replay of general text)
            pools["new"] = studying() if exam_mode and not a.review_steps else buf  # ... or every buffer in study
        if old_pool:
            pools["old"] = old_pool
        train(params, a.stage_steps, [new], 1.0 - a.replay, lr)
        if a.review_steps and len(study) > 1 and study[-1]["released"] is None:  # first night kept whole: earlier
            carry = [r for s in study[:-1] if s["released"] is None for r in s["rows"]]  # buffers in study review
            if carry:  # on top, with the same share of general text
                pools["new"] = carry
                train(params, a.review_steps, [new], 1.0 - a.replay, lr)
                reviewed.append(dict(stage=st + 1, steps=a.review_steps, windows=len(carry)))
        if a.overgrow > 1:  # most new neurons die; the ones that proved useful survive
            pruned.append(prune(keys[0], a.per_stage, new))
            print(f"  pruned {keys[0]}: harm {pruned[-1][0]} -> silenced {pruned[-1][1]}", flush=True)
        if a.sleep_priority == "dissonance":
            tag_for_sleep(seen)
        if a.consolidate_steps:  # sleep: replay only, the new material seen so far mixed with general text
            if a.sleep_distill:  # ... carrying memory's predictions, to be distilled into this stage's modules
                distilled.append(prepare_distill(st))
            if buf is not None:  # sleep replays every night's buffer so far (as C2's sleep replays seen specialties)
                pools["new"] = studying() if exam_mode else [r for b in buffers for r in b]
                if a.rebuild_cue:  # ... earlier nights as rebuilt from their cues, in the same proportion
                    n_prev = sum(len(b) for b in buffers[:-1])
                    pools["new"] = list(buf) + ([rebuilt[i % len(rebuilt)] for i in range(n_prev)] if rebuilt else [])
            if old_pool:
                pools["old"] = old_pool
            train(params, a.consolidate_steps, seen, a.sleep_new, lr / 3)
        pools.clear()
        if buf is not None and a.rebuild_cue:  # tonight's windows are kept only as cues from now on
            buffers[-1] = [r[:a.rebuild_cue] for r in buffers[-1]]
        if a.grow:
            for p in params:
                p.requires_grad_(False)  # matured: this stage's modules stop changing
        if exam_mode:  # the night's exams: release what tonight's study no longer improved
            report = []
            for s in study:
                if s["released"] is None:
                    s["u"].append(score(model, s["exam"], token_bytes))
                    s["nights"] += 1
                    gain = s["u"][-2] - s["u"][-1]
                    if s["nights"] >= a.study_nights if a.study_nights else gain < a.master_tol:
                        s["released"] = st + 1
                    report.append(f"day {s['day']} ({s['name'][:7]}) {s['u'][0]:.4f}->{s['u'][-1]:.4f}, "
                                  f"tonight {gain:+.4f}" + (" released" if s["released"] else ""))
            print(f"  exams: " + " | ".join(report) + f"; still in study: {len(studying())} windows", flush=True)
        curve.append(evaluate(f"stage {st + 1} ({','.join(keys)})"))
        if mem is not None:  # the model changed: re-key the memory (and release what it has now learned)
            held = mem.reencode(prune=a.memory_prune or None)
            mem_curve.append(mem_evaluate(f"stage {st + 1}", held))
    audit = []  # each part is removable: silence it and see what it held
    for keys in layout:
        for key in keys:
            silence(key, range(model.grown_gates[key].out_features))
        audit.append(evaluate(f"without {','.join(keys)}"))
        for key in keys:
            for j in range(model.grown_gates[key].out_features):
                if (key, j) in saved:
                    silence(key, [j], on=False)
    res = dict(name=a.name, ckpt=a.ckpt, mode="continual", domains=a.domains, grow=a.grow, adapter=a.adapter,
               replay_source=a.replay_source, sleep_priority=a.sleep_priority, multi_depth=a.multi_depth,
               recall_prefix=a.recall_prefix, recall_temp=a.recall_temp,
               stages=a.stages,
               per_stage=a.per_stage,
               placement=a.placement, init=a.init, consolidate_steps=a.consolidate_steps, overgrow=a.overgrow,
               stage_steps=a.stage_steps, layout=layout, curve=curve, pruned=pruned, audit=audit,
               replay=a.replay, sleep_new=a.sleep_new, memory_tokens=a.memory_tokens, memory_prune=a.memory_prune,
               memory_k=a.memory_k, memory_curve=mem_curve, sleep_distill=a.sleep_distill, distill_rows=a.distill_rows,
               distilled=distilled, stream=a.stream, surprise=a.surprise, surprise_q=a.surprise_q,
               surprise_k=a.surprise_k,
               stream_share=a.stream_share, buffer_max=a.buffer_max, stream_oracle=a.stream_oracle,
               stream_stats=stream_stats, old_pool=a.old_pool, old_from_days=a.old_from_days, old_read=old_read,
               old_refresh=a.old_refresh, old_generate=a.old_generate, rebuild_cue=a.rebuild_cue, rebuild_n=a.rebuild_n, gen_stats=gen_stats,
               master_tol=a.master_tol, exam_share=a.exam_share, study_nights=a.study_nights,
               review_steps=a.review_steps, reviewed=reviewed,
               study=[dict(day=s["day"], name=s["name"], rows=len(s["rows"]), exam_rows=len(s["exam"]), u=s["u"],
                           nights=s["nights"], released=s["released"]) for s in study])
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(res, open(os.path.join(OUT_DIR, f"continual_{a.name}.json"), "w"), indent=1)
    if a.save:  # the grown parts (safetensors), to load on top of the base export with load_grown (samples.py)
        from safetensors.torch import save_file
        os.makedirs(a.save, exist_ok=True)
        parts = nn.ModuleDict({"grown_at": model.grown_at, "grown_gates": model.grown_gates})
        save_file({k: v.detach().contiguous() for k, v in parts.state_dict().items()},
                  os.path.join(a.save, "grown.safetensors"))
        json.dump(dict(name=a.name, base=a.ckpt, domains=a.domains, layout=layout, keys=list(model.grown_at.keys()),
                       k={key: g.out_features for key, g in model.grown_gates.items()}),
                  open(os.path.join(a.save, "grown.json"), "w"), indent=1)
        print(f"saved the grown parts to {a.save}", flush=True)


def summary():
    """Forgetting grows faster than learning, so a run that learned little looks good per unit
    learned; the fair comparison is the loss at the same amount learned (the smallest any run
    reached)."""
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT_DIR, "*.json")))]
    target = min(r["curve"][0]["docs"] - r["curve"][-1]["docs"] for r in rows)
    print(f"| run | trains | docs before -> after | learned | english lost | code lost | lost per unit learned | "
          f"lost at {target:.3f} learned |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        c0, c1 = r["curve"][0], r["curve"][-1]
        learned = c0["docs"] - c1["docs"]
        lost_e, lost_c = c1["fineweb-edu"] - c0["fineweb-edu"], c1["code"] - c0["code"]
        ratio = (lost_e + lost_c) / 2 / learned if learned > 0 else float("nan")
        trains = r["train"] + (f" +{r['replay']:.0%} replay" if r.get("replay") else "")
        print(f"| {r['name']} | {trains} ({r['trainable_params'] / 1e6:.1f}M) | {c0['docs']:.4f} -> {c1['docs']:.4f} | "
              f"{learned:.4f} | {lost_e:+.4f} | {lost_c:+.4f} | {ratio:.3f} | {lost_at(r['curve'], target):+.4f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "run", "summary", "continual"])
    ap.add_argument("--stages", type=int, default=3, help="continual: parts the new material arrives in")
    ap.add_argument("--per_stage", type=int, default=2, help="continual: new modules grown per stage")
    ap.add_argument("--stage_steps", type=int, default=200, help="continual: training steps per stage")
    ap.add_argument("--consolidate_steps", type=int, default=50, help="continual: replay-only steps after each stage")
    ap.add_argument("--placement", default="need", choices=["need", "need_spread", "fixed", "dissonance"],
                    help="continual: where new modules go")
    ap.add_argument("--init", default="copy", choices=["copy", "copy_active", "fresh"],
                    help="continual: new module internals")
    ap.add_argument("--domains", default="", help="continual: comma list from data/<vocab>/domains, one stage each")
    ap.add_argument("--multi_depth", type=int, default=0, help="continual: grow per_stage modules at every --depths each stage")
    ap.add_argument("--sleep_priority", default="none", choices=["none", "dissonance"],
                    help="continual: sleep replays the windows the model is most conflicted about")
    ap.add_argument("--tag_new", type=int, default=1024, help="continual, sleep priority: new windows kept (of 2x)")
    ap.add_argument("--tag_old", type=int, default=4096, help="continual, sleep priority: old windows kept (of 2x)")
    ap.add_argument("--recall_prefix", type=int, default=0, help="continual, recall: cue length (real tokens) per memory")
    ap.add_argument("--recall_temp", type=float, default=1.0, help="continual, recall: sampling temperature")
    ap.add_argument("--replay_source", default="data", choices=["data", "recall"],
                    help="continual: old material for replay: stored text, or the model's own samples")
    ap.add_argument("--recall_seqs", type=int, default=512, help="continual, recall: sequences sampled before learning")
    ap.add_argument("--adapter", type=int, default=0, help="continual: 1 = add bottleneck adapters instead (baseline)")
    ap.add_argument("--grow", type=int, default=1, help="continual: 0 = fine-tune the whole model instead (baseline)")
    ap.add_argument("--lr_full", type=float, default=3e-4, help="continual --grow 0: AdamW LR for the whole model")
    ap.add_argument("--sleep_new", type=float, default=0.5, help="continual: share of new material in sleep replay")
    ap.add_argument("--depths", default="", help="continual, fixed placement: depths per stage (default 1,3,5,...)")
    ap.add_argument("--overgrow", type=int, default=1, help="continual: grow this many times per_stage, keep the most useful")
    ap.add_argument("--stream", type=int, default=0, help="continual --domains: label-free arrival (day: read an "
                    "unlabeled stream, buffer the surprising windows; night: grow, learn from the buffer, sleep)")
    ap.add_argument("--stream_share", type=float, default=0.5, help="stream: share of the day's windows that are new")
    ap.add_argument("--day_windows", type=int, default=0, help="stream: windows per day (0 = the specialty's whole "
                    "training text, plus familiar text to match --stream_share)")
    ap.add_argument("--surprise", default="today@6", help="stream: today@D (over-represented today compared with "
                    "familiar text, state after block D), or single-window loss, maha@D, knn@D (surprise_probe.py)")
    ap.add_argument("--surprise_k", type=int, default=30, help="stream, today@D: neighbours compared")
    ap.add_argument("--surprise_q", type=float, default=0.9, help="stream: keep windows above this quantile of "
                    "familiar text's surprise")
    ap.add_argument("--surprise_ref", type=int, default=4000, help="stream: familiar windows read before each day "
                    "(half picture familiar text, half set the threshold)")
    ap.add_argument("--buffer_max", type=int, default=0, help="stream: buffer capacity in windows (0 = unlimited)")
    ap.add_argument("--buffer_min", type=int, default=64, help="stream: below this, no growth or learning that night")
    ap.add_argument("--stream_oracle", type=int, default=0, help="stream control: the buffer keeps exactly the new "
                    "windows (a perfect detector)")
    ap.add_argument("--old_refresh", type=int, default=0, help="stream: redraw the --old_pool windows from the corpus "
                    "every night (the control for --old_generate)")
    ap.add_argument("--old_generate", type=int, default=0, help="stream: replay by reconstruction, old material is this "
                    "many windows the model writes itself each night (temperature 1, from document starts)")
    ap.add_argument("--rebuild_cue", type=int, default=0, help="stream: earlier nights' buffers are kept only as their "
                    "first N tokens; sleep replays windows the model rebuilds from them")
    ap.add_argument("--rebuild_n", type=int, default=1024, help="stream, rebuild: windows rebuilt per night")
    ap.add_argument("--master_tol", type=float, default=0.0, help="stream: study until learned (STRUCT_PLAN 4a); a "
                    "buffer stays in study each night until a night improves its held-back exam by less than this (bpb)")
    ap.add_argument("--exam_share", type=float, default=0.1, help="stream, master_tol/study_nights: share of each "
                    "buffer held back as its exam (never studied)")
    ap.add_argument("--study_nights", type=int, default=0, help="stream control for --master_tol: each buffer is "
                    "studied exactly this many nights (the exam is measured, not used)")
    ap.add_argument("--review_steps", type=int, default=0, help="stream, master_tol/study_nights: today's buffer gets "
                    "the whole wake (first night kept whole); earlier buffers still in study get this many extra steps")
    ap.add_argument("--gen_batch", type=int, default=32, help="stream: sequences written at once (old_generate/rebuild)")
    ap.add_argument("--old_pool", type=int, default=0, help="continual: replay old material from this many windows "
                    "kept from before (0 = the whole general corpus)")
    ap.add_argument("--old_from_days", type=int, default=0, help="stream, old pool: the windows each day judged familiar "
                    "join the pool (reservoir sampling keeps --old_pool)")
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--train", default="all")
    ap.add_argument("--docs", default=None, help="prepare: JSONL file of documents ({\"text\": ...} per line)")
    ap.add_argument("--name")
    ap.add_argument("--save", default="", help="continual: directory for the grown parts at the end (load_grown)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4, help="micro-batches per step (8 x 1024 x 4 = 32k tokens)")
    ap.add_argument("--lr_muon", type=float, default=0.01)
    ap.add_argument("--lr_adam", type=float, default=0.002)
    ap.add_argument("--keep_tokens", type=int, default=400_000, help="English and code tokens scored for forgetting")
    ap.add_argument("--eval_at", default="25,50,100,200,300")
    ap.add_argument("--replay", type=float, default=0.0, help="share of each batch drawn from general English/code")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", type=int, default=1, help="continual: compile the training steps (0 = eager)")
    ap.add_argument("--memory_tokens", type=float, default=0, help="continual --domains: tokens of each specialty read "
                    "into episodic memory as it arrives (0 = no memory)")
    ap.add_argument("--memory_prune", type=float, default=0, help="continual, memory: after each stage release episodes "
                    "the model now predicts with p(next) >= this (0 = keep all)")
    ap.add_argument("--memory_k", type=int, default=32, help="continual, memory: neighbours recalled")
    ap.add_argument("--memory_eval_tokens", type=int, default=100_000, help="continual, memory: tokens scored per text")
    ap.add_argument("--sleep_distill", type=float, default=0, help="continual, memory: sleep distils memory's predictions "
                    "(scale on the blend weight chosen per specialty; 1 = the best blend's lam; 0 = off)")
    ap.add_argument("--distill_rows", type=int, default=1024, help="continual, sleep distillation: windows in the pool "
                    "(small pools get memorised: 16 windows hurt in a short test, as 64/256-window sleep pools did)")
    a = ap.parse_args()
    a.eval_at = {int(s) for s in a.eval_at.split(",")}
    if a.command == "prepare":
        if not a.docs:
            raise SystemExit("prepare needs --docs FILE.jsonl (one {\"text\": ...} object per line)")
        prepare(a.docs)
    elif a.command == "run":
        a.name = a.name or (f"{os.path.basename(os.path.dirname(a.ckpt))}_{a.train.replace(':', '').replace(',', '-')}"
                            + (f"_replay{round(a.replay * 100)}" if a.replay else ""))
        run(a)
    elif a.command == "continual":
        continual(a)
    else:
        summary()


if __name__ == "__main__":
    main()
