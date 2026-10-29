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
    --stages 1 --placement fixed is the one-shot baseline (per_stage modules spread over blocks 1, 3, 5)."""
    from brain import ModuleBank
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    model, ck = load_model(a.ckpt)
    cfg, vocab = model.cfg, ck["args"]["vocab"]
    for p in model.parameters():
        p.requires_grad_(False)
    model.grown_at, model.grown_gates = nn.ModuleDict(), nn.ModuleDict()
    ids_train = np.load(os.path.join(HERE, "data", vocab, "crawl", "train.npy"))
    stages = np.array_split(ids_train, a.stages)
    docs_val = rows_from(np.load(os.path.join(HERE, "data", vocab, "crawl", "val.npy")), 10 ** 9)
    keep = {src: rows_from(load_shard(os.path.join(HERE, "data", vocab, src, "val_000000.bin"))[0], a.keep_tokens // T)
            for src in ("fineweb-edu", "code")}
    general = [load_shard(os.path.join(HERE, "data", vocab, src, "train_000000.bin"))[0]
               for src in ("fineweb-edu", "code")]
    token_bytes = torch.from_numpy(token_byte_lengths(VOCABS[vocab], model.cfg.vocab_size)).cuda()
    rng = np.random.default_rng(a.seed)

    def evaluate(tag):
        r = dict(tag=tag, docs=score(model, docs_val, token_bytes),
                 **{k: score(model, v, token_bytes) for k, v in keep.items()})
        print(f"{tag:24s} | docs {r['docs']:.4f} | english {r['fineweb-edu']:.4f} | code {r['code']:.4f}", flush=True)
        return r

    def window(src):
        st = rng.integers(0, len(src) - T - 1)
        return src[st:st + T + 1]

    def batch(new_sources, new_share):
        rows = []
        for _ in range(a.batch):
            if rng.random() < new_share:
                rows.append(window(new_sources[rng.integers(len(new_sources))]))
            else:
                rows.append(window(general[0] if rng.random() < 0.6 else general[1]))
        return torch.from_numpy(np.stack(rows).astype(np.int64)).cuda()

    def train(params, steps, new_sources, new_share, lr):
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.8, 0.95), eps=1e-10)
        model.train()
        for step in range(1, steps + 1):
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, step / 10)
            for _ in range(a.accum):
                r = batch(new_sources, new_share)
                with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPA_BACKENDS, set_priority=True):
                    out = model(r[:, :-1])
                loss = model.train_loss(out, r[:, 1:], draft_weight=0.0, critic_weight=0.0, balance_weight=0.0)[0]
                (loss / a.accum).backward()
            opt.step()
            model.zero_grad(set_to_none=True)

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

    curve = [evaluate("start")]
    fixed = [int(d) for d in a.depths.split(",")] if a.depths else list(range(1, len(model.blocks) - 1, 2))
    seen, layout, pruned = [], [], []
    for st in range(a.stages):
        new = stages[st]
        seen.append(new)
        if a.stages == 1 and a.placement == "fixed":  # one-shot: per_stage modules spread over 3 depths
            params, keys = [], []
            for d in fixed[:3]:
                p_, k_ = add_modules(d, 0, a.per_stage // 3)
                params += p_
                keys.append(k_)
        else:
            depth = neediest_depth(new)[0] if a.placement in ("need", "need_spread") else fixed[st % len(fixed)]
            params, key = add_modules(depth, st, a.per_stage * a.overgrow)
            keys = [key]
        for p in params:
            p.requires_grad_(True)
        layout.append(keys)
        train(params, a.stage_steps, [new], 1.0 - a.replay, a.lr_adam)
        if a.overgrow > 1:  # most new neurons die; the ones that proved useful survive
            pruned.append(prune(keys[0], a.per_stage, new))
            print(f"  pruned {keys[0]}: harm {pruned[-1][0]} -> silenced {pruned[-1][1]}", flush=True)
        if a.consolidate_steps:  # sleep: replay only, the new material seen so far mixed with general text
            train(params, a.consolidate_steps, seen, 0.5, a.lr_adam / 3)
        for p in params:
            p.requires_grad_(False)  # matured: this stage's modules stop changing
        curve.append(evaluate(f"stage {st + 1} ({','.join(keys)})"))
    audit = []  # each part is removable: silence it and see what it held
    for keys in layout:
        for key in keys:
            silence(key, range(model.grown_gates[key].out_features))
        audit.append(evaluate(f"without {','.join(keys)}"))
        for key in keys:
            for j in range(model.grown_gates[key].out_features):
                if (key, j) in saved:
                    silence(key, [j], on=False)
    res = dict(name=a.name, ckpt=a.ckpt, mode="continual", stages=a.stages, per_stage=a.per_stage,
               placement=a.placement, init=a.init, consolidate_steps=a.consolidate_steps, overgrow=a.overgrow,
               stage_steps=a.stage_steps, layout=layout, curve=curve, pruned=pruned, audit=audit)
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(res, open(os.path.join(OUT_DIR, f"continual_{a.name}.json"), "w"), indent=1)


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
    ap.add_argument("--placement", default="need", choices=["need", "need_spread", "fixed"],
                    help="continual: where new modules go")
    ap.add_argument("--init", default="copy", choices=["copy", "copy_active", "fresh"],
                    help="continual: new module internals")
    ap.add_argument("--depths", default="", help="continual, fixed placement: depths per stage (default 1,3,5,...)")
    ap.add_argument("--overgrow", type=int, default=1, help="continual: grow this many times per_stage, keep the most useful")
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--train", default="all")
    ap.add_argument("--docs", default=None, help="prepare: JSONL file of documents ({\"text\": ...} per line)")
    ap.add_argument("--name")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4, help="micro-batches per step (8 x 1024 x 4 = 32k tokens)")
    ap.add_argument("--lr_muon", type=float, default=0.01)
    ap.add_argument("--lr_adam", type=float, default=0.002)
    ap.add_argument("--keep_tokens", type=int, default=400_000, help="English and code tokens scored for forgetting")
    ap.add_argument("--eval_at", default="25,50,100,200,300")
    ap.add_argument("--replay", type=float, default=0.0, help="share of each batch drawn from general English/code")
    ap.add_argument("--seed", type=int, default=0)
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
