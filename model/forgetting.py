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


def grow(model, k):
    """Neurogenesis: a copy of a sparse-bank model with k new routed modules. The old modules and their
    router rows keep their weights and are frozen (gradient masks: they share stacked tensors with the new
    ones); only the new modules and their router rows learn. Returns (model, [(param, grad mask)])."""
    import dataclasses
    cfg = model.cfg
    assert cfg.module_mode != "dense" and cfg.n_modules, "growing needs a sparse module bank"
    R, de = cfg.n_modules - cfg.n_shared, cfg.module_dim
    # same slots per old module (capacity is a share of the routed tokens); revival brings new modules in
    cfg2 = dataclasses.replace(cfg, n_modules=cfg.n_modules + k, revive=cfg.revive or 0.1,
                               capacity=cfg.capacity * (R + k) / R)
    with torch.device("cuda"):
        new = BrainGPT(cfg2)
    old_state, new_state = model.state_dict(), new.state_dict()
    for name, t in old_state.items():
        tgt = new_state[name]
        if tgt.shape == t.shape:
            tgt.copy_(t)
        else:  # bank tensors grow along the module dimension: old modules first
            tgt[tuple(slice(0, n) for n in t.shape)].copy_(t)
    with torch.no_grad():  # new modules start silent: never chosen until revival lifts their bias
        new.router.weight[R:] = 0
        new.route_bias[R:] = -3.0
    masks = []
    for name, p in new.named_parameters():
        if name.startswith("bank.") or name == "router.weight":
            m = torch.ones_like(p)
            if name in ("bank.w_in", "bank.w_out"):
                m[:R * de] = 0
            else:
                m[:R] = 0
            masks.append((p, m))
    return new, masks


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
    if a.train.startswith("grow:"):
        model, masks = grow(model, int(a.train[5:]))
        for p in model.parameters():
            p.requires_grad_(False)
        params = [p for p, _ in masks]
    else:
        params = trainable(model, a.train)
    for p in params:
        p.requires_grad_(True)
    for p, m in masks:  # frozen old modules: their slices get no gradient
        p.register_hook(lambda g, m=m: g * m)
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
        if masks:  # growing: revival gives the new modules tokens to learn from
            model.update_after_step(parts)
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
    ap.add_argument("command", choices=["prepare", "run", "summary"])
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
    else:
        summary()


if __name__ == "__main__":
    main()
