"""Sample replies from a continual run saved with --save (forgetting.py): the same prompts answered by the base
model, by the model with some stages' modules switched off, and by the full model, side by side.

    python model/samples.py runs/forgetting/C1_save                      # -> runs/forgetting/C1_save/samples.md
    python model/samples.py runs/forgetting/C1_save --variants base,-chat,all --n 4
    python model/samples.py runs/forgetting/C2s_spec_instruct_chat --prompts model/prompts/specialties.txt \
        --variants base,-instruct+chat,-chat,all --n 2 --out samples_specialties.md

Prompts: --n held-out first turns from each instruct/chat domain in the run (val.npy, never trained on; the
person's reply is shown for reference) plus a few fixed ones, or the lines of a --prompts file instead of those. A variant is "base" (no grown modules), "all", or
"-a+b" (all but the modules grown in the stages of domains a and b). Sampling: temperature --temp, top-k --top_k,
the same seed for every variant; a reply ends at <|endoftext|> or where the model starts the next "User:" turn."""
import argparse
import os
import re

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel

from forgetting import load_grown
from tokenizer import VOCABS, Tokenizer
from train import SDPA_BACKENDS

HERE = os.path.dirname(os.path.abspath(__file__))
FIXED = ["What causes the seasons on Earth?",
         "Give me three tips for sleeping better.",
         "Write a function in Rust that returns the largest number in a vector.",
         "Explain what a monad is in Haskell, in simple terms."]


def held_out(name, n, tok, rng):
    """n (prompt, reference reply) pairs: the first turn of random held-out documents."""
    ids = np.load(os.path.join(HERE, "data", "ec32k", "domains", name, "val.npy"))
    cuts = np.flatnonzero(ids == tok.eot)
    docs = [ids[a + 1:b] for a, b in zip(cuts, list(cuts[1:]) + [len(ids)])]
    out = []
    for i in rng.permutation(len(docs)):
        text = tok.decode(docs[i].tolist())
        m = re.match(r"User: (.*?)\n\nAssistant: (.*?)(?:\n\nUser: |$)", text, re.S)
        if m and len(m.group(1)) < 1200:  # prompts that leave room for the reply in the context
            out.append((m.group(1), m.group(2)))
        if len(out) == n:
            return out
    return out


@torch.no_grad()
def reply(model, tok, prompt, max_new, temp, top_k, seed):
    ids = [tok.eot] + tok.encode(f"User: {prompt}\n\nAssistant:").tolist()
    start, g = len(ids), torch.Generator(device="cuda").manual_seed(seed)
    for _ in range(max_new):
        x = torch.tensor([ids[-model.cfg.seq_len:]], device="cuda")
        with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPA_BACKENDS, set_priority=True):
            h = model(x)
        logits = model.logits(h[0, -1:].float())[0] / temp
        if top_k:
            kth = torch.topk(logits, top_k).values[-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        t = int(torch.multinomial(torch.softmax(logits, -1), 1, generator=g))
        if t == tok.eot:
            break
        ids.append(t)
        if "\n\nUser:" in tok.decode(ids[-6:]):
            break
    return tok.decode(ids[start:]).split("\n\nUser:")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="directory written by forgetting.py continual --save")
    ap.add_argument("--variants", default="", help="comma list (default: base, each domain switched off, all)")
    ap.add_argument("--n", type=int, default=3, help="held-out prompts per instruct/chat domain")
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompts", default="", help="file of prompts, one per line (# comments), instead of the fixed ones")
    ap.add_argument("--out", default="samples.md", help="file name, written in the run's directory")
    a = ap.parse_args()
    model, meta = load_grown(a.run)
    tok = Tokenizer(VOCABS["ec32k"])
    names = meta["domains"].split(",")
    stage_of = {key: int(key.split("_")[1]) for key in meta["keys"]}
    talk = [d for d in names if d in ("instruct", "chat")]
    variants = a.variants.split(",") if a.variants else ["base"] + [f"-{d}" for d in talk] * (len(talk) > 1) + ["all"]
    rng = np.random.default_rng(a.seed)
    prompts = [(f"{d} (held out)", p, ref) for d in talk for p, ref in held_out(d, a.n, tok, rng)]
    if a.prompts:
        lines = [x.strip() for x in open(a.prompts, encoding="utf-8")]
        prompts += [("prompt file", x, None) for x in lines if x and not x.startswith("#")]
    else:
        prompts += [("fixed", p, None) for p in FIXED]
    bias = {key: g.bias.detach().clone() for key, g in model.grown_gates.items()}

    def use(variant):  # switch modules off by gate bias, as the audit in forgetting.py does
        off = set(meta["keys"]) if variant == "base" else set() if variant == "all" else \
            {key for key in meta["keys"] if names[stage_of[key]] in variant.lstrip("-").split("+")}
        with torch.no_grad():
            for key, g in model.grown_gates.items():
                g.bias.copy_(torch.full_like(g.bias, -30.0) if key in off else bias[key])

    out = [f"# Sample replies: {meta['name']}", "",
           f"Domains in order: {meta['domains']}. Temperature {a.temp}, top-k {a.top_k}, seed {a.seed}, up to "
           f"{a.max_new} tokens. Variants: " + ", ".join(variants) + " (-x = without the modules grown for x).", ""]
    for i, (src, p, ref) in enumerate(prompts):
        out += [f"## {i + 1}. {src}", "", "**User:** " + p.replace("\n", "\n> "), ""]
        if ref is not None:
            out += ["*Written reply (reference):* " + (ref[:600] + (" ..." if len(ref) > 600 else "")), ""]
        for v in variants:
            use(v)
            r = reply(model, tok, p, a.max_new, a.temp, a.top_k, a.seed + i)
            out += [f"**{v}:** {r}", ""]
            print(f"[{i + 1}/{len(prompts)}] {v}: {r[:100]!r}", flush=True)
    use("all")
    path = os.path.join(a.run, a.out)
    open(path, "w").write("\n".join(out))
    print("wrote", path)


if __name__ == "__main__":
    main()
