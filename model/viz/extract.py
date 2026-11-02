"""Record BrainGPT's internals for the 3D visualization (viz/template.html): the residual stream after every stage,
attention per layer and head, the router's choice per token (with capacity drops), the grown modules' gates,
layer-by-layer predictions (logit lens), final predictions, module usage per dataset and parameter counts.

    python model/viz/extract.py                       # base model + C1s's grown modules -> runs/viz/data.json
    python model/viz/extract.py --run runs/forgetting/C2s_spec_instruct_chat
    python model/viz/build.py                         # -> runs/viz/index.html (data inlined, one file)

--run is a continual run saved with --save (forgetting.py); its base export is the base model. Each prompt is run
twice, with the grown modules switched off (gate bias -30, as forgetting.py's audit does) and on. ~1.5 GB of GPU
memory for about a minute."""
import argparse
import base64
import json
import math
import os
import sys

MODEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MODEL)
os.chdir(MODEL)
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel

from forgetting import load_grown, rows_from
from prepare_data import load_shard
from tokenizer import VOCABS, Tokenizer
from train import SDPA_BACKENDS

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/forgetting/C1s_instruct_chat", help="continual run saved with --save")
ap.add_argument("--out", default="runs/viz/data.json")
a = ap.parse_args()
OUT = a.out
os.makedirs(os.path.dirname(OUT), exist_ok=True)
model, meta = load_grown(a.run)
model.eval()
cfg = model.cfg
tok = Tokenizer(VOCABS["ec32k"])
L, H, KV = cfg.n_layer, cfg.n_head, cfg.n_kv_head
bias0 = {k: g.bias.detach().clone() for k, g in model.grown_gates.items()}


def grown(on):
    with torch.no_grad():
        for k, g in model.grown_gates.items():
            g.bias.copy_(bias0[k] if on else torch.full_like(g.bias, -30.0))


rec = {"attn": [], "rec": False}
_sdpa = F.scaled_dot_product_attention


def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, **kw):
    if rec["rec"] and q.dim() == 4 and q.size(1) == H and q.size(0) == 1:
        qf, kf = q.float(), k.float()
        if kf.size(1) != H:
            kf = kf.repeat_interleave(H // kf.size(1), dim=1)
        s = (qf @ kf.transpose(-1, -2)) * (scale if scale is not None else 1 / math.sqrt(q.size(-1)))
        T = s.size(-1)
        s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=s.device).triu(1), float("-inf"))
        rec["attn"].append(torch.softmax(s, -1)[0].cpu().numpy())  # [H, T, T]
    return _sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale,
                 enable_gqa=enable_gqa, **kw)


F.scaled_dot_product_attention = sdpa
cap = {}
hooks = []
for i, b in enumerate(model.blocks):
    hooks.append(b.register_forward_pre_hook(lambda m, a, i=i: cap.__setitem__(f"in{i}", a[0].detach().float())))
    hooks.append(b.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(f"out{i}", o.detach().float())))
hooks.append(model.router.register_forward_hook(lambda m, a, o: cap.__setitem__("router", o.detach().float())))
for k in model.grown_gates:
    hooks.append(model.grown_gates[k].register_forward_hook(
        lambda m, a, o, k=k: cap.__setitem__(f"gate{k}", torch.sigmoid(o.detach().float()))))
    hooks.append(model.grown_at[k].register_forward_hook(
        lambda m, a, o, k=k: cap.__setitem__(f"grow{k}", o[0].detach().float())))


def q8(x):
    """int8 per tensor, base64; returns (scale, data)."""
    s = float(np.abs(x).max()) or 1.0
    return {"scale": s, "shape": list(x.shape),
            "b64": base64.b64encode(np.clip(np.round(x / s * 127), -127, 127).astype(np.int8).tobytes()).decode()}


def u8(x):  # probabilities 0..1
    return {"shape": list(x.shape), "b64": base64.b64encode(np.round(np.clip(x, 0, 1) * 255).astype(np.uint8).tobytes()).decode()}


def top(logits, n):
    p = torch.softmax(logits.float(), -1)
    v, i = p.topk(n, -1)
    return [[[tok.decode([int(t)]), round(float(pp), 4)] for t, pp in zip(ii, vv)] for ii, vv in zip(i.tolist(), v.tolist())]


@torch.no_grad()
def run(text, on):
    grown(on)
    ids = [tok.eot] + tok.encode(text).tolist()
    x = torch.tensor([ids], device="cuda")
    cap.clear()
    rec["attn"], rec["rec"] = [], True
    with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPA_BACKENDS, set_priority=True):
        h = model(x)
    rec["rec"] = False
    T = len(ids)
    emb = F.rms_norm(model.wte(x).float(), (cfg.d_model,))[0]
    snaps = [("embedding", emb)]
    for i in range(L):
        if i == cfg.trunk_layers:
            snaps.append(("after module bank", cap[f"in{i}"][0]))
        snaps.append((f"block {i}", cap[f"out{i}"][0]))
    lens = []
    for i in range(L):
        lens.append(top(model.logits(F.rms_norm(cap[f"out{i}"][0], (cfg.d_model,))), 3))
    final = top(model.logits(h[0].float()), 5)
    # router: logits over the routed modules, choice with the balance bias, gates, capacity drops
    lg = cap["router"][0]
    choose = lg + model.route_bias.float()
    ti = choose.topk(cfg.top_k, -1).indices
    gv = torch.softmax(lg.gather(-1, ti), -1)
    R = cfg.n_modules - cfg.n_shared
    C = min(T, math.ceil(cfg.capacity * T * cfg.top_k / R))
    count = [0] * R
    route = []
    for t in range(T):
        picks = []
        for j in range(cfg.top_k):
            e = int(ti[t, j])
            kept = count[e] < C
            count[e] += 1
            picks.append({"m": e, "g": round(float(gv[t, j]), 3), "kept": kept})
        route.append(picks)
    bank = (cap[f"in{cfg.trunk_layers}"][0] - cap[f"out{cfg.trunk_layers - 1}"][0]).norm(dim=-1)
    out = {"text": text, "tokens": [tok.decode([i]) for i in ids], "ids": ids,
           "snaps": [{"name": n, **q8(s.cpu().numpy())} for n, s in snaps],
           "attn": [u8(a) for a in rec["attn"]], "lens": lens, "final": final,
           "router": {"logits": [[round(float(v), 2) for v in row] for row in lg.tolist()], "route": route,
                      "capacity": C, "bank_norm": [round(float(v), 3) for v in bank.tolist()]},
           "stream_norm": [[round(float(v), 2) for v in s.norm(dim=-1).tolist()] for _, s in snaps]}
    if on:
        out["grown"] = {k: {"gates": [[round(float(v), 3) for v in row] for row in cap[f"gate{k}"][0].tolist()],
                            "norm": [round(float(v), 3) for v in cap[f"grow{k}"][0].norm(dim=-1).tolist()]}
                        for k in model.grown_gates}
    assert len(rec["attn"]) == L, len(rec["attn"])
    return out


PROMPTS = [("chat", "User: What causes the seasons on Earth?\n\nAssistant: The seasons are caused by the"),
           ("code", "fn largest(v: &[i32]) -> i32 {\n    let mut m = v[0];\n    for &x in v {\n        if x >"),
           ("prose", "The heart is a muscular organ that pumps blood through the")]
prompts = []
for name, text in PROMPTS:
    prompts.append({"name": name, "base": run(text, False), "grown": run(text, True)})
    print(name, len(prompts[-1]["base"]["tokens"]), "tokens; next:", prompts[-1]["base"]["final"][-1][:3],
          "| with grown:", prompts[-1]["grown"]["final"][-1][:3])

# module usage per dataset (base model; the router sits below the grown modules, so it's the same with them)
grown(False)
sets = {}
dd = os.path.join(MODEL, "data", "ec32k", "domains")
for n in "haskell rust lua fortran medical law philosophy logic music space instruct chat".split():
    sets[n] = np.load(os.path.join(dd, n, "val.npy"))
for src in ("fineweb-edu", "code"):
    sets[src] = load_shard(os.path.join(MODEL, "data", "ec32k", src, "val_000000.bin"))[0]
usage = {}
R = cfg.n_modules - cfg.n_shared
with torch.no_grad():
    for n, ids in sets.items():
        rows = rows_from(ids, 16)
        cnt = torch.zeros(R, device="cuda")
        for i in range(0, 16, 4):
            x = torch.from_numpy(rows[i:i + 4, :-1].astype(np.int64)).cuda()
            with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPA_BACKENDS, set_priority=True):
                model(x)
            ti = (cap["router"] + model.route_bias.float()).topk(cfg.top_k, -1).indices
            cnt += torch.bincount(ti.flatten(), minlength=R).float()
        usage[n] = [round(float(v), 4) for v in (cnt / cnt.sum()).tolist()]
        print(n, "top modules", np.argsort(-np.array(usage[n]))[:4].tolist())

# parameter counts
groups = {}
for k, v in model.state_dict().items():
    if k.startswith("grown"):
        g = "grown"
    elif k.startswith("blocks."):
        g = "blocks"
    elif k.startswith("bank."):
        g = "bank"
    elif k.startswith("shared."):
        g = "shared"
    elif k.startswith("router"):
        g = "router"
    elif k.startswith("wte"):
        g = "wte"
    else:
        g = "other:" + k.split(".")[0]
    groups[g] = groups.get(g, 0) + v.numel()
blk = {k.split(".", 2)[2]: list(v.shape) for k, v in model.state_dict().items() if k.startswith("blocks.0.")}
bank = {k.split(".", 1)[1]: list(v.shape) for k, v in model.state_dict().items() if k.startswith("bank.")}
print(groups)
cfgd = {k: getattr(cfg, k) for k in ("n_layer", "n_head", "n_kv_head", "d_model", "mlp_ratio", "trunk_layers", "gate_layers",
                                      "n_modules", "n_shared", "module_dim", "module_layers", "module_heads",
                                      "module_kv_heads", "top_k", "capacity", "logit_softcap", "attn_scale", "seq_len",
                                      "vocab_size", "head_dim")}
json.dump({"config": cfgd, "params": groups, "block_shapes": blk, "bank_shapes": bank, "skip_w":
           [round(float(v), 3) for v in model.skip_w.tolist()], "grown_layout": meta["layout"],
           "grown_domains": meta["domains"].split(","), "prompts": prompts, "usage": usage,
           "route_bias": [round(float(v), 3) for v in model.route_bias.tolist()]}, open(OUT, "w"))
print("wrote", OUT, os.path.getsize(OUT) // 1024, "KB")
