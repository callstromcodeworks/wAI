"""Weight size vs KV cache report for trained checkpoints.

    python model/analyze.py runs/sweep_v64k/ckpt.pt runs/sweep_v200k/ckpt.pt --out runs/analysis.json

For each checkpoint:
- memory: weights at bf16 / int8 / int4, and KV cache at bf16 / fp8 / int8 / int4, per token
  and per MB of text (a smaller vocabulary needs more tokens, so more cache, per byte)
- accuracy: validation bits per byte with weights and/or KV cache quantized (simulated by
  quantize-dequantize, which gives the exact accuracy of the quantized format)
- speed: CUDA-graph decode (fastdecode.py) at batch 1 and 32, in tokens/s and bytes of text/s
- decode traffic: bytes read from VRAM per byte of generated text at a given context length,
  (weights + context x KV per token) / bytes per token. Batch-1 decode is memory-bandwidth
  bound, so this is the cost the weight-vs-cache trade-off actually sets.
"""
import argparse
import copy
import json
import os

import numpy as np
import torch

from fastdecode import StaticDecoder
from generate import load_model
from train import evaluate, token_byte_lengths

HERE = os.path.dirname(os.path.abspath(__file__))
MIX = {"fineweb-edu": 0.6, "code": 0.4}


# ----------------------------------------------------------------------------- quantization

def quantize_rows(w, bits, group=None):
    """Symmetric round-to-nearest quantization of each row (or each group of `group` columns)."""
    shape = w.shape
    x = w.float().reshape(-1, group) if group else w.float()
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / qmax
    return ((x / scale).round().clamp(-qmax, qmax) * scale).reshape(shape).to(w.dtype)


def quantize_weights(model, body_bits, emb_bits, group=128):
    """Copy of the model with Linear weights at body_bits and the (tied) embedding at emb_bits."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, torch.nn.Linear) and body_bits < 16:
                mod.weight.copy_(quantize_rows(mod.weight, body_bits, group if body_bits <= 4 else None))
        if emb_bits < 16:
            m.wte.weight.copy_(quantize_rows(m.wte.weight, emb_bits, group if emb_bits <= 4 else None))
    return m


def kv_quantizer(kind):
    """Quantize-dequantize K or V ([B, H, T, D]) per token and head, as a cache would store them."""
    if kind == "fp8":
        def f(x):
            s = x.float().abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
            return ((x.float() / s).to(torch.float8_e4m3fn).float() * s).to(x.dtype)
        return f
    bits, group = {"int8": (8, None), "int4": (4, 32)}[kind]

    def g(x):
        B, H, T, D = x.shape
        return quantize_rows(x.reshape(B * H * T, D), bits, group).reshape(B, H, T, D)
    return g


def set_kv_transform(model, fn):
    for b in model.blocks:
        b.attn.kv_transform = fn


# ----------------------------------------------------------------------------- report

def mixed_bytes_per_token(vocab):
    bpt = {s: json.load(open(os.path.join(HERE, "data", vocab, s, "stats.json")))["bytes_per_token"] for s in MIX}
    return 1.0 / sum(w / bpt[s] for s, w in MIX.items())


def score(model, data_dir, n_ranks, eval_bytes, batch=8):
    token_bytes = torch.from_numpy(token_byte_lengths(n_ranks, model.cfg.vocab_size)).cuda()
    res = evaluate(model, model, data_dir, list(MIX), model.cfg.seq_len, batch, eval_bytes, token_bytes)
    return dict(eng_bpb=res["fineweb-edu"]["bpb"], code_bpb=res["code"]["bpb"],
                mix_bpb=sum(MIX[n] * res[n]["bpb"] for n in MIX))


def decode_speed(model, bytes_per_token, prompt_len=512, n_new=256):
    """CUDA-graph decode speed at context ~prompt_len: batch 1 (latency, bound by reading the
    weights) and batch 32 (throughput, where KV-cache reads grow)."""
    out = {}
    for batch in (1, 32):
        dec = StaticDecoder(model, kv_window=prompt_len + n_new, batch=batch)
        tps = dec.benchmark(prompt_len, n_new)
        out[f"batch{batch}"] = dict(tok_s=tps, text_KB_s=tps * bytes_per_token / 1e3)
        del dec
        torch.cuda.empty_cache()
    return out


def analyze(path, eval_bytes, quant, speed):
    model, ckpt = load_model(path)
    cfg = model.cfg
    vocab = ckpt["args"]["vocab"]
    data_dir = os.path.join(HERE, "data", vocab)
    bpt = mixed_bytes_per_token(vocab)
    n, emb = model.num_params(), model.embedding_params()
    kv_tok = {p: model.kv_bytes_per_token(b) for p, b in (("bf16", 2), ("fp8", 1), ("int8", 1), ("int4", 0.5))}
    r = dict(ckpt=path, vocab=vocab, n_ranks=cfg.n_ranks,
             arch=f"L{cfg.n_layer} d{cfg.d_model} h{cfg.n_head} kv{cfg.n_kv_head} share{cfg.kv_share}",
             params=n, embedding_params=emb, bytes_per_token=bpt,
             weights_MiB={"bf16": n * 2 / 2**20, "int8": n / 2**20, "int4": (n - emb) / 2 / 2**20 + emb / 2**20},
             kv_bytes_per_token=kv_tok,
             kv_MiB_per_MB_text={p: v * 1e6 / bpt / 2**20 for p, v in kv_tok.items()},
             tokens_trained=ckpt["tokens_seen"])
    # Bytes read per generated byte of text at batch 1 (bf16 weights and cache).
    r["decode_traffic_per_text_byte"] = {
        f"{ctx_kb}KB_context": (n * 2 + ctx_kb * 1e3 / bpt * kv_tok["bf16"]) / bpt for ctx_kb in (4, 64, 1024)}
    r["accuracy"] = {"bf16": score(model, data_dir, cfg.n_ranks, eval_bytes)}
    if quant:
        for name, (wb, eb) in {"w8 (all int8)": (8, 8), "w4 body + int8 embedding": (4, 8)}.items():
            r["accuracy"][name] = score(quantize_weights(model, wb, eb), data_dir, cfg.n_ranks, eval_bytes)
        for kind in ("fp8", "int8", "int4"):
            set_kv_transform(model, kv_quantizer(kind))
            r["accuracy"][f"kv {kind}"] = score(model, data_dir, cfg.n_ranks, eval_bytes)
        set_kv_transform(model, None)
    if speed:
        r["speed"] = decode_speed(model, bpt)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--eval_bytes", type=int, default=4_000_000)
    ap.add_argument("--no_quant", action="store_true")
    ap.add_argument("--no_speed", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    results = []
    for path in a.ckpts:
        r = analyze(path, a.eval_bytes, not a.no_quant, not a.no_speed)
        results.append(r)
        acc = r["accuracy"]
        print(f"\n== {path}  [{r['vocab']}, {r['arch']}, {r['tokens_trained'] / 1e6:.0f}M tokens trained]")
        print(f"weights: {r['params'] / 1e6:.1f}M params ({r['embedding_params'] / 1e6:.1f}M embedding) = "
              + ", ".join(f"{v:.0f} MiB {k}" for k, v in r["weights_MiB"].items()))
        print(f"KV cache: {r['kv_bytes_per_token']['bf16'] / 1024:.2f} KiB/token bf16 = "
              + ", ".join(f"{v:.1f} MiB {k}" for k, v in r["kv_MiB_per_MB_text"].items()) + " per MB of text "
              f"({r['bytes_per_token']:.3f} bytes/token)")
        print("decode traffic per text byte: " + ", ".join(f"{v / 1e6:.2f} MB @ {k}"
                                                          for k, v in r["decode_traffic_per_text_byte"].items()))
        base = acc["bf16"]["mix_bpb"]
        for k, v in acc.items():
            print(f"  {k:26s} eng {v['eng_bpb']:.4f}  code {v['code_bpb']:.4f}  mix {v['mix_bpb']:.4f}"
                  + ("" if k == "bf16" else f"  ({(v['mix_bpb'] / base - 1) * 100:+.2f}%)"))
        for k, v in r.get("speed", {}).items():
            print(f"decode {k} (CUDA graph): {v['tok_s']:,.0f} tok/s = {v['text_KB_s']:,.1f} KB/s of text")
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
