"""Sample text from a trained checkpoint (CUDA-graph decoding, see fastdecode.py).

    python model/generate.py runs/<run>/ckpt.pt --prompt "def quicksort(items):" --tokens 200
    python model/generate.py runs/<run>/ckpt.pt --prompt "..." --tokens 5000 --kv_window 256 --neural_cache

--kv_window caps the KV cache (a ring buffer), so generation can run far past the trained
context in fixed memory. --neural_cache loads the gate cache_lab.py --save_gate fitted
(neural_cache.json next to the checkpoint), which recovers much of the long-range recall that
a small window loses.
"""
import argparse
import json
import os
import sys
import time

import torch

from fastdecode import StaticDecoder
from gpt import GPT, GPTConfig
from tokenizer import Tokenizer


def load_model(path, device="cuda"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    with torch.device(device):
        model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--kv_window", type=int, default=0, help="KV cache slots (0 = the trained context)")
    ap.add_argument("--neural_cache", action="store_true", help="use neural_cache.json next to the checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    model, _ = load_model(a.ckpt)
    tok = Tokenizer(model.cfg.n_ranks)
    nc = None
    if a.neural_cache:
        nc = json.load(open(os.path.join(os.path.dirname(a.ckpt), "neural_cache.json")))
    dec = StaticDecoder(model, kv_window=a.kv_window or None, neural_cache=nc)
    ids = [tok.eot] + tok.encode(a.prompt).tolist()
    out, printed = [], [0]

    def stream(t):  # print whole UTF-8 characters as they complete
        out.append(t)
        text = tok.decode_bytes(out).decode("utf-8", errors="ignore")
        sys.stdout.write(text[printed[0]:])
        sys.stdout.flush()
        printed[0] = len(text)

    sys.stdout.write(a.prompt)
    t0 = time.perf_counter()
    dec.generate(ids, a.tokens, a.temperature, a.top_k, stop_id=tok.eot, on_token=stream)
    dt = time.perf_counter() - t0
    print(f"\n\n[{len(out)} tokens in {dt:.2f}s incl. compile: {len(out) / max(dt, 1e-9):.0f} tok/s]", file=sys.stderr)


if __name__ == "__main__":
    main()
