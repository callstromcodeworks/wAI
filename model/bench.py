"""Measure training throughput (full fwd + bwd + optimizer steps on random tokens).

    python model/bench.py --vocab ec64k --preset small --micro_batch 8         # one configuration
    python model/bench.py --sweep ablations                         # each optimization on/off
    python model/bench.py --sweep presets                           # model sizes x micro-batch

Each sweep entry runs in a fresh subprocess so compile caches and allocator state don't leak
between configurations.
"""
import argparse
import json
import subprocess
import sys
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

SWEEPS = {
    "ablations": [
        dict(preset="small", micro_batch=8),
        dict(preset="small", micro_batch=8, compile=0),
        dict(preset="small", micro_batch=8, attn="default"),
        dict(preset="small", micro_batch=8, fp32_reduce=0),
        dict(preset="small", micro_batch=8, ce_chunk=4096),
        dict(preset="small", micro_batch=8, fp8=1),
    ],
    "phase2": [
        dict(vocab="ec32k", micro_batch=16),
        dict(vocab="ec32k", micro_batch=16, fp8=1),
        dict(vocab="ec32k", micro_batch=16, value_residual=1, unet=1),
        dict(vocab="ec32k", micro_batch=16, n_kv_head=2),
        dict(vocab="ec32k", micro_batch=16, n_kv_head=1, kv_share=2),
        dict(vocab="ec32k", micro_batch=16, mla_rank=96),
    ],
    "presets": [
        dict(preset="small", micro_batch=4), dict(preset="small", micro_batch=8), dict(preset="small", micro_batch=12),
        dict(preset="medium", micro_batch=4), dict(preset="medium", micro_batch=6),
        dict(preset="deep", micro_batch=2), dict(preset="deep", micro_batch=4),
    ],
}


def run_one(a):
    from gpt import GPT, GPTConfig, PRESETS
    from tokenizer import VOCABS
    from train import SDPA_BACKENDS, build_optimizers, measure_peak_tflops

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not a.fp32_reduce
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(max(0.1, (free - 256 * 2**20) / total))
    with torch.device("cuda"):
        model = GPT(GPTConfig(n_ranks=VOCABS[a.vocab], seq_len=a.seq_len, n_kv_head=a.n_kv_head,
                              kv_share=a.kv_share, value_residual=bool(a.value_residual), unet=bool(a.unet),
                              mla_rank=a.mla_rank, **PRESETS[a.preset]))
    muon, adam = build_optimizers(model, argparse.Namespace(lr_muon=0.02, lr_embed=0.002, lr_scalar=0.02, wd=0.0))
    if a.fp8:
        import fp8
        fp8.enable(model)
    fwd = torch.compile(model, dynamic=False) if a.compile else model
    backends = SDPA_BACKENDS if a.attn == "cudnn" else [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                                                        SDPBackend.MATH]

    def step():
        for _ in range(a.accum):
            x = torch.randint(0, model.cfg.n_vocab, (a.micro_batch, a.seq_len), device="cuda", dtype=torch.int32)
            y = torch.randint(0, model.cfg.n_vocab, (a.micro_batch, a.seq_len), device="cuda", dtype=torch.int32)
            with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(backends, set_priority=True):
                h = fwd(x)
            model.loss(h, y, a.ce_chunk).backward()
        muon.step()
        adam.step()
        model.zero_grad(set_to_none=True)

    res = dict(vars(a))
    try:
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        res["first_step_s"] = round(time.perf_counter() - t0, 1)
        for _ in range(2):
            step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.steps):
            step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / a.steps
        tps = a.accum * a.micro_batch * a.seq_len / dt
        peak = measure_peak_tflops()
        res.update(params_M=round(model.num_params() / 1e6, 1), tok_s=round(tps), peak_tflops=round(peak, 1),
                   achieved_tflops=round(model.flops_per_token() * tps / 1e12, 1),
                   mfu=round(model.flops_per_token() * tps / 1e12 / peak, 3),
                   mem_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    except torch.OutOfMemoryError:
        res["error"] = "OOM"
    print("RESULT " + json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=sorted(SWEEPS))
    ap.add_argument("--vocab", default="ec64k")
    ap.add_argument("--preset", default="small")
    ap.add_argument("--n_kv_head", type=int, default=0)
    ap.add_argument("--kv_share", type=int, default=1)
    ap.add_argument("--value_residual", type=int, default=0)
    ap.add_argument("--unet", type=int, default=0)
    ap.add_argument("--mla_rank", type=int, default=0)
    ap.add_argument("--micro_batch", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--attn", choices=["cudnn", "default"], default="cudnn")
    ap.add_argument("--fp32_reduce", type=int, default=1)
    ap.add_argument("--ce_chunk", type=int, default=2048)
    ap.add_argument("--fp8", type=int, default=0)
    a = ap.parse_args()
    if not a.sweep:
        return run_one(a)
    rows = []
    for cfg in SWEEPS[a.sweep]:
        cmd = [sys.executable, __file__] + [f"--{k}={v}" for k, v in cfg.items()]
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        line = next((l for l in out.splitlines() if l.startswith("RESULT ")), None)
        r = json.loads(line[7:]) if line else dict(cfg, error="crashed")
        rows.append(r)
        print(json.dumps({k: r.get(k) for k in ("vocab", "preset", "micro_batch", "compile", "attn", "fp32_reduce", "ce_chunk", "fp8", "value_residual",
                                                 "unet", "n_kv_head", "kv_share", "mla_rank",
                                                 "params_M", "tok_s", "achieved_tflops", "mfu", "mem_GiB", "error")}),
              flush=True)


if __name__ == "__main__":
    main()
