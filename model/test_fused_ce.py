"""Check fused_ce against plain PyTorch, then time it at the real vocabulary size.

    python model/test_fused_ce.py
"""
import time

import torch
import torch.nn.functional as F

from fused_ce import linear_cross_entropy, per_token_loss


def reference(h, w, t, softcap):
    logits = (h.bfloat16() @ w.bfloat16().t()).float()
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    return F.cross_entropy(logits, t, ignore_index=-1)


def check(n, d, V, softcap, chunk, ignore):
    torch.manual_seed(0)
    h = torch.randn(n, d, device="cuda", requires_grad=True)
    w = (torch.randn(V, d, device="cuda") * 0.05).requires_grad_()
    t = torch.randint(0, V, (n,), device="cuda")
    if ignore:
        t[::7] = -1
    ref = reference(h, w, t, softcap)
    gh_ref, gw_ref = torch.autograd.grad(ref, (h, w))
    out = linear_cross_entropy(h, w, t, softcap, chunk)
    gh, gw = torch.autograd.grad(out * 2.0, (h, w))  # non-unit upstream grad
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    per_tok = per_token_loss(h, w, t, softcap, chunk)
    ok_pt = abs(per_tok.sum().item() / (t >= 0).sum().item() - ref.item()) < 1e-3
    res = (abs(out.item() - ref.item()), rel(gh, 2 * gh_ref), rel(gw, 2 * gw_ref))
    good = res[0] < 1e-3 and res[1] < 1e-2 and res[2] < 1e-2 and ok_pt
    print(f"n={n} d={d} V={V} softcap={softcap} chunk={chunk} ignore={ignore}: "
          f"|dloss|={res[0]:.2e} rel(dh)={res[1]:.2e} rel(dW)={res[2]:.2e} per_token={'ok' if ok_pt else 'BAD'} "
          f"{'PASS' if good else 'FAIL'}")
    return good


def bench(n, d, V, chunk, softcap=15.0):
    h = torch.randn(n, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = (torch.randn(V, d, device="cuda") * 0.02).requires_grad_()
    t = torch.randint(0, V, (n,), device="cuda")

    def fused():
        linear_cross_entropy(h, w, t, softcap, chunk).backward()

    def naive():
        reference(h, w, t, softcap).backward()

    flops = 6 * n * d * V
    for name, fn in [("fused", fused), ("naive", naive)]:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 5
            peak = (torch.cuda.max_memory_allocated() - base) / 2**30
            print(f"  {name:5s} n={n} chunk={chunk}: {dt * 1e3:6.1f} ms  {flops / dt / 1e12:5.1f} TFLOPS  "
                  f"extra peak mem {peak:.2f} GiB")
        except torch.OutOfMemoryError:
            print(f"  {name:5s} n={n}: out of memory")
        h.grad = w.grad = None


if __name__ == "__main__":
    ok = True
    for args in [(300, 64, 1000, 0.0, 128, False), (300, 64, 1000, 15.0, 128, True),
                 (1000, 128, 8195, 30.0, 4096, False), (2048, 768, 200064, 15.0, 1024, True)]:
        ok &= check(*args)
    print("ALL PASS" if ok else "SOME FAILED")
    print("speed at the real vocabulary size (d=768, V=200064):")
    for n, chunk in [(8192, 1024), (8192, 2048), (8192, 4096), (16384, 4096)]:
        bench(n, 768, 200064, chunk)
