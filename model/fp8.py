"""FP8 training for the transformer's linear layers (RTX 40-series / sm_89 and newer).

Ada's FP8 tensor cores run at twice the bf16 rate (124 vs 61 TFLOPS measured here). Each
linear layer's three GEMMs run in FP8 with per-tensor dynamic scaling:

    forward   y  = x @ W^T        x, W in e4m3
    backward  dx = dy @ W         dy in e5m2 (wider range for gradients), W in e4m3
              dW = dy^T @ x       accumulated and returned in fp32

The LM head (fused_ce.py), attention and normalization stay in bf16/fp32, and the master
weights and optimizer state are unchanged. enable(model) switches the block Linears over.

Layouts: _scaled_mm wants its first operand row-major and its second column-major, so the
backward needs transposed copies of x, W and dy. Inductor miscompiles copies of fp8 data into a
transposed layout (torch 2.15: NaN gradients; 2.14: at some shapes), so no fp8 tensor is ever
transposed here: each transposed operand is cast from the bf16 transpose with the same scale,
which Inductor fuses into one kernel. The GEMMs go through mm8, an op tagged needs_exact_strides
so Inductor hands it exactly those row-major layouts.
"""
import torch
import torch.nn.functional as F

E4M3, E5M2 = torch.float8_e4m3fn, torch.float8_e5m2
E4M3_MAX, E5M2_MAX = 448.0, 57344.0
_CHECKED = False


def _cast(x, scale, dtype, maxval):
    return (x.float() * scale).clamp(-maxval, maxval).to(dtype)


def _quantize(x, dtype, maxval):
    """x -> (fp8 tensor, inverse scale) with the tensor's absmax mapped to the format's max."""
    amax = x.abs().amax().float().clamp(min=1e-12)
    scale = maxval / amax
    return _cast(x, scale, dtype, maxval), scale.reciprocal()


@torch.library.custom_op("worksai::mm8", mutates_args=(), tags=(torch.Tag.needs_exact_strides,))
def mm8(a: torch.Tensor, bt: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, out_fp32: bool) -> torch.Tensor:
    """a @ bt.T for fp8 a [M, K] and bt [N, K], both row-major."""
    if not (a.is_contiguous() and bt.is_contiguous()):  # never seen; a copy here is slow but correct
        a, bt = a.contiguous(), bt.contiguous()
    return torch._scaled_mm(a, bt.t(), scale_a=scale_a, scale_b=scale_b,
                            out_dtype=torch.float32 if out_fp32 else torch.bfloat16)


@mm8.register_fake
def _(a, bt, scale_a, scale_b, out_fp32):
    return a.new_empty((a.shape[0], bt.shape[0]), dtype=torch.float32 if out_fp32 else torch.bfloat16)


class _FP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        x8, sx = _quantize(x, E4M3, E4M3_MAX)
        w8, sw = _quantize(w, E4M3, E4M3_MAX)
        y = mm8(x8, w8, sx, sw, False)  # x @ W^T
        # the backward's operands, transposed in bf16 and cast with the same scales
        ctx.save_for_backward(_cast(x.t().contiguous(), sx.reciprocal(), E4M3, E4M3_MAX), sx,
                              _cast(w.t().contiguous(), sw.reciprocal(), E4M3, E4M3_MAX), sw)
        ctx.w_dtype = w.dtype
        return y

    @staticmethod
    def backward(ctx, gy):
        xt8, sx, wt8, sw = ctx.saved_tensors
        g8, sg = _quantize(gy, E5M2, E5M2_MAX)
        gt8 = _cast(gy.t().contiguous(), sg.reciprocal(), E5M2, E5M2_MAX)
        dx = mm8(g8, wt8, sg, sw, False)  # dy @ W
        dw = mm8(gt8, xt8, sg, sx, True)  # dy^T @ x
        return dx, dw.to(ctx.w_dtype)


def fp8_linear(x, weight):
    shape = x.shape
    y = _FP8Linear.apply(x.reshape(-1, shape[-1]).to(torch.bfloat16), weight)
    return y.view(*shape[:-1], weight.shape[0])


def check_compiled(tol=5e-2):
    """Compile an FP8 linear and compare its gradients with the same layer run eagerly (always correct).
    Different builds round the quantization differently (2.14: ~1-3% apart, 2.15: <0.1%); a miscompile
    shows up as NaN or garbage. Returns the worst relative error, or raises on a bad build."""
    torch.manual_seed(0)
    worst = 0.0
    for m, n_in, n_out in ((512, 128, 256), (512, 1024, 6144), (4096, 768, 3072)):
        x = torch.randn(m, n_in, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n_out, n_in, device="cuda") * 0.05
        grads = []
        for fn in (fp8_linear, torch.compile(fp8_linear, dynamic=False)):
            xr, wr = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
            fn(xr, wr).float().pow(2).sum().backward()
            grads.append((xr.grad.float(), wr.grad.float()))
        for e, c in zip(*grads):
            err = ((c - e).norm() / e.norm()).item()
            if not (bool(torch.isfinite(c).all()) and err <= tol):
                raise RuntimeError(f"fp8: this torch build miscompiles the FP8 linear ({m}x{n_in}->{n_out}: "
                                   f"relative gradient error {err:.3g}); train with --fp8 0")
            worst = max(worst, err)
    return worst


def enable(model):
    """Route the nn.Linear layers inside the transformer blocks through FP8 GEMMs. Small heads with a
    bias or a size FP8 can't tile (memory.py's per-head gates and write strengths) stay in bf16."""
    global _CHECKED
    if not _CHECKED:
        check_compiled()
        _CHECKED = True
    n = 0
    for mod in model.blocks.modules():
        if isinstance(mod, torch.nn.Linear):
            if mod.bias is not None or mod.in_features % 16 or mod.out_features % 16:
                continue
            mod.forward = (lambda m: (lambda x: fp8_linear(x, m.weight)))(mod)
            n += 1
    return n
