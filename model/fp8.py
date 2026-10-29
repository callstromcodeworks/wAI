"""FP8 training for the transformer's linear layers (RTX 40-series / sm_89 and newer).

Ada's FP8 tensor cores run at twice the bf16 rate (124 vs 61 TFLOPS measured here). Each
linear layer's three GEMMs run in FP8 with per-tensor dynamic scaling:

    forward   y  = x @ W^T        x, W in e4m3
    backward  dx = dy @ W         dy in e5m2 (wider range for gradients), W in e4m3
              dW = dy^T @ x       accumulated and returned in fp32

The LM head (fused_ce.py), attention and normalization stay in bf16/fp32, and the master
weights and optimizer state are unchanged. enable(model) switches the block Linears over.
"""
import torch
import torch.nn.functional as F

E4M3, E5M2 = torch.float8_e4m3fn, torch.float8_e5m2
E4M3_MAX, E5M2_MAX = 448.0, 57344.0
# torch 2.15's Inductor drops the copies that lay out both operands of the weight-gradient GEMM, and the
# gradient comes out NaN (2.14 compiles it correctly, and the direct form is ~20% faster there)
_INDUCTOR_DROPS_COPIES = tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 15)


def _quantize(x, dtype, maxval):
    """x -> (fp8 tensor, inverse scale) with the tensor's absmax mapped to the format's max."""
    amax = x.abs().amax().float().clamp(min=1e-12)
    scale = maxval / amax
    return (x.float() * scale).clamp(-maxval, maxval).to(dtype), scale.reciprocal()


def _col_major(t):
    """Same values, column-major layout (what _scaled_mm wants for its second operand)."""
    return t.t().contiguous().t()


@torch.library.custom_op("worksai::fp8_mm", mutates_args=())
def fp8_mm(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, out_fp32: bool) -> torch.Tensor:
    """a @ b.T for fp8 a [M, K] and b [N, K] (any layout), opaque to torch.compile so the layouts
    _scaled_mm needs are always made here. Used for the weight gradient on torch 2.15+ only (see
    _INDUCTOR_DROPS_COPIES); the other two GEMMs compile correctly and stay visible to the compiler."""
    return torch._scaled_mm(a.contiguous(), b.contiguous().t(), scale_a=scale_a, scale_b=scale_b,
                            out_dtype=torch.float32 if out_fp32 else torch.bfloat16)


@fp8_mm.register_fake
def _(a, b, scale_a, scale_b, out_fp32):
    return a.new_empty((a.shape[0], b.shape[0]), dtype=torch.float32 if out_fp32 else torch.bfloat16)


class _FP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        x8, sx = _quantize(x, E4M3, E4M3_MAX)
        w8, sw = _quantize(w, E4M3, E4M3_MAX)
        y = torch._scaled_mm(x8, w8.t(), scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16)  # x @ W^T
        ctx.save_for_backward(x8, sx, w8, sw)
        ctx.w_dtype = w.dtype
        return y

    @staticmethod
    def backward(ctx, gy):
        x8, sx, w8, sw = ctx.saved_tensors
        g8, sg = _quantize(gy, E5M2, E5M2_MAX)
        dx = torch._scaled_mm(g8, _col_major(w8), scale_a=sg, scale_b=sw, out_dtype=torch.bfloat16)  # dy @ W
        if _INDUCTOR_DROPS_COPIES:  # dW = dy^T @ x through the opaque op (see fp8_mm)
            dw = fp8_mm(g8.t(), x8.t(), sg, sx, True)
        else:
            dw = torch._scaled_mm(g8.t().contiguous(), _col_major(x8), scale_a=sg, scale_b=sx, out_dtype=torch.float32)
        return dx, dw.to(ctx.w_dtype)


def fp8_linear(x, weight):
    shape = x.shape
    y = _FP8Linear.apply(x.reshape(-1, shape[-1]).to(torch.bfloat16), weight)
    return y.view(*shape[:-1], weight.shape[0])


def enable(model):
    """Route the nn.Linear layers inside the transformer blocks through FP8 GEMMs. Small heads with a
    bias or a size FP8 can't tile (memory.py's per-head gates and write strengths) stay in bf16."""
    n = 0
    for mod in model.blocks.modules():
        if isinstance(mod, torch.nn.Linear):
            if mod.bias is not None or mod.in_features % 16 or mod.out_features % 16:
                continue
            mod.forward = (lambda m: (lambda x: fp8_linear(x, m.weight)))(mod)
            n += 1
    return n
