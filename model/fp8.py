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


def _quantize(x, dtype, maxval):
    """x -> (fp8 tensor, inverse scale) with the tensor's absmax mapped to the format's max."""
    amax = x.abs().amax().float().clamp(min=1e-12)
    scale = maxval / amax
    return (x.float() * scale).clamp(-maxval, maxval).to(dtype), scale.reciprocal()


def _col_major(t):
    """Same values, column-major layout (what _scaled_mm wants for its second operand)."""
    return t.t().contiguous().t()


class _FP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        x8, sx = _quantize(x, E4M3, E4M3_MAX)
        w8, sw = _quantize(w, E4M3, E4M3_MAX)
        y = torch._scaled_mm(x8, w8.t(), scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16)
        ctx.save_for_backward(x8, sx, w8, sw)
        ctx.w_dtype = w.dtype
        return y

    @staticmethod
    def backward(ctx, gy):
        x8, sx, w8, sw = ctx.saved_tensors
        g8, sg = _quantize(gy, E5M2, E5M2_MAX)
        dx = torch._scaled_mm(g8, _col_major(w8), scale_a=sg, scale_b=sw, out_dtype=torch.bfloat16)
        dw = torch._scaled_mm(g8.t().contiguous(), _col_major(x8), scale_a=sg, scale_b=sx,
                              out_dtype=torch.float32)
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
