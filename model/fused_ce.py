"""Fused LM head + cross-entropy for a 200k vocabulary.

With V = 200,064, full logits for one 16k-token micro-batch would take 6.5 GB in bf16 and
twice that in fp32. Instead, tokens are processed in chunks, and the loss and both
gradients are produced during the forward pass:

    logits  = h_c @ W^T                (bf16 GEMM into a reused buffer)
    kernel  : softcap, log-sum-exp, loss, and dL/dlogits written over the logits in place
    dh_c    = dlogits @ W              (bf16 GEMM with fp32 output)
    dW     += dlogits^T @ h_c          (bf16 GEMM accumulating into fp32 via out_dtype)

The backward pass only scales the stored gradients by the incoming grad. That is the same
three GEMMs an unfused head does, with no recompute and one chunk of logits alive.

dh_c reduces over all 200k vocabulary entries, and the sum cancels heavily
(sum_v p_v W_v - W_target). With a bf16 output, cuBLAS may pick a split-K kernel that reduces
partial sums in bf16 (PyTorch allows it by default), which measured 1-2% relative error.
An fp32 output keeps the reduction in fp32, and on an RTX 4070 it is also faster.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _tanh(x):
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def _ce_kernel(logits_ptr, stride, targets_ptr, loss_ptr, scale_ptr, V, softcap,
               HAS_SOFTCAP: tl.constexpr, COMPUTE_GRAD: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * stride
    tgt = tl.load(targets_ptr + row).to(tl.int64)
    valid = tgt >= 0
    tgt_safe = tl.where(valid, tgt, 0)

    # Pass 1: online log-sum-exp, one running (max, sum) per lane.
    m = tl.full([BLOCK], -1e30, tl.float32)
    s = tl.zeros([BLOCK], tl.float32)
    for off in range(0, V, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < V
        x = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_SOFTCAP:
            x = softcap * _tanh(x / softcap)
        x = tl.where(mask, x, float("-inf"))
        m_new = tl.maximum(m, x)
        s = s * tl.exp(m - m_new) + tl.exp(x - m_new)
        m = m_new
    m_all = tl.max(m, axis=0)
    lse = m_all + tl.log(tl.sum(s * tl.exp(m - m_all), axis=0))

    xt = tl.load(base + tgt_safe).to(tl.float32)
    if HAS_SOFTCAP:
        xt = softcap * _tanh(xt / softcap)
    tl.store(loss_ptr + row, tl.where(valid, lse - xt, 0.0))

    if COMPUTE_GRAD:
        # Pass 2: d(loss)/d(logits) = (softmax - onehot) * d(softcap), overwriting the logits.
        g = tl.where(valid, tl.load(scale_ptr), 0.0)
        for off in range(0, V, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < V
            x = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
            if HAS_SOFTCAP:
                t = _tanh(x / softcap)
                p = tl.exp(softcap * t - lse)
            else:
                p = tl.exp(x - lse)
            p = tl.where(cols == tgt, p - 1.0, p)
            if HAS_SOFTCAP:
                p = p * (1.0 - t * t)
            tl.store(base + cols, (p * g).to(logits_ptr.dtype.element_ty), mask=mask)


def _launch(logits, targets, loss_rows, scale, softcap, compute_grad):
    n, V = logits.shape
    _ce_kernel[(n,)](logits, logits.stride(0), targets, loss_rows, scale, V, float(softcap or 1.0),
                     HAS_SOFTCAP=bool(softcap), COMPUTE_GRAD=compute_grad, BLOCK=4096, num_warps=16)


class _LinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, weight, targets, softcap, chunk):
        n, d = h.shape
        hb = h.to(torch.bfloat16)
        wb = weight.to(torch.bfloat16)  # one cast per call, not per chunk
        targets = targets.contiguous()
        valid = (targets >= 0).sum()
        scale = (1.0 / valid.clamp(min=1).float()).reshape(1)  # mean over valid tokens, on device
        loss_rows = torch.empty(n, dtype=torch.float32, device=h.device)
        need_grad = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        logits = torch.empty(min(chunk, n), weight.shape[0], dtype=torch.bfloat16, device=h.device)
        if need_grad:
            grad_h = torch.empty(n, d, dtype=torch.float32, device=h.device)
            grad_w = torch.zeros(weight.shape, dtype=torch.float32, device=h.device)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            hc, lg = hb[s:e], logits[:e - s]
            torch.mm(hc, wb.t(), out=lg)
            _launch(lg, targets[s:e], loss_rows[s:e], scale, softcap, need_grad)
            if need_grad:
                torch.mm(lg, wb, out_dtype=torch.float32, out=grad_h[s:e])
                torch.addmm(grad_w, lg.t(), hc, out_dtype=torch.float32, out=grad_w)
        if need_grad:
            ctx.save_for_backward(grad_h, grad_w)
            ctx.dtypes = (h.dtype, weight.dtype)
        return loss_rows.sum() * scale[0]

    @staticmethod
    def backward(ctx, grad_loss):
        grad_h, grad_w = ctx.saved_tensors
        h_dtype, w_dtype = ctx.dtypes
        return (grad_h * grad_loss).to(h_dtype), (grad_w * grad_loss).to(w_dtype), None, None, None


def linear_cross_entropy(h, weight, targets, softcap=0.0, chunk=4096):
    """Mean cross-entropy of softcap(h @ weight^T) against targets; targets < 0 are ignored.

    h: [N, d], weight: [V, d] (any float dtype; math runs in bf16 with fp32 accumulation),
    targets: [N] integer ids.
    """
    return _LinearCrossEntropy.apply(h, weight, targets, softcap, chunk)


def per_token_loss(h, weight, targets, softcap=0.0, chunk=4096):
    """Per-token cross-entropy (nats, fp32), no gradients. Used for evaluation."""
    with torch.no_grad():
        n = h.shape[0]
        hb, wb = h.to(torch.bfloat16), weight.to(torch.bfloat16)
        targets = targets.contiguous()
        out = torch.empty(n, dtype=torch.float32, device=h.device)
        one = torch.ones(1, dtype=torch.float32, device=h.device)
        logits = torch.empty(min(chunk, n), weight.shape[0], dtype=torch.bfloat16, device=h.device)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            lg = logits[:e - s]
            torch.mm(hb[s:e], wb.t(), out=lg)
            _launch(lg, targets[s:e], out[s:e], one, softcap, False)
        return out
