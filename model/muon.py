"""Muon optimizer (Keller Jordan et al.) for the transformer's hidden weight matrices.

Each update is the momentum buffer, orthogonalized with a quintic Newton-Schulz iteration in
bf16. Muon reaches a given loss in fewer tokens than AdamW on these matrices. The embedding
is not a hidden matrix and stays on AdamW.

All matrices of the same shape are stacked and orthogonalized in one batched, compiled call.
A parameter can hold several logical matrices as row blocks (a fused qkv weight holds q, k
and v), and each block is orthogonalized on its own. A 3-D parameter [n, rows, cols] is n
stacked matrices (brain.py's module bank); its row splits apply to each of them.
"""
from collections import defaultdict

import torch
import torch._dynamo

# One compiled specialization per (matrices in bucket, rows, cols); brain.py's module bank adds
# several shapes, more than the default limit of 8 (past it, the function silently runs uncompiled).
torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)


@torch.compile(dynamic=False)
def newton_schulz5(G, steps: int = 5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.05, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0, splits=None):
        """splits maps a parameter to its row-block sizes, e.g. [q_rows, k_rows, v_rows]."""
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                                      weight_decay=weight_decay))
        self.splits = {id(p): list(s) for p, s in (splits or {}).items()}

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            buckets = defaultdict(list)  # matrix shape -> [(param, stack index, first row, matrix)]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad, 1 - mom)
                g = p.grad.lerp(buf, mom) if group["nesterov"] else buf
                if wd:
                    p.mul_(1 - lr * wd)
                for e, ge in enumerate(g.unbind(0) if g.dim() == 3 else (g,)):
                    row = 0
                    for m in ge.split(self.splits.get(id(p), [ge.shape[0]]), dim=0):
                        buckets[tuple(m.shape)].append((p, e if g.dim() == 3 else None, row, m))
                        row += m.shape[0]
            for (rows, cols), items in buckets.items():
                U = newton_schulz5(torch.stack([m for *_, m in items]), group["ns_steps"])
                scale = max(1.0, rows / cols) ** 0.5
                for (p, e, row, _), u in zip(items, U):
                    (p if e is None else p[e])[row:row + rows].add_(u, alpha=-lr * scale)
