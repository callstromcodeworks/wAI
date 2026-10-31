"""FP8 linear layers (fp8.py): compiled gradients match eager, a full-size layer stays finite, on-grid values
multiply exactly, and a small model learns under compile."""
import sys
import os
MODEL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MODEL)
import torch
import fp8

ok = lambda c, name, extra="": print(("PASS " if c else "FAIL ") + name + (f"  ({extra})" if extra else ""))
try:
    worst = fp8.check_compiled()
    ok(True, "compiled FP8 linear matches eager (gradients finite, within 5%)", f"worst {worst:.2e}")
except RuntimeError as e:
    ok(False, "compiled FP8 linear matches eager", str(e))

# a full-size layer
x = torch.randn(4096, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
w = (torch.randn(2304, 768, device="cuda") * 0.05).requires_grad_(True)
torch._dynamo.reset()
f = torch.compile(fp8.fp8_linear, dynamic=False)
f(x, w).float().pow(2).sum().backward()
ok(bool(torch.isfinite(x.grad).all()) and bool(torch.isfinite(w.grad).all()), "compiled 4096x768->2304 step is finite")

# exactness where nothing is rounded: values already on the fp8 grid, power-of-two absmax
a = torch.tensor([[1.0, -2.0, 0.5, 4.0] * 4] * 16, device="cuda", dtype=torch.bfloat16)
b = torch.tensor([[0.25, 1.0, -1.0, 2.0] * 4] * 32, device="cuda")
y = f(a.clone().requires_grad_(True), b.clone().requires_grad_(True))
ok(torch.allclose(y.float(), a.float() @ b.t(), rtol=0, atol=0), "on-grid values give the exact product")

# a tiny MLP learns with FP8 layers under compile (loss falls, stays finite)
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(64, 256, bias=False), torch.nn.ReLU(), torch.nn.Linear(256, 64, bias=False)).cuda()
for m in net:
    if isinstance(m, torch.nn.Linear):
        m.forward = (lambda mod: (lambda t: fp8.fp8_linear(t, mod.weight)))(m)
step = torch.compile(lambda t: net(t))
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
xs = torch.randn(512, 64, device="cuda", dtype=torch.bfloat16)
ys = xs.float() @ torch.randn(64, 64, device="cuda")
losses = []
for i in range(200):
    loss = (step(xs).float() - ys).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
ok(all(v == v for v in losses) and losses[-1] < 0.2 * losses[0], "an FP8 MLP learns under compile",
   f"loss {losses[0]:.3f} -> {losses[-1]:.3f}")
