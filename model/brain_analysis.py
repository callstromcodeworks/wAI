"""What a trained BrainGPT learned: module specialisation, critic quality, memory use.

    python model/brain_analysis.py runs/sweep_b32k_full/ckpt.pt [--tokens 400000]

Runs validation text from each source through the model and reports:
- modules: the share of English and code tokens each module fires on, and the tokens each
  fires on most often relative to how common they are (what it specialised in)
- critic: correlation between its error estimate and the draft's actual next-token loss, and
  its contribution: the final loss with the critic's output switched off minus the loss with it
  on, grouped by the critic's estimate. If the gate uses the critic to fix what it flags, the
  contribution is largest where the estimate is highest.
- memory: how open its gate is on average, and how many of its slots get read
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel

from brain import BrainConfig, BrainGPT
from prepare_data import load_shard
from tokenizer import VOCABS, Tokenizer
from train import SDPA_BACKENDS

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tokens", type=int, default=400_000, help="validation tokens per source")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")  # module top tokens can include bytes the console code page lacks
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = BrainConfig(**ck["config"])
    with torch.device("cuda"):
        model = BrainGPT(cfg)
    model.load_state_dict(ck["model"])
    model.train()  # train-mode forward returns the draft and the critic's estimate (no dropout anywhere)
    tok = Tokenizer(cfg.n_ranks)
    data_dir = os.path.join(HERE, "data", ck["args"]["vocab"])
    T, B = cfg.seq_len, a.batch

    captured = {}
    if model.router is not None:
        model.router.register_forward_hook(lambda m, i, o: captured.__setitem__("router", o.float()))
    if model.memory is not None:
        model.memory.register_forward_hook(lambda m, i, o: captured.__setitem__("mem_in", i[0]))

    report = {"config": {k: v for k, v in ck["config"].items()}, "sources": {}}
    E = cfg.n_modules
    for source in ("fineweb-edu", "code"):
        val, _ = load_shard(os.path.join(data_dir, source, "val_000000.bin"))
        n_rows = min(a.tokens // T, (len(val) - 1) // T) // B * B
        fire = np.zeros(E)
        tok_count, tok_fire = Counter(), [Counter() for _ in range(E)]
        est, d_loss, f_loss, off_loss, say = [], [], [], [], []
        gate_sum, slots_used, n_tok = 0.0, set(), 0
        for i in range(0, n_rows, B):
            rows = torch.from_numpy(np.stack([val[r * T:r * T + T + 1] for r in range(i, i + B)]).astype(np.int64)).cuda()
            x, y = rows[:, :-1], rows[:, 1:]
            with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPA_BACKENDS, set_priority=True):
                hidden, draft, err, _, _, logvar, _ = model(x)
            f_loss.append(model.token_losses(hidden, y).flatten().cpu())
            d_loss.append(model.token_losses(draft, y).flatten().cpu())
            if err is not None:
                est.append(err.flatten().cpu())
                if cfg.critic_gate:  # how much of its output reached the gate, per token
                    say.append(model.critic_say(logvar).float().expand(*err.shape, 1).flatten().cpu())
                saved = model.critic.w_out.clone(), model.err_to_resid.clone()
                model.critic.w_out.zero_(); model.err_to_resid.zero_()  # critic silent: no correction reaches the gate
                with torch.autocast("cuda", torch.bfloat16), sdpa_kernel(SDPA_BACKENDS, set_priority=True):
                    off_loss.append(model.token_losses(model(x)[0], y).flatten().cpu())
                model.critic.w_out.copy_(saved[0]); model.err_to_resid.copy_(saved[1])
            ids = x.flatten().cpu().numpy()
            tok_count.update(ids.tolist())
            n_tok += ids.size
            if E:
                k = cfg.top_k or (E - cfg.n_shared)
                scores = captured["router"].flatten(0, 1)  # routed modules only, as brain.route chooses
                if cfg.sigmoid_router:
                    scores = torch.sigmoid(scores)
                if cfg.balance_bias:
                    scores = scores + model.route_bias
                top = scores.topk(k, dim=-1).indices.cpu().numpy() + cfg.n_shared
                for e in range(E):
                    hit = np.ones(len(top), bool) if e < cfg.n_shared else (top == e).any(1)
                    fire[e] += hit.sum()
                    tok_fire[e].update(ids[hit].tolist())
            if model.memory is not None:
                mem = model.memory
                xin = captured["mem_in"]
                from gpt import norm
                xn = norm(xin).reshape(-1, xin.size(-1)).float()
                gate_sum += torch.sigmoid(mem.gate(xn).float()).sum().item()
                q = norm(mem.query(xn).float()).view(-1, 2, mem.dk // 2)
                v1, i1 = (q[:, 0] @ mem.keys[0].float().t()).topk(mem.k, dim=-1)
                v2, i2 = (q[:, 1] @ mem.keys[1].float().t()).topk(mem.k, dim=-1)
                best = (v1[:, :, None] + v2[:, None, :]).flatten(1).topk(mem.k, dim=-1).indices
                slot = i1.gather(1, best // mem.k) * mem.n + i2.gather(1, best % mem.k)
                slots_used.update(torch.unique(slot).tolist())
        f_loss, d_loss = torch.cat(f_loss), torch.cat(d_loss)
        src = {"tokens": n_tok, "final_loss": f_loss.mean().item(), "draft_loss": d_loss.mean().item()}
        if E:
            src["module_fire_share"] = (fire / n_tok).round(3).tolist()
            spec = []
            for e in range(E):
                total_e = sum(tok_fire[e].values())
                lift = [((tok_fire[e][t] / total_e) / (tok_count[t] / n_tok), t)
                        for t in tok_fire[e] if tok_count[t] >= 30]
                lift.sort(reverse=True)
                spec.append([(tok.decode([t]), round(l, 1)) for l, t in lift[:12]])
            src["module_top_tokens"] = spec
        if est:
            est, off_loss = torch.cat(est), torch.cat(off_loss)
            src["critic_corr"] = torch.corrcoef(torch.stack([est, d_loss]))[0, 1].item()
            src["loss_without_critic"] = off_loss.mean().item()
            q = torch.quantile(est, torch.tensor([0.2, 0.4, 0.6, 0.8]))
            bucket = torch.bucketize(est, q)
            src["by_critic_estimate"] = [
                dict(quintile=b + 1, estimate=est[bucket == b].mean().item(), draft=d_loss[bucket == b].mean().item(),
                     final=f_loss[bucket == b].mean().item(),
                     gain=(d_loss[bucket == b] - f_loss[bucket == b]).mean().item(),
                     critic_contribution=(off_loss[bucket == b] - f_loss[bucket == b]).mean().item(),
                     **({"say": torch.cat(say)[bucket == b].mean().item()} if say else {}))
                for b in range(5)]
            if say:
                say_all = torch.cat(say)
                miss = (est - d_loss).abs()  # how far off its estimate was, per token
                src["critic_trust"] = model.trust.item()
                src["critic_say_mean"] = say_all.mean().item()
                # negative = it speaks up more where its estimates are accurate (what confidence should do)
                src["say_vs_miss_corr"] = torch.corrcoef(torch.stack([say_all, miss]))[0, 1].nan_to_num(0.0).item()
        if model.memory is not None:
            src["memory_gate_mean"] = gate_sum / n_tok
            src["memory_slots_used"] = f"{len(slots_used)} of {cfg.memory_sub ** 2}"
        report["sources"][source] = src

    for source, src in report["sources"].items():
        print(f"\n== {source}: final loss {src['final_loss']:.4f}, draft loss {src['draft_loss']:.4f} "
              f"({src['tokens']:,} tokens)")
        if "module_fire_share" in src:
            print("module fire share:", "  ".join(f"m{e}:{s:.0%}" for e, s in enumerate(src["module_fire_share"])))
            for e, toks in enumerate(src["module_top_tokens"]):
                print(f"  m{e} fires most on: " + ", ".join(f"{t!r}x{l}" for t, l in toks[:8]))
        if "critic_corr" in src:
            print(f"critic: correlation with the draft's real loss {src['critic_corr']:.3f}; final loss with the "
                  f"critic switched off {src['loss_without_critic']:.4f} (on: {src['final_loss']:.4f})")
            if "critic_trust" in src:
                print(f"  earned say: trust {src['critic_trust']:.2f}, average say {src['critic_say_mean']:.2f}, "
                      f"say vs how far off its estimate was: corr {src['say_vs_miss_corr']:+.3f} (negative = speaks "
                      f"up where it's accurate)")
            for r in src["by_critic_estimate"]:
                print(f"  quintile {r['quintile']}: estimate {r['estimate']:.2f} | draft {r['draft']:.3f} -> final "
                      f"{r['final']:.3f} | critic's contribution {r['critic_contribution']:+.4f} nats"
                      + (f" | say {r['say']:.2f}" if "say" in r else ""))
        if "memory_gate_mean" in src:
            print(f"memory: gate open {src['memory_gate_mean']:.1%} on average, slots read {src['memory_slots_used']}")
    out = os.path.join(os.path.dirname(a.ckpt), "analysis.json")
    json.dump(report, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
