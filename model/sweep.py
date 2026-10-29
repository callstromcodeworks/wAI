"""Equal-time training sweeps over the weight-size vs KV-cache knobs, plus a summary table.

    python model/sweep.py vocab --minutes 20     # run a group (finished runs are skipped)
    python model/sweep.py --summary              # table of every run in model/runs/sweep_*

Every run gets the same wall-clock training budget on this GPU and is scored in bits per byte
on the same validation text. That folds in everything a design choice changes: tokens per
byte, FLOPs per token, achieved throughput, and how well the model learns.

Equal time is only fair when each run has the GPU to itself. train.py benchmarks a bf16 GEMM at
startup; the summary shows it (gpu_TF) and marks runs below GPU_SHARED_BELOW as "GPU shared"
(their tok/s and final scores are low; step-matched scores are unaffected).
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MIX = {"fineweb-edu": 0.6, "code": 0.4}
GPU_SHARED_BELOW = 62.0  # TFLOPS; the RTX 4070 measures 64-65 when nothing else is using it

GROUPS = {
    # Vocabulary size at a fixed transformer body (12 x 768, full multi-head attention).
    "vocab": {
        "v64k": "--vocab ec64k --preset small",
        "v200k": "--vocab ec200k --preset small",
        "v32k": "--vocab ec32k --preset small",
        "v16k": "--vocab ec16k --preset small",
        "v100k": "--vocab ec100k --preset small",
    },
}


def combo_group(vocab):
    """Phase 3: the winning recipe (value residual + U-net + FP8) at three K/V cache sizes."""
    v = vocab[2:]
    base = f"--vocab {vocab} --preset small --value_residual 1 --unet 1 --fp8 1"
    return {
        f"c{v}_mha": base,                                       # 36 KiB/token
        f"c{v}_kv2": f"{base} --n_kv_head 2",                    # 12 KiB/token
        f"c{v}_kv1_s2": f"{base} --n_kv_head 1 --kv_share 2",    # 3 KiB/token
    }


def arch_group(vocab):
    """Phase 2, at the chosen vocabulary: near-free accuracy tricks, and K/V-cache reductions."""
    v = vocab[2:]  # "ec32k" -> "32k"
    base = f"--vocab {vocab} --preset small"
    return {
        f"a{v}_vr_unet": f"{base} --value_residual 1 --unet 1",
        f"a{v}_kv2": f"{base} --n_kv_head 2",                    # GQA: 3x less KV
        f"a{v}_kv1_s2": f"{base} --n_kv_head 1 --kv_share 2",    # MQA + cross-layer: 12x less KV
        f"a{v}_mla96": f"{base} --mla_rank 96 --mla_rope_dim 32",  # latent attention: 12x less KV
        f"a{v}_fp8": f"{base} --fp8 1",                          # FP8 GEMMs: +22% tokens/s
    }


def brain_group(vocab):
    """brain.py against the best GPT recipe at equal time, then one piece removed per run."""
    v = vocab[2:]
    common = f"--vocab {vocab} --preset small --value_residual 1 --unet 1 --fp8 1 --n_kv_head 2"
    brain = f"{common} --arch brain --n_layer 8"
    return {
        f"b{v}_ref": common,                                    # the c32k_kv2 recipe, re-run alongside
        f"b{v}_full": brain,                                    # modules (top-2 of 8) + memory + critic
        f"b{v}_nocritic": f"{brain} --critic 0 --draft_weight 0",
        f"b{v}_nomem": f"{brain} --memory_sub 0",
        f"b{v}_dense": f"{brain} --top_k 0",                    # every module writes, softmax-weighted
        f"b{v}_nobank": f"{brain} --n_modules 0",
        # round 2: separate what each piece costs and gives
        f"b{v}_ref_s2": f"{common} --seed 7",                   # run-to-run noise
        f"b{v}_plain8": f"{brain} --n_modules 0 --memory_sub 0 --critic 0 --draft_weight 0",  # the 8-block stack alone
        f"b{v}_bank": f"{brain} --memory_sub 0 --critic 0 --draft_weight 0",  # 8 blocks + module bank only
        f"b{v}_draftonly": f"{brain} --memory_sub 0 --critic 0",  # bank + draft loss, no critic
        f"b{v}_critic_nodraft": f"{brain} --memory_sub 0 --draft_weight 0",  # bank + critic, draft not trained
        # round 3: bias balancing (no dead modules), and a critic that observes without reshaping the draft
        # bank2: one shared module (training made one module fire on nearly every token anyway) +
        # top-2 of 7 routed, balanced by the router bias on bounded sigmoid scores
        f"b{v}_bank2": f"{brain} --memory_sub 0 --critic 0 --draft_weight 0 --balance_weight 0 --balance_bias 0.01 "
                       f"--n_shared 1 --sigmoid_router 1",
        f"b{v}_critic2": f"{brain} --memory_sub 0 --balance_weight 0 --balance_bias 0.01 --critic_detach 1",
        f"b{v}_critic2_nodraft": f"{brain} --memory_sub 0 --balance_weight 0 --balance_bias 0.01 --critic_detach 1 "
                                 f"--draft_weight 0",
        # bank3: the bias moved a token-independent "default" role between routed modules and starved the
        # evicted ones; here the shared module takes that role and the loss balances the routed seven
        f"b{v}_bank3": f"{brain} --memory_sub 0 --critic 0 --draft_weight 0 --n_shared 1",
        # bank4: bank3 kept one routed module dead (the loss gives a module that never fires no push);
        # the bias on top revives it, while the shared module and the loss stop a default role forming
        f"b{v}_bank4": f"{brain} --memory_sub 0 --critic 0 --draft_weight 0 --n_shared 1 --balance_bias 0.01",
        # round 4: equal tokens instead of equal time (--tokens fixes the step count, so --hours is unused),
        # each with its own full schedule: is the modular model more accurate for the same training?
        f"b{v}_ref_t100m": f"{common} --tokens 1e8",
        f"b{v}_plain8_t100m": f"{brain} --n_modules 0 --memory_sub 0 --critic 0 --draft_weight 0 --tokens 1e8",
        f"b{v}_bank3_t100m": f"{brain} --memory_sub 0 --critic 0 --draft_weight 0 --n_shared 1 --tokens 1e8",
        # round 5: a critic that earns its say (observes first; trust grows with its track record),
        # on the bank3 base at equal tokens, against the same critic with full say from step 0
        f"b{v}_critic2_t100m": f"{brain} --memory_sub 0 --draft_weight 0 --n_shared 1 --critic_detach 1 --tokens 1e8",
        f"b{v}_critic3_t100m": f"{brain} --memory_sub 0 --draft_weight 0 --n_shared 1 --critic_detach 1 "
                               f"--critic_gate 2 --tokens 1e8",
        f"b{v}_critic3t_t100m": f"{brain} --memory_sub 0 --draft_weight 0 --n_shared 1 --critic_detach 1 "
                                f"--critic_gate 1 --tokens 1e8",
        # round 6: is the earned-say critic's 0.003 real? two more seeds of it and of its no-critic base;
        # if it holds, both at 1B tokens to see whether the gain grows with training
        **{f"b{v}_{name}_t100m_s{seed}": f"{flags} --tokens 1e8 --seed {seed}" for seed in (7, 13) for name, flags in (
            ("bank3", f"{brain} --memory_sub 0 --critic 0 --draft_weight 0 --n_shared 1"),
            ("critic3", f"{brain} --memory_sub 0 --draft_weight 0 --n_shared 1 --critic_detach 1 --critic_gate 2"))},
        f"b{v}_bank3_t1b": f"{brain} --memory_sub 0 --critic 0 --draft_weight 0 --n_shared 1 --tokens 1e9",
        f"b{v}_critic3_t1b": f"{brain} --memory_sub 0 --draft_weight 0 --n_shared 1 --critic_detach 1 --critic_gate 2 "
                             f"--tokens 1e9",
        # 1B replicates: does the earned-say critic's -0.0024 at 1B hold on other seeds?
        **{f"b{v}_{name}_t1b_s{seed}": f"{flags} --tokens 1e9 --seed {seed}" for seed in (7, 13) for name, flags in (
            ("bank3", f"{brain} --memory_sub 0 --critic 0 --draft_weight 0 --n_shared 1"),
            ("critic3", f"{brain} --memory_sub 0 --draft_weight 0 --n_shared 1 --critic_detach 1 --critic_gate 2"))},
        # round 7: sparse modules, each routed module computes only on its own tokens (bank3 shape:
        # 1 shared + top-2 of 7), two seeds each, against bank3's three seeds (mean 1.1455)
        **{f"b{v}_sparse{tag}_t100m" + ("" if seed == 1337 else f"_s{seed}"):
           f"{brain} --memory_sub 0 --critic 0 --draft_weight 0 --n_shared 1 --module_mode {mode} --tokens 1e8 --seed {seed}"
           for seed in (1337, 7) for tag, mode in (("B", "routed_attn"), ("C", "ffn"))},
        # round 8: more routed modules at the same compute per token (still top-2), with dead-module
        # revival; B8r separates the revival rule from the module count
        **{f"b{v}_sparseB{n}_t100m" + ("" if seed == 1337 else f"_s{seed}"):
           f"{brain} --memory_sub 0 --critic 0 --draft_weight 0 --n_shared 1 --module_mode routed_attn "
           f"--n_modules {n + 1} --revive 0.1 --tokens 1e8 --seed {seed}"
           for n, seeds in ((16, (1337, 7)), (32, (1337, 7))) for seed in seeds},
        f"b{v}_sparseB8r_t100m": f"{brain} --memory_sub 0 --critic 0 --draft_weight 0 --n_shared 1 "
                                 f"--module_mode routed_attn --revive 0.1 --tokens 1e8",
        # bigbank2: 32 and 64 routed modules with room for them (capacity 2.0x, round 8 dropped a quarter
        # of the firings at 1.25x), plus the 32-module bank and plain8 at 1B tokens: does the edge grow?
        **{f"b{v}_sparseB{n}c2_t100m" + ("" if seed == 1337 else f"_s{seed}"):
           f"{brain} --memory_sub 0 --critic 0 --draft_weight 0 --n_shared 1 --module_mode routed_attn "
           f"--n_modules {n + 1} --revive 0.1 --capacity 2.0 --tokens 1e8 --seed {seed}"
           for n in (32, 64) for seed in (1337, 7)},
        f"b{v}_sparseB32c2_t1b": f"{brain} --memory_sub 0 --critic 0 --draft_weight 0 --n_shared 1 "
                                 f"--module_mode routed_attn --n_modules 33 --revive 0.1 --capacity 2.0 --tokens 1e9",
        f"b{v}_plain8_t1b": f"{brain} --n_modules 0 --memory_sub 0 --critic 0 --draft_weight 0 --tokens 1e9",
        # same-hardware baselines: H200 runs score lower than the RTX 4070 runs at equal settings
        f"b{v}_plain8_t100m_s7": f"{brain} --n_modules 0 --memory_sub 0 --critic 0 --draft_weight 0 --tokens 1e8 --seed 7",
    }


def memory_group(vocab):
    """Long-context memory (memory.py, PLAN_concept_memory.md): 8 plain blocks, 4,096-token sequences,
    300M tokens each. Score with long_context.py after the runs finish."""
    v = vocab[2:]
    base = (f"--vocab {vocab} --preset small --value_residual 1 --unet 1 --fp8 1 --n_kv_head 2 --n_layer 8 "
            f"--seq_len 4096 --tokens 3e8")
    concept = "--attn_mode concept --window 512 --concept_block 64 --concept_topk 4"
    return {
        f"m{v}_full": base,                                             # every past token: upper bound
        f"m{v}_window": f"{base} --attn_mode window --window 512",     # the last 512 only: lower bound
        f"m{v}_concept": f"{base} {concept}",                          # window + concepts + expansion
        f"m{v}_concept_noexp": f"{base} {concept} --concept_topk 0",   # does expanding detail matter?
        f"m{v}_window_fast": f"{base} --attn_mode window --window 512 --fast_heads 2",  # Hebbian memory alone
        f"m{v}_concept_fast": f"{base} {concept} --fast_heads 2",      # both together
        f"m{v}_concept_s7": f"{base} {concept} --seed 7",
        f"m{v}_concept_fast_s7": f"{base} {concept} --fast_heads 2 --seed 7",
    }


def memory2_group(vocab):
    """Phase 2 memory: position-free concepts (fixes the breakdown past the training length) and 16k context,
    where full attention gets expensive. 8 plain blocks, 300M tokens each; score with long_context.py."""
    v = vocab[2:]
    base = f"--vocab {vocab} --preset small --value_residual 1 --unet 1 --fp8 1 --n_kv_head 2 --n_layer 8 --tokens 3e8"
    concept = "--attn_mode concept --window 512 --concept_block 64 --concept_topk 4 --concept_rope 0"
    return {
        f"m{v}_concept_nopos_fast": f"{base} --seq_len 4096 {concept} --fast_heads 2",
        f"m16_full": f"{base} --seq_len 16384",
        f"m16_window": f"{base} --seq_len 16384 --attn_mode window --window 512",
        f"m16_concept_nopos": f"{base} --seq_len 16384 {concept}",
        f"m16_concept_nopos_fast": f"{base} --seq_len 16384 {concept} --fast_heads 2",
    }


def scale_group(vocab):
    """Phase 2 scale: does what helped small models pay off at ~230M parameters (16 x 1024)? 1.5B tokens each."""
    common = "--preset medium --value_residual 1 --unet 1 --fp8 1 --n_kv_head 2 --tokens 1.5e9"
    bank = ("--arch brain --n_layer 16 --memory_sub 0 --critic 0 --draft_weight 0 --n_shared 1 --n_modules 33 "
            "--module_mode routed_attn --revive 0.1 --capacity 2.0")
    return {
        "s_med_plain": f"--vocab {vocab} {common}",
        "s_med_brain": f"--vocab {vocab} {common} {bank}",
        "s_med_plain_ec64k": f"--vocab ec64k {common}",
    }


def combo_group(vocab):
    """Phase 3: the brain parts together (sparse module bank + position-free concept memory + fast weights),
    which added up at 100M tokens locally; 600M tokens each (budgeted to fit the rental), small and medium."""
    common = "--value_residual 1 --unet 1 --fp8 1 --n_kv_head 2 --seq_len 4096"
    memory = "--attn_mode concept --window 512 --concept_block 64 --concept_topk 4 --concept_rope 0 --fast_heads 2"
    bank = ("--arch brain --memory_sub 0 --critic 0 --draft_weight 0 --n_shared 1 --n_modules 33 "
            "--module_mode routed_attn --revive 0.1 --capacity 2.0")
    small, med = f"--vocab {vocab} --preset small --n_layer 8 {common}", f"--vocab {vocab} --preset medium {common}"
    return {
        "c_med_combo": f"{med} {memory} {bank} --n_layer 16 --tokens 6e8",
        "c_med_full4k": f"{med} --tokens 6e8",
        "c_combo_full_t1b": f"{small} {memory} {bank} --tokens 6e8",
        "c_combo_mem_t1b": f"{small} {memory} --tokens 6e8",
        "c_full4k_t1b": f"{small} --tokens 6e8",
    }


def run_dir(name):
    return os.path.join(HERE, "runs", f"sweep_{name}")


def finished(name):
    log = os.path.join(run_dir(name), "log.jsonl")
    if not os.path.exists(log):
        return False
    return "done" in open(os.path.join(run_dir(name), "stdout.txt"), errors="replace").read()


def bytes_per_token(vocab):
    """Bytes of text per token for the 60/40 English/code byte mix, from the tokenized data stats."""
    bpt = {s: json.load(open(os.path.join(HERE, "data", vocab, s, "stats.json")))["bytes_per_token"] for s in MIX}
    return 1.0 / sum(w / bpt[s] for s, w in MIX.items())


def brain_label(m):
    bank = (f" bank{m['n_modules']}x{m['module_dim']}" + (f"s{m['n_shared']}+" if m.get("n_shared") else "") + (f"top{m['top_k']}" if m["top_k"] else "dense")
                                                + {"routed_attn": " sparse-attn", "ffn": " sparse-ffn"}.get(m.get("module_mode"), ""))         if m["n_modules"] else " nobank"
    return f" brain{bank}" + (f" mem{m['memory_sub'] ** 2 // 1024}k" if m["memory_sub"] else " nomem")         + (" critic" if m["critic"] else " nocritic")


def gpu_tflops(d):
    """The startup GEMM benchmark from a run's stdout, scaled so both methods share one threshold:
    the old 0.2 s burst ("peak") read 64-65 on an idle card, the sustained 2 s timing reads about
    59 at the 200 W power cap. None for runs older than the benchmark."""
    path = os.path.join(d, "stdout.txt")
    text = open(path, errors="replace").read() if os.path.exists(path) else ""
    if found := re.search(r"measured bf16 GEMM sustained ([0-9.]+) TFLOPS", text):
        return float(found.group(1)) * 64.5 / 59.3
    found = re.search(r"measured bf16 GEMM peak ([0-9.]+) TFLOPS", text)
    return float(found.group(1)) if found else None


def summarize():
    rows = []
    for d in sorted(glob.glob(os.path.join(HERE, "runs", "sweep_*"))):
        cfg_path, log_path = os.path.join(d, "config.json"), os.path.join(d, "log.jsonl")
        if not (os.path.exists(cfg_path) and os.path.exists(log_path)):
            continue
        cfg = json.load(open(cfg_path))
        recs = [json.loads(l) for l in open(log_path)]
        evals = [r for r in recs if "eval" in r]
        speeds = [r["tok_s"] for r in recs if "tok_s" in r]
        if not evals:
            continue
        last = evals[-1]
        vocab = cfg["args"]["vocab"]
        bpt = bytes_per_token(vocab)
        m = cfg["model"]
        rows.append(dict(
            run=os.path.basename(d)[6:], vocab=vocab,
            arch=f"L{m['n_layer']} d{m['d_model']} h{m['n_head']} kv{m['n_kv_head']} s{m['kv_share']}"
                 + (f" mla{m['mla_rank']}" if m.get("mla_rank") else "")
                 + (" vr" if m.get("value_residual") else "") + (" unet" if m.get("unet") else "")
                 + (" fp8" if cfg["args"].get("fp8") else "")
                 + (brain_label(m) if m.get("arch") == "brain" else "")
                 + (f" {m['attn_mode']}{m['window']}" + (f" top{m['concept_topk']}" if m["attn_mode"] == "concept" else "")
                    if m.get("attn_mode", "full") != "full" else "")
                 + (f" fast{m['fast_heads']}" if m.get("fast_heads") else ""),
            params_M=cfg["params"] / 1e6, emb_M=cfg["embedding_params"] / 1e6,
            weights_MiB=cfg["params"] * 2 / 2**20, kv_KiB_tok=cfg["kv_bytes_per_token"] / 1024,
            kv_MiB_per_MB=cfg["kv_bytes_per_token"] * 1e6 / bpt / 2**20, bytes_tok=bpt,
            tok_s=sorted(speeds)[len(speeds) // 2] if speeds else 0,
            text_KB_s=(sorted(speeds)[len(speeds) // 2] if speeds else 0) * bpt / 1e3,
            tokens_M=last["tokens"] / 1e6, eng_bpb=last["eval"]["fineweb-edu"]["bpb"],
            code_bpb=last["eval"]["code"]["bpb"], mix_bpb=last["mix_bpb"], final=finished(os.path.basename(d)[6:]),
            gpu_TF=gpu_tflops(d)))
    if not rows:
        print("no runs yet")
        return rows
    hdr = ("run", "vocab", "arch", "params_M", "emb_M", "weights_MiB", "kv_KiB_tok", "kv_MiB_per_MB", "bytes_tok",
           "gpu_TF", "tok_s", "text_KB_s", "tokens_M", "eng_bpb", "code_bpb", "mix_bpb")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in sorted(rows, key=lambda r: r["mix_bpb"]):
        cells = []
        for k in hdr:
            v = r[k]
            cells.append("-" if v is None else f"{v:.4f}" if k.endswith("bpb") else f"{v:,.0f}" if k in ("tok_s",) else
                         f"{v:.2f}" if isinstance(v, float) else str(v))
        notes = [] if r["final"] else ["running"]
        if r["gpu_TF"] is not None and r["gpu_TF"] < GPU_SHARED_BELOW:
            notes.append("GPU shared")
        print("| " + " | ".join(cells) + " |" + (f" ({', '.join(notes)})" if notes else ""))
    json.dump(rows, open(os.path.join(HERE, "runs", "sweep_summary.json"), "w"), indent=1)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group", nargs="?", choices=sorted(GROUPS) + ["arch", "combo", "brain", "memory"])
    ap.add_argument("--vocab", default="ec32k", help="vocabulary for the arch group")
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--only", default=None, help="comma-separated run names within the group")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.group:
        group = {"arch": arch_group, "combo": combo_group, "brain": brain_group, "memory": memory_group}.get(
            a.group, lambda v: GROUPS[a.group])(a.vocab)
        for name, flags in group.items():
            if a.only and name not in a.only.split(","):
                continue
            if finished(name):
                print(f"{name}: already done")
                continue
            os.makedirs(run_dir(name), exist_ok=True)
            cmd = [sys.executable, "-u", os.path.join(HERE, "train.py"), *flags.split(), "--run", f"sweep_{name}",
                   "--hours", str(a.minutes / 60), "--eval_every", "100", "--save_every", "1000000",
                   "--log_every", "20"]
            print(f"{name}: {' '.join(cmd[2:])}", flush=True)
            with open(os.path.join(run_dir(name), "stdout.txt"), "w") as out:
                subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT)
            summarize()
    if a.summary or not a.group:
        summarize()


if __name__ == "__main__":
    main()
