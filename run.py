#!/usr/bin/env python3
"""One entry point for a rented Linux GPU box: set up, check, train several runs at once, evaluate.

    python run.py all                       # setup, data, test, train (default queue), eval
    python run.py setup                     # build fasttok, check packages, GPU and vocabulary
    python run.py data [--download]         # check the token shards (or rebuild them from Hugging Face)
    python run.py test                      # tokenizer + model correctness suites, in parallel
    python run.py train [--queue memory,critic1b] [--parallel 4]
    python run.py status                    # progress of every queued run
    python run.py eval                      # long-context scores + the sweep summary table
    python run.py export                    # finished runs -> safetensors + JSON, safe to take home

Training runs several experiments at the same time on one big GPU (an H200 NVL has 141 GB): each
run gets an equal share of VRAM (--vram_gb, default: 92% of the card / --parallel) and the GPU's
compute is shared. Every queued run has a fixed token budget, so sharing changes how long a run
takes, never its result. Runs checkpoint every --save_every steps and pick up where they left off
if the box restarts (interruptible rentals): just run the same command again.

Queues (from model/sweep.py):
  memory    the long-context round: full / window / concept memory / fast weights (8 runs x 300M tokens)
  critic1b  1B-token replicates of the earned-say critic vs its no-critic base (seeds 7, 13; 4 runs)
  bigbank   more routed modules, sparse mode B, with revival (5 runs x 100M tokens; done at home)
  bigbank2  32 and 64 routed modules at capacity 2.0 (2 seeds x 100M), and 32 vs plain8 at 1B tokens
  smoke     two tiny runs in parallel, a few minutes: checks training and resuming before the real queue
"""
import argparse
import hashlib
import importlib
import os
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(ROOT, "model")
RUNS = os.path.join(MODEL, "runs")
PY = sys.executable
VOCAB_FILE = os.path.join(ROOT, "vocab", "ec200k_base.tiktoken")
VOCAB_SHA256 = "7a17729c50552022f725e9ea880f40697933a7318c9e9cb2e0d50cd7e45b0b15"
DATA = os.path.join(MODEL, "data", "ec32k")
SOURCES = {"fineweb-edu": 3, "code": 6}  # upstream parquet files: the same text as the home training set
TEST_SUITES = ["brain_checks", "gate_checks", "bias_check", "sparse_checks", "memory_checks", "revive_check",
               "depth_checks", "spike_checks"]
DEFAULT_QUEUE = "memory,critic1b,bigbank2"


def say(msg=""):
    print(msg, flush=True)


def sh(cmd, **kw):
    say("$ " + " ".join(cmd))
    return subprocess.run(cmd, **kw)


# ---- queues ------------------------------------------------------------------

def queues():
    sys.path.insert(0, MODEL)
    import sweep  # noqa: E402  (model/sweep.py holds every run's flags)
    brain = sweep.brain_group("ec32k")
    pick = lambda names: {n: brain[n] for n in names}
    return {
        "memory": sweep.memory_group("ec32k"),
        "memory2": sweep.memory2_group("ec32k"),
        "scale": sweep.scale_group("ec32k"),
        # medium brain diverged at step 1660 (same seed reproduces it): does FP8 or the unbounded router cause it?
        # same seed and schedule as s_med_brain, all instrumented (router max logit, bank/stream rms, grad norm),
        # stopped at step 2600 (it diverged at 1660): which change keeps the medium bank stable?
        "diag2": {f"d_{tag}": f"{sweep.scale_group('ec32k')['s_med_brain']} --stop_at 2600 {extra}"
                  for tag, extra in (("base", ""), ("z", "--router_z 0.001"), ("norevive", "--revive 0"),
                                     ("lr_half", "--lr_muon 0.025"))},
        # the cause: biased top-k picked modules whose softmax probs underflowed, and gates = 0/0. d_fix = d_base + the fix
        "diag3": {"d_fix": f"{sweep.scale_group('ec32k')['s_med_brain']} --stop_at 2600"},
        # s_med_brain with the gate fix, resumed from d_fix's step-2600 checkpoint (identical config and schedule)
        "scale2": {"s_med_brain_fix": sweep.scale_group("ec32k")["s_med_brain"]},
        # one judging module beside the others: its per-token error estimate sets how many routed modules each token
        # gets (same average). Controls: the same bank without a judge (b32k_sparseB32c2_t100m, 2 seeds), judge
        # without allocation, and allocation by a random score
        "judge": {name: f"{sweep.brain_group('ec32k')['b32k_sparseB32c2_t100m']} {extra}"
                  for name, extra in (("j_alloc", "--judge 2"), ("j_alloc_s7", "--judge 2 --seed 7"),
                                      ("j_only", "--judge 1"), ("j_random", "--judge 3"))},
        # judge round 2: round 1's judge reached corr ~0.37 at the routing depth, short of the trust bar (0.45 for
        # half say), so it never allocated. Bar scaled to what's achievable this early (half say at 0.25, full at 0.35);
        # j2_surprise also feeds it each token's surprise; c_bank_rerun measures run-to-run noise of the control
        "judge2": {name: f"{sweep.brain_group('ec32k')['b32k_sparseB32c2_t100m']} {extra}"
                   for name, extra in (("j2_alloc", "--judge 2 --gate_lo 0.15 --gate_hi 0.35"),
                                       ("j2_random", "--judge 3 --gate_lo 0.15 --gate_hi 0.35"),
                                       ("j2_surprise", "--judge 2 --judge_surprise 1 --gate_lo 0.15 --gate_hi 0.35"),
                                       ("c_bank_rerun", ""))},
        # judge round 3, the owner's aim: cognitive dissonance. The judge anticipates conflict between the model's first
        # impression and its considered view (no answer needed) and sees its surprise at new evidence; anticipated
        # conflict brings in the router's next-choice module (an alternative view). Compare j2_surprise (error target)
        "judge3": {name: f"{sweep.brain_group('ec32k')['b32k_sparseB32c2_t100m']} --judge 2 --judge_target dissonance "
                         f"--gate_lo 0.15 --gate_hi 0.35 {extra}"
                   for name, extra in (("j3_dissonance", "--judge_surprise 1"), ("j3_dissonance_s7", "--judge_surprise 1 --seed 7"),
                                       ("j3_dissonance_nosurprise", ""))},
        # the decisive judge test at 1B tokens (100M runs have ~0.005 noise): dissonance judge vs the same bank, 2 seeds
        # each, both on today's code
        "judge1b": {name: f"{sweep.brain_group('ec32k')['b32k_sparseB32c2_t1b']} {extra}"
                    for name, extra in (("j1b_dissonance", "--judge 2 --judge_target dissonance --judge_surprise 1 --gate_lo 0.15 --gate_hi 0.35"),
                                        ("j1b_dissonance_s7", "--judge 2 --judge_target dissonance --judge_surprise 1 --gate_lo 0.15 --gate_hi 0.35 --seed 7"),
                                        ("j1b_bank", ""), ("j1b_bank_s7", "--seed 7"))},
        "diag": {f"{k}_{tag}": f"{v} {extra}" for k, v in sweep.scale_group("ec32k").items() if k == "s_med_brain"
                 for tag, extra in (("nofp8", "--fp8 0"), ("sigmoid", "--sigmoid_router 1"))},
        "h200base": pick(["b32k_plain8_t100m", "b32k_plain8_t100m_s7"]),
        "combo_med": {k: v for k, v in sweep.combo_group("ec32k").items() if k.startswith("c_med")},
        "combo_small": {k: v for k, v in sweep.combo_group("ec32k").items() if not k.startswith("c_med")},
        "critic1b": pick([f"b32k_{m}_t1b_s{s}" for s in (7, 13) for m in ("bank3", "critic3")]),
        "bigbank": pick(["b32k_sparseB16_t100m", "b32k_sparseB16_t100m_s7", "b32k_sparseB32_t100m",
                         "b32k_sparseB32_t100m_s7", "b32k_sparseB8r_t100m"]),
        "bigbank2": pick(["b32k_sparseB32c2_t100m", "b32k_sparseB32c2_t100m_s7", "b32k_sparseB64c2_t100m",
                          "b32k_sparseB64c2_t100m_s7", "b32k_sparseB32c2_t1b", "b32k_plain8_t1b"]),
        # a few minutes: two tiny runs side by side, to check setup, parallel training and resuming
        "smoke": {"smoke_plain": "--vocab ec32k --preset small --n_layer 4 --fp8 1 --n_kv_head 2 --tokens 4e6",
                  "smoke_memory": "--vocab ec32k --preset small --n_layer 4 --fp8 1 --n_kv_head 2 --seq_len 4096 "
                                  "--attn_mode concept --fast_heads 2 --tokens 4e6"},
    }


def run_dir(name):
    return os.path.join(RUNS, f"sweep_{name}")


def run_state(name):
    """diverged / done / partial (a checkpoint to resume) / new. A non-finite loss or eval anywhere in the
    output makes a run DIVERGED, whatever else it printed: its results and checkpoint can't be trusted."""
    out = os.path.join(run_dir(name), "stdout.txt")
    text = open(out, errors="replace").read() if os.path.exists(out) else ""
    if re.search(r"loss nan|bpb nan|non-finite|diverged", text, re.I):
        return "DIVERGED"
    if re.search(r"^done:|^stopped at step", text, re.M):
        return "done"
    return "partial" if os.path.exists(os.path.join(run_dir(name), "ckpt.pt")) else "new"


# ---- setup / data / test -------------------------------------------------------

def cmd_setup(a):
    ok = True
    missing = []
    for mod in ("torch", "numpy", "pyarrow", "huggingface_hub", "triton", "tiktoken"):
        try:
            m = importlib.import_module(mod)
            say(f"  {mod} {getattr(m, '__version__', '')}")
        except ImportError:
            missing.append(mod.replace("_", "-"))
    if missing:
        ngc = os.path.exists("/etc/pip/constraint.txt")  # NVIDIA's NGC containers pin their packages
        say(f"  MISSING {', '.join(missing)}: pip install {' '.join(missing)}"
            + ("  (NGC container: install just these, never torch/numpy)" if ngc else ""))
        ok = False
    if not ok:
        return 1
    import torch
    if not torch.cuda.is_available():
        say("  no CUDA GPU visible")
        return 1
    free, total = torch.cuda.mem_get_info()
    say(f"  GPU {torch.cuda.get_device_name()} | {total / 2**30:.0f} GiB ({free / 2**30:.0f} free) | "
        f"compute capability {'.'.join(map(str, torch.cuda.get_device_capability()))}")
    if torch.cuda.get_device_capability() < (8, 9):
        say("  note: FP8 needs compute capability 8.9+ (Ada/Hopper/Blackwell); the runs use --fp8 1")
    digest = hashlib.sha256(open(VOCAB_FILE, "rb").read()).hexdigest()
    if digest != VOCAB_SHA256:
        say(f"  vocabulary hash mismatch: {digest}")
        return 1
    say("  vocabulary ec200k_base: hash ok")
    if not shutil.which("cmake"):
        say("  MISSING cmake (apt install cmake build-essential)")
        return 1
    build = os.path.join(ROOT, "build")
    r = sh(["cmake", "-S", ROOT, "-B", build, "-DCMAKE_BUILD_TYPE=Release"], stdout=subprocess.DEVNULL)
    r = r.returncode or sh(["cmake", "--build", build, "-j", str(os.cpu_count())], stdout=subprocess.DEVNULL).returncode
    lib = os.path.join(build, "libfasttok.so")
    if r or not os.path.exists(lib):
        say("  fasttok build failed")
        return 1
    say(f"  built {lib}")
    return 0


def cmd_data(a):
    missing = [s for s in SOURCES if not os.path.exists(os.path.join(DATA, s, "val_000000.bin"))
               or not any(f.startswith("train_") for f in os.listdir(os.path.join(DATA, s)))]
    if missing and not a.download:
        say(f"  no token shards for {', '.join(missing)} in {DATA}.")
        say("  Either copy model/data/ec32k from the home machine (8.3 GB), or rebuild them here from")
        say("  Hugging Face (FineWeb-Edu, github-code-clean; a few GB download): python run.py data --download")
        return 1
    for s, files in SOURCES.items():
        if s in missing:
            r = sh([PY, os.path.join(MODEL, "prepare_data.py"), "--source", s, "--files", str(files), "--vocab", "ec32k"])
            if r.returncode:
                return r.returncode
    for s in SOURCES:
        shards = [f for f in os.listdir(os.path.join(DATA, s)) if f.startswith("train_")]
        tokens = sum((os.path.getsize(os.path.join(DATA, s, f)) - 1024) // 2 for f in shards)  # uint16 after a 1 KB header
        say(f"  {s}: {len(shards)} train shards, {tokens / 1e9:.2f}B tokens")
    crawl = os.path.join(DATA, "crawl")
    say("  forgetting-test docs: " + ("present" if os.path.exists(os.path.join(crawl, "train.npy")) else
                                     "absent (optional: copy model/data/ec32k/crawl/*.npy)"))
    return 0


def cmd_test(a):
    # fasttok against tiktoken on the bundled vocabulary and the ec32k prefix the models use (no download;
    # tests/unit.cpp also needs OpenAI's o200k/cl100k files, so it's left to tokenizer development)
    jobs = {"tokenizer = tiktoken": [PY, os.path.join(ROOT, "tests", "test_vs_tiktoken.py"),
                                     "--lib", os.path.join(ROOT, "build", "libfasttok.so"),
                                     "--vocab-dir", os.path.join(ROOT, "vocab"),
                                     "--encodings", "ec200k_base,ec200k_base:32763", "--fuzz", "20000",
                                     "--skip-codepoints"]}
    jobs["fused loss"] = [PY, os.path.join(MODEL, "test_fused_ce.py")]
    for t in TEST_SUITES:
        jobs[t] = [PY, os.path.join(MODEL, "tests", f"{t}.py")]
    start = lambda cmd: subprocess.Popen(cmd, cwd=MODEL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # the fused-loss benchmark fills the GPU on purpose, so it runs alone first; the rest run together
    procs = {"fused loss": start(jobs.pop("fused loss"))}
    procs["fused loss"].wait()
    procs.update({name: start(cmd) for name, cmd in jobs.items()})
    jobs = procs
    failed = 0
    for name, p in procs.items():
        out = p.communicate()[0]
        passes, fails, skips = (len(re.findall(rf"^{w}\b", out, re.M)) for w in ("PASS", "FAIL", "SKIP"))
        bad = p.returncode != 0 or fails > 0 or "Traceback" in out
        failed += bad
        if passes or fails:
            detail = f"{passes} passed, {fails} failed" + (f", {skips} skipped" if skips else "")
        else:  # suites with their own report format: show their last line
            detail = (out.strip().splitlines() or ["no output"])[-1][:100]
        say(f"  {'FAIL' if bad else 'ok  '} {name}: {detail}")
        if bad:
            say("\n".join("      " + l for l in out.strip().splitlines()[-15:]))
    say(f"\n{len(jobs) - failed} of {len(jobs)} suites passed")
    return 1 if failed else 0


# ---- training ------------------------------------------------------------------

def cmd_train(a):
    import torch
    allq = queues()
    names = []
    for q in a.queue.split(","):
        if q not in allq:
            raise SystemExit(f"unknown queue {q!r}; choose from {', '.join(allq)}")
        names += [n for n in allq[q] if n not in names]
    flags = {n: f for q in allq.values() for n, f in q.items()}
    for n in names:
        if "--tokens" not in flags[n]:
            raise SystemExit(f"{n} has no --tokens budget: time-budgeted runs can't share a GPU fairly")
    total = torch.cuda.mem_get_info()[1] / 2**30
    vram = a.vram_gb or round(total * 0.92 / a.parallel, 1)
    todo = [n for n in names if run_state(n) not in ("done", "DIVERGED")]  # a diverged run is not retried blindly
    say(f"{len(names)} runs, {len(names) - len(todo)} already done; {a.parallel} at a time, {vram} GiB VRAM each "
        f"(GPU: {total:.0f} GiB)")
    os.makedirs(RUNS, exist_ok=True)
    running, results = {}, {}

    def launch(name):
        d = run_dir(name)
        os.makedirs(d, exist_ok=True)
        common = ["--eval_every", "100", "--save_every", str(a.save_every), "--log_every", "20", "--vram_gb", str(vram)]
        if run_state(name) == "partial":
            cmd = [PY, "-u", os.path.join(MODEL, "train.py"), "--resume", os.path.join(d, "ckpt.pt"), *common]
            mode = "a"
        else:
            cmd = [PY, "-u", os.path.join(MODEL, "train.py"), *flags[name].split(), "--run", f"sweep_{name}", *common]
            mode = "w"
        out = open(os.path.join(d, "stdout.txt"), mode)
        say(f"[{time.strftime('%H:%M')}] start {name}" + (" (resuming)" if mode == "a" else ""))
        running[name] = (subprocess.Popen(cmd, cwd=MODEL, stdout=out, stderr=subprocess.STDOUT), out, time.time())

    def stop_all(*_):
        say("\nstopping: runs keep their last checkpoint; run the same command to resume")
        for p, out, _ in running.values():
            p.terminate()
        for p, out, _ in running.values():
            try:
                p.wait(30)
            except subprocess.TimeoutExpired:
                p.kill()
            out.close()
        sys.exit(130)

    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)
    queue = list(todo)
    while queue or running:
        while queue and len(running) < a.parallel:
            launch(queue.pop(0))
            if queue and len(running) < a.parallel:
                time.sleep(a.stagger)  # let each run compile and size its micro-batch before the next starts
        time.sleep(10)
        for name, (p, out, t0) in list(running.items()):
            if p.poll() is not None:
                out.close()
                state = run_state(name)
                results[name] = state if p.returncode == 0 else f"failed (exit {p.returncode})"
                say(f"[{time.strftime('%H:%M')}] {name}: {results[name]} after {(time.time() - t0) / 60:.0f} min")
                del running[name]
    failed = [n for n, r in results.items() if r.startswith("failed")]
    say(f"\nfinished: {len(results) - len(failed)} ok, {len(failed)} failed" + (f" ({', '.join(failed)})" if failed else ""))
    return 1 if failed else 0


def cmd_status(a):
    for q, runs in queues().items():
        say(f"== {q}")
        for n in runs:
            out = os.path.join(run_dir(n), "stdout.txt")
            last = ""
            if os.path.exists(out):
                steps = re.findall(r"^step (\d+)/(\d+) .*?\| ([\d,]+) tok/s", open(out, errors="replace").read(), re.M)
                if steps:
                    s, t, tps = steps[-1]
                    last = f"step {s}/{t} ({int(s) / int(t):.0%}), {tps} tok/s"
            say(f"  {run_state(n):8s} {n:28s} {last}")
    return 0


def cmd_eval(a):
    q = queues()
    mem = [os.path.join(run_dir(n), "ckpt.pt") for n in list(q["memory"]) + list(q["memory2"]) + list(q["combo_small"])
           + list(q["combo_med"]) if run_state(n) == "done"]
    if mem:
        with open(os.path.join(RUNS, "long_context.txt"), "w") as f:
            r = subprocess.run([PY, os.path.join(MODEL, "long_context.py"), *mem, "--length", "8192", "--docs", "40"],
                               cwd=MODEL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            f.write(r.stdout)
        say("\n".join(l for l in r.stdout.splitlines() if l.startswith("|") or "documents" in l))
    else:
        say("no finished memory runs to score yet")
    r = subprocess.run([PY, os.path.join(MODEL, "sweep.py"), "--summary"], cwd=MODEL, stdout=subprocess.PIPE, text=True)
    open(os.path.join(RUNS, "sweep_summary.txt"), "w").write(r.stdout)
    say(f"\nsweep summary: {os.path.join(RUNS, 'sweep_summary.txt')}")
    return 0


def cmd_export(a):
    """Finished runs -> model.safetensors + model.json (weights only, safe to load anywhere)."""
    sys.path.insert(0, MODEL)
    import safe_checkpoint
    n = 0
    for runs in queues().values():
        for name in runs:
            ck = os.path.join(run_dir(name), "ckpt.pt")
            done = os.path.join(run_dir(name), "model.safetensors")
            if not os.path.exists(ck):  # checkpoint removed after its export was brought home (disk space)
                continue
            if run_state(name) == "done" and (not os.path.exists(done) or os.path.getmtime(done) < os.path.getmtime(ck)):
                safe_checkpoint.export(ck)
                n += 1
                say(f"  exported {name}")
    say(f"{n} runs exported. To bring results home without any pickle files:")
    say("  rsync -av --include='*/' --include='*.safetensors' --include='*.json' --include='*.jsonl' "
        "--include='*.txt' --exclude='*' <host>:wAI/model/runs/ ./runs/")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["all", "setup", "data", "test", "train", "status", "eval", "export"])
    ap.add_argument("--download", action="store_true", help="data: rebuild missing shards from Hugging Face")
    ap.add_argument("--queue", default=DEFAULT_QUEUE, help=f"train: comma-separated queues (default {DEFAULT_QUEUE})")
    ap.add_argument("--parallel", type=int, default=4, help="train: runs at once on the GPU")
    ap.add_argument("--vram_gb", type=float, default=0, help="train: VRAM per run (0 = 92%% of the card / --parallel)")
    ap.add_argument("--save_every", type=int, default=200, help="train: checkpoint interval in steps (for resuming)")
    ap.add_argument("--stagger", type=int, default=90, help="train: seconds between starting runs")
    a = ap.parse_args()
    steps = {"setup": cmd_setup, "data": cmd_data, "test": cmd_test, "train": cmd_train, "status": cmd_status,
             "eval": cmd_eval, "export": cmd_export}
    order = ["setup", "data", "test", "train", "eval", "export"] if a.command == "all" else [a.command]
    for name in order:
        say(f"\n=== {name} ===")
        rc = steps[name](a)
        if rc:
            say(f"{name} failed; stopping")
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
