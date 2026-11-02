# Handoff: brain-model research (state as of Oct 31, 2026, morning Central)

Read this first, then `model/H200_PLAN.md` (the full results log, every run and decision in order) and
`model/LOCAL_PLAN.md` (the local RTX 4070 phase, from Oct 31). The owner's
memory files (copied to the new machine's Claude memory folder) carry standing rules: security, time format
(Central, 12-hour), divergence guards, rental details.

## The project in one paragraph
A small GPT (modded-nanogpt lineage: RoPE, QK-norm, ReLU^2, U-net skips, Muon, FP8) as a strong baseline, and
"brain" parts tested against it at equal tokens: a routed module bank (shared module + top-k routed modules),
concept memory, fast weights, a judge/critic, and neurogenesis + sleep for continued learning. Owner's goal: find
mechanisms that are more efficient and more brain-like, building on known work where it fits but not reinventing
it. Budget is small: no large paid runs (the owner declined ~$1,500 scale-up for now).

## What is established (with numbers in H200_PLAN.md)
- **Routed module bank** beats plain: medium (230M, 1.5B tokens) 0.8522 vs 0.8587 mix bpb after the gate fix
  (wAI 87400b6: renormalised gates were 0/0 = NaN when the revive bias picked underflowed modules).
- **Neurogenesis + sleep** is the strongest result. Frozen base; each stage grows 2 copy-initialised, zero-output
  gated modules; wake on new material + 25% replay; sleep replaying mostly old material.
  - 3 seeds: forgetting per unit learned 0.005 (90% old in sleep) vs 0.134 for full fine-tuning (27x).
  - 10 specialties in sequence: keeps 100% of what it learns; full fine-tuning keeps 37% and damages general
    code; ~2x the learning of adapters at the same parameter budget.
  - Weakness: prose specialties (philosophy, law, medical, logic) learn little. Spreading modules over 3 depths
    per stage doubled prose learning (A_prose_multidepth); more modules at one depth barely helped.
- **Judge** (owner's design: ONE judging module beside the others). Dissonance target (first impression vs
  considered view) is predictable (corr ~0.65) but using it to allocate +/-1 module gave no gain at 100M or 1B.
  Its remaining candidate roles: sleep priority, placement of new modules, when to consult memory.
- **Prose diagnostics:** word choice is ~12% of loss everywhere; register is already represented; tokenization
  explains ~15% of the prose gap. The prose difficulty is mostly understanding (meaning/knowledge).
- **Negative:** concept memory loses to full attention at 4k and doesn't scale to 16k; fast weights hurt at 16k;
  generative replay from the model's own free samples degrades everything; episodic kNN memory over general text
  hurts slightly (the memory repeats the model's own errors).

## Added Oct 30 midday (details in H200_PLAN.md)
- Episodic memory of NEW specialty text helps instantly, no training (Lua +8.3%, Rust +5.2%, Fortran +3.6%);
  memory of general text does not. Prose barely benefits from memory either.
- Prose: growing at 3 depths per stage = 3x longer training (0.0058 vs 0.0057); placement by dissonance puts prose
  at depth 6 and lifts prose 2.6x. E runs combine the two.
- Dissonance-prioritised sleep needs large pools (tiny pools memorised); cued recall still harmful.

## Added Oct 30 afternoon (details in H200_PLAN.md)
- Best neurogenesis setup: multi-depth growth at the most conflicted depths (0.0273 kept, prose 2.9x, no added
  forgetting). Dissonance-prioritised sleep: no benefit (closed). Cued recall: harmful (closed).
- The bank's gain is worth ~10% more data, not 2x (plain at 2B tokens 0.9010 vs bank at 1B ~0.930). 128 modules
  are worse than 32. Treat the bank as a small, real improvement, not the big win.
- Resume bug fixed in train.py: the real medium brain weights are d_fix's export (now also in s_med_brain_fix,
  stale copy labelled). Anything evaluated on the old s_med_brain_fix export before 3 PM Oct 30 is invalid.

## Added Oct 30 evening
- Medium size (230M): neurogenesis with 2 modules per depth keeps less than full fine-tuning (0.012 vs 0.016),
  but with 8 per depth it nearly matches (0.0150) with ~no forgetting (code +0.0025 vs +0.0329). Added capacity
  must scale with the model. Plain at 3B tokens 0.8865 (bank at 1B = ~1.1B tokens of plain).
- Rental work ended 8:12 PM Central Oct 30; keeper final pull 10 PM Central. All results: H200_PLAN.md.

## Rental: finished
All H200 work finished 8:12 PM Central Oct 30; the final pull (9:10 PM) brought everything home to
D:\Claude\worksAI_runs\h200 (/mnt/d/Claude/worksAI_runs/h200 on Linux). Every result is in H200_PLAN.md, including
A, B, C, D, D2, E, the data-worth runs and the specialty-memory test. #3 (sleep during pretraining) did not run.

## Local phase (Ubuntu 26.04, RTX 4070 12 GB; details in model/LOCAL_PLAN.md)
- Repo at ~/claude/wAI, .venv on Python 3.14 (torch 2.15.0+cu132); `run.py test` passes (11 suites).
- FP8 fix (31cf051): the torch 2.15 workaround made eager fp8 transpose copies (~8% of step time); transposed
  operands are now cast from the bf16 transpose. Plain small 71.3k -> 77.0k tok/s, 1 GiB less memory.
- Card: persistence mode on (permanent); a 216 W power limit and the performance CPU governor gave no gain.
- The owner runs an openbox desktop on the 4070: tell them before GPU-memory-heavy jobs so they can close it.
- continual harness: training steps compiled (-33% wall time); episodic memory beside neurogenesis
  (model/episodic.py, `--memory_tokens`); queue in model/cls_queue.sh (tmux `cls`).

## Next steps, in order
1. Complementary-learning loop (step 2, running locally): each specialty enters episodic memory as it arrives;
   after wake + sleep the memory is re-keyed and releases what the modules now predict (p >= 0.5). Questions: does
   memory add on top of consolidated modules; does release keep the gain; which gate (fixed, entropy, dissonance)
   decides best when to consult memory. Then: sleep that distils memory's predictions into the modules.
   M1 (neurogenesis + memory) done Oct 31: memory adds 1.4-3.0% on top of the modules for code, ~0.2% for prose;
   release at p >= 0.5 loses Lua's gain; a fixed blend beats the entropy and dissonance gates.
   M2 (multi-depth at the most conflicted depths + memory) done Nov 1: better on every count (modules 0.0319 vs
   0.0277 kept, prose 3x, memory on top 1.00% vs 0.87% and 0.82% vs 0.46% after release); still a fixed blend
   best. Numbers in LOCAL_PLAN.md.
   Topics overlap (philosophy's held-out text recurs in logic's training text, 18% of windows): expected and
   harmless to the model, so no deduplication; only read memory's philosophy gain after logic arrives (15%) as
   verbatim recall, not generalisation.
2. Prose: answered on the H200 (multi-depth at the most conflicted depths, prose 2.9x; 3x longer stages = depth
   spread). Revisit only if the memory loop changes the picture.
3. #3 sleep during pretraining (fixed data, several passes, dissonance-prioritised replay).

## Where things are
- Code: GitHub callstromcodeworks/wAI (this repo; the Linux version). Older Windows tree: worksAI (agent, phone
  cluster, home server; contains key files that must not be read or copied casually).
- Key tools: `run.py` (setup, data, test, train --queue, status, eval, export), `model/forgetting.py` (continual
  learning harness: `continual` with --domains, --multi_depth, --placement, --sleep_priority, --replay_source),
  `model/build_domains.py` (the 10-specialty benchmark), `model/memory_lm.py`, `model/register_diag.py`,
  `model/register_probe.py`, `model/tokenization_diag.py`, `model/tests/*_checks.py`, `model/episodic.py`
  (episodic memory for `continual --memory_tokens`), `model/cls_queue.sh` (the memory-loop runs).
- Local data: token shards ec32k/ec64k and the raw parquet (copied from worksAI/model/data on C:); the 10-specialty
  domains are rebuilt from the raw files by build_domains.py (offline, deterministic). Built in one pass they match
  the H200's only for Haskell, medical and Lua (the H200 added philosophy and logic later with --only); see
  LOCAL_PLAN.md for the H200-identical build.
  The base model for the continual runs is the H200 export runs/sweep_b32k_sparseB32c2_t1b (safetensors).
- Rental: vast.ai H200, SSH through the proxy only (`ssh -i ~/.ssh/vastai_ed25519 -o IdentitiesOnly=yes
  -o ForwardAgent=no -p 20275 root@ssh7.vast.ai`), code at /workspace/wAI. Never -A/-R/-X; never bring back or
  load a pickled ckpt.pt; only safetensors/json/text come home.

## On the Linux machine: the Windows drives stay attached
C: and D: remain readable from Ubuntu (NTFS, e.g. mounted at /mnt/c and /mnt/d; set them in /etc/fstab with the
ntfs3 driver). Windows paths map over: `C:\Users\Ryan\Claude\...` -> `/mnt/c/Users/Ryan/Claude/...`,
`D:\Claude\worksAI_runs\h200` -> `/mnt/d/Claude/worksAI_runs/h200`.
- Turn off Windows Fast Startup (and don't hibernate) before booting Ubuntu, or the NTFS drives mount read-only.
  If BitLocker is on, those drives need unlocking first.
- NTFS ignores Linux permissions: copy `~/.ssh/vastai_ed25519` to the Linux `~/.ssh/` and `chmod 600` it there
  (ssh refuses a key it sees as world-readable).
- Results already pulled (safetensors/json) can be read in place. For training, copy the token shards
  (model/data/ec32k, ec64k, ~8 GB each) to the Linux drive: ext4 on the NVMe is faster than NTFS for the data
  loader, and keep new runs on Linux too.
- Claude memory: copy `/mnt/c/Users/Ryan/.claude/projects/C--Users-Ryan-Claude/memory/` into the new project's
  memory folder; its notes name Windows paths, which map as above.
- Key files in worksAI (`ai.lan.key`, `ai.local.key`, `cluster/secrets/`) are visible through the mount too:
  the rule not to read them still applies.

## Lessons to keep
- Every training loop needs a non-finite guard, and status checks must read loss values, not just progress.
- Test any new logging or option end to end (a short real run) before launching a queue.
- Small uncompiled jobs sharing the GPU with a compiled run slow it ~6x; run them separately or locally.
- 100M-token runs have ~0.005 bpb run-to-run noise: use them as screens, confirm at 1B with 2+ seeds.
- Ask before starting GPU runs outside an explicit autonomous mandate.
- The H200 continual runs used `--lr_adam 0.006` (the default is 0.002): copy settings from cls_queue.sh or the
  H200_PLAN commands, not from defaults.
- Never kill a tmux queue's shell to "pause" it: the pane closes and SIGHUP kills the running job too.
