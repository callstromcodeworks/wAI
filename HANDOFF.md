# Handoff: brain-model research (state as of Oct 30, 2026, afternoon Central)

Read this first, then `model/H200_PLAN.md` (the full results log, every run and decision in order). The owner's
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

## In flight at handoff (on the vast.ai H200; results land in model/runs and D:\Claude\worksAI_runs\h200)
- `abc` tmux: A (prose capacity; A_prose_long still running), B (dissonance-prioritised sleep; placement by
  dissonance), C (cued recall sleep: the model continues real 32-token fragments).
- `worth` tmux: plain small at 2B and 3B tokens (what the bank's gain is worth in data), 128-module bank at 1B.
- `mem` tmux: episodic memory of NEW specialty text (the real test of the hippocampus idea).
- Keeper (local) final pull: 10 PM Central Oct 30. Credit runs out ~2 AM Central Oct 31.
- Not started (deferred for budget): D (neurogenesis at medium size), #3 (sleep during pretraining).

## Next steps, in order
1. Collect the in-flight results into H200_PLAN.md (summary scripts are in the scratch notes; each job writes
   JSON: runs/forgetting/continual_*.json, runs/memory_lm.json, sweep_*/log.jsonl).
2. If specialty memory helps: build the full complementary-learning loop (new text enters episodic memory
   instantly; sleep consolidates it into neurogenesis modules; the judge's dissonance decides when to consult
   memory).
3. Prose: combine multi-depth growth with dissonance placement; longer stages if A_prose_long shows time matters.
4. #3 sleep during pretraining (fixed data, several passes, dissonance-prioritised replay).
5. Local Linux setup: Ubuntu 24.04, `python run.py setup`, then `python run.py test` before any training.

## Where things are
- Code: GitHub callstromcodeworks/wAI (this repo; the Linux version). Older Windows tree: worksAI (agent, phone
  cluster, home server; contains key files that must not be read or copied casually).
- Key tools: `run.py` (setup, data, test, train --queue, status, eval, export), `model/forgetting.py` (continual
  learning harness: `continual` with --domains, --multi_depth, --placement, --sleep_priority, --replay_source),
  `model/build_domains.py` (the 10-specialty benchmark), `model/memory_lm.py`, `model/register_diag.py`,
  `model/register_probe.py`, `model/tokenization_diag.py`, `model/tests/*_checks.py`.
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
