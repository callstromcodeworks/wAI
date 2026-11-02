# Best so far (Nov 2, 2026)

The tree holds many variants: switches in `train.py` and `forgetting.py`, and dozens of runs in H200_PLAN.md and
LOCAL_PLAN.md. Most of them lost. This page lists the winners with their numbers, the variants that lost, and which
files to keep. Full history: H200_PLAN.md (Oct 29-30) and LOCAL_PLAN.md (from Oct 31).

**Reading the numbers**
- bpb is bits per byte of held-out text; lower is better.
- "mix" means 60% English (FineWeb-Edu) and 40% code.
- "kept" is the mean bpb by which a specialty improved, measured after all later stages have trained.
- Noise: about 0.005 bpb at 100M tokens, 0.003-0.004 between seeds at 1B, and about 0.005 per specialty in
  continual runs (C2 against M2). Smaller differences are ties.

## The current best system

There is one lineage, built in four layers. Each layer won its own comparison (see the sections below).

| layer | what | run | result |
|---|---|---|---|
| 1. base model | **BrainGPT small**: 8 blocks x 768, GQA (2 KV heads), value residual, U-net skips, ec32k vocabulary, FP8 training. A module bank after block 3: 1 shared module + top-2 of 32 routed modules, each attending over its own tokens, capacity 2.0. 136M parameters, 1B tokens. | `runs/sweep_b32k_sparseB32c2_t1b` | mix **0.9296** (English 1.0751, code 0.7112). Plain 8 blocks at the same tokens: 0.9341 |
| 2. learning new material | **Multi-depth neurogenesis at the most conflicted depths.** Each stage grows 2 modules at each of the 3 depths where the new material most disagrees with the model's first impression. They are copied from the bank; the base and earlier stages stay frozen. Each stage: 400 steps awake on the new material with 25% replay, then 200 steps of sleep with 90% old material. | `M2_mddis_mem` (settings: `X` in cls_queue.sh, plus `--multi_depth 1 --placement dissonance`) | 10 specialties: **0.0319** bpb kept per specialty (100% of what was learned); prose (medical, law, logic) 0.0065. English -0.0003, code +0.0012 (no forgetting) |
| 3. memory on top | **Episodic memory** reads each specialty as it arrives (500k tokens) and is blended with a fixed weight. After each stage it releases the episodes the model now predicts at p >= 0.5. | `M2_mddis_mem` (`--memory_tokens 5e5 --memory_prune 0.5`) | **+1.00%** bpb on top of the modules after all 10 stages (Rust 4.0%, Lua 3.2%, Haskell 1.3%, prose 0-0.2%); 0.82% after release |
| 4. answering | Two more stages: **instruct** (Dolly-15k), then **chat** (oasst2 English), each in its own modules. | `C2_spec_instruct_chat`; its saved rerun `C2s_spec_instruct_chat` (modules on disk, within ~0.005 of C2) | instruct 1.1254 -> **1.0419**, chat 0.9249 -> **0.8709** (C2s: 1.0414, 0.8707); the 10 specialties moved by at most 0.0004 |

Layer 3 runs beside the model as a separate component. C2 and C2s were run without it. **The best model you can
load and sample is C2s** (`forgetting.load_grown("runs/forgetting/C2s_spec_instruct_chat")`, or `samples.py` on
that directory); the 3D page shows it.

**Size and cost of the full stack (C2):**
- Parameters: the 136M base plus 12 stages x 6 modules x 1.83M = 132M grown, so **268M** in all.
- Compute: grown modules run on every token and are gated afterwards. Each stage adds about 14% to the work per
  token, so after 12 stages the work is about 2.7x the base's (estimated, not measured).
- Fix (code change, not built): route the grown modules sparsely, as the bank's own modules are.
- Training time: C2 took 2 h on the 4070. M2, with memory scoring, took 4.8 h.

**Talking quality** (`samples.py`; C1s = base + instruct + chat, C2s = the same plus the 10 specialties):
- The base model never answers.
- The instruct modules alone answer short tasks.
- Assistant style ("Sure! Here's ...", numbered tips) appears only with both groups.
- **Content is wrong, with or without the specialty modules.** On one question per specialty, no variant of C2s or
  C1s answers correctly, and none writes code.
- The specialty modules alone write their specialty's kind of text (β-cells for diabetes, NASA missions for Mars)
  but don't answer. On short chat questions they change little: for 6 of 12 the reply opens exactly as the base
  model's does.
- At 136M parameters the model learns the form, and the specialties add style and terms, not knowledge
  (LOCAL_PLAN.md "C2s").

## Best per question

### Which base model (pretraining)

| model | params | tokens | mix bpb | English | code |
|---|---:|---:|---:|---:|---:|
| medium plain, **ec64k** vocabulary (`s_med_plain_ec64k`) | 243M | 1.5B | **0.8489** | 0.9906 | 0.6363 |
| medium + module bank, 32 routed (`s_med_brain_fix`) | 275M | 1.5B | 0.8522 | 0.9966 | 0.6358 |
| medium plain (`s_med_plain`) | 210M | 1.5B | 0.8587 | 1.0023 | 0.6434 |
| 12-block plain, home Windows run (`final_ec32k_gqa`) | 101M | 3.39B | 0.8642 | 1.011 | 0.645 |
| small plain 8 blocks (`w_plain8_t3b`, `_t2b`) | 75.5M | 3B / 2B | 0.8865 / 0.9010 | 1.0339 (3B) | 0.6653 (3B) |
| small + earned-say critic (`b32k_critic3_t1b_s7`, `_s13`) | 92.0M | 1B | 0.9266 (2 seeds) | | |
| small dense bank, 1 shared + top-2 of 7 (`b32k_bank3_t1b_s7`, `_s13`) | 90.2M | 1B | 0.9274 (2 seeds) | | |
| **small sparse bank of 32 (the base above)** (`b32k_sparseB32c2_t1b`, `j1b_bank`, `_s7`) | 136M | 1B | 0.9301 (3 runs: 0.9296 / 0.9318 / 0.9288) | 1.0751 | 0.7112 |
| small sparse bank of 128 (`w_sparseB128c2_t1b`) | 312M | 1B | 0.9333 | 1.0776 | 0.7169 |
| small plain 8 blocks (`b32k_plain8_t1b`) | 75.5M | 1B | 0.9341 | 1.0792 | 0.7164 |

**Verdict**
- **Lowest bpb overall:** the medium models, 16 blocks x 1024 trained on 1.5B tokens.
  - ec64k is the best at 0.8489, but each of its tokens covers about 6% more text, so at equal tokens it saw more
    data.
  - The bank is the best brain model: 0.8% better than plain at equal tokens.
  - A medium bank with ec64k has not been trained.
- **At the small size** the 8- and 32-module banks beat plain by 0.004-0.008. That gain is worth about 10% more data, not 2x
  (plain at 2B is 0.9010).
  - Dense bank, sparse bank and critic tie within seed noise.
  - 64 or 128 routed modules don't help.
- **Why the small 32-bank is the base anyway:** every continual result (H200 and local) is on this model.
  - At medium size, continual learning needs 4x the added modules to stay competitive: D2's 8 per depth keeps
    0.0150, against 0.0157 for full fine-tuning.
- The 12-block Windows run was scored at home on Windows. Earlier home runs scored about 0.007 worse than the same
  settings on the H200, so compare it loosely.
- Deployment, measured on the Windows 12-block plain model (README.md):
  - int8 weights cost +0.09%; int8 KV cache +0.00%.
  - int4 weights cost +6.6%, so avoid them.
  - A KV window of 128 plus the neural cache keeps full accuracy.
  - None of this has been re-measured on BrainGPT.

### How to learn new material without forgetting (10-specialty benchmark)

There are two copies of the benchmark. Seven of the 10 specialties have different documents in the H200 build and
the local rebuild, so compare within a block only. "learned" is each specialty's gain right after its own stage.

| method | learned | kept at end | share kept | prose kept (medical, law, logic) | English | code |
|---|---:|---:|---:|---:|---:|---:|
| *H200, small base* | | | | | | |
| full fine-tuning + replay (`DB_fullft_old90`) | 0.0568 | 0.0213 | 37% | 0.0009 | +0.0121 | +0.0447 |
| adapters, same parameter budget (`DB_adapter_old90`) | 0.0142 | 0.0130 | 92% | 0.0022 | -0.0002 | +0.0010 |
| neurogenesis, 2 per stage at depths 1, 3, 5 in turn (`DB_neuro_old90`) | 0.0252 | 0.0252 | 100% | 0.0018 | +0.0000 | +0.0014 |
| same, depth by dissonance (`B_place_dissonance`) | 0.0234 | 0.0234 | 100% | 0.0054 | -0.0004 | +0.0009 |
| **multi-depth at the most conflicted depths (`E_all_multidepth_dissonance`)** | 0.0271 | **0.0273** | 101% | 0.0050 | -0.0003 | +0.0011 |
| *local, small base* | | | | | | |
| M1, depths 1, 3, 5 in turn | 0.0279 | 0.0277 | 99% | 0.0019 | -0.0001 | +0.0008 |
| **M2, multi-depth by dissonance** | 0.0318 | **0.0319** | 100% | **0.0065** | -0.0003 | +0.0012 |
| M3 = M2 + sleep that distils memory (shelved) | 0.0316 | 0.0311 | 99% | 0.0064 | -0.0000 | +0.0016 |
| C2's first 10 stages (M2's settings again) | 0.0313 | 0.0314 | 101% | 0.0067 | -0.0003 | +0.0009 |
| *H200, medium base (275M)* | | | | | | |
| full fine-tuning + replay | 0.0389 | 0.0157 | 40% | 0.0015 | +0.0085 | +0.0329 |
| multi-depth by dissonance, 2 / 4 / 8 modules per depth | 0.0123 / 0.0137 / 0.0153 | 0.0121 / 0.0136 / **0.0150** | 98-99% | 0.0025-0.0027 | -0.0001 | +0.0020 / +0.0020 / +0.0025 |

**Verdict: multi-depth growth at the most conflicted depths, with a 90%-old sleep (E on the H200, M2 locally).**
- It keeps everything it learns and ends with more specialty knowledge than full fine-tuning.
- It learns about 3x more prose than single-depth growth.
- It doesn't forget. Full fine-tuning learns 2x more at each stage, then loses 63% of it and damages general code
  30-40x more.
- Caveat: it grows 6 modules per stage, against 2 for the single-depth rows.
- Sleep is the key part. The share of old material in sleep is a dial: at 90% old, forgetting per unit learned is
  27x below full fine-tuning (3 seeds, crawled docs).
- At medium size the added modules must scale with the model.

### Memory (on top of M2's modules, final, mean of 9 specialties)

| how memory is consulted | gain |
|---|---:|
| each text's best blend, chosen on its first half | **1.00%** (the figure in the layer table) |
| **fixed blend** | **0.93%** (best on 8 of 9 in M1) |
| dissonance gate | 0.63% |
| entropy gate | 0.50% |

- **Before any training:** memory of a new specialty's text helps at once (Lua +8.3%, Rust +5.2%, Fortran +3.6%).
- **General text:** memory of general text hurts slightly; it repeats the model's own errors.
- **Prose:** memory barely helps prose (0.2-0.8%).
- **Distilling memory in sleep (M3):** no measurable effect; shelved.

### Instruction and conversation as modules

| run | instruct | chat | specialties |
|---|---:|---:|---|
| base model | 1.1254 | 0.9249 | |
| C1: instruct, then chat, on the base | **1.0270** | **0.8606** | none grown |
| C2: the 10 specialties, then instruct, then chat | 1.0419 | 0.8709 | 0.0314 kept; moved by at most 0.0004 |

- **No forgetting.** Instruction and conversation training cost the specialties nothing, unlike usual instruction
  tuning, which wears knowledge away.
- **What each group holds.** The instruct modules carry the shared skill of answering (44% of chat's gain); the
  chat modules hold only what is specific to conversation.
- **Cost after specialties.** C2 learned about 15% less than C1 (one run each).
- **Use C2's layout** if the model needs both specialties and answers. C2s is the saved copy.

## Tested and lost (leave these off)

| variant | switch | result |
|---|---|---|
| concept memory (long context) | `--attn_mode concept` | about half of full attention's long-range gain at 4k; 26-28% at 16k; at medium size 3% behind full attention |
| window attention | `--attn_mode window` | worse than full attention at every length |
| Hebbian fast weights | `--fast_heads` | small gain with concepts at 4k, hurts at 16k, nothing on a window |
| spiking MLP | `--mlp_mode spiking` | 1-bit spikes cost 3.5-4.5% and are slower on a GPU |
| depth routing (skip blocks per token) | `--depth_every` | +2.7-3.6% at 100M; not fast enough to pay |
| latent attention (MLA) | `--mla_rank` | +5.2% at 3 KiB/token; MQA + K/V sharing costs +1.3% for the same cache |
| product-key memory in the bank | `--memory_sub` (default 128) | worse; the best base uses `--memory_sub 0` |
| critic, trained jointly | `--critic 1` (default) | hurts (1.426 vs 1.353 with the critic detached, step 300); the best base uses `--critic 0` |
| earned-say critic | `--critic_gate 2` | -0.0013 at 1B over 3 seeds, within noise; not in the best base |
| draft loss | `--draft_weight` (default 0.1) | no effect; the best base uses 0 |
| judge choosing modules per token | `--judge` | no gain at 1B (+0.0015 / +0.0007, 2 seeds) |
| bounded sigmoid router | `--sigmoid_router` | stable only because 87.5% of routed tokens were dropped (bank effectively off) |
| 64 or 128 routed modules | `--n_modules` | no better than 32 |
| full fine-tuning / adapters for new material | `--grow 0` / `--adapter 1` | see the table above |
| placement by gradient size | `--placement need` | worse than spreading over depth |
| copy the modules the new text routes to | `--init copy_active` | no gain |
| grow extra modules, keep the most useful | `--overgrow 2` | no gain at the same final size |
| no sleep | `--consolidate_steps 0` | 25-30% less learning, about 3x the forgetting |
| longer sleep (400 steps) | `--consolidate_steps 400` | no better than 200 |
| sleep on the most conflicted windows | `--sleep_priority dissonance` | no benefit with large pools; small pools are catastrophic |
| replay of the model's own samples | `--replay_source recall` | degrades everything, cued or free |
| sleep that distils memory | `--sleep_distill` | no measurable effect (M3) |

**Defaults are not the best settings.**
- `train.py`'s brain defaults build the old full model (product-key memory, critic and draft loss all on).
- `forgetting.py`'s defaults are the early continual settings (200 + 50 steps, no replay, `--lr_adam 0.002`,
  placement by need, single depth, 50% new material in sleep).
- Copy the flags from the base model's config.json and cls_queue.sh instead.

## Which files to keep

**Run names**
- **Pretraining.**
  - `sweep_b32k_*` is small (8 x 768, ec32k); `_t100m` / `_t1b` mean 100M / 1B tokens, `_s7` / `_s13` other seeds.
  - `sweep_s_med_*` is medium (16 x 1024).
  - `sweep_m32k_*` and `sweep_m16_*` are long-context memory runs, at 4k and 16k context.
  - `sweep_c_*` are combinations, `sweep_j*` the judge, `sweep_d_*` divergence diagnostics, `sweep_w_*` the "what
    is data worth" runs, `smoke_*` tests.
- **Continual, H200** (results in `/mnt/d/Claude/worksAI_runs/h200/forgetting`):
  - A is prose capacity, B dissonance placement and sleep, C cued recall.
  - DB is the 10-specialty baselines, D and D2 the medium size, E multi-depth by dissonance.
  - Single letters (E2 ... Z6) are crawled-docs runs.
- **Continual, local** (`runs/forgetting`): M1-M3 are the memory loop, C1 and C2 instruction and chat, `T_*` smoke
  tests. A name ending in `s` (C1s, C2s) is the same run with `--save`, so its modules can be loaded.

| keep | where |
|---|---|
| best small base (all continual work) | `model/runs/sweep_b32k_sparseB32c2_t1b` (copy on D:) |
| best medium brain model | `/mnt/d/Claude/worksAI_runs/h200/sweep_s_med_brain_fix/model.safetensors` (the real final weights; `model.STALE_step2600` is not) |
| lowest-bpb model | `/mnt/c/Users/Ryan/Claude/worksAI/model/runs/h200/sweep_s_med_plain_ec64k` |
| best saved model (specialties + instruct + chat) | `runs/forgetting/C2s_spec_instruct_chat` (grown.safetensors 529 MB, grown.json); its samples in `samples_specialties.md` there |
| the same without specialties | `runs/forgetting/C1s_instruct_chat` |
| 3D page | `runs/viz/index.html` (C2s; C1s's data in `runs/viz/data_C1s.json`); published as the private artifact "BrainGPT, Inside" |
| continual results | `runs/forgetting/continual_*.json` (local); the H200 folder above |

## Rebuild the best system (from model/)

```bash
# 1. Base model: already trained (H200 export). To retrain it under a new name (never into the
#    existing runs/sweep_b32k_sparseB32c2_t1b, which every continual run loads):
../.venv/bin/python train.py --vocab ec32k --preset small --value_residual 1 --unet 1 --fp8 1 --n_kv_head 2 \
  --arch brain --n_layer 8 --memory_sub 0 --critic 0 --draft_weight 0 --n_shared 1 --module_mode routed_attn \
  --n_modules 33 --revive 0.1 --capacity 2.0 --tokens 1e9 --run b32k_sparseB32c2_t1b_rerun

# 2-4. Specialties, then instruct and chat, keeping the modules (this is C2s, 2 h on the 4070). Already done:
#      runs/forgetting/C2s_spec_instruct_chat. Rerun only under another --save / --name.
#      Add the memory flags ($M in cls_queue.sh) for layer 3 (memory scoring brings it to ~5 h).
B=runs/sweep_b32k_sparseB32c2_t1b
D=haskell,medical,lua,philosophy,law,fortran,logic,music,rust,space
../.venv/bin/python forgetting.py continual $B --domains $D,instruct,chat --per_stage 2 --stage_steps 400 \
  --consolidate_steps 200 --lr_adam 0.006 --init copy --replay 0.25 --sleep_new 0.1 --depths 1,3,5 \
  --multi_depth 1 --placement dissonance --save runs/forgetting/C2s_spec_instruct_chat --name C2s_spec_instruct_chat

# Sample replies: base, specialties only, without chat, everything
../.venv/bin/python samples.py runs/forgetting/C2s_spec_instruct_chat --prompts prompts/specialties.txt \
  --variants base,-instruct+chat,-chat,all --n 2 --out samples_specialties.md

# 3D visualization of the saved model (~1.5 GB of GPU memory for about a minute; not while a run is training)
../.venv/bin/python viz/extract.py          # default --run is C2s -> runs/viz/data.json
../.venv/bin/python viz/build.py            # -> runs/viz/index.html
```
