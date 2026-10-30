# H200 research run (Oct 29-30, 2026)

Rented H200 NVL (141 GB), $4.03/h, $100 of credit: ~24.8 h from ~05:00 UTC Oct 29. Final pull of
results 02:00 UTC Oct 30 (h200_keeper.sh in the home repo pulls safetensors + logs every 90 min).

Goal, from the owner: keep testing what we've worked on, then what was dismissed for budget, then make
the AMN ideas work efficiently. Follow what tests as **more efficient and more brain-like**, not standard
paths just because they're proven.

## Phase 1: the queued rounds (started 05:47 UTC)
`memory` (concept memory, fast weights, 8 runs x 300M), `critic1b` (earned-say critic 1B replicates),
`bigbank2` (32/64 routed modules at capacity 2.0; 32-module bank vs plain8 at 1B).

## Phase 2: chosen from phase 1 results
Candidates, in order of fit with the goals:
1. **Combine what works, at a size budget used to rule out**: the best memory + bank + critic variants
   on a bigger model (~300M) for 2-3B tokens. The open question from home: do the brain parts pay off
   once the model is big enough?
2. **Signal / no-signal routing through depth** (event-driven compute): each block processes only the
   tokens its router sends it; the rest pass straight through the residual. Brain regions don't all
   process every input; this saves compute in proportion to what's skipped.
3. **Sparse firing inside the blocks** (k-winners-take-all): only the most active fraction of MLP
   neurons fire per token, as in cortex. Tests whether brain-like activation sparsity costs accuracy.
4. **Longer context**: concept memory trained at 16k, where full attention gets expensive.
5. **Vocabulary at scale**: ec32k vs ec64k on the bigger model (rejected at 100M for weight cost).

## Phase 3: AMN, made efficient
Its ideas that are testable here: local (Hebbian) learning during use (fast weights already do this
per conversation), a coordinator that rewires modules (our router), sparse event-driven units (2, 3
above). If time remains: units that keep a membrane state and fire only above threshold (spiking
dynamics) trained with surrogate gradients, compared against the dense blocks at equal tokens.

## Log
(results and decisions get appended here as runs finish)

**07:20 UTC.** Phase 1 memory round (300M tokens, 4,096-token sequences, ordinary validation):
full 0.9727 | concept 0.9747 / 0.9845 (seed 7) | concept no-expansion 0.9845 | window 0.9913 |
window + fast weights 0.9944 | concept + fast weights ~0.977. Concept memory sits between window and
full; fast weights don't help (yet). Seed noise ~0.01, so the long-document scores decide.

Local (RTX 4070, 100M tokens) depth routing: 8 blocks with routing at 25% / 50%: 1.1902 / 1.1802 vs
plain8 1.1490 (29% / 16% faster, not enough). Equal compute, 12 blocks routed every 2nd at 25%: 1.1604
(104.6k tok/s) vs plain8 1.1490 (107k). MLP-only routing (everyone gets attention): 1.1457 at 90k tok/s,
better per token but slower. Not a win at this scale; keep for the larger-scale test only.

**07:50 UTC. Long-document scores (long_context.py, 40 English + 40 code docs of 8k+ tokens; trained at 4k):**

| run | keys/query | Eng 2-4k | Eng 4-8k | code 2-4k | code 4-8k |
|---|---:|---:|---:|---:|---:|
| full | 8,192 | 1.1621 | 1.1851 | 0.6377 | 0.6350 |
| window 512 | 576 | 1.1961 | 1.2060 | 0.7256 | 0.7125 |
| concept | 960 | 1.1797 | 1.6716 | 0.6815 | 1.1475 |
| concept, no expansion | 704 | 1.1886 | 1.3654 | 0.7123 | 0.8424 |
| concept + fast weights | 960 | 1.1773 | 1.2018 | 0.6766 | 0.7564 |
| window + fast weights | 576 | 1.1992 | 1.2083 | 0.7279 | 0.7158 |

Concept memory recovers ~half of full attention's long-range gain at 8.5x fewer keys; expansion is half
of that; fast weights add a small, consistent gain on top of concepts (both seeds) but nothing on a bare
window. Past the training length concept memory breaks (English 4-8k 1.67): its summaries pool
RoPE-rotated keys, so distant concepts look like nothing seen in training. Fix for phase 2: position-free
concepts (pooled from pre-RoPE keys, scored with pre-RoPE queries); window and expanded detail keep
positions. Then retrain at 16k.

**08:00 UTC. Local spiking MLP** (8 blocks, 100M tokens): 1.1999 vs plain8 1.1490 (+4.4%), 82k vs 107k
tok/s, 21-33% of neurons firing. Diagnosing: spike_window 1 (binary threshold, no integration) and
mlp_ratio 8 (twice the neurons, as brains offset 1-bit spikes with numbers).

**09:15 UTC.** Spiking diagnostics (local, 100M tokens): binary threshold, no integration 1.1891;
integration window 8, mlp_ratio 8 (twice the neurons) 1.1886 at 45k tok/s. 1-bit spikes cost ~3.5-4.5%
whatever the integration or neuron count; on a GPU they're slower, not faster. Next: rate coding
(0-3 spikes per token, 2 bits), the brain's own answer to 1-bit spikes.
H200: b32k_bank3_t1b_s7 done; first bigbank2 run done (13 min); ec64k shards ready.

**09:50 UTC. Phase 1 results.** Earned-say critic at 1B tokens, 3 seeds: no critic 0.9343 / 0.9269 / 0.9278
(mean 0.9297), critic 0.9319 / 0.9279 / 0.9253 (mean 0.9284): -0.0013, 2 of 3 seeds; suggestive, within
seed spread (~0.004). bigbank2 (100M): 32 routed at capacity 2.0: 1.1429 / 1.1405 (mean 1.1417); 64 routed:
1.1418 / 1.1411 (1.1415); capacity 1.25 was 1.1464, so capacity helped, 64 adds nothing over 32. H200
runs score lower than RTX 4070 runs at equal settings (no-critic 1B: 0.927 vs 0.934), so plain8 baselines
get rerun on the H200 (queue h200base). Phase 2 started 09:44 (scale first, 2 slots).

**11:06 UTC. Phase 1 complete (18/18).** At 1B tokens on the H200: plain8 0.9341, 32-routed sparse bank
(capacity 2.0, revival) 0.9296, dense bank3 0.9269 / 0.9278. The module bank's edge grows with training
(level at 100M, ~0.005-0.007 better at 1B): the first brain component that clearly beats plain on the
same hardware. Phase 2 restarted at 4 parallel: scale (medium plain / brain / ec64k), h200base, memory2.

**11:45 UTC. Position-free concepts fix the long-context breakdown** (local combo models, 100M tokens,
trained at 4k, 20 + 20 docs at 8k): English 2-4k -> 4-8k 1.284 -> 1.300 (+1.2%), code 0.889 -> 0.931
(+4.7%), vs phase 1's RoPE concepts +42% / +68% and full attention +2% / 0%. Combination (bank + concepts
+ fast weights) adds up: 1.1096 vs 1.1130 without the bank at 100M tokens. Phase 3 queued on the H200:
combo at medium size (1.5B) and at 1B, with full-attention references.

**12:00 UTC. Forgetting at 1B tokens** (local, on the H200 models via safetensors). Lost at equal learning
(0.052 bpb of docs): plain all 0.0051, 32-bank all 0.0053, 32-bank bank-only 0.0104, bank3 bank-only
0.0128, plain + 25% replay 0.0040, bank + 25% replay 0.0040. 1B models forget far less than 100M ones
(English -0.08 vs -0.20); updating only existing modules still forgets ~2x more; replay still best.
Next: neurogenesis (forgetting.py --train grow:K): add K fresh routed modules, freeze the old ones (and
their router rows), let revival bring the new ones in; growing starts output-identical to the old model.

**12:40 UTC. Neurogenesis + replay: the least forgetting measured.** 1B 32-module bank fine-tuned on docs
(300 steps): add 8 new modules (separate gated bank, old model frozen) + 25% replay: English lost +0.0016,
code +0.0027 (vs +0.019 / +0.023 for all weights + replay), lost at equal learning 0.0021 (best of all;
all+replay 0.0027, all 0.0038). Without replay new modules alone forget a lot (0.0126). Learns less in 300
steps (0.039 vs 0.154 bpb on docs; 14.7M trainable). Brain analogue: new neurons for new material plus
replay of old (sleep). A safe way to teach the agent new facts: grow modules, train with replay.
Rental credit at 12:35 UTC: $70.42 ($0 at ~06:05 UTC Oct 30). Combo runs cut to 600M tokens to fit.

**13:35 UTC. Neurogenesis follow-ups.** 900 steps: all weights + replay overfits the 3.3M-token docs
(learned 0.099 < 0.154 at 300 steps, forgetting doubles); grow:8 + replay keeps improving (0.050) with
forgetting flat (+0.0013 / +0.0026). grow:16 adds little (0.056). Adam lr 0.006 for the new modules:
0.052 in 300 steps, same forgetting; 0.02 worse. Plateau ~0.05 vs 0.15 for full fine-tuning at ~10x less
forgetting: new modules attach at one depth only. Follow-up (not this session): grow modules at several depths.
Multi-depth growth (2 new modules after blocks 1, 3, 5, 7; same 14.7M params; replay 25%, lr 0.006): learned
0.058 vs 0.052 single-depth, forgetting +0.0020 / +0.0038. Small gain; the module width (256) is the likelier
limit. Recipe that works: grow new modules, train them with replay, keep the base frozen.

**14:10 UTC. Same-hardware baselines (H200, 100M):** plain8 1.1447 / 1.1408 (mean 1.1428) vs 32-bank c2.0
1.1417, 64-bank 1.1415: level at 100M (the earlier -0.007 was home-vs-H200 hardware). The bank's advantage
appears only with longer training (1B: plain 0.9341, 32-bank 0.9296, bank3 0.927). Queue on schedule:
s_med_plain 53%, s_med_brain 45%, s_med_plain_ec64k 28%, c_med_combo 20%; disk 59/100 GB.

**14:50 UTC.** Local: combination + earned-say critic (100M, 4k): 1.1142 vs 1.1096 without the critic
(within noise; the critic only helped at 1B before). Critic stays out of the combination at this scale.
Next local: working-memory span 128 instead of 512 (keys/query 512 vs 960), concepts + expansion + fast weights.
Working memory 128 (vs 512; concepts + fast weights, 100M, 4k): 1.1417 vs 1.1130, worse on long documents
too (code 2-4k 0.965 vs 0.889), and no faster at this size (51.8k vs 50.5k tok/s). Next: same key budget
spent on recall instead: window 256 + 8 expanded blocks (896 keys) vs window 512 + 4 (960).
Recall vs working memory at equal key budget (100M, 4k): window 256 + 8 recalled blocks 1.1120 vs window
512 + 4 blocks 1.1130; code 2-4k 0.878 vs 0.889. Halving working memory and doubling recall costs nothing
(slightly better on code, where exact earlier detail matters); recall can substitute for working memory.

**16:05 UTC. Owner's note on neurogenesis:** promising as a long-running "continued training" mechanism:
few new modules relative to the whole, base frozen (neurons don't change function), new ones integrating
with replay. Owner is considering next steps; nothing started on it.

**16:45 UTC. Continual neurogenesis (owner's 5-part design), 1B 32-bank, 6 new modules, ~750 steps, 25% replay:**
A one-shot spread fresh: learned 0.063, English +0.0027, code +0.0045 | B staged + consolidation, spread,
fresh: 0.046, +0.0008, +0.0026 | C staged, placement by need, fresh: 0.039, +0.0009, +0.0018 | D staged, by
need, copied features: 0.077, +0.0013, +0.0036. Copied features double learning at tiny forgetting (D vs C);
D beats one-shot on both learning and English forgetting; staging + consolidation trades learning for less
forgetting; gradient placement chose block 4 every time and did slightly worse than spreading. Next: E =
staged, spread, copied features.
E staged, spread (blocks 1, 3, 5), copied features: learned 0.080, English +0.0011, code +0.0030: best on both.
Next: F = 6 stages x 1 module (smaller, more frequent additions), same total.
F 6 stages x 1 module (100 + 25 steps each): learned 0.064, +0.0015, +0.0041: worse than E's 3 x 2. Stages too
small to integrate; gains stop after ~4. Sweet spot: a few modules per stage, enough steps to mature them.
Next: E with twice the steps per stage.
E2 (E with 400 + 100 steps per stage): learned 0.096 (vs 0.080), English +0.0013, code +0.0030: more training
raises learning with forgetting flat, unlike full fine-tuning (which overfit and forgot 2x more at 900 steps).
Safe to run long, as the owner proposed for continued training.

### 19:55 UTC: medium brain runs diverged; guard added
s_med_brain (230M, routed bank 1 shared + top-2 of 32, FP8) went NaN at step 1660 (0.22B of 1.5B tokens) and
c_med_combo (same bank + concept memory + fast weights) at step 1940 (0.25B of 0.6B). Both at peak learning
rate, both preceded by a jump in dropped tokens (0.09 -> 0.66-0.88) and balance NaN; a resume from step 1600
diverged again at the same step, so it is deterministic instability, not a hardware fault. Last good evals:
s_med_brain 1.0668 at step 1600 vs s_med_plain 1.0716 at the same step (the bank was ahead), c_med_combo 1.0164 at step 1900.
No small brain run ever did this. train.py had no non-finite guard, so both kept stepping NaN weights and
saving them (checkpoints unusable; ~13 GPU-hours lost). Fix (wAI f7f952c): a non-finite loss or gradient skips
the step without touching weights; 20 in a row stops the run without saving. c_med_combo killed to free its
slot. A rerun does not fit the rental; diagnose locally afterwards (router logit growth at medium width,
FP8 in the bank, capacity drops).

### Neurogenesis round 2 (local, 1B bank32 base; 3 stages x 2 modules, 400 + 100 steps, replay 0.25)
learned / English lost / code lost; each run ends with a removal audit (silence one stage's part at a time):
- E2b control (spread 1,3,5; random copy; sleep): 0.0973 / +0.0013 / +0.0034
- H copy the modules the new material routes to: 0.0964 / +0.0013 / +0.0045 (no gain)
- G need with crowding (chose 4, 4, 5) + copy-active: 0.0845 / +0.0014 / +0.0052 (worse: gradient size is a
  poor measure of need; early blocks matter more)
- I grow 4, keep the 2 whose removal hurts most: 0.0956 / +0.0014 / +0.0042 (no gain at the same final size;
  usefulness varied 10x between modules, so selection works, but survivors lose partners they trained with)
- J no sleep, same steps (500 + 0): 0.0753 / +0.0032 / +0.0101: sleep is the most important part: +29%
  learning and a third of the forgetting. Without it stage 3 made the new material worse.
Audit: each part removes cleanly (the old skill returns to within 0.0002-0.0013 of start). The block-1 part
holds most of the new knowledge (removing it loses 0.05 of 0.097); later parts hold 0.006-0.017 each, and
removing stage 2 costs less than expected because stage 3 compensated.
Next (round 3): depth (all at block 1; 1,2,3; 0,1,2), more sleep (200), and a seed replicate for noise.

### Neurogenesis round 3 (same setup; learned / English lost / code lost)
- E2c (E2b, seed 1): 0.0953 / +0.0012 / +0.0043 -> seed noise ~0.002 learned, ~0.001 forgetting
- L spread + sleep 200: 0.0996 / +0.0011 / +0.0033 (best: more sleep, more learning AND less forgetting)
- K all at block 1: 0.0882 / +0.0010 / +0.0024 (least forgetting, less learning)
- M blocks 1,2,3: 0.0831 / +0.0011 / +0.0025;  N blocks 0,1,2: 0.0835 / +0.0016 / +0.0036
Spreading over depth learns most; the third stage at block 5 adds 0.017 where blocks 1-3 add ~0.006.
Round 4: reversed order (5,3,1) to separate depth from stage order, sleep 400, all at block 5, and sleep
replay with 75% old material.

### 21:15 UTC: scale results
s_med_plain_ec64k (64k vocabulary) 0.8489 mix bpb vs s_med_plain (32k) 0.8587: 1.1% better (English 0.9906 vs
1.0023, code 0.6363 vs 0.6434) at equal tokens. Not equal data: a 64k token covers more bytes, so it saw more
text, and has a larger embedding/head. Wall time 7.9 h vs 5.4 h, though runs shared the GPU so times are rough.
memory2: m32k_concept_nopos_fast 0.983, m16_full 0.9545 (short-context eval; the long_context eval decides).

### Neurogenesis round 4 (sleep 200 unless noted; learned / English lost / code lost)
- R reversed order 5,3,1: 0.1017 / +0.0012 / +0.0033 (spread matters, order doesn't; the last stage always
  holds a lot, since its sleep consolidates everything seen so far)
- S sleep 400: 0.0974 / +0.0012 / +0.0032 (no better than 200: sleep saturates)
- T all at block 5: 0.0892 / +0.0011 / +0.0042
- U sleep replay 75% old (general) material: 0.0983 / +0.0004 / +0.0009: forgetting down 3-4x for ~1% less
  learning. 0.0066 lost per unit learned vs 0.134 for full fine-tuning with replay (20x) and 0.055 for one-shot
  growth with replay. Sleeping on old memories is what protects them.
Round 5: replicate U (seed 1), 90% old in sleep, awake on new material only (all protection in sleep), U + 400.

### 22:30 UTC: medium combo vs plain, 16k memory runs
c_med_full4k (plain medium, full attention at 4k, 0.6B tokens): 0.8633 final; at step 1900 0.9847 vs
c_med_combo 1.0164 at the same step (same schedule) before it diverged: at medium, the combo was 3% BEHIND,
unlike small (1.1096 vs 1.1130 ahead). Likely the 512 window + 4 concept blocks costs more at 4k context than the
bank gains; the bank alone at medium (s_med_brain) was 0.5% ahead at step 1600. Memory is the part that does not
scale as is.
16k context, 0.3B tokens, short eval: m16_full 0.9545, m16_concept_nopos 0.9798, m16_window 0.9870 (concepts
recover ~23% of the window-to-full gap here; long_context eval pending).
H200 queue: m16_concept_nopos_fast 52%, combo_small 17-39%; all expected done by ~00:30 UTC.

### Neurogenesis round 5 (sleep 200; learned / English lost / code lost)
- Y old-75% sleep, seed 1: 0.0976 / +0.0006 / +0.0022 (U's +0.0009 code was partly luck; still ~2x below 50/50)
- V old-90% sleep: 0.0923 / +0.0002 / +0.0002: forgetting at noise level, 94% of the learning. A dial: the
  share of old material in sleep trades a little learning for protection.
- X awake on new material only (wake replay 0): 0.0975 / +0.0010 / +0.0032: replay while awake still matters.
- W old-75% sleep 400: 0.0963 / +0.0005 / +0.0010 (no better than 200).
Best recipe so far: staged growth (2 modules per stage) spread over depth, copied features, frozen base and
earlier stages, 25% replay while learning, then a sleep of half the stage's length replaying 75-90% old material.

### 23:41 UTC: divergence diagnostic (rental idle time, hard stop 01:15)
s_med_brain reproduced its NaN at step 1660 from the same seed, so two variants with the identical seed, data
order and VRAM cap (same micro-batching) test the cause: s_med_brain_nofp8 (--fp8 0) and s_med_brain_sigmoid
(bounded router). Passing step 1660 cleanly points at the changed part; the trainer guard stops a diverged one.

### 00:25 UTC: combo_small done; divergence diagnostic, first answer
combo_small (8 x 512 small, 4k context, 0.6B tokens, mix bpb): c_full4k_t1b (plain, full attention) 0.9190,
c_combo_full_t1b (bank + concept memory + fast weights) 0.9291, c_combo_mem_t1b (memory only) 0.9317. Full
attention wins at 4k context by 1.1%; the bank adds 0.0026 on top of memory. Long-context eval pending.
Diagnostic: s_med_brain_nofp8 diverged EARLIER (non-finite from step 1444, vs 1660 with FP8), with intermittent
skips and then every step non-finite: an oversized but finite update makes activations overflow afterwards.
FP8 is not the cause. s_med_brain_sigmoid (bounded router) at step 920, healthy so far.

### 00:32 UTC: long-context eval (local, from pulled safetensors)
4k-trained models at 8k (bpb by position; 40 docs per source):
| run | keys/query | Eng 2-4k | Eng 4-8k | code 2-4k | code 4-8k |
|---|---:|---:|---:|---:|---:|
| c_full4k_t1b (full attention) | 8,192 | 1.1167 | 1.1526 | 0.5764 | 0.5901 |
| c_combo_mem_t1b (concepts + fast) | 960 | 1.1422 | 1.1483 | 0.6281 | 0.6213 |
| c_combo_full_t1b (+ bank) | 960 | 1.1410 | 1.1860 | 0.6273 | 0.6460 |
Past its training length, full attention degrades on English (1.117 -> 1.153) and the memory model edges it
(1.148); on code full attention still wins clearly. The bank version degrades past 4k (English 1.186, code
0.646), likely because the routed modules use RoPE attention at positions they never saw (untested). Within 4k, full attention wins everywhere past 1k.
16k-trained at 16k (fineweb has only 9 docs this long; code 40):
| run | keys/query | Eng 4-16k | code 4-16k |
|---|---:|---:|---:|
| m16_full | 16,384 | 1.1611 | 0.5688 |
| m16_window (512) | 576 | 1.2151 | 0.6748 |
| m16_concept_nopos | 1,088 | 1.2010 | 0.6448 |
| m16_concept_nopos_fast | 1,088 | 1.2110 | 0.6523 |
Concepts recover 26% (English) / 28% (code) of the window-to-full gap with 7% of full attention's keys; less
than the ~half recovered at 4k. Fast weights hurt at 16k (they helped with concepts at 4k). Conclusion: the
concept memory as built (4 recalled blocks of 64) does not scale with context; recall needs to grow with
context (more or adaptive blocks), or concepts need richer summaries than a pooled vector.

### 00:40 UTC: diagnostic result; rental work finished
s_med_brain_sigmoid (bounded router) never diverged, but only because its routed bank was effectively off:
87.5% of routed tokens dropped from step 200 on (all modules over capacity), and at step 2000 it scored 1.0287 vs
s_med_plain 1.0286. FP8-off diverged at 1444 (stopped by the guard at 20 skips, last checkpoint kept). So: not
FP8; the only stable variant was the one without working routing, which points (weakly) at the routed bank's
updates at medium width. Next, locally: log router logit and per-module output norms near the blow-up, and try
router z-loss / lower bank learning rate / gradient clipping on the bank. Rental GPU idle from 00:40; no new
work. Final export + pull at 02:00 UTC.

### Neurogenesis round 6: long-standing continued training (6 stages x 2 modules, 1,3,5,1,3,5; sleep 200)
| run | learned | English lost | code lost | by stage (docs) |
|---|---:|---:|---:|---|
| Z6_old90 | 0.0788 | +0.0002 | +0.0009 | 1.018 1.008 0.998 0.988 0.985 0.982 |
| Z6_old75 | 0.0776 | +0.0008 | +0.0025 | 1.015 1.007 1.002 0.989 0.985 0.983 |
Forgetting does not pile up: with 90%-old sleep it stays at +0.0002 English and creeps on code to +0.0009 by
stage 4, then flat through 12 modules. Learning is lower than 3 x 2 (0.092): each stage sees a sixth of the
material, and stages 5-6 add only ~0.003 each; the audit shows their modules hold almost nothing (removing one
costs 0.000-0.003). Useless late modules are exactly what survival-by-usefulness pruning should remove
(round 2's prune test was at fixed size; here it would shrink the model for free). Takeaway for the owner's
design: frequent small additions are safe indefinitely with old-memory sleep; how much each addition learns
depends on how much genuinely new material it gets.

## Rental, second $100 (from 02:10 UTC Oct 30; est. $0 ~06:30 UTC Oct 31; final pull 02:00 UTC Oct 31)
Pulls now go to D:\Claude\worksAI_runs\h200. Rental disk: removed 35 checkpoints whose safetensors were verified
home (83 -> 50 GB used).

### Medium bank divergence: cause found
diag2 (s_med_brain's seed and schedule, instrumented, stop at 2600): d_base diverged at 1523 (guard stopped it
at 1535); residual rms climbed ~150 (step 200) -> 10k (1520) in every full-LR variant, 8x slower at half LR.
But finished plain medium models have residual rms up to 190k and trained fine, so growth is not the cause.
Cause, in BrainGPT.route: with a router bias (revive), top-k by score + bias can pick modules whose full-softmax
probability underflows to 0 in float32 once router logits spread past ~100 (router max logit was 108-120 at
each failure); the renormalised gates probs.gather / sum were then 0/0 = NaN. Explains: late onset (logit
growth), medium more than small, independent of FP8, and the sigmoid router (bounded) never failing.
Fix (wAI 87400b6): gates = softmax over the chosen modules' logits (identical in exact math, 1e-7 in practice);
regression test fails on the old code. Running d_fix (= d_base + fix) to step 2600; d_norevive (no bias, so
immune to the bug) should also survive while d_z (z-loss, still biased) fails.
Confirmed: d_fix and d_z ran to 2600 with zero skipped steps (d_base, identical but unfixed, diverged at 1523).
At 2600: d_fix 1.0130, d_z 1.0126, d_norevive 1.0127 (but 30% of routed tokens dropped), d_lr_half 1.0154,
plain 1.0181.

### s_med_brain_fix (230M, routed bank, fix; resumed from d_fix at 2600): 0.8522 mix bpb vs s_med_plain 0.8587
English 0.9966 vs 1.0023, code 0.6358 vs 0.6434: the bank is 0.8% better at equal tokens (more parameters than
plain, about the same compute per token), zero skipped steps in 11,445. (Plain with the 64k vocabulary: 0.8489.)

### The judge (owner's design: one judging module beside the others, not a judge per module)
The shared module gets a head (on detached features) estimating each token's final loss and its uncertainty;
once its track record earns trust, the estimate sets each token's routed-module count (easiest third 1, middle 2,
hardest 3, same average; cut-offs are running averages, so causal). judge queue at 100M tokens vs the same bank
without a judge (b32k_sparseB32c2_t100m, 2 seeds): j_alloc (2 seeds), j_only (no allocation), j_random (random
allocation).
Judge round 1 (100M tokens): the judge reached corr ~0.37 with the real loss at the routing depth, short of the
trust bar (0.45 for half say), so it never allocated; j_alloc 1.1465/1.1438, j_only 1.1465, j_random 1.1471 vs
control 1.1429/1.1405 (+0.003 to investigate; c_bank_rerun measures noise). Round 2 (judge2): bar scaled to the
routing depth (half say at corr 0.25, full at 0.35), random-allocation control, and a surprise input (prediction
error of the arriving token at the routing depth, causal).

### Owner's aim for the judge: cognitive dissonance (own idea vs own idea, and own idea vs new evidence)
The judge's target becomes the model's own internal conflict, KL(final prediction || first impression at the
routing depth), needing no answer; surprise at the arriving token is its input (belief vs evidence); anticipated
conflict brings in the router's next choice, an alternative view. Smoke run: dissonance grows to ~1.2 nats in 150
steps and the judge anticipates it at corr 0.65 (vs 0.37 for the real error), earning trust 0.97. judge3 queue.
Later uses of the same signal: dissonance-prioritised sleep replay, dissonance-placed neurogenesis, more memory
recall when conflicted.
Judge round 2 (100M): c_bank_rerun (identical config to the control's seed 1337) 1.1475 vs the original 1.1429:
run-to-run noise ~0.005 at this size (today's code changes shift numerics; runs aren't bit-reproducible). So
round 1's +0.003 was noise. j2_alloc (error judge, trust reached 0.97) 1.1488, j2_random 1.1471, j2_surprise
1.1517: no difference from random allocation resolvable at 100M tokens. 100M is a screen only; anything promising
goes to 1B tokens with 2-3 seeds.
Judge round 3 (100M): the dissonance judge anticipates the model's own conflict at corr ~0.68 and earns full
trust, but its allocation (+/- one routed module per token) is not distinguishable from noise: j3 1.1490/1.1459,
without surprise 1.1506, vs controls 1.1405-1.1475 and random allocation 1.1471. Decisive test at 1B tokens
(judge1b: dissonance judge vs bank, 2 seeds each, today's code), started 09:00.

### 10-specialty benchmark, interim (7 of 10 stages; change from start, negative = better)
neurogenesis (sleep 90% old): Haskell -0.088, Lua -0.070, Fortran -0.033, philosophy/logic/medical/law -0.001 to
-0.003; English +0.0000, code +0.0016. Full fine-tune: Haskell -0.077, Lua -0.045, Fortran -0.037, philosophy
-0.008, logic -0.020, but English +0.015, code +0.043, and later specialties damage earlier ones (law, music,
rust worse). Neurogenesis learns code languages as well or better with ~zero forgetting, but barely moves the
prose specialties. Replay of the model's own samples (recall, temperature 1) degraded everything (English +0.037,
code +0.053 by stage 4): a 110M model's free samples are poor text; stopped. Next form: cued recall (replay
prompted by real fragments).

### Owner's idea: meaning first, word choice gated by register/impact. Diagnostic (register_diag.py)
Loss split exactly into meaning (group of near-synonyms, k-means on the model's output embeddings) and word choice
within the group, base 1B bank model, 200k tokens per text type. At 4096 groups word choice is a near-constant
~12% of the loss for every text type (general English 12.0%, philosophy 12.1, logic 12.1, law 12.2, medical
11.5, code languages 10-12%); 1024 and 8192 groups show the same uniformity (20-23%, 8-10%). The prose
specialties' extra difficulty over general English is mostly meaning (philosophy +0.027 of +0.032 bpb, logic
+0.034 of +0.041). So the prose weakness in neurogenesis looks like missing content, not word choice; a register
gate could touch only a small slice at this scale. Caveats: embedding groups are crude (role-based: "big" groups
with small/large, "casual" with "raw"); register may act across a passage, not per token; a 110M model may not
represent register yet. Not building the register-gated module for now; possible direct probe: formal vs casual
contexts and the model's hello/hi, therefore/so preference.

### Neurogenesis replicates (3 seeds each, local 4070; crawl docs, 3 stages x 2 modules, 400 + 200 steps)
learned / English lost / code lost / lost per unit learned (mean +/- sd):
- full fine-tune + 25% replay, 1800 steps: 0.153+/-0.001 / +0.0184 / +0.0226 / 0.134
- neurogenesis, no sleep (600 + 0):        0.067+/-0.010 / +0.0030 / +0.0103 / 0.101
- sleep 50% old:                           0.097+/-0.002 / +0.0012 / +0.0034 / 0.024
- sleep 75% old:                           0.097+/-0.001 / +0.0005 / +0.0017 / 0.012
- sleep 90% old:                           0.091+/-0.002 / +0.0002 / +0.0007 / 0.005
Sleep is the key part; the old share in sleep is a smooth dial (each step halves forgetting for <=6% of
learning); at 90% old, forgetting per unit learned is 27x below full fine-tuning. Full fine-tuning still learns
1.6x more in absolute terms.

### Direct register probe (register_probe.py; 12 hand-written formal/casual context pairs, same-meaning word pairs)
Mean register shift (log-odds of formal vs casual word, formal context minus casual context): small plain 1B
+4.06 nats (58x odds, 11/12 right way), small bank 1B +4.42 (83x, 12/12), medium plain +4.82 (124x, 12/12),
medium bank fixed +2.80 (16x, 11/12; driven by 2 items), medium plain ec64k +4.81 (122x, 12/12). Matching word
preferred in 17/24 contexts for every model: casual contexts always pick the casual word (12/12); formal contexts
still pick the commoner word for assist, approximately, commence, inquire, "Good morning" (frequency beats
register at the formal end; consistent with the owner's point that even the formal training text uses the common
words). Register is represented even at 110M; with the loss split (word choice ~12% everywhere, prose difficulty
mostly meaning), an explicit register gate is unlikely to fix the prose weakness at this scale; next for prose:
content capacity per stage.

### Judge at 1B tokens (judge1b): dissonance allocation gives no gain
bank 0.9318 / 0.9288 (seeds 1337 / 7; original 0.9296) vs dissonance judge 0.9333 / 0.9295: paired +0.0015 /
+0.0007, within the 0.003 seed spread. The judge anticipates the model's own conflict at corr 0.61-0.67 with full
trust throughout, but +/-1 routed module per token is not a useful response. Untested uses of the signal: sleep
priority, neurogenesis placement, a stronger response (memory consult / weighting the alternative view).

### 10-specialty benchmark, final (Haskell, medical, Lua, philosophy, law, Fortran, logic, music, Rust, space)
mean bpb gain per specialty right after its stage / left at the end / kept / English lost / code lost:
- neurogenesis, sleep 90% old: 0.0252 / 0.0252 / 100% / +0.0000 / +0.0014
- neurogenesis, sleep 75% old: 0.0278 / 0.0280 / 101% / +0.0002 / +0.0023
- adapters, same parameter budget: 0.0142 / 0.0130 / 92% / -0.0002 / +0.0010
- full fine-tuning: 0.0568 / 0.0213 / 37% / +0.0121 / +0.0447 (Haskell 0.175 -> 0.063, Lua 0.120 -> 0.027,
  law ends worse than it started)
Neurogenesis keeps everything it learns across ten specialties, ends with more specialty knowledge than full
fine-tuning, and learns ~2x more than adapters at the same parameter budget. Prose specialties gain only
0.000-0.004 (code languages 0.03-0.10): the open problem. Generative replay (model's own samples) stopped at stage
4 after degrading everything.

### 13:00 UTC block (owner approved A-D)
A prose capacity (medical, philosophy, law, logic): base 2 modules / 8 modules / 2 per depth at 1,3,5 together /
3x longer stages. B (10 specialties): sleep replays the windows most conflicted between first impression and
final view (tagged before each sleep: top quarter of 4x candidates, new and old); placement at the depth where
new material is most conflicted relative to familiar text (smoke: Haskell -> depth 1, medical prose -> depth 6).
C (10 specialties): cued recall, the model continues 32-token fragments of real old text (temperature 1.0 and
0.7) instead of free sampling. D (starts when A ends): neurogenesis vs full fine-tuning on the fixed medium brain
(230M, depths 2,6,10). Rental disk 83 -> 62% (17 verified checkpoints removed).

### Tokenization diagnostics (tokenization_diag.py; medium plain 32k vs 64k vocabulary, same text)
Fragmentation (32k): prose specialties split no more than general English (words in 2+ tokens: philosophy 12.6%,
logic 12.5%, law 10.8%, medical 15.1%, general English 13.4%; code languages 12-17%); loss sits ~3/4 on whole
words in all prose, as in general English (continuation pieces 5-7%; code ~17-21%). Split words cost ~2.3x a whole
word everywhere. Same text, 64k gain over 32k: general English +1.0%, philosophy +1.5, law +1.5, medical +1.9,
logic +0.7, space +1.8, music +1.5, Haskell +2.6, Fortran/Lua +1.9, Rust +0.2, general code +0.1. Tokenization
explains a real but small part (~15% of philosophy's gap over general English); with the register and loss-split
results, most of the prose difficulty is understanding (meaning), not spelling or word choice.
Credit: $70 left at 8:35 AM Central (13:35 UTC) Oct 30 at $4.028/h -> ~2 AM Central Oct 31; keeper final pull
moved to 10 PM Central (03:00 UTC Oct 31), no new work after 9 PM Central.

### Results collected 11:50 AM Central Oct 30 (times below in Central)
- Episodic memory (memory_lm.py, 1B bank model, no training; settings chosen on one half, reported on the other):
  a memory of general text (4M episodes) slightly HURTS at every setting (it repeats the model's own errors).
  A memory that also read each specialty's training text (1M tokens each): Lua +8.3%, Rust +5.2%, Fortran +3.6%,
  Haskell +2.5%, music +2.0%, prose 0.2-0.8%, general English +0.1%: instant learning of new code languages on a
  par with neurogenesis's trained gains, from reading once; prose barely helps.
- A prose capacity (mean prose gain kept): 2 modules 0.0028, 8 modules at one depth 0.0034, 2 at each of 3 depths
  0.0058, 3x longer stages 0.0057. Depth spread equals tripling the time; no forgetting in any.
- B placement by dissonance (10 specialties): prose modules went to depth 6, code to depths 1-2; prose gain 0.0045
  vs 0.0017 with fixed spread (2.6x), total 0.0234 vs 0.0252, English -0.0004 / code +0.0009 (less forgetting).
- B dissonance-prioritised sleep: catastrophic (English +0.144, code +0.148) because the tagged pools were tiny
  (64 new / 256 old windows, each replayed ~25 times: memorised; the most conflicted windows are often odd text).
  Retest with 1024 / 4096-window pools running (E_sleep_dissonance_bigpool).
- C cued recall (32-token real cues, temperature 1.0 / 0.7): still harmful (English +0.033 / +0.015, code +0.096 /
  +0.144). Replaying a 110M model's reconstructions is not viable at this scale.
- Running: worth (plain at 2B/3B tokens, 128-module bank), E (multi-depth growth at the most conflicted depths:
  prose-only and all 10 specialties; the big-pool sleep retest).

### Results collected 2:50 PM Central Oct 30
- E (multi-depth growth at the 3 most conflicted depths), 10 specialties: mean gain kept 0.0273 (best of all;
  standard neurogenesis 0.0252, dissonance placement 0.0234), prose 0.0049 (vs 0.0017), kept 101%, English
  -0.0003, code +0.0011. Caveat: 3x the modules per stage. Prose-only set: 0.0052 (fixed multi-depth 0.0058), so
  the dissonance part matters when stages mix code and prose.
- Dissonance-prioritised sleep, large pools (1024 new / 4096 old): 0.0251 total (= standard), prose -0.0010,
  English +0.0018 / code +0.0016: no benefit; idea closed.
- What the bank is worth (CORRECTION of my earlier ">2x data" estimate): plain small at 2B tokens 0.9010 vs plain
  1B 0.9341 and bank 1B 0.9288-0.9318. Doubling data is worth ~0.033; the bank's ~0.004 is worth ~10% more data.
  (Plain at 3B still running.)
- 128-module bank at 1B: 0.9333, worse than 32 modules (0.929-0.932): wider sparsity doesn't help here.
- RESUME BUG (fixed, wAI train.py): resume copied the run name from the checkpoint, so s_med_brain_fix (resumed
  from a copy of d_fix's checkpoint) trained into sweep_d_fix; s_med_brain_fix's export was the stale step-2600
  checkpoint (English 1.139). The real final medium brain is d_fix's export (English 1.0024 by the same eval; plain
  1.0074), now also placed as s_med_brain_fix/model.safetensors (stale file kept as model.STALE_step2600). The
  logged 0.8522 result stands (it came from the training process). Invalidated and redone: register probe on the
  medium brain is +4.54 nats (94x odds; was +2.80 on the stale weights); D (medium neurogenesis) relaunched on the
  correct weights.
