# Local phase: RTX 4070 12 GB, Ubuntu (from Oct 31, 2026; times Central)

Continues H200_PLAN.md. The machine: i7-10700K, 31 GB RAM, RTX 4070 (driver 595.99.02, CUDA 13.2), Ubuntu 26.04.1
(kernel 7.0, Python 3.14.4), torch 2.15.0+cu132, triton 3.9.0. Data on ext4: ec32k, ec64k, raw parquet, the
10-specialty domains (rebuilt by build_domains.py in one 10-domain pass). The H200 built 8 domains first (be99b1c)
and added philosophy and logic later with `--only` (fresh seed), so the shared shuffle differs after Lua: only
Haskell, medical and Lua have the H200's exact texts (start scores equal); the other 7 have the same sources but
other documents (Fortran starts 1.073 here, 1.519 there). Compare those 7 within local runs only. To reproduce the
H200's (untested): `build_domains.py --only haskell,medical,lua,law,fortran,music,rust,space`, then
`--only philosophy,logic`, into a copy so running jobs keep their files.
The owner runs an openbox desktop on the card: give notice before GPU-memory-heavy jobs.

## Throughput (plain small, L12 d768 kv2, 16 x 1024 x 8 accum, FP8 + compile)
| setup | tok/s |
|---|---|
| Windows, torch 2.14.1, Oct 25 code (final_ec32k_gqa log) | 80,100 |
| Linux, torch 2.15, 1432088 (row_major workaround) | 71,300 |
| Linux, torch 2.14.1, Oct 28 code 557869e (before the workaround) | 77,100 |
| Linux, torch 2.15, FP8 fix 31cf051 (Ubuntu 24.04 and 26.04 alike) | 77,000 |

The torch 2.15 workaround copied every transposed fp8 operand eagerly, twice (contiguous + clone): 15.7 ms of
222 ms per micro-batch. Compiled fp8-to-fp8 transposed copies return NaN gradients on 2.15 (at some shapes on 2.14),
so the fix casts each transposed operand from the bf16 transpose with the same scale (one fused kernel); memory
5.6 -> 4.6 GiB. Windows never paid this: its 80k run predates the workaround (Oct 25) and ran torch 2.14.1, which
only fails at shapes this model doesn't use.

Card settings (same benchmark): persistence mode, the performance CPU governor and a 216 W limit (from 200 W) all
gave 71.3-71.4k: the card never drew more than 199 W even when allowed 216 W, at 2,520-2,715 MHz and ~70 C.

continual harness (forgetting.py): the training forward now runs compiled; two stages of the 10-specialty
benchmark took 785 s eager vs 525 s compiled with the same numbers (to 1e-4). The exact H200 continual settings
include `--lr_adam 0.006` (default 0.002); cls_queue.sh copies them.

## Complementary learning: episodic memory beside neurogenesis (handoff step 2)
episodic.py: each specialty is read into memory as it arrives (500k tokens) and every text is scored with memory
before any training; after wake + sleep the memory is re-keyed with the changed model and episodes it now predicts
with p >= 0.5 are released (kept stored for comparison). Each score: model alone / full memory / only held
episodes, plus the best blend of each gate (fixed, entropy, dissonance), chosen on the first half of each text and
reported on the second.
- M1_neuro_mem: DB_neuro_old90 settings + memory.
- M2_mddis_mem: E_all_multidepth_dissonance settings + memory.

Scoring cost: each scoring searches every stored episode, so a stage's memory scoring grows with memory (M1: ~10 min
for stage 1, ~30 for stage 5, ~50 for stage 10; ~6 h in all). After M1 started, two speedups that change no
number (tests/episodic_checks.py; M1's code and the new code give identical scores on a stand-in model): an arrival
encodes only the new text (the model that keyed the rest is unchanged, checked by a weight fingerprint), and
scoring right after an arrival reuses each text's previous search and searches only the new episodes. Keys are
pinned per text (M1 held 15 GiB of pinned blocks by stage 6; now at most one 1 GiB block per specialty). M2 runs
with this code.

### M1_neuro_mem results (7:10 AM - 1:08 PM Oct 31)
Modules: mean kept 0.0277, prose 0.0017, English -0.0001, code +0.0008 (no forgetting). On the three specialties
whose texts match the H200's, the modules learned the same (Haskell -0.089 vs -0.088, Lua -0.070 vs -0.070).
Memory gain in bpb (second half of each held-out text; "on top" = after the stage's wake + sleep, "held" = only
episodes not released at p >= 0.5; final = after all 10 stages):

| specialty | instant | on top | on top, held | final | final, held | held share |
|---|---|---|---|---|---|---|
| Haskell | 2.5% | 1.5% | 1.3% | 1.4% | 1.2% | 55% |
| medical | 0.2% | 0.2% | 0.0% | 0.2% | 0.0% | 68% |
| Lua | 5.0% | 2.6% | 0.0% | 2.6% | 0.0% | 46% |
| law | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 75% |
| Fortran | 1.5% | 0.0% | 0.0% | 0.0% | 0.0% | 45% |
| logic | 0.2% | 0.2% | 0.0% | 0.2% | 0.0% | 75% |
| music | 0.2% | 0.2% | 0.1% | 0.2% | 0.1% | 80% |
| Rust | 6.0% | 3.0% | 2.8% | 3.0% | 2.8% | 45% |
| space | 0.3% | 0.2% | 0.0% | 0.2% | 0.0% | 76% |

- Memory adds on top of consolidated modules for code (Rust 3.0%, Lua 2.6%, Haskell 1.4%), ~0.2% for prose,
  0 for general English and code (never harms: the blend turns itself off). Fortran: memory's 1.5% was absorbed
  by the modules (Fortran 1.089 -> 1.014).
- Release at p >= 0.5 keeps most of Rust's and Haskell's gain but all of Lua's is lost: memory helps most on
  tokens the model already half-knows. Next: a stricter threshold (p >= 0.9) or release by memory's own benefit.
- Gate (final, mean over 9 specialties): fixed lam 0.87%, entropy 0.44%, dissonance 0.34%; fixed best on 8 of 9,
  entropy on Rust. Dissonance is not a useful "when to consult memory" signal here.
- 5.0M episodes stored, 3.2M held (64%).
- **Data overlap (philosophy excluded above):** 18% of philosophy's held-out 64-token windows occur verbatim in
  logic's training text (pages matching both filters), so memory "gains" 15% on philosophy from stage 7 (logic)
  on, and logic training leaks into philosophy's score. Within a specialty: Rust 2.2%, Lua 1.3%, Haskell and
  medical 0.4% of held-out windows recur in its own training text (shared boilerplate), so part of the code gains
  is verbatim recall. Overlap between topics is expected and doesn't hurt the model (owner's call: no
  deduplication); it only matters when reading memory gains on a text whose twin is already in memory.
- Checked (base model, philosophy's held-out second half): 36% of its tokens end a 32-token passage found verbatim
  in logic's memory text. Memory of philosophy + logic: +15.1% overall (as M1), +41.8% on those copied tokens,
  +0.0% on the rest (philosophy's own text alone: +0.1% overall, -0.1% on the rest). The 15% is recall of the
  shared passages; memory of logic gives no measurable topical help to the other philosophy text. As retrieval it
  works well: memory finds passages read under another specialty, cut at different row offsets with different
  preceding text, and uses them (+41.8%) without harming anything else; just don't count it as philosophy learned. (The modules'
  small philosophy gain at the logic stage, 0.0008, is partial training, possibly helped by the same passages.)

### M2_mddis_mem results (12:06 - 4:52 PM Nov 1; arrival scoring ~3 min at every stage, M1's grew to ~25)
Multi-depth growth at the most conflicted depths (code went to depths 1-3, prose to 4-6, as on the H200) + memory.

| | M1 (fixed depth) | M2 (multi-depth, dissonance) |
|---|---|---|
| modules: mean kept | 0.0277 | 0.0319 |
| prose kept (medical, law, logic) | 0.0019 | 0.0065 |
| English / general code drift | -0.0001 / +0.0008 | -0.0003 / +0.0012 |
| memory on top, final (mean of 9, philosophy excluded) | 0.87% | 1.00% |
| ... only held episodes (release p >= 0.5) | 0.46% | 0.82% |
| model + memory, final mean bpb (9) | 0.9752 | 0.9708 |
| gates: fixed / entropy / dissonance | 0.87 / 0.44 / 0.34% | 0.93 / 0.50 / 0.63% |

- Per specialty (final, all / held): Rust 4.0 / 3.7%, Lua 3.2 / 2.6%, Haskell 1.3 / 1.2%, Fortran 0, prose 0-0.2%.
  M2 keeps most of memory's gain after release (Lua too, which M1 lost entirely).
- Modules: M2 learns more Haskell (0.105 vs 0.088), Fortran (0.073 vs 0.059) and 3x the prose; a little less
  Lua (0.060 vs 0.069) and Rust (0.046 vs 0.050). No forgetting in either.
- Fixed blend still the best gate; dissonance improves (0.63%) but stays behind.
- Philosophy/logic overlap, consolidated: M2 learned logic 4x as much as M1, including the passages shared with
  philosophy; memory released them (philosophy final 14.9% with all episodes, 0.0% with held ones; M1 kept
  14.9% held) and the modules' philosophy score improved 0.0017 at the logic stage (M1 0.0004). What memory
  recalled verbatim, the modules took over.

### M3_mddis_distill: sleep that distils memory into the modules (6:22 PM Nov 1 - 12:18 AM Nov 2; shelved)
M2's settings + `--sleep_distill 1`. Before each sleep, memory is re-keyed with the woken model and a pool of 1,024
windows is drawn from the specialties seen so far, beyond the text memory read (a window never finds itself).
For every position memory's 32 neighbours are stored; per specialty the blend (lam, tau) is the best on its own
pool windows, as in scoring. Sleep's new-material rows come from the pool, with target (1 - lam) one-hot + lam
memory's distribution (kNN distillation, the model + memory as teacher); where memory doesn't help, lam = 0.
Short tests (2 specialties, 20-step stages): a 16-window pool hurt (Haskell 1.1424 vs 1.1239 without
distillation) and did so with distillation off too, so the pool was memorised (as the H200's 64/256-window sleep
pools were); with 512 or 1,024 windows Haskell matched no distillation (1.1256, 1.1247). Peak 8.9 GB.
Test of success vs M2: modules alone better on the code specialties, memory on top smaller, model + memory no worse.

| | M2 | M3 (distillation) |
|---|---|---|
| modules: mean kept | 0.0319 | 0.0311 |
| prose kept (medical, law, logic) | 0.0065 | 0.0064 |
| English / general code drift | -0.0003 / +0.0012 | -0.0000 / +0.0016 |
| modules alone, final: Haskell / Lua / Rust / Fortran | 1.0299 / 0.8837 / 0.7209 / 1.0002 | 1.0378 / 0.8854 / 0.7175 / 0.9961 |
| memory on top, final (mean of 9, philosophy excluded) | 1.00% | 0.86% |
| ... only held episodes (release p >= 0.5) | 0.82% | 0.47% |
| model + memory, final mean bpb (9) | 0.9708 | 0.9722 |
| gates: fixed / entropy / dissonance | 0.93 / 0.50 / 0.63% | 0.86 / 0.44 / 0.29% |

- **No measurable effect: within run-to-run noise (~0.005 bpb) on every count.** Two code specialties a little
  better (Rust, Fortran), two a little worse (Haskell 0.008, the largest gap, opened at its first sleep; Lua).
  Memory on top is smaller, but not because the modules took its knowledge over: model + memory is no better.
  The held-only drop is mostly Lua (2.6% -> 0.0%, as M1) and philosophy's shared passages staying held (14.6%,
  as M1; M2 released them): release-threshold effects, which swing between runs.
- Why it could hardly move anything: the dose was tiny (memory's distribution at 5% weight on the 10% of sleep
  rows that are new material, ~0.5% of what sleep trains on), and the teacher barely beats the student: on the
  pool windows model + memory beat the model by 0-2.7%, and the blend search chose the smallest weight offered
  (0.05) at 46 of 55 specialty-sleeps and none (memory doesn't help) at the other 9.
- **Shelved (owner, Nov 1).** A real answer would need a much larger dose and 2+ seeds of both M2 and M3 (~6 h a
  run). Memory stays a separate component on top of the modules (as in M2). `--sleep_distill` stays in the code,
  off by default.
