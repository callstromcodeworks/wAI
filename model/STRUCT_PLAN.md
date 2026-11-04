# Branch wai_struct: brain-like structure first (from Nov 2, 2026; times Central)

This branch starts from main 165c1b4 and lives in its own worktree, ~/claude/wai_struct.
- Data, the virtualenv, the tokenizer library and the base model are links to ~/claude/wAI.
- Its runs go to its own model/runs.
- main's finished continual results are readable at model/runs/main_forgetting.

The owner's rules for the branch:
- Lean towards structure before performance. A more brain-like change stays unless it makes the model inoperable.
- Make sacrifices where the hardware doesn't fit the brain, choosing speed there (sparse long-range wiring,
  continuous feedback loops).
- Nothing from this branch is pushed to main; main stays focused on efficiency.
- Rebase the branch only where necessary. The branch itself may be pushed (origin/wai_struct), with local commits
  at milestones.

## Work queue (owner: "pursue the items you mentioned (label-free arrival, selective hippocampus, etc.)")
The brain-like learning life cycle, built on the organization that already works:
1. **Label-free arrival.** One unlabeled stream mixes familiar text with the 10 specialties, in their usual order
   but with no boundaries given. During the "day" the model only reads, and passages it finds surprising go to a
   hippocampal buffer. At "night" it grows modules at its most conflicted depths, learns from the buffer in sleep
   with replay, and clears the buffer.
   - Score with the same 10-specialty evaluation as C2 (labeled, same step budget).
   - Also score detection: how much of the buffer was specialty text, and how much specialty text was missed
     (the true labels are known for analysis only).
2. **Selective hippocampus.** Episodic memory stores only surprising windows instead of everything that arrives,
   and its benefit is measured per stored token.
3. **Experience as the only data.** Sleep replays a small fixed pool of old text kept from before, instead of an
   unlimited corpus. Sweep the pool size: tiny pools were catastrophic before.
   - Added Nov 3 (owner: "Both and before item 4"), from the owner's account of human memory: recall isn't
     verbatim; the brain stores patterns and reconstructs them, a pattern used more is recalled closer to its
     source, and mnemonics help by tying new material to what is strongly held (like quantization: precision traded
     for memorability).
   - **2b. Pattern hippocampus.** Memory keeps patterns, not copies: each episode's key becomes pointers into
     learned codebooks of prototypes plus a small correction (product/residual quantization). Precision follows use:
     refined while it is retrieved, coarsened and then released when it isn't. Scored by gain per byte against item
     2's exact 16-bit keys (1.5 KB per episode).
   - **3b. Replay by reconstruction.** No stored old text: at the start of each night the model generates old
     material from its own weights (sampled from a document start), and the night replays that. Well-learned
     patterns come back accurately, rarely used ones loosely. A second run also rebuilds the earlier days'
     specialty text from short cues kept from their buffers, instead of whole windows (the hippocampus keeps an
     index, the cortex fills it in). Needs incremental decoding for the grown model (BrainGPT had none).
   - **Dropped for now (owner, Nov 3), after 3b's control.** The owner: humanity learned from generations of
     shared knowledge, not each individual's own; replaying the shared record (the corpus) is part of learning.
     S3b_gen2k was stopped at day 3; S3b_gen2k_cue32 was not run.
4. **Surprise-gated plasticity (neuromodulation).** Each token's training is weighted by how unexpected it was to
   the model.
   - **4a, first version (owner, Nov 3: "start A as 4's first version"): study until learned.** From the owner:
     students learn over several sessions until their uncertainty falls below a threshold, and only then are said
     to have learned the material. 10% of each night's buffer is held back as its exam. The buffer stays in study
     (wake and sleep) on the nights that follow, and is released after a night that improved its exam by less than
     0.002 bpb. Old general text is replayed as before. Run S4a_learned; then a control that studies every buffer
     for the same fixed number of nights (`--study_nights`, set from S4a's average), so the test is what decides.
   - The original form (each token's training weighted by its surprise) stays for a later version.
   - **B, planned after 4a (owner: "plan for B afterwards"):** use makes recall easier because the brain gets better
     at what it uses. Plan below, under "4b plan".
5. **Pruning and recycling.** Remove grown modules that are rarely used and reuse their capacity. Its priority
   depends on R1 (main: reverse order, does learning slow as knowledge accumulates?).
6. **Capstone.** Tell it facts during the day; after a night's sleep, does it know them without the facts in
   context?
Then the structural changes below, best value first.

## Results log

### 1. Label-free arrival (Nov 2)
**Detector probe** (model/surprise_probe.py, base model, inference only):
- **Single-window novelty fails.** Tried: loss, dissonance per depth, Mahalanobis distance of the state, distance to
  the nearest familiar windows, unusual routing pattern.
  - The code specialties stand out (Haskell, Fortran, Lua, Rust: AUC 0.75–0.89).
  - The English ones (medical, law, philosophy, music, space) score *below* familiar text on every signal
    (AUC 0.2–0.5). One window of them is no stranger to the model than familiar text.
- **What sets a new topic apart is that it keeps coming up.** The signal is today@D: a window whose nearest
  neighbours (state after block D) come mostly from today's stream rather than from familiar text.
  - In a simulated 50/50 day at depth 6, k=30: AUC 0.958 (worst specialty, law, 0.880).
  - At a threshold that keeps 10% of familiar text: precision 0.878, recall 0.933.
  - Structurally, this is the hippocampus comparing the day's episodes with what the cortex knows. The day's states
    are held until night, and the night selects what recurred.

**Stream mode** (forgetting.py `--stream`):
- **Day:** the specialty's whole training text as consecutive windows, mixed 50/50 into familiar text, shuffled,
  unlabeled. The model only reads.
- **Night:**
  - The buffer keeps windows above the 0.9 quantile of familiar windows held outside the day.
  - Grow 6 modules at the 3 depths most conflicted for the buffer (vs familiar text).
  - 400 steps learning from the buffer with 25% general replay.
  - 200 steps of sleep replaying every night's buffer so far, with the same shares as C2.
- **Smoke test** (rust, space): precision 0.87 and 0.95, recall 0.96 and 0.92. The depths matched C2's labeled
  choice (rust 1–3, space 4–6).
- **Runs:** S1_stream, then its control S1o_stream_oracle with a perfect detector (struct_queue.sh, started
  4:37 PM). Compare: `python model/stream_compare.py C2_spec_instruct_chat C2s_spec_instruct_chat S1_stream
  S1o_stream_oracle`.

**S1_stream result** (4:36–6:32 PM, no errors): **without labels it learns as much as labeled C2.**
- Mean total kept per specialty: 0.0319, against 0.0314 for C2 and 0.0304 for C2s. English -0.0002, code
  +0.0010: no forgetting, as in C2.
- Buffer: mean precision 0.922, recall 0.953.
  - Code specialties: recall 0.99–1.00.
  - English specialties: recall 0.87–0.95. Law was lowest (0.871), then philosophy, music and space (~0.91).
  - Familiar text kept by mistake: 6–10% of it.
- Per specialty, total at stage 10 against C2/C2s, where the C2/C2s noise is sd 0.0027:
  - Haskell 0.119 against 0.101, well above noise. Space and medical are a little higher.
  - Law (0.0015 against 0.0025), philosophy (0.0047 against 0.0074) and logic (0.0062 against 0.0089) are lower.
    These are the specialties with the lowest recall. Philosophy and logic share text (main LOCAL_PLAN, M1), so
    they also pass less to each other.
- The oracle (S1o) separates the cost of detection from the other differences: fixed windows instead of random
  ones, and familiar text in the buffer acting as extra replay.

**S1o_stream_oracle** (perfect detector, 6:32–8:25 PM): **detection costs nothing.**
- Mean total 0.0317, against S1's 0.0319. English -0.0001, code +0.0013.
- Law (0.0008), music (0.0026), philosophy (0.0055) and logic (0.0071) are low here too, so the prose dip against C2
  is not detection.
- Haskell is high in both stream runs (0.115 and 0.119, against 0.10).
- Both come from the night, which repeats the same consecutive windows ~3× instead of drawing windows at random
  offsets. That seems to help a code specialty that is learned from scratch, and to cost the prose a little, whose
  gains are small refinements.
- Item 1 done: the model finds what is new by itself and learns it as well as when told.

### 2. Selective hippocampus (Nov 2)
**Probe** (model/selective_memory.py, 8:25–10:34 PM, inference only, no errors):
- **Cortex:** main's C2s model (base + 36 grown parts), so memory sits on top of consolidated modules, as in M2's
  "final" numbers.
- **Days:** each specialty arrives as a day of 1,500 of its training windows mixed with 1,500 familiar windows,
  unlabeled. The buffer is today@6, as in S1: precision 0.91–0.94, recall 0.89–1.00 (law lowest).
- **Equal capacity:** every policy stores 500k episodes per day (5M in all) and is scored like M2 (gain over the
  model alone, blend chosen on the first half of each held-out text).

| policy: what is stored | text read for 5M episodes | mean gain (9, no philosophy) | per M episodes |
|---|---|---|---|
| all: every window as it arrives | 5.0M | 0.38% | 0.075% |
| buffer: windows the detector kept | 5.0M | 0.77% | 0.153% |
| buffer, tokens with p(next) < 0.9 | 6.4M | 0.94% | 0.189% |
| buffer, tokens with p(next) < 0.5 | 8.3M | **1.17%** | **0.234%** |
| labeled: the specialty's windows (perfect detector) | 5.0M | 0.85% | 0.170% |
| labeled, p(next) < 0.5 | 8.3M | 1.14% | 0.228% |

- **Storing only what surprised it triples the value of each stored episode** (0.234% against 0.075% per M).
  - Choosing windows (the day's detector) doubles it, because half of what arrives is familiar text.
  - Choosing tokens (p < 0.5) adds half again, because the same capacity covers 1.65× as much new text.
- **Label-free matches labeled again.** With p < 0.5 the buffer even edges out the perfect detector (1.17% against
  1.14%, within noise). The familiar windows the detector kept by mistake are well predicted, so token selection
  drops them anyway. Storing whole windows, the buffer is a little behind (0.77% against 0.85%).
- **Lua carries the mean** (5.7% with buffer p < 0.5, against 2.5% storing everything). Without Lua the order is
  the same: all 0.12%, buffer 0.51%, labeled 0.45%, buffer p < 0.5 0.60%, labeled p < 0.5 0.60%.
  - From the buffer or the labels: Rust 1.7–2.6%, Haskell ~0.8%, space up to 1.0%.
  - Law and logic stay at ~0 under every policy: the blend chooses not to use memory there.
  - Fortran is the exception to token selection: +0.4–0.5% storing whole windows, 0 with p < 0.9 or 0.5.
- **No harm:** English and code +0.00% under every policy (the blend gives memory no weight there).
- **Against M2** (labeled, every token, on M2's model): 1.00% with 5M episodes, 0.82% for the episodes held after
  release at p ≥ 0.5. Here the labeled all-token policy gives 0.85% on C2s's model with a different day sample, so
  only the comparison inside this probe is like for like.
- **Control (P2c, Nov 3 9:02–10:21 AM, runs/selective/P2c_C2s.log):** a random 60% of tokens covers as much text
  as p < 0.5 without choosing by surprise.

  | policy | text read | mean gain | per M episodes |
  |---|---|---|---|
  | buffer, random 60% of tokens | 8.34M | 0.79% | 0.157% |
  | labeled, random 60% of tokens | 8.34M | 0.95% | 0.189% |
  | buffer, p < 0.5 (rerun, another reading order) | 8.27M | 1.10% | 0.219% |

  - **Surprise is what helps, not coverage.** Covering 1.65× the text at random adds almost nothing to the buffer
    (0.77% → 0.79%); choosing the same amount by surprise gives 1.10–1.17%. The reading above ("adds half again,
    because the same capacity covers 1.65× as much new text") was wrong about the cause.
  - Noise between reading orders: 1.17% against 1.10%, so ~0.07. Surprise beats random by ~0.3, about 4× that.

### 3. Experience as the only data (Nov 2–3)
**Sweep** (struct_queue.sh, 10:34 PM – 8:26 AM, no errors). S1, but old material in the wake's 25% replay and in
sleep comes from a fixed pool of N windows kept from before (familiar text, English/code 60/40), instead of fresh
windows from the whole corpus. A night draws ~9,000 old windows, so over the run each pool window is replayed about
90,000 / N times.

| old material from | replays per window | mean total kept | English | code |
|---|---|---|---|---|
| the whole corpus (S1) | — | 0.0319 | -0.0002 | +0.0010 |
| 16,384 windows | ~5 | **0.0314** | **+0.0006** | **+0.0015** |
| 4,096, refreshed with each day's familiar windows (reservoir) | ~22 | 0.0262 | +0.0058 | +0.0062 |
| 4,096 | ~22 | 0.0258 | +0.0085 | +0.0090 |
| 1,024 | ~90 | -0.0264 | +0.1119 | +0.1185 |
| 256 | ~350 | -0.0233 | +0.0911 | +0.1244 |

- **16k windows (~17M tokens) is enough:** within noise of S1 on every specialty and on English/code.
- **4k drifts steadily** (English ~+0.0009 a night). The prose specialties, closest to general English, end below
  their start (law -0.0068, music -0.0048). The code specialties learn as much as in S1 or more (Fortran 0.073,
  Rust 0.047), perhaps because the narrower replay constrains the new modules less.
- **Refreshing the pool with what the day read** cuts the forgetting by about a third, but no more learning is
  kept.
- **1k and 256 collapse:** English and code ~0.1 worse, every prose specialty below its start. The own-stage gains
  look large (Lua 0.49 with 256) only because each stage repairs damage done before it.
- **Reading:** what matters is how often each kept window is replayed, not the pool itself. Below ~5 replays per
  window over the run it holds, by ~20 it drifts, and by ~90 the model memorizes the pool and loses general text.
- **Limits of the analogy** (discussed with the owner, Nov 3): the pool keeps raw text verbatim, which the brain
  doesn't, and four of the five pools are corpus text chosen at the start rather than anything the model read. Item
  3b replaces the stored text with text the model writes itself.
- **Earlier H200 results this explains:** main's replay of the model's own samples ("recall", Oct 30, closed as
  harmful in BEST.md) drew 512 sequences once, before learning, and replayed them for all 10 stages: a fixed pool of
  512, which this sweep shows is catastrophic even for real text. So that verdict may be about pool size, not about
  the samples; 3b retests it with fresh samples every night.

### 2b and 3b: setup (Nov 3)
- **Decoding for the grown model** (brain.py BrainCache, reconstruct.py Decoder): BrainGPT had no incremental
  decoding. A fixed-shape step (position as a GPU tensor, masked attention over the whole buffer) is captured as a
  CUDA graph.
  - Check against a full forward (C2s, 36 grown parts, 4 × 1,024 tokens each of English, code, Haskell): loss
    3.2324 / 1.9778 / 2.6179 forward vs 3.2314 / 1.9783 / 2.6180 decoding, top-1 agreement 99.0–99.3% (bf16
    rounding).
  - Speed with 36 parts: 1,811 tokens/s at batch 32 (eager decoding was 414, bound by ~2,000 small kernels per
    token). It is now memory-bound: every grown part keeps its own K/V. A stream run has 0–27 parts at night, so
    2,048 windows take about 4–15 minutes a night.
- **Pattern memory check** (Rust alone, 500k episodes): exact keys 2.53%, fine codes (96 B, 1/16 of the bytes)
  2.02%, coarse (32 B) 0.64%, precision by use (48 B average, 32 use windows) 0.78%; releasing never-recalled
  episodes (51%) lost it all (-0.07%). The full probe uses 64 use windows per specialty.
- **Queue** (struct_queue.sh, tmux s5, from 9:02 AM): P2c (item 2 control) → P2b (pattern memory) → S3b_fresh2k
  (control: 2,048 fresh corpus windows every night) → S3b_gen2k (2,048 windows the model writes every night) →
  S3b_gen2k_cue32 (also earlier nights' buffers kept as 32-token cues and rebuilt, 1,024 a night).

### 2b. Pattern hippocampus (Nov 3, 10:21 AM–12:22 PM, runs/selective/P2b_C2s.log)
Item 2's best store (buffer, p < 0.5, 5M episodes on C2s), with keys as product-quantized codes. "Use": the 25% of
episodes recalled most during a use period (1,280 windows, 1.3M queries) keep fine codes, the rest coarse.

| keys | bytes per episode | key memory | mean gain (9, no philosophy) |
|---|---|---|---|
| exact (fp16) | 1,536 | 7,324 MB | 1.12% |
| fine code | 96 | 458 MB | **0.96%** |
| coarse code | 32 | 153 MB | 0.05% |
| use: most-recalled quarter fine, rest coarse | 48 | 229 MB | 0.08% |
| random quarter fine, rest coarse | 48 | 229 MB | 0.08% |
| use, never-recalled episodes released (14%) | — | — | 0.05% |

- **Patterns instead of verbatim keys work.** Fine codes keep 86% of the gain at 1/16 of the bytes (2.1% per GB
  against 0.16% for exact keys).
- **Precision following use did nothing as built.** Use and random tie at 0.08%, both close to all-coarse. A
  shared kNN search across mixed precision is dominated by its coarse majority: noisy coarse keys crowd in as
  false neighbors, so sharpening the most-used quarter can't show.
- **Releasing the never-recalled 14%** cost the little that was left (0.08% → 0.05%).
- Not tested: keep only the most-used quarter (fine codes) and release the rest, which is closer to "used patterns
  stay sharp, unused ones fade".

### 3b. Replay by reconstruction: the control (Nov 3, 12:22–2:22 PM)
S3b_fresh2k: as S1, but old material is 2,048 corpus windows redrawn every night (~9,000 drawn per night, so each
window ~4.4 times in one night and never again).

| run | old material | mean kept | English | code |
|---|---|---|---|---|
| S1_stream | whole corpus | 0.0319 | -0.0002 | +0.0010 |
| S3_pool16k | fixed 16k windows | 0.0314 | +0.0006 | +0.0015 |
| S3_pool4k | fixed 4k windows | 0.0258 | +0.0085 | +0.0090 |
| S3b_fresh2k | 2k fresh windows each night | **0.0173** | +0.0130 | +0.0186 |

- **Fresh is worse than fixed at a larger size.** 2k fresh windows a night forget more than a fixed 4k pool, though
  each window is replayed only ~4 times in all (against ~22). So item 3's "replays per window" should read
  **replays per window per night**: 4.4 here, between the fixed 4k (2.2) and the fixed 1k (8.8).
- **What this means for 3b:** the generated runs write 2,048 windows a night, so this is their ceiling if the
  model's text were as good as real. The question they answer is unchanged (written against read text at an
  equal count), but matching S1 would need ~16k written windows a night, ~0.5–2 h of decoding per night at
  today's speed.
- 3b's generated runs were then dropped for now (owner, Nov 3; see the work queue).

### 4a. Study until learned: setup (Nov 3)
- **forgetting.py `--master_tol X`** (stream mode):
  - Each night's buffer is split 90/10. The 10% is the exam, never studied, and is scored (bpb) before anything
    is learned from it.
  - Every buffer still in study is pooled for the night's wake and sleep, in place of S1's "today's buffer by
    day, every buffer so far in sleep".
  - After each night every buffer in study is examined again, and released if tonight improved it by less than
    X. Released buffers are no longer replayed.
- **`--study_nights K`** is the control: each buffer is studied exactly K nights, whatever its exam says.
- **Threshold:** 0.002 bpb is about 2% of Haskell's own-stage gain and about the size of the smallest
  specialties' whole gain (law, logic), so those are released after their first night.
- Smoke test (3 specialties, 20 + 10 steps): exams, pooling and release behave as intended. The log line is
  `exams: day 1 (haskell) 1.1892->1.1737, tonight +0.0038 | ...`.
- S4a_learned ran 3:12–5:10 PM (struct_queue.sh, tmux s5).

**Result (S4a_learned against S1_stream; gain in bpb at stage 10):**

| specialty | nights in study | S1 | S4a |
|---|---|---|---|
| haskell | 4 | 0.1191 | 0.1022 |
| medical | 2 | 0.0097 | 0.0028 |
| lua | 5 | 0.0601 | 0.0553 |
| philosophy | 1 | 0.0047 | 0.0013 |
| law | 1 | 0.0015 | 0.0007 |
| fortran | 5 | 0.0652 | **0.0880** |
| logic | 2 | 0.0062 | 0.0022 |
| music | 1 | 0.0038 | 0.0014 |
| rust | 2, still in study at the end | 0.0427 | **0.0572** |
| space | 1 | 0.0064 | 0.0019 |
| **mean** | 2.4 | **0.0319** | **0.0313** |
| English / code drift | | -0.0002 / +0.0010 | +0.0002 / +0.0015 |

- **Equal on average** (0.0313 against 0.0319; noise ~0.0027), and English and code drift no more.
- **Study moved to what kept improving.** The code specialties stayed in study 2–5 nights: Fortran and Rust both
  end ~35% above S1. The prose specialties were released after 1–2 nights and keep about a third of S1's (small)
  gains: 0.0018 against 0.0055 mean (medical, law, logic, music, space).
- **Why prose lost:** less study on its first night, not forgetting (each prose total is close to its own-stage
  gain). The buffers in study share the night equally, so a new day's buffer gets a third to a half of the wake,
  while S1 gives it all of it (medical's own-stage gain 0.0036 against 0.0099).
- **Released material slips without review:** Haskell's held-out score worsened 0.016 bpb between its release
  (night 4) and night 10, and Lua's 0.004 between nights 7 and 10.
- In terms of interest: the rule behaves like interest as learning progress (Oudeyer and Kaplan), with effort
  following progress. The control decides whether the exam's choice matters or just studying several nights:
  S4a_nights2, every buffer exactly 2 nights (S4a's mean 2.4), queued after P2d.

**Control (S4a_nights2, 6:35–8:39 PM):** every buffer studied exactly 2 nights.

| | S1 | S4a (exam decides) | S4a_nights2 (2 nights each) |
|---|---|---|---|
| mean | 0.0319 | 0.0313 | 0.0319 |
| haskell / lua | 0.1191 / 0.0601 | 0.1022 / 0.0553 (4 and 5 nights) | 0.0940 / 0.0502 |
| fortran / rust | 0.0652 / 0.0427 | 0.0880 / 0.0572 | 0.0906 / 0.0656 |
| prose mean (medical, law, logic, music, space) | 0.0055 | 0.0018 | 0.0032 |
| English / code drift | -0.0002 / +0.0010 | +0.0002 / +0.0015 | +0.0002 / +0.0016 |

- **The exam decided nothing a fixed 2 nights doesn't.** All three runs have the same mean. The exam's extra nights
  gave Haskell and Lua ~0.005–0.008, and releasing prose after one night cost about as much.
- **What both 4a runs change against S1 is the schedule, not the test:** a second night lifts the code specialties
  (Fortran +0.025, Rust +0.015–0.023; Rust's second night alone added 0.02–0.03), while sharing the first night
  costs the prose specialties and Haskell.
- **Not tried:** each new buffer gets its whole first night as in S1, and the following nights' study comes on top.
  That keeps the gain for code without the cost for prose.

**First night kept whole (owner, Nov 3: "First night variant"; S4a_first, from 11:52 PM):**
- forgetting.py `--review_steps R`: today's buffer gets the whole wake as in S1. Earlier buffers still in study
  then get R extra steps (75% their windows, 25% general text), and sleep replays every buffer in study. Release
  by exam (0.002 bpb) as in S4a.
- R = 200, about S4a's dose for carried-over buffers. The extra steps are on top, so the run trains more than S1
  (up to 1,800 extra steps against S1's 6,000; the JSON's `reviewed` gives the count). If it wins, the control is
  the same extra steps spent reviewing every earlier buffer, chosen without the exam.

**Result (S4a_first, 11:52 PM–2:12 AM; 1,600 review steps on 8 nights, +27% training):**

| | S1 | S4a | S4a_nights2 | S4a_first |
|---|---|---|---|---|
| mean | 0.0319 | 0.0313 | 0.0319 | **0.0298** |
| fortran / rust | 0.0652 / 0.0427 | 0.0880 / 0.0572 | 0.0906 / 0.0656 | 0.0850 / 0.0547 |
| haskell | 0.1191 | 0.1022 | 0.0940 | 0.0910 |
| prose mean (medical, law, logic, music, space) | 0.0055 | 0.0018 | 0.0032 | 0.0026 |
| code drift | +0.0010 | +0.0015 | +0.0016 | +0.0023 |

- **No better, with more training.** The mean is a little lower (within noise), and code drifts a little more.
- **My explanation for the prose loss was wrong.** Today's buffer had the whole wake again, yet prose own-stage
  gains stayed below every S1-type run (medical 0.0042 here; 0.0075–0.0099 in S1, S1o, S3_pool16k and S3_pool4k,
  which share S1's first night). Sharing the wake was not the cause.
- **Likely cause: who does the re-study.** Earlier parts are frozen after their night, so an earlier buffer can
  only be studied again by *tonight's* parts, grown and placed for tonight's material. Here the review retrained
  them on other material right after their wake. In S4a and S4a_nights2 they split the wake. Either way, tonight's
  specialty loses part of its parts, and the old one is studied by parts grown for something else. That still
  helps code (more capacity on it) and costs prose. Likely, not shown.
- **What this means:** in this design, "several sessions" can't work as it does in the brain, where re-study
  strengthens the circuits that already hold the material. That is 4b's mechanism (unfreeze the parts that hold
  the topic). A direct test fits the 10-day benchmark: an earlier buffer still in study is re-studied by the parts
  grown on its own night (label-free: the night that grew them), not by tonight's parts.
- **The owner's correction (Nov 4):** "the human brain is much more fallible. Where it may take the organic brain
  a few retries to parse new information, the digital brain gets it in much fewer steps." Repeated sessions are
  the brain's workaround for its own fallibility, not part of its structure, so 4a shouldn't be expected to help.
  The data agree: with its whole first night, a topic learns 70–88% of everything it ever learns in that night
  (S4a_first: Haskell 86%, Fortran 88%, Rust 78%, Lua 70%), and one night already passes over its buffer ~3 times.
  The re-study test proposed above is on hold until the owner chooses the next step.

### 2c. Interest: is "too surprising" worth storing? (setup, Nov 3)
The owner: interest is critical to human learning, and surprise usually draws interest. But curiosity peaks at
intermediate confidence (Kang et al. 2009) and infants look away from the too simple and the too random (Kidd et al.
2012). Item 2's rule (p < 0.5) has no upper limit on surprise: it keeps noise-like tokens such as IDs and numbers.
- episodic.py `store(min_p=M)` and the policy `buffer_mMpP` keep tokens with M <= p(next) < P.
- P2d (queued 5:12 PM, ~80 min): `buffer_p0.5` again, the band `buffer_m0.01p0.5`, and a stricter `buffer_p0.1`.
  Same capacity per day (500k episodes), so the band reads more text to fill it.
- Prediction either way: an inverted U favors the band, but kNN memory helps most on rare tokens (names, library
  identifiers), which may be exactly the p < 1% ones.

**Result (P2d, 5:12–6:35 PM):**

| policy | text read | episodes | mean gain | per M episodes |
|---|---|---|---|---|
| p < 0.5 (third run; earlier 1.17%, 1.10%) | 8.30M | 5.00M | 0.97% | 0.194% |
| band 0.01 <= p < 0.5 | 12.67M | 4.96M | **0.26%** | 0.053% |
| p < 0.1 | 12.14M | 4.86M | **1.24%** | 0.254% |

- **The "too surprising" tokens carry most of the memory's value.** Without them the gain falls by three
  quarters (Lua 4.36% → 0.00%, Rust 2.69% → 1.42%, Haskell 0.98% → 0.39%). They are ~21% of all tokens, about a
  third of the p < 0.5 ones.
- **Stricter is better:** p < 0.1 beats p < 0.5 (1.24% against 0.97%, about 2 noise widths). The value of a stored
  episode rises with surprise all the way; no inverted U.
- **Noise:** p < 0.5 has now given 0.97%, 1.10% and 1.17%, so ~0.1 between reading orders (not 0.07).
- **Reading:** the hippocampus is for what the cortex can't predict: names and identifiers that recur across
  documents. In people that matches the isolation (von Restorff) effect: the item that stands out is the one
  remembered. The inverted U belongs to choosing what to *study*, and 4a's progress rule is that half. So the
  two halves of interest split across the two systems: novelty decides what the hippocampus keeps; progress decides
  what the cortex studies.

### 4b plan: recurring topics and practice (owner: "plan for B afterwards"; not built)
The owner's account: what gets used a lot becomes easier to recall, because the brain itself gets better at it.
The current benchmark can't test that: each specialty arrives on one day, so nothing is used more than anything
else.
Owner's answers (Nov 3):
- **Schedule:** "Repeat once, 4 times, and then the full 8", meaning three runs (confirmed: not one run with three
  tiers).
- **What a repeat is:** "less mixing things and more applying the new knowledge to what was already there". A repeat
  isn't old and new text mixed together: new material on a known topic is worked into the knowledge that already
  holds it.

1. **Benchmark: three runs with rising repeats.** Every specialty appears 2, then 4, then 8 times. Each run splits
   the specialty's whole training text into that many disjoint parts, so every appearance is new text on a known
   topic and the run as a whole reads the same text as S1.
   - Days are 20, 40 and 80. Appearances go in rounds, in the usual specialty order (round k brings part k of each
     specialty), so a topic returns every 10 days. The order is my default, not the owner's.
   - Night steps scale with the part (400/n wake, 200/n sleep), so every run studies each window as much as S1
     does. Only the number of sessions changes.
   - Scored as now: each specialty's gain at the end against S1 (the same text in one session) and across 2/4/8,
     plus English/code drift and parts count.
   - **Drift and savings (added Nov 3, owner: "what we don't use will drift"; accepted).** Each topic's held-out
     score just before every return shows how far it drifted while unused. The model's frozen parts don't decay,
     so drift here is interference from later learning. On each return, the steps needed to win back the lost
     ground, against the steps the topic first took to reach that level, give Ebbinghaus's savings: if relearning
     is faster, the knowledge was still in the parts, only covered by interference. Needs the topic scored a few
     times within a night, not only at its end.
2. **Applying new knowledge to what was there (the mechanism).** On a day whose buffer matches a topic already
   held, no new parts grow. The parts that hold that topic are unfrozen and learn the new text.
   - Matching is label-free: state neighbors between today's buffer and earlier days' buffers, as the today@6
     detector does within a day. A labeled match (oracle) is the control.
   - New parts grow only for material nothing matches, normally a topic's first appearance.
   - The update must keep what the parts already did for the topic. Where that topic's earlier material is
     still in study (4a), it is replayed with the new text. Otherwise the parts' earlier outputs are matched
     (distillation against the parts as they were). To settle when building.
   - Brain-like in structure: practice revises existing circuits, and new tissue serves new skills.
3. **Baseline:** the current recipe on the same streams, growing 3 frozen parts every day. At 8 repeats that is 240
   parts, likely too slow and too big for the card; then the baseline runs only for 2 and 4 repeats.
4. **Recall drives consolidation (the hippocampus), after 1–3.** Episodic memory joins the stream (today
   `--stream` refuses `--memory_tokens`). Episodes recalled while reading a day count as used. Sleep replays the
   windows behind the most-used episodes, and episodes the cortex now predicts unaided (p(next) ≥ 0.5) are
   released, as M2's prune did.
5. **Order:** the 2-repeat stream with the baseline and with the mechanism, then 4, then 8.
6. **Cost:** total training steps equal S1's; per-day overhead (reading, calibration, placement, scoring every
   specialty) repeats 20/40/80 times. To be timed on the 2-repeat run before estimating the others.
7. **With 4a:** a night of 1/n the steps improves an exam less, so 4a's 0.002 bpb release threshold would need
   scaling by n if 4a is kept.
8. **Built (Nov 4; owner: "Yes but don't run it yet"). Not run, not smoke-tested on the GPU.**
   - forgetting.py `--repeats N`: each specialty's text in N disjoint parts, round r brings part r of every
     specialty in the usual order (10N days). Night lengths are set in the command (`--stage_steps 400/N
     --consolidate_steps 200/N`).
   - `--practice 1`: before growth, today's buffer is matched against the topics already held. Each window votes
     for the topic holding most of its 30 nearest neighbors (states at depth 6; 256 windows sampled per held topic,
     plus the familiar reference, which can win the vote). A topic with at least half the votes (`--match_share`)
     is a match: nothing grows, and the parts grown on that topic's first day are unfrozen and learn tonight (wake
     on today's buffer; sleep replays every buffer so far, the topic's earlier ones included, which keeps what the
     parts already did). `--practice_oracle 1` matches by name (control). Every match is logged with its votes and
     whether it was right.
   - `--track_every 25`: the day's specialty is scored on 64 held-out windows every 25 wake steps (labels for
     analysis only), for savings.
   - recur_report.py: gain kept at the end (beside S1), after each appearance, drift while away, savings on return,
     matching accuracy. Checked on synthetic data.
   - Queue (struct_queue.sh, commented out): R2_grow, then R2_practice. ~2–2.5 h each (20 days; per-day overhead
     still to be timed).
   - **Risk:** R2_grow ends with ~60 grown parts (S1: 30). Training memory grows with parts (S3 runs reached ~8.4
     GB at ~27–30), so it may not fit in 12 GB. If not: gradient checkpointing first (owner's order), or run the
     baseline at fewer parts per day.
   - Before the real runs: a ~5-minute smoke test (3 specialties × 2 appearances, short nights).

## Brain-like structure roadmap (moved from main's LOCAL_PLAN.md, Nov 2)
The owner wants to "push the metaphor to the extreme". Clarified: brain-like in *structure*, not mimicking biology
(neuron mechanics) and not about ability or sensation. The question is how closely the model's organization can
match the brain's large-scale organization.

**Map: brain structure -> BrainGPT today**

| brain | role | BrainGPT | status |
|---|---|---|---|
| cortical areas | specialized processors, repeated canonical circuit | routed bank: 32 identical small transformers + 1 shared | have; specialized without being assigned |
| areas ordered in a hierarchy | surface features low, abstract high | one bank at block 3; grown modules at several depths | partial: grown modules sorted themselves, code at depths 1-3, prose at 6 |
| global workspace | specialists compete to write to a shared broadcast | residual stream + router (top-2 of 32 write) | have, close match |
| thalamus | hub routing and gating traffic between areas | one router, at one depth | partial |
| basal ganglia | selects which program proceeds | top-k selection inside the router | partial (merged with routing) |
| hippocampus | fast episodic store, consolidation into cortex | episodic memory (kNN, blended at the output) | have, but outside the network |
| cerebellum | learned corrections to cortical output | critic (reads the draft, predicts its error, corrects) | built; off in the best base (no gain) |
| prefrontal cortex | working memory held across time, top-down control | none: nothing persists beyond the 1,024-token window | missing |
| feedback connections | as dense as feedforward; later areas modulate earlier ones | none: strictly bottom-up in depth | missing |
| neuromodulators | global broadcast setting gain and plasticity | none (the critic's trust is one global scalar) | missing |
| slow upper areas | higher areas integrate over longer spans | every block runs per token | missing (concept memory tried, for long context only) |
| structural plasticity | new neurons, pruning | growth at conflicted depths; no pruning | half |

**Structural changes, best value first.** Screen each at 100M tokens (~20 min, noise ~0.005), confirm winners at 1B
(~3-4 h), each against an equal-compute baseline.
1. **Areas across the hierarchy:** several smaller banks at different depths, each with its own router, instead of
   one bank at block 3. Same total parameters. The grown modules already sorted themselves by depth. Code: BrainGPT
   supports one bank; the grown-module code shows the pattern for several depths.
2. **Feedback (top-down) connections:** two-pass settling, where the second pass's lower blocks also read the first
   pass's top output. Compare against a plain model of equal compute (2 x 8 blocks against 16). This is the
   brain's biggest structural feature the model lacks.
3. **Prefrontal working memory:** a few state vectors carried from one segment to the next, updated through a gate
   (basal-ganglia-like). Needs training on consecutive segments of documents rather than random windows.
4. **Hippocampus wired into the network:** retrieved episodes enter the residual stream at mid-depth (as the
   hippocampus feeds back into cortex), instead of only blending output probabilities.
5. **Separate basal ganglia and thalamus:** selection (which areas act) split from routing (where information
   goes), and later area-to-area routing that doesn't pass through the shared stream. Highest risk.
6. **Cerebellum revisited:** the critic as a fast-learning corrector during continual learning, rather than in
   pretraining where it gained nothing.
7. **Neuromodulatory broadcast:** one small unit reads surprise and uncertainty and sets gains for every area.
8. **Slow upper areas:** the top blocks run on 8-token chunks.
9. **Pruning** to pair with growth (more urgent if R1 shows plasticity falling).

**Limits of the metaphor on a GPU:**
- Sparse long-range wiring and continuous recurrence run slowly on dense matrix hardware, so feedback is
  approximated in passes.
- The brain runs its areas asynchronously and in parallel; here they run in a fixed order.

**Decided (owner, Nov 2):** structure first. Keep the more brain-like change unless it makes the model inoperable,
and choose speed where the hardware doesn't fit the brain (sparse long-range wiring, continuous feedback loops).
