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

### Instruction and conversation as modules (C1, C2; data and queue Nov 2)
Data (build_domains.py --only instruct,chat; raw files in data/raw_chat): instruct = Dolly-15k (15,011
single-turn instructions written by Databricks staff, CC BY-SA 3.0; 2.74M train tokens), chat = OpenAssistant
oasst2 English (5,400 conversation trees written by volunteers, Apache-2.0, one conversation per tree along the
best-ranked reply; 2.26M train tokens). Both as "User: ...\n\nAssistant: ..." turns, split by document.
M2's settings (multi-depth at the most conflicted depths, sleep), no memory.

**C1_instruct_chat (base model, instruct then chat; 2:24 - 2:34 AM Nov 2)**

| | instruct | chat | English | code |
|---|---|---|---|---|
| start | 1.1254 | 0.9249 | 1.0777 | 0.7038 |
| after instruct stage | 1.0272 | 0.8967 | 1.0779 | 0.7039 |
| after chat stage | 1.0270 | 0.8606 | 1.0780 | 0.7039 |
| without the instruct modules | 1.1148 | 0.8891 | | |
| without the chat modules | 1.0272 | 0.8967 | | |

- Both placed at depths 4-6 (where prose goes; the dissonance gap was strongly negative at depths 1-3).
- Instruct learned 0.098 bpb (8.7%). Transfer one way: the instruct modules alone gave chat 0.028, 44% of
  chat's total gain (0.064); the chat modules added nothing to instruct (0.0002).
- The two module groups stack cleanly: switching off either removes just its own share (the instruct modules
  still supply chat's 0.028 after the chat stage; the chat modules didn't relearn it). The shared part (answering
  a request in the User/Assistant format) lives in the instruct modules; the chat modules hold what is specific to
  multi-turn conversation.
- No forgetting: English +0.0003, code +0.0001 (noise).

**C2_spec_instruct_chat (the 10 specialties, then instruct, then chat; 2:34 - 4:36 AM Nov 2)**

| | instruct | chat | specialties: mean kept | English / code drift |
|---|---|---|---|---|
| start | 1.1254 | 0.9249 | | |
| after the 10 specialties (stage 10) | 1.1252 | 0.9255 | 0.0313 | -0.0003 / +0.0009 |
| after instruct (stage 11) | 1.0473 | 0.8998 | 0.0313 | -0.0002 / +0.0009 |
| after chat (stage 12) | 1.0419 | 0.8709 | 0.0314 | -0.0003 / +0.0009 |
| C1 final (base model only) | 1.0270 | 0.8606 | | |

- **Instruction and conversation training cost the specialties nothing:** from stage 10 to the end every
  specialty moved by at most 0.0004 bpb (Haskell; mean 0.0001). The usual cost of instruction tuning (knowledge
  worn away) doesn't appear when it goes into its own modules.
- Separation stays clean: switching off the instruct modules removes instruct's gain (1.1138) and its share of
  chat (0.8952); switching off the chat modules removes only chat's stage gain; the specialty modules don't touch
  either (switching off space's modules leaves instruct and chat unchanged).
- Placement: instruct at depths 4-6 as in C1; chat this time at depths 2-4 (C1: 4-6), and here it gave instruct a
  little (0.0054; C1 0.0002).
- Learned ~15% less than on the bare base model: instruct 0.084 (C1 0.098), chat 0.054 (C1 0.064). Single runs,
  3x the noise level; possibly the router sharing depths 4-6 with the specialty modules already there (new modules
  are copied from the base bank, not from the specialties, so not the init). Unchecked.
- **Noise check:** C2's first 10 stages use M2's training settings (memory only reads and scores, it doesn't
  train), so they are a rerun of M2's modules: Haskell 1.0350 vs 1.0299 (0.005), Rust 0.7187 vs 0.7209, Fortran
  1.0012 vs 1.0002, mean kept 0.0314 vs 0.0319. Confirms ~0.005 bpb run-to-run noise per specialty, and that M3's
  differences from M2 (Haskell 0.008, M3 vs C2 0.002) are noise.

**Sample replies (C1s_instruct_chat: C1 rerun with `--save`, 6:58 - 7:07 AM Nov 2)**
`continual --save DIR` writes the grown modules (safetensors + grown.json); `forgetting.load_grown(DIR)` puts them
back on the base export (checked: the reloaded model reproduces the run's scores exactly); `samples.py DIR` answers
held-out first turns and fixed prompts with the base model, each module group switched off, and the full model
(temperature 0.7, top-k 40) -> DIR/samples.md (runs/forgetting/C1s_instruct_chat/samples.md, not in git).
The rerun matched C1 (instruct 1.0277 vs 1.0270, chat 0.8603 vs 0.8606).
- Base model: never answers; continues as web text or repeats the question.
- Chat modules only (instruct switched off): imitates the transcript, writes "User:"/"Assistant:" turns itself.
- Instruct modules only: answers short tasks in Dolly's form ("Identify which car manufacturer is British or
  American: Aston Martin, Cadillac" -> "Aston Martin is British, Aston Martin is American"; pulls the right
  sentence out of a given passage: "The species can be found in Africa ...").
- Both: assistant style appears only here ("Certainly! Here's a possible paragraph:", "Sure! Here's a quick
  step-by-step guide", numbered tips: "Choose a quiet place: Get your bedroom dark and away from any
  electronics ...", "Practice mindfulness: ... meditation, deep breathing"). The chat modules build on the
  instruct modules rather than standing alone, as the scores showed.
- Content is mostly wrong or looping (watermelon "a great source of protein", a Haskell monad explained as an
  automobile; C1 has no specialty modules): a 136M model at 1B tokens learns the form, not the knowledge.

### 3D visualization (model/viz, Nov 2)
`viz/extract.py` runs the base model plus a saved run's grown modules (default C2s since Nov 2, C1s before) on three prompts (chat, Rust,
prose) and records the residual stream after every stage, attention per block and head, the router's picks with
capacity drops, grown-module gates, the logit lens and module usage on 16k tokens of each of 14 datasets;
`viz/build.py` inlines it into `viz/template.html` (three.js) -> runs/viz/index.html. Published as the private
artifact "BrainGPT, Inside". Found along the way:
- The routed modules specialise without being told: module 29 is first for Haskell, Rust, Lua, Fortran and general
  code (12.6% of Lua's firings, even share 3.1%); 28 web prose (medical, law, FineWeb-Edu); 14 and 10 philosophy
  and logic; 25 music, space and conversation; 8 instructions.
- U-net skip into block 4 has weight -0.93 and its source is the stream itself (block 3's output), so the stream
  shrinks to 7% before block 4: blocks 4-7 write with ~14x more relative weight.
- Short prompts squeeze the bank: capacity is 2 x T x 2 / 32 slots per module, 3 on a 19-token prompt (128 in
  training), so some routes are dropped at inference on short inputs.
- Nov 2, after C2s: the page reads the grown modules from the run (one centred row per depth, a shade per stage, the
  run's own scores in the chapter), so any saved run works; republished with C2s's 72 modules (C1s's data kept as
  runs/viz/data_C1s.json). The camera animation now clamps its progress at 0 (a frame time earlier than the click
  sent it backwards). Headless screenshots of the page can come out black (an idle canvas, not a render failure):
  copy the canvas into an <img> after a render to check it.

### C2s_spec_instruct_chat: do the specialties come through in chat? (10:01 AM - 12:02 PM Nov 2)
**Question:** does specialty knowledge show when the model answers in chat form? C1s's replies had the assistant
form but wrong content, and C1s has no specialty modules. C2s is C2 again with `--save` (queue line in
cls_queue.sh), so its modules can be loaded and sampled. Prompts: model/prompts/specialties.txt (one question per
specialty, plus two general ones); samples in runs/forgetting/C2s_spec_instruct_chat/samples_specialties.md and,
for chat without specialties, runs/forgetting/C1s_instruct_chat/samples_specialties.md (not in git).

**Scores: a second noise check, passed.** The same settings as C2 put every stage at the same depths. Final scores
are within ~0.005 bpb of C2 on everything: Rust +0.0054 and Fortran +0.0049 (the largest), Haskell -0.0030, Lua
+0.0030, the rest within 0.0006. Instruct 1.0414 (C2 1.0419), chat 0.8707 (0.8709); English and code unchanged
(1.0774 / 0.7049). Mean kept 0.0303 (C2 0.0314), prose 0.0065 (0.0067). The audit separates cleanly again:
switching off one stage's modules removes only its own specialty's gain (Haskell 1.032 -> 1.143, Lua 0.887 ->
0.948, Fortran 1.006 -> 1.088, Rust 0.724 -> 0.770); switching off instruct takes instruct back to 1.111 and chat
to 0.893; switching off chat leaves instruct at 1.045.

**Replies: no.** On the 12 prompt-file questions (temperature 0.7, same seed per prompt in every variant):
- **Answers are wrong with or without the specialty modules.** No variant of C2s gets any of the 12 right, and
  neither does C1s. C1s's diabetes list happens to name two real symptoms (fatigue, weight loss); C2s says "high
  blood sugar levels and weight gain" inside an otherwise wrong answer (an autoimmune disease of the colon).
- **Specialty modules alone (`-instruct+chat`)** continue the specialty's kind of text instead of answering.
  Diabetes gets insulin-producing cells, β-cells and blood sugar (wrong: type 1). Mars gets NASA missions and the
  Mars Science Laboratory. Kant gets philosophy-sounding prose. Fortran gets subroutine documentation. The
  vocabulary is topical, but there is still no answer and no code.
- **They change little on short chat questions.** For 6 of the 12 the first 30 characters of the reply are the
  same as the base model's (same seed); the modules learned ~3% on their own texts, a small shift in the odds.
- **Everything on (`all`)** answers in form, like C1s; for 4 of 12 the reply opens exactly as C1s's does ("There
  are two ways to play a C major chord on the guitar", "Type 2 diabetes is an autoimmune disease that usually
  affects the lining of ..."). The instruct and chat modules decide the reply. C2s's answers are a little plainer
  than C1s's (fewer numbered lists).
- **No variant writes code.** The Rust, Lua and Fortran questions get prose about functions and vectors.

**Reading:** at 136M parameters and 1B tokens of pretraining, a few million tokens per specialty moves its
held-out text by 6-9% in bpb for code languages and under 1% for prose (style, terms, common phrasing), not enough knowledge for correct answers to come out
in any form. The modular split still works as measured (separate, removable, no forgetting); what is missing is
knowledge, which more scale or more data per specialty would have to supply. Untested ways to check that the
knowledge is there but not reachable: prompts written in the specialty's own style (a code comment, a textbook
sentence to complete) instead of chat questions.

### Future plan, unscheduled (owner, Nov 2): more languages as modules
Not queued; nothing to run until the owner schedules it. Measured Nov 2 on the base model (held-out samples,
~60k tokens each; oasst2 messages for spoken languages, the 6 local github-code-clean files for code):

| | base bpb | bytes / token |
|---|---|---|
| English (oasst2) | 0.99 | 4.2 |
| code the base knows: C, Java, HTML, CSS, C# | 0.27-0.70 | 3.4-4.6 |
| less familiar code: Perl 0.98, Makefile 1.00, Shell 1.04, PowerShell 1.05, Julia 1.12, Haskell 1.14, Assembly 1.16, Dockerfile 1.33, Fortran 1.34, TeX 1.38 | 0.98-1.38 | 2.4-3.3 |
| Spanish 1.84, French 1.94, Portuguese 2.00, German 2.22, Italian 2.25, Catalan 2.47 | 1.84-2.47 | 2.9-3.2 |
| Russian / Chinese (2-3 bytes per character, so bpb flatters them) | 1.50 / 2.19 | 2.9 / 1.8 |

Estimates (to be replaced by measurement):
- **Programming languages (fairly confident):** the four code specialties gained 6-9% each (Haskell from 1.13: 9%;
  Rust from 0.77: 6%), more the less familiar the language, with no forgetting. Best candidates: TeX, Assembly,
  Shell, Makefile, Perl (expect 6-10%); C/Java/HTML/CSS would gain little. Local data (6 of 880 files) covers TeX,
  Shell, Perl, Makefile, Go, Ruby, Scala; Assembly and PowerShell are borderline (~2M tokens); Julia and Dockerfile
  would need more files (a download).
- **Spoken languages (less confident):** Latin-script languages start at ~2x English's bpb, so expect larger
  gains than code (guess 15-30% per 3M tokens; outside anything measured), learning common words and phrasing,
  not fluency. The tokenizer is English-heavy and frozen in continual learning (Chinese gets under one character
  per token), so Chinese and Russian should learn less per token than Spanish or French. Spanish in oasst2
  (28k messages, on disk) is enough for one stage; other languages need a download (e.g. FineWeb-2).
- **Costs, either kind:** ~10 min per language on the 4070; 6 modules (11M parameters, ~8% of the base) per
  stage. Grown modules run densely (every one on every token, gated afterwards), so each language adds ~14% to
  the work per token; ten would more than double it. Sparse routing for grown modules would fix that (code change).
  Crowding: instruct/chat learned ~15% less after 10 specialties (one run); code languages all land at depths 1-3.

**Cheapest test when scheduled:** one continual run, Spanish (oasst2 `es`, needs a build_domains source like
`chat` filtered by language) then TeX (github-code-clean `TeX`, add to CODE in build_domains.py), M2/C2 settings,
no memory: ~25 min, no download.

### Memory switches for bigger base models (Nov 2; not used by any run yet)
The owner asked for both as switches, "with the plan not to use either unless we get close to hitting the
ceiling": `train.py --grad_ckpt 1` (activation checkpointing) first, then `--bf16_state 1` (bf16 gradients and
Muon momentum) only as a last resort for the largest models. Calibration pair on hold. Details and estimated
ceilings are in BEST.md, "Memory ceiling".
- Either switch compiles block by block instead of the whole model. With a whole-model graph, every weight's
  gradient arrives together at the end of the backward pass. Per block they arrive as each block finishes, so the
  bf16 buffers can take them one block at a time.
  - Measured on a 4 x 1024 model: compiling per block alone lowered the peak during gradient accumulation by
    87 MiB (449 -> 362 MiB above steady state).
- Checkpointing gives the same results:
  - The gradients are bit-identical to the normal path, eager and compiled. That covers K/V shared across layers,
    value residual and U-net skips (state passed between blocks, so the recompute gets a copy of what each block
    saw), and a brain model with a sparse bank.
  - The backward pass runs under the forward's choice of attention kernels, so the recompute uses the same ones.
  - Only the main blocks are checkpointed, not the brain's bank.
- Smoke runs of 30 steps on tiny models (1.5 GB cap):

  | model | switches | bpb | tokens/s | peak memory |
  |---|---|---|---|---|
  | plain | off | 2.8686 | 214k | 0.9 GiB |
  | plain | `--grad_ckpt` | 2.8683 | 175k | 0.6 GiB |
  | plain | both | 2.8679 | 176k | 0.6 GiB |
  | brain, 9-module bank, FP8 | off | 2.8989 | 111k | 1.1 GiB |
  | brain, 9-module bank, FP8 | both | 2.8980 | 65k | 0.9 GiB |

  At full size the slowdown should be smaller (a tiny model is dominated by launch overhead); not measured yet.

### R1_spec_reverse: does accumulated knowledge make new learning harder? (started 2:35 PM Nov 2)
Owner's hypothesis: the more (conflicting) knowledge a mind holds, the harder new knowledge is to take in. The run
uses the 10 specialties in reverse order, with C2's settings exactly (cls_queue.sh, no memory, ~1 h 40 min). The
comparison is each specialty's gain in its own stage (bpb just before minus just after) against C2 and C2s, whose
first 10 stages are the forward order. Haskell moves from position 1 to 10 and space from 10 to 1.
- **Noise:** between C2 and C2s, the gain per specialty differs by sd 0.0027 (max 0.0062).
- **Confound:** a specialty learned late may already have improved through transfer (Haskell after Lua, Fortran and
  Rust). So the total gain from the start is compared too.
- **Script:** model/order_effect.py (gain = curve[s-1] - curve[s] for the specialty learned at stage s;
  total = curve[0] - curve[10]).
- **Result (done 4:17 PM, no errors): later means less for the English-text specialties, not for code.**
  - Order doesn't change the total: mean gain 0.0328 vs 0.0321 and 0.0311 forward. Order moves learning between
    specialties.
  - **Text specialties** (medical, philosophy, law, logic, music, space):
    - All six moved in the predicted direction, by 3–12× their noise (rms 0.0005 between C2 and C2s).
    - Space learns 0.0101 at position 1 against 0.0039 at position 10 (2.6×). Medical learns half as much at
      position 9 as at position 2.
    - Correlation -0.90, about -12% per position later.
  - **Code specialties** (haskell, lua, fortran, rust): -1.3% per position, within noise (rms 0.0044). Haskell learns
    the same at position 10 as at position 1.
  - **Not transfer or placement:** every specialty gets the same depths in both orders. The text specialties start
    their stage from the same bpb (space 1.0787 forward, 1.0788 reverse), so nothing learned earlier had helped
    them.
  - **Not capacity:** old modules are frozen and each stage trains 6 new ones, so capacity per stage is constant.
    Two things do accumulate:
    - (a) the frozen modules of earlier stages, all active in the forward pass;
    - (b) sleep. Its 10% specialty share is split over every specialty seen so far: the current one gets 10% at
      stage 1 and 1% at stage 10, and the new modules spend sleep fitting older material. In exposure this is
      small: 20 against 2 batch-equivalents out of ~320 per stage.
    - (b) is this architecture's version of the owner's "combine it with everything learned before". These runs
      can't separate (a) from (b): bpb is measured only after sleep.
  - **Reading:** accumulation hurts subtle learning (small refinements on text the base already reads well). Big
    gains (a new programming language) hold up.
  - **Total** (after all 10 stages, forgetting included) shows the same pattern: -39% per 9 positions later
    overall, against -46% for gain.
  - **Possible follow-up (not queued):** evaluate after wake as well as after sleep, to split (a) from (b). If (a)
    dominates, pruning and recycling (wai_struct STRUCT_PLAN item 5) moves up.
