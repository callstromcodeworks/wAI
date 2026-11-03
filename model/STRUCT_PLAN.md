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
4. **Surprise-gated plasticity (neuromodulation).** Each token's training is weighted by how unexpected it was to
   the model.
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
