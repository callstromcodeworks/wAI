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
