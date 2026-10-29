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
