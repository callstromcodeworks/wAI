# Plan: concept memory (compress old context, expand on demand)

## Idea
People don't hold everything they've read word for word. They keep compressed concepts, and
expand one back into detail from memory when the current context calls for it (the
hippocampus's job). A transformer today keeps every past token in full (the KV cache), so long
context costs memory and compute in proportion to its length.

The model version: older context is kept as **concepts**, one learned summary per block of 64
tokens. Each token reads three things:
1. **Recent text in full:** a sliding window of the last 512 tokens.
2. **Older text as concepts:** one summary per older block, so 4,096 tokens of history become 64
   concept slots.
3. **Expanded detail on demand:** for the few concepts it attends to most (top 4), the original
   tokens of those blocks are read in full for this step.

Expected payoff: near full-attention accuracy on long documents, with about 16x less memory for
old context and attention cost that grows far slower than the context.

## Prior work it builds on
- **Native Sparse Attention** (DeepSeek, 2025): the same three branches (compressed blocks,
  selected blocks, sliding window), shown to match full attention at scale. It's the closest
  existing design.
- Landmark Attention (2023): one landmark per block; attending to it retrieves the block.
- Compressive Transformer (DeepMind, 2019): old memories squeezed into fewer slots.
- This repo: section 3 of README.md (KV trimming + the neural cache) is the untrained version.

## Fast weights: learning during use (added)
The trainable form of Hebbian/STDP learning (from the AMN review): each block gets a small
associative memory, M <- decay * M + w_t k_t v_t^T ("strengthen the link between what this token's
key and value"), read by later tokens as q_t M. The write strength is chosen per token; each head
forgets at its own learned rate (horizons from ~20 to ~3,000 tokens). **Per conversation:** M starts
empty for every sequence, so nothing a user or a web page writes persists; a persistent version
would have to go through the admin review queue like facts and fine-tuning. Relatives: DeltaNet,
test-time-training layers, Titans. Implemented chunk-parallel in `memory.py` (`--fast_heads`).

## Data: no download needed
Train on 4,096-token windows from the existing shards (fineweb-edu + code, 60/40). Long-document
evaluation from the validation shards: documents of 8k+ tokens hold 14% of English and 50% of
code validation tokens. Score bits per byte by position (0-1k, 1-2k, 2-4k, 4-8k); a model that
uses long context gets better further in. The 4-8k bucket tests going past the training length.

## Implementation
- `model/concept.py`: `ConceptAttention`, a drop-in for `gpt.Attention`, built on FlexAttention
  (`torch.nn.attention.flex_attention`), which runs block-sparse attention as one compiled Triton
  kernel:
  - **Concepts:** after each layer's K/V projection, each completed 64-token block gets a concept
    key/value by learned attention pooling over the block (a small learned query per head).
  - **Selection** (the "hippocampus"): queries score the concept keys; each query block (64
    queries share one choice, which keeps it block-sparse) takes its top 4 older blocks.
  - **One attention call:** keys = window + concepts of all older blocks + the selected blocks'
    raw tokens, expressed as a `BlockMask` built each step from the selection
    (`BlockMask.from_kv_blocks`), so nothing outside it is computed.
  - **Mixing:** per-head learned gates weigh the three branches (as in NSA).
  - **Causal:** a block's concept exists only once the block is complete, and only blocks wholly
    before the query's window are summarised or selected.
- `gpt.py`: `GPTConfig.attn_mode` = `full` | `window` | `concept`, plus `window`, `concept_block`
  and `concept_topk`; the blocks pick the attention class from it. Old checkpoints default to full.
- `train.py`: matching flags. `--seq_len 4096` already exists; the micro-batch picker adapts.
- `model/long_context.py` (new): the position-bucket evaluation above, plus KV bytes per token
  and decode memory at 8k.
- `sweep.py`: a `concept` group.

**Step 0:** confirm FlexAttention compiles and runs on this Windows Triton build (fused_ce
already uses Triton). If it doesn't, fall back to masked SDPA for a proof of concept (slower),
or move this to WSL2 with the rest of the Linux setup.

## Experiment
Base: `plain8` (the 8-block recipe, the fastest), 4,096-token sequences, 300M tokens per run,
equal tokens.

| run | attention | role |
|---|---|---|
| `full4k` | full attention over 4,096 | quality upper bound, biggest KV cache |
| `window512` | the last 512 tokens only | lower bound: what long context is worth |
| `concept` | window + concepts + top-4 expansion | the idea |
| `concept_noexpand` | window + concepts, no expansion | does expanding detail on demand matter? |
| `window_fast` | window + fast weights | does learning during use help on its own? |
| `concept_fast` | concept + fast weights | do they add up? |

Plus second seeds of `concept` and `concept_fast`. As built: sweep group `memory` (runs `m32k_*`),
then `long_context.py` at 8,192 tokens. Measured speed at 4,096: full attention 86.9k tok/s, concept +
fast weights 64.9k (10.2 GB); about 9 hours for the round.

About 5 hours of GPU time. Success: `concept` closes most of the gap between `window512` and
`full4k` on the 2-4k and 4-8k buckets, with a much smaller KV cache. `concept_noexpand` against
`concept` answers your hippocampus question directly: does re-expanding detail add anything
beyond keeping the gist?

## Verification
- Unit checks: causality (changing token t never changes earlier outputs, including the
  selection), FlexAttention output equal to a dense masked reference on small inputs, every
  parameter gets gradient, old checkpoints load.
- A position-bucket sanity check: `full4k` should beat `window512` beyond 512 tokens, or the
  evaluation isn't measuring long-context use.

## Later, if it works
- Decoding with a concept KV cache in fastdecode.py: the inference-memory payoff.
- Longer training sequences (8-16k) and longer documents (books).
- Combine with the sparse module bank, if round 7 finds a good variant.
