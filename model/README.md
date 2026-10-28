# worksAI model

A GPT language model trained on fasttok's ec200k vocabulary family, built to find the best
trade-off between **weight size** and **KV cache** for accuracy and speed on one consumer GPU
(RTX 4070, 12 GB, Windows).

Accuracy is always reported in **bits per byte** (bpb) of UTF-8 text, which doesn't depend on
the tokenizer, so models with different vocabularies compare fairly. Every number below was
measured on this machine.

## Files

| file | what it does |
|---|---|
| `tokenizer.py` | ctypes wrapper over `fasttok.dll`. `Tokenizer(n_ranks)` loads a rank prefix of ec200k (named sizes in `VOCABS`). Parallel batch encoding: 221 MB/s vs tiktoken's 49 MB/s. |
| `prepare_data.py` | Downloads FineWeb-Edu (English) and github-code-clean (code) and writes token shards per vocabulary (uint16 when the vocabulary fits). |
| `gpt.py` | The model, with knobs for vocabulary size, GQA (`n_kv_head`), cross-layer K/V sharing (`kv_share`), value residuals and U-net skips. Pluggable KV caches. |
| `fused_ce.py` | Fused LM head + cross-entropy (Triton kernel + cuBLAS). The `[tokens, vocab]` logits are never materialized. |
| `muon.py` | Muon optimizer with batched Newton-Schulz and per-block (q/k/v) orthogonalization. |
| `train.py` | Training: torch.compile, cuDNN attention, auto micro-batch, VRAM guard, WSD schedule, `--hours` budget, checkpoint/resume. |
| `sweep.py` | Equal-time training sweeps and the summary table. |
| `analyze.py` | Weight/KV memory, accuracy under int8/int4 weights and fp8/int8/int4 KV, CUDA-graph decode speed. |
| `cache_lab.py` | KV-cache trimming policies, the neural cache, and the trimmed-KV + neural-cache hybrid. |
| `fastdecode.py` | CUDA-graph decoding with a ring-buffer KV cache (streaming past the trained context) and the neural cache. |
| `fp8.py` | FP8 GEMMs for the transformer's linear layers (`train.py --fp8 1`): +22% tokens/s on this card. |
| `brain.py` | Experimental modular model (`train.py --arch brain`): a trunk, a bank of small modules where each token fires its top-k, a product-key memory, and a critic that estimates the draft's error ahead of final gate blocks. Section 7. |
| `brain_analysis.py` | What a trained brain model learned: which tokens each module fires on, how well the critic predicts error, memory use. |
| `forgetting.py` | Fine-tunes a checkpoint on crawled documentation (Python docs, Rust book) and measures what it learns there against what it loses on English and code. |
| `memory.py` | Long-context memory (`train.py --attn_mode window|concept`, `--fast_heads`): concept memory (window + one learned summary per older block + the most-attended blocks expanded back to full detail, block-sparse FlexAttention) and Hebbian fast weights that learn within a sequence and reset after it. See `PLAN_concept_memory.md`. |
| `long_context.py` | Bits per byte by position (0-1k ... 4-8k) in long validation documents: does a model use distant context? |
| `generate.py` | Sample text from a checkpoint. |
| `bench.py`, `test_fused_ce.py` | Throughput benchmarks and the fused-loss correctness test. |

## Setup (Windows, already done on this machine)

```powershell
# Python 3.12 venv managed by uv, kept in the repo (.venv, .python are gitignored)
python -m venv $env:TEMP\uvboot; & $env:TEMP\uvboot\Scripts\pip install uv
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.python"
& $env:TEMP\uvboot\Scripts\uv venv .venv --python 3.12 --seed
& $env:TEMP\uvboot\Scripts\uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu130
& $env:TEMP\uvboot\Scripts\uv pip install --python .venv\Scripts\python.exe numpy tiktoken pyarrow huggingface_hub triton-windows
```

Build fasttok with Visual Studio's clang-cl. `CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS` is needed
because the C API has no `__declspec(dllexport)`; without it `fasttok.dll` exports nothing:

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang-cl ^
      -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON "-DCMAKE_CXX_FLAGS=/EHsc /clang:-march=native /clang:-O3"
cmake --build build
```

## Usage

```powershell
cd model
..\.venv\Scripts\python.exe prepare_data.py --source fineweb-edu --files 1 --vocab ec32k
..\.venv\Scripts\python.exe prepare_data.py --source code --files 4 --vocab ec32k
..\.venv\Scripts\python.exe train.py --vocab ec32k --preset small --hours 8 --run main
..\.venv\Scripts\python.exe generate.py runs\main\ckpt.pt --prompt "def merge_sort(items):"
..\.venv\Scripts\python.exe analyze.py runs\main\ckpt.pt
..\.venv\Scripts\python.exe cache_lab.py runs\main\ckpt.pt --trim --hybrid
```

## GPU efficiency (measured)

- bf16 GEMM ceiling on this card: ~61 TFLOPS. FP8 `_scaled_mm`: 124 TFLOPS.
- The Windows PyTorch build has **no FlashAttention**. cuDNN attention was 1.56x faster than
  the memory-efficient kernel that SDPA picks by default on sm_89, so training pins cuDNN.
- Fused LM head + CE at d=768, V=200k: **~50 TFLOPS** (82% of the GEMM ceiling) in <3 GB.
  The unfused version runs out of memory.
- `allow_bf16_reduced_precision_reduction` (on by default in PyTorch) let cuBLAS reduce
  split-K partial sums in bf16, giving **1-2% relative error** in the LM-head input gradient.
  The fused head uses an fp32-output GEMM, which is both exact (2e-6) and faster here, and
  training turns the flag off.
- Training MFU against the measured ceiling: 68-74% (12-layer/768 model, 32k-200k vocabularies).
- On Windows, CUDA silently spills to system RAM when VRAM runs out (10x slower, no error).
  `train.py` caps the allocator so this becomes a clean out-of-memory error instead.

## Results

All runs: 12 layers x 768 wide, trained for the same **20 minutes** on this GPU (131,072-token
batches, Muon + AdamW, WSD schedule), evaluated on the same validation text (60/40 English/code
mix by rows). bpb = bits per byte (lower is better).

### 1. Vocabulary size (weight size vs tokens per byte)

| vocab | params | weights (bf16) | bytes/token | train tok/s | text/s | eng bpb | code bpb | mix bpb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **ec32k** | **110M** | **210 MiB** | 3.94 | 61.2k | 241 KB | 1.332 | 1.073 | **1.228** |
| ec16k | 98M | 186 MiB | 3.64 | 67.3k | 245 KB | 1.338 | 1.089 | 1.239 |
| ec64k | 135M | 258 MiB | 4.19 | 51.8k | 217 KB | 1.340 | 1.155 | 1.266 |
| ec100k | 162M | 309 MiB | 4.30 | 44.1k | 190 KB | 1.359 | 1.232 | 1.308 |
| ec200k | 239M | 455 MiB | 4.40 | 30.0k | 132 KB | 1.450 | 1.568 | 1.497 |

ec32k wins: 2.2x smaller than ec200k and 22% better at equal training time, for 11.5% more
KV cache per byte of text. The 200k embedding is 64% of the model's weights and 64% of its
training FLOPs at this size. (Short runs favor smaller models; longer training narrows the
gap, but not by 22%.)

### 2. Quantization (ec32k model)

| | mix bpb | change | size |
|---|---:|---:|---|
| bf16 | 1.2512 | | 210 MiB weights, 36 KiB KV/token |
| int8 weights | 1.2512 | +0.00% | 105 MiB |
| int4 weights (int8 embedding) | 1.2637 | +1.00% | 64 MiB |
| fp8 KV cache | 1.2513 | +0.01% | 18 KiB/token |
| int8 KV cache | 1.2512 | -0.00% | 18 KiB/token |
| int4 KV cache | 1.2524 | +0.09% | 9 KiB/token |

Decode (CUDA graph, context ~512): 868 tok/s at batch 1, 9,089 tok/s at batch 32.

### 3. KV trimming and the neural cache (ec32k model, turns of 128 tokens, 1024-token rows)

Trimming the KV cache after each turn, change in bpb vs the full cache:

| KV budget | English (window / snap / pool) | code (window / snap / pool) |
|---:|---|---|
| 128 | +1.65% / +1.53% / +2.25% | +7.11% / +7.18% / +9.42% |
| 256 | +0.64% / +0.66% / +1.08% | +3.16% / +3.13% / +4.73% |
| 512 | +0.11% / +0.11% / +0.23% | +0.71% / +0.63% / +1.23% |

Attention sinks matched plain windowing to within 0.03% (QK-norm models don't seem to form
sinks), and score-based selection (snap) barely beats it. Pooling old K/V is always worst.

The **neural cache** (one 128-value vector per token, mixed in word by word through a learned
gate) on top of a trimmed KV cache, vs the full KV cache:

| config | English | code | KV-side memory at 1024 tokens |
|---|---:|---:|---:|
| full KV + neural cache | -0.94% | -2.56% | 102% |
| 512 KV slots + neural cache | -0.83% | -2.31% | 51% |
| 256 KV slots + neural cache | -0.43% | -1.45% | 26% |
| **128 KV slots + neural cache** | **+0.09%** | **-0.23%** | **13%** |

A 7.6x smaller cache keeps full accuracy, and stacked with int4 KV storage that is ~30x less
cache memory at equal accuracy, with no retraining. `generate.py --kv_window 128 --neural_cache`
runs this as streaming generation with fixed memory.

### 4. Architecture (ec32k, same 20-minute budget)

| variant | weights | KV/token | train tok/s | eng bpb | code bpb | mix bpb | vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (MHA, bf16) | 210 MiB | 36 KiB | 61.2k | 1.332 | 1.073 | 1.228 | |
| **value residual + U-net** | 210 MiB | 36 KiB | 60.3k | 1.285 | 0.979 | **1.162** | **-5.4%** |
| **FP8 GEMMs** | 210 MiB | 36 KiB | **75.1k** | 1.301 | 1.048 | **1.200** | **-2.3%** |
| GQA, 2 KV heads | 192 MiB | 12 KiB | 66.4k | 1.331 | 1.083 | 1.232 | +0.3% |
| MQA + 2-layer K/V sharing | 185 MiB | 3 KiB | 68.5k | 1.342 | 1.103 | 1.246 | +1.5% |
| MLA (latent 96 + RoPE 32) | 188 MiB | 3 KiB | 65.6k | 1.384 | 1.153 | 1.292 | +5.2% |

- Value residuals + U-net skips are the largest single gain, at no memory cost.
- FP8 doesn't hurt accuracy; its 22% speedup becomes 2.3% better bpb in the same time.
- At the same 3 KiB/token cache, MQA + cross-layer sharing beats MLA by a wide margin at this
  scale and training length. GQA with 3x less cache is almost free.

### 5. Streaming decode (ec32k model, CUDA graph, batch 1)

Greedy CUDA-graph decoding agrees 100% with a full forward pass. With a ring-buffer KV cache,
generation ran 3,000 tokens (3x the trained context) at:

| KV cache | tok/s |
|---|---:|
| 1024-slot window | 639 |
| 128-slot window | 1,005 |
| 128-slot window + neural cache | 902 |

### 6. Combined recipe (ec32k + value residual + U-net + FP8, same 20-minute budget)

| K/V cache | weights | KV/token | train tok/s | eng bpb | code bpb | mix bpb |
|---|---:|---:|---:|---:|---:|---:|
| MHA | 210 MiB | 36 KiB | 73.4k | 1.259 | 0.964 | 1.1410 |
| **GQA, 2 KV heads** | **192 MiB** | **12 KiB** | **80.2k** | 1.257 | 0.968 | **1.1414** |
| MQA + 2-layer sharing | 185 MiB | 3 KiB | 83.2k | 1.268 | 0.988 | 1.1556 |

Under the combined recipe, GQA's 3x smaller cache costs nothing (+0.04%), and the 12x smaller
MQA + sharing cache costs 1.3%.

### 7. Brain architecture (`brain.py`, ec32k, combined recipe)

The idea: a trunk, then a bank of small modules where each token fires its top-k (specialisation
is learned, not assigned), a product-key memory, and a critic that estimates the draft's error
before final gate blocks. Each piece was tested against an 8-block model (`plain8`, the same
stack without the extras) and the 12-block recipe from section 6 (`ref`).

**Equal tokens** (100M tokens each, each with its own full learning-rate schedule;
`sweep.py brain`, runs `*_t100m`):

| model | params | mix bpb | train tok/s |
|---|---:|---:|---:|
| ref (12 blocks) | 101M | **1.1348** | 77.1k |
| plain8 (8 blocks) | 76M | 1.1490 | 107.0k |
| plain8 + module bank (1 shared + top-2 of 7) | 90M | 1.1504 | 78.3k |

**Step 300** (39M tokens, all runs still at full learning rate; run-to-run noise about 0.017):

| run | mix bpb | what it showed |
|---|---:|---|
| ref | 1.339-1.359 | best per token |
| plain8 | 1.350-1.352 | |
| + module bank (any balancing) | 1.353-1.357 | neutral per token, 25-35% slower |
| + bank + memory + critic (full) | 1.395 | worse |
| + critic trained jointly | 1.426 | its loss pushed into the main stack (3x the gradient it sent its own weights) |
| + critic detached (observes only) | 1.353 | damage gone, no gain; 6-9% slower |
| + draft loss | +0.007 | no effect |

**A critic that earns its say** (`--critic_gate`, equal tokens, 100M, bank3 base). The critic
starts out only observing (its output scaled by 0 for 100 steps), then gets a say equal to its
trust (its record: correlation of its error estimates with the draft's real errors, 0 at 0.3, 1 at
0.6), times (gate 2) a per-token confidence from a predicted variance (Gaussian NLL):

| critic | mix bpb | final say | critic switched off (Eng / code) |
|---|---:|---:|---|
| none (bank3) | 1.1504 | | |
| full say from step 0 (detached) | 1.1513 | 1 | +0.016 / +0.021 |
| track record only (gate 1) | 1.1506 | 0.92 | |
| **track record x confidence (gate 2)** | **1.1475** | 0.41 | +0.005 / +0.004 |

Trust stayed 0 through step 100, then rose with the record (say 0.04 at step 120, 0.27 at 200,
0.33 at 300; trust 0.90 at the end). The confidence did what it should: the critic speaks up most
where its estimate is accurate (say vs its own miss: corr -0.37 English, -0.51 code; say 0.72-0.84
on the tokens it expects to be easy, 0.13-0.27 on the hardest). Gate 2 is the only critic that
beat having no critic, by 0.003-0.004 bpb; that is about the size of run-to-run noise at this
budget, so it needs seed replicates before it counts. Odd detail: the gated critic's own
contribution is smaller than the ungated one's, so the gain (if real) comes from how the rest of
the model trains around a critic that stays quiet until it is reliable, not from bigger corrections.

At 1B tokens (one seed each) the earned-say critic trailed early (+0.005 at step 500), crossed
over near step 2,000 and stayed ahead at every evaluation after: 0.9319 vs 0.9343 at the end
(-0.0024 bpb, -0.26%; English 1.0778 vs 1.0795, code 0.7131 vs 0.7166). Consistent, but within
single-seed noise; it needs replicates at 1B to confirm.

**Sparse modules** (`--module_mode`, equal tokens, 100M; bank3 shape: 1 shared + top-2 of 7). Each
routed module computes only on its own tokens, with 1.25x its fair share of slots per sequence:

| module bank | mix bpb, seed 1337 / 7 | train tok/s | dropped (over capacity) |
|---|---|---:|---:|
| dense (every module on every token) | 1.1504 / 1.1480 | 78.3k | |
| **B:** attention over its own tokens only | 1.1463 / 1.1498 | 91.2k | 3% / 18% |
| **C:** feed-forward only, its own tokens | 1.1514 / 1.1508 | 97.0k | 5% / 2% |
| plain8 (no bank) | 1.1490 | 107.0k | |

Both sparse modes keep the dense bank's accuracy (all within noise; B's mean is the lowest) while
cutting its speed cost: B is 17% faster than the dense bank, C 24%. B learned fine despite each
module seeing only a fragment of the text. Dead routed modules still occur in some seeds (B seed
7: two, which pushed the others past capacity and dropped 18% of firings).

What the parts learned (`brain_analysis.py` on the full model): modules specialised without
being told to (numbers, symbols and tables, technical words, verbs/line ends, indentation, word
fragments); the critic's error estimate correlated 0.55 (English) / 0.64 (code) with the real
error and the gate used it most where the estimate was high; the memory gate stayed 99% open.

**Balancing the module bank:**

| method | result |
|---|---|
| Switch balance loss | 6-8 of 8 modules alive (a dead one in some seeds) |
| router bias on raw logits | collapsed onto 2 modules: the router's logits outgrew the bias |
| router bias on sigmoid scores (DeepSeek-V3) | one module fired on every token; evicting it passed that role from module to module and the loss suffered (4.56 vs 4.03 at step 160) |
| 1 shared module + bias | the hand-offs continued among the routed modules |
| **1 shared module + balance loss** | **stable**; 6-7 of 7 routed alive |
| + bias on top | did not revive a dead module |

Training wants one module active on every token (a general-purpose path); giving it one is what
made routing stable. A module that dies early stays dead under every method tried.

**Forgetting** (`forgetting.py`): fine-tune for 300 steps (9.8M tokens, about 3 passes) on the
crawled documentation (Python docs, Rust book; 3.3M train / 0.27M held-out tokens), then measure
held-out docs, English and code. Forgetting grows faster than learning, so the fair column is the
loss at the same amount learned:

| model, weights trained | docs learned | English lost | code lost | lost at 0.098 learned |
|---|---:|---:|---:|---:|
| plain8, all | 0.226 | +0.201 | +0.156 | +0.022 |
| plain8, blocks 3-4 (12.6M) | 0.158 | +0.074 | +0.057 | +0.022 |
| bank3, all | 0.258 | +0.238 | +0.167 | +0.021 |
| bank3, module bank only (14.7M) | 0.143 | +0.069 | +0.043 | +0.030 |
| **plain8, all + 25% replay** | **0.234** | **+0.032** | **+0.024** | **+0.012** |

Updating only the modules forgets about 40% more for the same learning, and learns slower.
Mixing 25% general text into the fine-tuning batches cut end-of-run forgetting 6x (1.8x at equal
learning) while learning slightly more: that, not the architecture, is what protects old knowledge.

**Verdict:** at this scale the modular pieces are neutral (bank, detached critic) or harmful
(memory, joint critic, bank-only updates), and they cost 25-35% throughput. The one possible
exception is the critic that earns its say with per-token confidence (-0.003 bpb vs no critic,
within noise until replicated). The 12-block plain
model stays the recommendation. Replay should be part of any fine-tuning run.

Measurement notes: runs named `*_stockclk` ran before the GPU's overclock profile was applied;
the others ran with it (no faster: the card sits at its 200 W limit). `train.py` now benchmarks
a sustained 2 s GEMM at startup; the old 0.2 s burst could read before boost (28 TFLOPS on an
idle card for `b32k_ref`), so its "GPU shared" flag in the summary is a false positive.

## Recommended configuration

```powershell
python train.py --vocab ec32k --preset small --n_kv_head 2 --value_residual 1 --unet 1 --fp8 1 --hours 8
```

Against the starting point (ec200k vocabulary, plain multi-head attention, bf16) at equal
training time on this GPU:

| | ec200k baseline | recommended | |
|---|---:|---:|---|
| validation bpb (mix) | 1.497 | 1.141 | 23.8% better |
| weights | 455 MiB | 192 MiB (96 MiB int8, lossless) | 2.4x smaller (4.7x int8) |
| KV cache per token | 36 KiB | 12 KiB (3 KiB as int4, +0.09%) | 3x smaller (12x int4) |
| training throughput | 30.0k tok/s | 80.2k tok/s | 2.7x faster |

At inference, add `--kv_window 128 --neural_cache` to generation to cap the cache at 128 slots
with no accuracy loss vs the full context (section 3). For the tightest cache budget, use
`--n_kv_head 1 --kv_share 2` instead (3 KiB/token, +1.3% bpb).

## Final model (12-hour run)

`runs/final_ec32k_gqa/ckpt.pt`: ec32k vocabulary, 12 layers × 768, GQA (2 KV heads), value
residuals, U-net skips, FP8 training. 25,849 steps, 3.39B tokens, 5.1 h of training after a
crash and resume (it lost about 30 minutes to a Windows bugcheck and resumed from a checkpoint).

| | final | 20-minute run, same recipe | 20-minute ec200k baseline |
|---|---:|---:|---:|
| validation bpb (60/40 mix) | **0.864** | 1.141 | 1.497 |
| English / code bpb | 1.011 / 0.645 | 1.257 / 0.968 | 1.450 / 1.568 |

| deployment | bpb | change | size |
|---|---:|---:|---|
| bf16 | 0.8642 | | 192 MiB weights, 12 KiB KV/token |
| int8 weights | 0.8650 | +0.09% | 96 MiB |
| int4 weights (int8 embedding) | 0.9210 | +6.57% | 60 MiB |
| int8 KV cache | 0.8643 | +0.00% | 6 KiB/token |
| int4 KV cache | 0.8677 | +0.40% | 3 KiB/token |

Decode (CUDA graph): 882 tok/s at batch 1, 10,482 tok/s at batch 32.

Two things changed with full training. int4 weights went from +1.0% to +6.6%: a fully trained
model packs more into its weights, so int8 is the format to deploy. And the neural cache's gain
with a full context fell from -1.5% to -0.05%: the model learned to copy from its own context.
The neural cache now matters mainly alongside a trimmed KV cache.
