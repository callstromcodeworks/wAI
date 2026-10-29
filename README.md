# fasttok

A fast, header-only C++17 BPE tokenizer that produces **exactly the same token ids as
OpenAI's [tiktoken](https://github.com/openai/tiktoken)** for `o200k_base` (GPT-4o / GPT-5
family), `cl100k_base`, `p50k_base` and `r50k_base`.

Single-threaded it runs **2.8–11× faster than tiktoken**. Multi-threaded, it splits one
large text across cores with output identical to the single-threaded encoder.

It also ships its own vocabulary, **`ec200k_base`** (`vocab/ec200k_base.tiktoken`). It has
200k tokens trained for English prose and source code, and needs **3.7% fewer tokens than
o200k_base and cl100k_base** on held-out English and code. Any rank prefix of it is a
smaller exact BPE vocabulary (`Tokenizer::ec_prefix`). See [vocab/README.md](vocab/README.md).

[model/](model/README.md) trains GPT models on these vocabularies and measures the trade-off
between weight size and KV cache.

## Running the experiments on a rented Linux GPU

This tree is the tokenizer and the model code only: no agent server, crawl cluster, keys or user
data. `run.py` does everything:

Recommended image: NVIDIA's NGC PyTorch container (`nvcr.io/nvidia/pytorch:26.09-py3` or newer; pick a
host whose driver supports its CUDA version). Disk: 100 GB (the image is ~20 GB, data ~16 GB,
checkpoints for the default queue ~17 GB).

```bash
git clone https://github.com/callstromcodeworks/wAI.git && cd wAI
pip install pyarrow huggingface-hub tiktoken   # NGC: only these (torch/numpy are pinned there)
# other images: apt-get install -y cmake build-essential && pip install -r requirements.txt
python run.py setup                           # build fasttok, check GPU, packages, vocabulary hash
python run.py data --download                 # or copy model/data/ec32k/ from home (8.3 GB)
python run.py test                            # correctness suites, in parallel
python run.py train --queue memory,critic1b --parallel 4
python run.py status                          # from another shell, any time
python run.py eval                            # long-context table + sweep summary
```

`python run.py all` runs setup, data, test, train and eval in order. FP8 checks the torch build on first use:
some builds (PyPI 2.15) miscompile one FP8 GEMM, and training then routes it through a safe op
(it prints a line saying so; ~7% slower, correct). Training runs several
experiments at once, each with an equal share of VRAM, checkpointing every 200 steps; if the box
is interrupted, the same command resumes them. Use `tmux` (or `nohup`) so a dropped SSH session
doesn't stop the queue. Bring back `model/runs/` (checkpoints, logs, `long_context.txt`,
`sweep_summary.txt`) when done.

## Benchmarks

Throughput in MB/s of UTF-8 input (best of 5). Measured on a 2-core Intel Xeon VM at
2.8 GHz with gcc 13 `-O3 -march=native`, against tiktoken 0.14.0 on Python 3.13. Runs
on this VM varied by about ±10%.

| encoding | corpus | tiktoken 1T | tiktoken 1T (numpy) | **fasttok 1T** | speedup | tiktoken 2T batch | **fasttok 2T** | speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| o200k_base | English books, 14.8 MB | 16.0 | 17.6 | **96.9** | 5.5× | 21.6 | **185.5** | 8.6× |
| o200k_base | Python source, 8.0 MB | 12.6 | 13.6 | **117.3** | 8.6× | 17.8 | **197.9** | 11.1× |
| o200k_base | 380-language UDHR, 5.7 MB | 12.2 | 13.4 | **37.2** | 2.8× | 20.0 | **77.4** | 3.9× |
| cl100k_base | English books, 14.8 MB | 11.0 | 12.3 | **106.8** | 8.7× | 15.6 | **175.3** | 11.3× |
| cl100k_base | Python source, 8.0 MB | 9.1 | 9.7 | **109.7** | 11.3× | 11.4 | **190.2** | 16.7× |
| cl100k_base | 380-language UDHR, 5.7 MB | 9.6 | 10.5 | **46.3** | 4.4× | 15.4 | **93.5** | 6.1× |

- **Corpora:** nine distinct Project Gutenberg novels, the Python 3.13 standard library,
  and the Universal Declaration of Human Rights in about 380 languages (NLTK `udhr2`).
- **Speedup columns** compare against the faster of the two single-threaded tiktoken
  calls, and against `encode_ordinary_batch(num_threads=2)` over 64 chunks.
- **fasttok timing** excludes Python. tiktoken's includes building the Python list or
  numpy array, so part of the gap is binding overhead, not tokenization. The numpy
  column narrows that.
- **Multilingual text gains least.** About half its pieces are not a single token and
  need actual BPE merging, which is a chain of dependent table lookups.
- **Very long pieces:** tiktoken is not quadratic on them either. On 200,000 spaces or
  200,000 random letters, fasttok is about 1.5–1.9× faster.
- **Load time:** 150 ms for o200k_base (tiktoken: 338 ms) and 64 ms for cl100k_base
  (tiktoken: 150 ms).

Reproduce with `python3 bench/compare.py --bin build/fasttok --vocab-dir VOCAB_DIR FILES...`.

## Correctness

`tests/test_vs_tiktoken.py` runs fasttok and tiktoken side by side and requires identical
ids (`--encodings ec200k_base` covers the bundled vocabulary too). It found zero mismatches in **1.34 million cases per encoding** (6.7 million total across all five).
The cases are:
- every Unicode code point (all 1.1 million), embedded in a template that puts it next to
  letters of each case, digits, apostrophes, contractions, tabs, CR/LF and itself;
- whole corpora (English, code, multilingual, pathological long runs), plus 3,000
  random slices of each;
- 200,000 random strings built from characters picked to hit every regex class boundary
  (all letter, mark and number categories, Unicode spaces, `'ſ`, emoji, unassigned and
  private-use code points), plus long random strings;
- special tokens, with all specials allowed;
- `encode_ordinary_parallel` checked against the sequential encoder, and decode round trips.

`tests/unit.cpp` covers the C++ API. It checks `count`, the batch and parallel paths, the
cache off, allow-listed specials, empty input, id bounds, and byte-exact round trips of
random invalid UTF-8. It passes under AddressSanitizer and UBSan.

**About Unicode versions:** tiktoken's Rust regex crate classifies characters with
**Unicode 16.0** data. Python 3.13's own tables are Unicode 15.1, and with those about
0.4% of the fuzz cases mismatched (on scripts added in 16.0). The shipped table is
generated from `unicodedata2==16.0.0`. If a future tiktoken moves to a newer Unicode
version, regenerate it with `tools/gen_unicode_tables.py --ucd unicodedata2` against
the matching `unicodedata2` release and re-run the test.

## How it gets its speed

1. **No regex engine.** Each encoding's pre-tokenization regex is hand-compiled into a
   scanner that reproduces its backtracking semantics exactly, including details like
   the regex `(?i)` matching U+017F `ſ`. It has an ASCII fast path, a 44 KB two-stage
   Unicode class table, and SSE2 skipping of letter and space runs.
2. **Prefetch.** Pieces are scanned 16 at a time, and each one's hash slot is prefetched
   before the lookup.
3. **One integer compare per lookup for most pieces.** About 90% of English pieces are
   whole tokens. Tokens of up to 7 bytes are keyed by their packed bytes, so a lookup
   compares one 64-bit integer and never reads the token bytes.
4. **Integer merge tables.** BPE looks up `(left id, right id) → merged id`, not byte
   slices. The tables are derived at load time from every split of every token. Pairs
   with both ids < 512 (about 72% of lookups) hit a dense 1 MB table. The rest sit in
   small hash groups per left token, which keeps related merges in the same cache lines.
5. **Per-thread memo.** A 512 KB cache of recent BPE results skips re-merging repeated
   words. It hits about 56% of the pieces that need merging on the multilingual corpus.
6. **Long pieces** switch to a heap and linked list (O(n log n)), with the same
   leftmost-wins tie-breaking as tiktoken.
7. **Parallel splitting.** A single text is split only right after a `\n` that is
   followed by an ASCII letter. No regex alternative can span that point, so the output
   is unchanged.

Things I tried and measured as slower, so they're not in the code:
- a branchless or SIMD minimum in the merge loop, which loses to store-forwarding stalls;
- fixed-position linked-list merging;
- interleaving the merges of several pieces to overlap memory latency, which loses to
  branch mispredictions;
- one global merge hash table, about 15–30% slower than grouping by left token.

## Build

```sh
cmake -S . -B build && cmake --build build -j
# or simply
g++ -std=c++17 -O3 -march=native -Iinclude -o build/fasttok src/fasttok_cli.cpp -pthread
```

Vocabulary files are OpenAI's `.tiktoken` files, for example
`https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken`. The SHA-256
hashes tiktoken expects are:
- o200k_base: `446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`
- cl100k_base: `223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7`

## Usage

```cpp
#include "fasttok.hpp"

auto tok = fasttok::Tokenizer::o200k_base("o200k_base.tiktoken");
// or cl100k_base(...), p50k_base(...), r50k_base(...), ec200k_base("vocab/ec200k_base.tiktoken")

std::vector<uint32_t> ids = tok.encode_ordinary("Hello world!");  // like tiktoken encode_ordinary
size_t n = tok.count(text);                                        // no allocation
auto with_special = tok.encode("a<|endoftext|>b");                 // allowed_special="all"
auto some = tok.encode(text, {"<|endoftext|>"});                   // others encoded as text
auto par = tok.encode_ordinary_parallel(big_text);                 // identical output, all cores
auto docs = tok.encode_ordinary_batch(views);                      // many documents, thread pool
std::string back = tok.decode(ids);
```

The `Tokenizer` is immutable after loading and safe to share across threads. Other
encodings using the same patterns (for example `o200k_harmony`) can be built with
`Tokenizer::load(path, fasttok::Pattern::O200K, {{"<|start|>", 200006}, ...})`.

**CLI:**

```sh
fasttok o200k_base o200k_base.tiktoken encode file.txt
fasttok o200k_base o200k_base.tiktoken count  file.txt
fasttok o200k_base o200k_base.tiktoken bench  file.txt 5 2   # iterations, threads
```

**C ABI:** `src/fasttok_capi.cpp` exposes `ft_new`, `ft_encode_batch`, `ft_count`,
`ft_decode` and `ft_free`, for use from Python via ctypes or any other FFI. The test
harness shows how.

## Behaviour notes and limits

- **Input** is UTF-8. Invalid bytes are tokenized one at a time as "other" characters
  and still decode back exactly. tiktoken only accepts valid strings.
- **Special tokens:** `encode(text, allowed)` treats non-allowed special-token text as
  ordinary text, which matches tiktoken with `disallowed_special=()`. It never raises.
- **Platform:** a 64-bit little-endian target is required, and SSE2 is used when
  available. Vocabularies can have up to 2²¹ ids.
- **Memory:** each thread keeps a 512 KB memo per tokenizer it has used, for up to four
  tokenizers. Disable it with `set_cache_enabled(false)`, or resize it with
  `-DFASTTOK_CACHE_BITS=N`.

## Layout

```
include/fasttok.hpp           the tokenizer (header-only)
include/fasttok_unicode.hpp   generated Unicode 16.0 class tables
tools/gen_unicode_tables.py   table generator
src/fasttok_cli.cpp           CLI and benchmark
src/fasttok_capi.cpp          C ABI
tests/test_vs_tiktoken.py     differential test against tiktoken
tests/unit.cpp                API tests
bench/compare.py              benchmark against tiktoken
vocab/ec200k_base.tiktoken    English + code 200k vocabulary (see vocab/README.md)
vocab/training/               the pipeline that trained it (data, counting, BPE, evaluation)
```
