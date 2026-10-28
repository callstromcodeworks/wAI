# ec200k_base: a 200k vocabulary for English prose and source code

`ec200k_base.tiktoken` is a byte-level BPE vocabulary of 200,000 tokens, trained to use
**as few tokens as possible on English prose and source code**. On held-out text it never
saw, it needs **3.7% fewer tokens than both o200k_base and cl100k_base** on an even mix of
English and code. It wins on every English category and every programming language tested.

It is a standard `.tiktoken` file. It loads in fasttok, and in OpenAI's tiktoken with the
cl100k pattern, and both produce identical token ids.

```cpp
auto tok = fasttok::Tokenizer::ec200k_base("vocab/ec200k_base.tiktoken");
auto ids = tok.encode_ordinary(text);
```

| | |
|---|---|
| Pre-tokenizer | `fasttok::Pattern::CL100K` (cl100k_base's regex) |
| Regular tokens | ids 0–199,999. Ids 0–100,255 are **byte-identical to cl100k_base**; 100,256–199,999 are new |
| Special tokens | `<\|endoftext\|>` 200000, `<\|fim_prefix\|>` 200001, `<\|fim_middle\|>` 200002, `<\|fim_suffix\|>` 200003, `<\|endofprompt\|>` 200004 |
| SHA-256 | `7a17729c50552022f725e9ea880f40697933a7318c9e9cb2e0d50cd7e45b0b15` |

Using it from tiktoken:

```python
tiktoken.Encoding(name="ec200k_base", pat_str=CL100K_PATTERN,
                  mergeable_ranks=tiktoken.load.load_tiktoken_bpe("ec200k_base.tiktoken"),
                  special_tokens={"<|endoftext|>": 200000, ...})
```

## Smaller vocabularies from rank prefixes

The first N ranks of a BPE vocabulary are exactly the vocabulary that training would have
produced after N merges, so any prefix of `ec200k_base.tiktoken` is itself a complete
vocabulary. `Tokenizer::ec_prefix(path, N)` (C API `ft_new_ec_prefix`) loads one and
renumbers the five specials to follow it (N .. N+4). `tests/test_vs_tiktoken.py
--encodings ec200k_base:N` checks it against tiktoken loaded with the same truncated ranks.

This is the lever between a model's weight size and its sequence length. The embedding is
vocabulary × d_model, while a smaller vocabulary needs more tokens (KV cache, compute) per
byte. Bytes per token on 20 MB each of FineWeb-Edu and github-code-clean documents:

| prefix N | English | code | 60/40 mix | tokens vs N=200k | embedding at d=768 |
|---:|---:|---:|---:|---:|---:|
| 16,379 | 3.91 | 3.31 | 3.65 | +22% | 12.6M |
| 32,763 | 4.28 | 3.59 | 3.97 | +12% | 25.2M |
| 65,531 | 4.57 | 3.79 | 4.22 | +5.2% | 50.3M |
| 100,256 (= cl100k regular tokens) | 4.70 | 3.89 | 4.34 | +2.4% | 77.0M |
| 200,000 | 4.82 | 3.98 | 4.44 | — | 153.6M |

The 16k, 32k and 64k sizes leave room for the specials inside 16,384 / 32,768 / 65,536 ids,
so the last of them still fits uint16 token storage.

## Results

All numbers are bytes of UTF-8 per token on held-out text (higher is better). None of
these books, repositories or news corpora were used in training.

| Held-out set | size | o200k_base | cl100k_base | **ec200k_base** | vs o200k | vs cl100k |
|---|---:|---:|---:|---:|---:|---:|
| English: 40 Gutenberg books | 17.5 MB | 4.099 | 4.051 | **4.331** | −5.4% tokens | −6.5% |
| English: tech docs (Kubernetes, PostgreSQL, Flask, and other held-out repos' docs) | 10.7 MB | 4.109 | 4.121 | **4.324** | −5.0% | −4.7% |
| English: news (Reuters, Brown, ABC) | 18.6 MB | 4.429 | 4.368 | **4.463** | −0.8% | −2.1% |
| Python (Flask, Requests, Black, httpx, FastAPI, Click) | 2.8 MB | 4.581 | 4.604 | **4.790** | −4.4% | −3.9% |
| JavaScript/TypeScript (Express, Lodash, Axios, Chart.js, Prettier, three.js) | 7.2 MB | 3.820 | 3.837 | **3.941** | −3.1% | −2.6% |
| C/C++ (SQLite, nlohmann/json) | 8.1 MB | 3.529 | 3.556 | **3.650** | −3.3% | −2.6% |
| Java/Kotlin (Retrofit, Okio) | 2.0 MB | 4.015 | 4.123 | **4.310** | −6.8% | −4.3% |
| Go (Gin, Cobra) | 1.2 MB | 3.643 | 3.647 | **3.801** | −4.2% | −4.1% |
| Rust (Serde, Clap) | 1.4 MB | 4.212 | 4.223 | **4.364** | −3.5% | −3.2% |
| C# (Newtonsoft.Json) | 2.7 MB | 5.331 | 5.469 | **5.673** | −6.0% | −3.6% |
| PHP (Composer) | 2.9 MB | 4.394 | 4.423 | **4.545** | −3.3% | −2.7% |
| Ruby (Jekyll) | 0.3 MB | 4.217 | 4.266 | **4.395** | −4.1% | −2.9% |
| Swift (Vapor) | 0.8 MB | 4.482 | 4.511 | **4.719** | −5.0% | −4.4% |
| **All English** | 46.8 MB | 4.227 | 4.188 | **4.381** | −3.5% | −4.4% |
| **All code** | 29 MB | 3.991 | 4.025 | **4.151** | −3.9% | −3.0% |
| **50/50 English + code** | | 4.105 | 4.105 | **4.263** | **−3.7%** | **−3.7%** |

- **Encoding speed** with fasttok (1 thread) is *faster* than o200k: 125 vs 100 MB/s on
  English books, and 117 vs 111 MB/s on Python source. More pieces are whole tokens, so
  fewer need merging.
- **The trade-off:** this vocabulary spends nothing on other languages. On the
  380-language UDHR corpus it needs 30% more tokens than o200k, about the same as cl100k
  (2% fewer). Use o200k_base for multilingual text.

### How close to the limit is this?

fasttok's pre-tokenizer splits text into pieces before BPE runs, and no token can span
two pieces. So the best any vocabulary could do, even an infinite one, is one token per
piece. For the cl100k pattern on this held-out mix, that ceiling is 4.635 bytes/token.
ec200k reaches 4.263, which is 92% of it. The remaining 8% is mostly long code
identifiers and rare words that no 200k vocabulary can give a token each.

## How it was built

1. **Data (1.75 GB).**
   - 901 MB of source code from 61 large open-source repos in about 20 languages
     (CPython, Django, Linux, LLVM, React, VS Code, TypeScript, Go, Kubernetes, Rust,
     Kotlin, Spring, .NET, Rails, Laravel, Swift, Flutter, …).
   - 849 MB of English: 1,100 random English Project Gutenberg books (license
     boilerplate removed) and about 420 MB of modern technical documentation (MDN,
     GitHub, Azure, SQL Server, .NET and PowerShell docs, Pro Git, the Rust book, …, plus
     the docs inside the code repos).
   - Vendored, minified, generated and lock files are skipped. Exact duplicates are
     removed, and held-out files are removed from training. No repo contributes more than
     30 MB of code.
2. **Pieces** are produced by fasttok's own cl100k scanner, so training sees exactly the
   pieces the encoder will.
3. **Source-damped counts.** Piece frequencies are counted per source, with each book
   and each repo as one source. A piece's most frequent source may contribute at most
   2× its second-most frequent source (per byte). This keeps one repo's macros
   (`{{domxref(…)}}` in MDN), one project's name, or one novel's character names from
   taking slots that help nowhere else. Words common everywhere are unaffected.
4. **Seeded BPE.**
   - Training starts from cl100k_base's 100,256 tokens. They were learned on far more
     data than is available here, and a rank prefix of a BPE vocabulary is self-contained.
   - Every piece is first encoded with that prefix using fasttok's rank-based BPE.
   - Greedy BPE then continues on the source-damped counts up to 200,000 tokens. Each
     step merges the most frequent adjacent pair; ties go to the smallest pair of ids.
   - The merge order becomes the ranks, which is exactly how tiktoken and fasttok replay
     them.

### What was tried

Each setting was trained at the full 200k size and scored on the same held-out 50/50 mix.
Training takes about 5 seconds per run, so every option below was actually measured.

| Variant | 50/50 bytes/token |
|---|---:|
| o200k_base / cl100k_base (reference) | 4.105 / 4.105 |
| From scratch, r50k pattern | 4.058 (best on news, worst on code) |
| From scratch, o200k pattern | 4.227 |
| From scratch, cl100k pattern | 4.230 |
| Seeded with o200k's first 50k / 100k tokens, cl100k pattern | 4.232 / 4.217 |
| Seeded with cl100k's first 20k / 50k / 80k / all tokens | 4.240 / 4.244 / 4.248 / 4.248 |
| **+ source-damped counts, κ = 2 (final)** | **4.263** |
| κ = 1, 1.5, 4, 8; also capping the second source; grouping the Microsoft docs repos | 4.261–4.263 |
| Half the training data (same method) | −0.17% vs full |
| Code weighted 0.6× / 1.6× vs English | +0.07% / −0.12% |
| Minimum piece count 1 / 2 / 5 | within 0.05% |

Doubling the data, reweighting and pruning each moved the score by less than 0.2%. The
pre-tokenizer ceiling is the binding limit, so I stopped there.

## Verification

- **Trainer:** `bpe_train` was checked against `ref_train.py`, a slow, straightforward
  reference that recounts every pair each step, and produced byte-identical vocabularies
  in three configurations.
- **Encoder:** `tests/test_vs_tiktoken.py --encodings ec200k_base` compares fasttok
  against tiktoken loaded with the same file and pattern. It found 0 mismatches in
  1,338,437 cases: every Unicode code point in context, whole corpora and random slices,
  200k fuzz strings, special tokens, and parallel encoding.
- **Unit tests** (including AddressSanitizer and UBSan) cover ec200k_base.
- **Determinism:** re-running the final counting and training steps reproduces the
  identical file (same SHA-256).
- **Reproduction:** a fresh end-to-end run of `train.sh` in a clean directory re-cloned
  every repo at its then-current HEAD. It produced a vocabulary scoring 4.264 on the
  50/50 mix, against 4.263 for the shipped file.

## Rebuilding

```sh
vocab/training/train.sh PATH/TO/cl100k_base.tiktoken
```

This fetches all the data, assembles the corpora, counts, trains and prints the held-out
evaluation. Upstream repos and books change over time, so a fresh run reproduces the
method rather than this exact file.

`vocab/training/` contains:
- `ft_count.cpp`: piece counter using fasttok's scanner
- `bpe_train.cpp`: trainer, with `--seed`
- `ref_train.py`: reference trainer
- `robust_counts.py`: source damping
- `ft_eval.cpp`, `summary.py`: evaluation
- `ft_usage.cpp`: per-token usage
- `mix_counts.py`: weighted counts
- `assemble.py`, `fetch_repos.sh`, `fetch_books.py`, `repos.txt`: data pipeline
