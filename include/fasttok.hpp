// fasttok.hpp -- a fast, tiktoken-compatible byte-pair-encoding tokenizer.
//
// Produces exactly the same token ids as OpenAI's tiktoken for o200k_base,
// cl100k_base, p50k_base and r50k_base (verified by tests/test_vs_tiktoken.py), and
// ships its own English+code vocabulary, ec200k_base (vocab/ec200k_base.tiktoken).
//
// Why it is fast:
//   * No regex engine. Each encoding's pre-tokenization regex is compiled by hand
//     into a small scanner that reproduces the regex's backtracking semantics exactly,
//     with an ASCII/SSE2 fast path and a 2-stage Unicode class table for the rest.
//   * Pieces are scanned 16 at a time and their hash slots prefetched before lookup.
//   * A piece that is a whole token (the vast majority) costs one hash probe. Tokens of
//     up to 7 bytes are keyed by their packed bytes, so the probe compares one integer
//     and never touches the token byte arena.
//   * BPE merges look up (left id, right id) -> merged id in integer-keyed tables
//     derived from the vocabulary at load time, instead of hashing byte slices: a dense
//     1 MB table for ids < 512 and per-left-token groups for the rest.
//   * A small per-thread memo of recent BPE results skips re-merging repeated words.
//   * Long pieces (e.g. huge whitespace runs) use a heap + linked list: O(n log n).
//   * Single large texts can be split at provably safe boundaries and encoded in
//     parallel; batches of documents are spread over a thread pool.
//
// Requirements: C++17, x86-64 or other little-endian 64-bit target (SSE2 used when
// available). Header-only; also include fasttok_unicode.hpp, generated from Unicode
// 16.0 by tools/gen_unicode_tables.py -- the version tiktoken's regex engine uses.
// Vocabularies may have up to 2^21 token ids.
//
// Input is UTF-8. Invalid UTF-8 bytes are tokenized one byte at a time as "other"
// characters (tiktoken only accepts valid Unicode strings, so it has no equivalent).
// Tokenizer objects are immutable after loading and safe to share across threads.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <functional>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "fasttok_unicode.hpp"

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define FT_INLINE inline __attribute__((always_inline))
#define FT_NOINLINE __attribute__((noinline))
#define FT_LIKELY(x) __builtin_expect(!!(x), 1)
#define FT_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define FT_INLINE inline
#define FT_NOINLINE
#define FT_LIKELY(x) (x)
#define FT_UNLIKELY(x) (x)
#endif

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "fasttok assumes a little-endian target"
#endif

namespace fasttok {

using Rank = uint32_t;
inline constexpr Rank kNone = 0xFFFFFFFFu;

// Pre-tokenization pattern families used by OpenAI's encodings.
enum class Pattern {
  O200K,   // o200k_base, o200k_harmony
  CL100K,  // cl100k_base
  R50K,    // r50k_base, p50k_base, p50k_edit, gpt2
};

namespace detail {

using namespace ucd;
inline constexpr size_t NPOS = ~size_t(0);

static_assert(sizeof(size_t) == 8, "fasttok requires a 64-bit target");

// ---------------------------------------------------------------------------
// Character classification
// ---------------------------------------------------------------------------

struct Ch {
  uint32_t len;  // UTF-8 byte length (1 for invalid bytes)
  uint32_t fl;   // F_* class bits
};

FT_INLINE uint32_t cp_flags(uint32_t cp) {
  return kStage2[(size_t)kStage1[cp >> 8] * 256 + (cp & 0xFF)];
}

inline Ch decode_slow(const uint8_t* s, size_t n, size_t i) {
  const uint32_t b0 = s[i];
  if (b0 < 0xC2) return {1, 0};  // stray continuation byte or overlong lead
  if (b0 < 0xE0) {
    if (i + 1 < n && (s[i + 1] & 0xC0) == 0x80)
      return {2, cp_flags(((b0 & 0x1F) << 6) | (s[i + 1] & 0x3F))};
    return {1, 0};
  }
  if (b0 < 0xF0) {
    if (i + 2 < n && (s[i + 1] & 0xC0) == 0x80 && (s[i + 2] & 0xC0) == 0x80) {
      uint32_t cp = ((b0 & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F);
      if (cp >= 0x800 && (cp < 0xD800 || cp > 0xDFFF)) return {3, cp_flags(cp)};
    }
    return {1, 0};
  }
  if (b0 < 0xF5) {
    if (i + 3 < n && (s[i + 1] & 0xC0) == 0x80 && (s[i + 2] & 0xC0) == 0x80 &&
        (s[i + 3] & 0xC0) == 0x80) {
      uint32_t cp = ((b0 & 0x07) << 18) | ((s[i + 1] & 0x3F) << 12) |
                    ((s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F);
      if (cp >= 0x10000 && cp <= 0x10FFFF) return {4, cp_flags(cp)};
    }
    return {1, 0};
  }
  return {1, 0};
}

// kStage1[0] == 0 by construction, so ASCII flags are simply kStage2[byte].
FT_INLINE Ch char_at(const uint8_t* s, size_t n, size_t i) {
  const uint32_t b = s[i];
  if (FT_LIKELY(b < 0x80)) return {1, kStage2[b]};
  return decode_slow(s, n, i);
}

FT_INLINE bool isP(uint32_t f) { return !(f & (F_L | F_N | F_NL)); }  // [^\r\n\p{L}\p{N}]
FT_INLINE bool isO(uint32_t f) { return !(f & (F_L | F_N | F_WS)); }  // [^\s\p{L}\p{N}]

#if defined(__SSE2__)
// Byte mask of x in [lo, lo + count) (unsigned), for 16 bytes at once.
FT_INLINE __m128i in_range16(__m128i x, char lo, int count) {
  const __m128i d = _mm_xor_si128(_mm_sub_epi8(x, _mm_set1_epi8(lo)), _mm_set1_epi8(char(0x80)));
  return _mm_cmplt_epi8(d, _mm_set1_epi8(char(count - 128)));
}
// Number of leading bytes of p[0..16) that are ASCII members of the class `Mask`
// (only F_L, F_UP, F_LO have ASCII members that are letters: A-Z / a-z).
template <uint32_t Mask>
FT_INLINE unsigned ascii_prefix16(const uint8_t* p) {
  const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  __m128i m;
  if constexpr (Mask == F_LO) m = in_range16(v, 'a', 26);
  else if constexpr (Mask == F_UP) m = in_range16(v, 'A', 26);
  else m = _mm_or_si128(in_range16(v, 'a', 26), in_range16(v, 'A', 26));
  return (unsigned)__builtin_ctz(~(unsigned)_mm_movemask_epi8(m));
}
FT_INLINE unsigned space_prefix16(const uint8_t* p) {
  const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  return (unsigned)__builtin_ctz(~(unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(v, _mm_set1_epi8(' '))));
}
#endif

// Advance over a maximal run of characters having any bit of `mask`.
template <uint32_t Mask>
FT_INLINE size_t run(const uint8_t* s, size_t n, size_t p) {
  for (;;) {
#if defined(__SSE2__)
    if constexpr (Mask == F_L || Mask == F_UP || Mask == F_LO) {
      // Skip ASCII letters 16 at a time; this also avoids a mispredicted branch at
      // the end of nearly every word.
      while (p + 16 <= n) {
        const unsigned k = ascii_prefix16<Mask>(s + p);
        p += k;
        if (k < 16) break;
      }
    }
#endif
    if (p >= n) return p;
    const Ch c = char_at(s, n, p);
    if (!(c.fl & Mask)) return p;
    p += c.len;
  }
}

FT_INLINE size_t run_other(const uint8_t* s, size_t n, size_t p) {
  while (p < n) {
    const Ch c = char_at(s, n, p);
    if (!isO(c.fl)) break;
    p += c.len;
  }
  return p;
}

FT_INLINE size_t up_to_3_numbers(const uint8_t* s, size_t n, size_t p) {
  for (int k = 0; k < 3 && p < n; ++k) {
    const Ch c = char_at(s, n, p);
    if (!(c.fl & F_N)) break;
    p += c.len;
  }
  return p;
}

struct WsRun {
  size_t end;     // end of the maximal \s run
  size_t last;    // start of its last character
  size_t nl_end;  // end of its last \r or \n, or NPOS
};

FT_INLINE WsRun ws_run(const uint8_t* s, size_t n, size_t i) {
  WsRun r{i, i, NPOS};
  size_t p = i;
  while (p < n) {
#if defined(__SSE2__)
    if (s[p] == ' ' && p + 16 <= n) {  // indentation and space runs, 16 at a time
      const unsigned k = space_prefix16(s + p);
      p += k;
      r.last = p - 1;
      if (k == 16) continue;
      if (p >= n) break;
    }
#endif
    const Ch c = char_at(s, n, p);
    if (!(c.fl & F_WS)) break;
    r.last = p;
    p += c.len;
    if (c.fl & F_NL) r.nl_end = p;
  }
  r.end = p;
  return r;
}

// ---------------------------------------------------------------------------
// Pre-tokenizers. Each `next(s, n, i)` returns the end of the piece that the
// encoding's regex would match at position i (i < n). Every position matches some
// alternative, so pieces tile the input exactly like tiktoken's find_iter.
// ---------------------------------------------------------------------------

// o200k:
//   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//  |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//  |\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
struct O200K {
  // (?i:'s|'t|'re|'ve|'m|'ll|'d)?  -- Unicode simple case folding adds U+017F (long s).
  static FT_INLINE size_t contraction(const uint8_t* s, size_t n, size_t e) {
    if (e + 1 < n && s[e] == '\'') {
      const uint8_t c1 = s[e + 1] | 0x20;
      if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return e + 2;
      if (e + 2 < n) {
        const uint8_t c2 = s[e + 2] | 0x20;
        if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l'))
          return e + 3;
        if (s[e + 1] == 0xC5 && s[e + 2] == 0xBF) return e + 3;
      }
    }
    return e;
  }

  // UP* LO+ C?  with the regex's backtracking: UP and LO overlap (Lm, Lo, M), so if
  // the greedy UP run is not followed by a LO char, the match ends at the last
  // overlapping char inside the run.
  static FT_INLINE size_t a1(const uint8_t* s, size_t n, size_t j) {
    size_t p = j, ab_end = NPOS;
    while (p < n) {
      const Ch c = char_at(s, n, p);
      if (!(c.fl & F_UP)) {
        if (c.fl & F_LO) return contraction(s, n, run<F_LO>(s, n, p + c.len));
        break;
      }
      p += c.len;
      if (c.fl & F_LO) ab_end = p;
    }
    return ab_end != NPOS ? contraction(s, n, ab_end) : NPOS;
  }

  // UP+ LO* C?
  static FT_INLINE size_t a2(const uint8_t* s, size_t n, size_t j) {
    const size_t p = run<F_UP>(s, n, j);
    if (p == j) return NPOS;
    return contraction(s, n, run<F_LO>(s, n, p));
  }

  static FT_INLINE size_t next(const uint8_t* s, size_t n, size_t i) {
    const Ch c0 = char_at(s, n, i);
    const uint32_t f = c0.fl;
    const size_t i1 = i + c0.len;
    const bool pre = isP(f) && i1 < n;
    size_t r;
    // Alternatives 1 and 2, tried in regex order: alt1 with the optional prefix char,
    // alt1 without it, alt2 with it, alt2 without it. Shortcuts below skip attempts
    // that provably fail.
    if (f & (F_UP | F_LO)) {  // c0 is a letter, or a mark (marks are also prefix chars)
      if (pre) {              // c0 is a mark
        if ((char_at(s, n, i1).fl & (F_UP | F_LO)) && (r = a1(s, n, i1)) != NPOS) return r;
        return a1(s, n, i);  // a mark is in both sets, so this cannot fail
      }
      if ((r = a1(s, n, i)) != NPOS) return r;
      return a2(s, n, i);  // a1 failed, so c0 is Lu/Lt and alt2 matches
    }
    if (pre && (char_at(s, n, i1).fl & (F_UP | F_LO))) {
      if ((r = a1(s, n, i1)) != NPOS) return r;
      return a2(s, n, i1);  // a1 failed, so the char after the prefix is Lu/Lt
    }
    // \p{N}{1,3}
    if (f & F_N) return up_to_3_numbers(s, n, i);
    //  ?[^\s\p{L}\p{N}]+[\r\n/]*
    size_t j = NPOS;
    if (s[i] == ' ') {
      if (i1 < n && isO(char_at(s, n, i1).fl)) j = i1;
    } else if (isO(f)) {
      j = i;
    }
    if (j != NPOS) {
      size_t p = run_other(s, n, j);
      while (p < n && (s[p] == '\r' || s[p] == '\n' || s[p] == '/')) ++p;
      return p;
    }
    // Only whitespace remains.
    const WsRun w = ws_run(s, n, i);
    if (w.nl_end != NPOS) return w.nl_end;  // \s*[\r\n]+
    if (w.end == n) return n;               // \s+(?!\S) at end of text
    if (w.last > i) return w.last;          // \s+(?!\S): give back one char
    return w.end;                           // \s+
  }
};

// cl100k:
//   '(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+
//  |\s++$|\s*[\r\n]|\s+(?!\S)|\s
struct CL100K {
  static FT_INLINE size_t next(const uint8_t* s, size_t n, size_t i) {
    if (s[i] == '\'' && i + 1 < n) {
      const uint8_t c1 = s[i + 1] | 0x20;
      if (c1 == 's' || c1 == 'd' || c1 == 'm' || c1 == 't') return i + 2;
      if (i + 2 < n) {
        const uint8_t c2 = s[i + 2] | 0x20;
        if ((c1 == 'l' && c2 == 'l') || (c1 == 'v' && c2 == 'e') || (c1 == 'r' && c2 == 'e'))
          return i + 3;
        if (s[i + 1] == 0xC5 && s[i + 2] == 0xBF) return i + 3;  // 'ſ
      }
    }
    const Ch c0 = char_at(s, n, i);
    const uint32_t f = c0.fl;
    const size_t i1 = i + c0.len;
    if (f & F_L) return run<F_L>(s, n, i1);
    if (isP(f) && i1 < n) {  // possessive prefix: no retry without it
      const Ch c1 = char_at(s, n, i1);
      if (c1.fl & F_L) return run<F_L>(s, n, i1 + c1.len);
    }
    if (f & F_N) return up_to_3_numbers(s, n, i);
    size_t j = NPOS;
    if (s[i] == ' ') {
      if (i1 < n && isO(char_at(s, n, i1).fl)) j = i1;
    } else if (isO(f)) {
      j = i;
    }
    if (j != NPOS) {
      size_t p = run_other(s, n, j);
      while (p < n && (s[p] == '\r' || s[p] == '\n')) ++p;
      return p;
    }
    const WsRun w = ws_run(s, n, i);
    if (w.end == n) return n;               // \s++$
    if (w.nl_end != NPOS) return w.nl_end;  // \s*[\r\n]
    if (w.last > i) return w.last;          // \s+(?!\S)
    return i1;                              // \s
  }
};

// r50k / p50k / gpt2:
//   '(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s
struct R50K {
  static FT_INLINE size_t next(const uint8_t* s, size_t n, size_t i) {
    if (s[i] == '\'' && i + 1 < n) {
      const uint8_t c1 = s[i + 1];
      if (c1 == 's' || c1 == 'd' || c1 == 'm' || c1 == 't') return i + 2;
      if (i + 2 < n) {
        const uint8_t c2 = s[i + 2];
        if ((c1 == 'l' && c2 == 'l') || (c1 == 'v' && c2 == 'e') || (c1 == 'r' && c2 == 'e'))
          return i + 3;
      }
    }
    size_t j = i;
    const Ch c0 = char_at(s, n, i);
    Ch c = c0;
    if (s[i] == ' ' && i + 1 < n) {
      const Ch c1 = char_at(s, n, i + 1);
      if (!(c1.fl & F_WS)) {
        j = i + 1;
        c = c1;
      }
    }
    if (c.fl & F_L) return run<F_L>(s, n, j + c.len);
    if (c.fl & F_N) return run<F_N>(s, n, j + c.len);
    if (isO(c.fl)) return run_other(s, n, j + c.len);
    const WsRun w = ws_run(s, n, i);
    if (w.end == n) return n;       // \s++$
    if (w.last > i) return w.last;  // \s+(?!\S)
    return i + c0.len;              // \s
  }
};

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

FT_INLINE uint64_t mum(uint64_t a, uint64_t b) {
  __uint128_t r = (__uint128_t)a * b;
  return (uint64_t)r ^ (uint64_t)(r >> 64);
}
FT_INLINE uint64_t rd4(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
FT_INLINE uint64_t rd8(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// wyhash-style hash for byte strings (used only for tokens of 8+ bytes).
FT_INLINE uint64_t hash_bytes(const uint8_t* p, size_t len) {
  constexpr uint64_t s0 = 0xa0761d6478bd642full, s1 = 0xe7037ed1a0b428dbull,
                     s2 = 0x8ebc6af09c88c6e3ull;
  uint64_t seed = s0, a, b;
  if (FT_LIKELY(len <= 16)) {
    if (len >= 4) {
      a = (rd4(p) << 32) | rd4(p + ((len >> 3) << 2));
      b = (rd4(p + len - 4) << 32) | rd4(p + len - 4 - ((len >> 3) << 2));
    } else if (len > 0) {
      a = ((uint64_t)p[0] << 16) | ((uint64_t)p[len >> 1] << 8) | p[len - 1];
      b = 0;
    } else {
      a = b = 0;
    }
  } else {
    size_t i = len;
    const uint8_t* q = p;
    while (i > 16) {
      seed = mum(rd8(q) ^ s1, rd8(q + 8) ^ seed);
      q += 16;
      i -= 16;
    }
    a = rd8(p + len - 16);
    b = rd8(p + len - 8);
  }
  return mum(s1 ^ len, mum(a ^ s1, b ^ seed ^ s2));
}

FT_INLINE uint64_t fib(uint64_t k) { return k * 0x9E3779B97F4A7C15ull; }

inline int ceil_log2(size_t x) {
  int b = 0;
  while ((size_t(1) << b) < x) ++b;
  return b;
}

// ---------------------------------------------------------------------------
// Sinks receive token ids from the encoder.
// ---------------------------------------------------------------------------

// Buffers ids locally so the hot loop keeps the write index in a register.
struct VecSink {
  explicit VecSink(std::vector<Rank>* out) : v(out) {}
  VecSink(const VecSink&) = delete;
  ~VecSink() { flush(); }
  FT_INLINE void push(Rank r) {
    buf[n++] = r;
    if (FT_UNLIKELY(n == kCap)) flush();
  }
  void flush() {
    v->insert(v->end(), buf, buf + n);
    n = 0;
  }
  static constexpr uint32_t kCap = 512;
  std::vector<Rank>* v;
  uint32_t n = 0;
  Rank buf[kCap];
};
struct CountSink {
  size_t c = 0;
  FT_INLINE void push(Rank) { ++c; }
};

// Direct-mapped memo of recent BPE results (piece bytes -> token ids). Natural text
// repeats words constantly, so most pieces that need merging have been merged before.
// Each thread keeps caches for up to 4 tokenizers (keyed by a unique tokenizer id).
struct PieceCache {
#ifndef FASTTOK_CACHE_BITS
#define FASTTOK_CACHE_BITS 12
#endif
  static constexpr size_t kBits = FASTTOK_CACHE_BITS;
  static constexpr size_t kMaxLen = 32;   // longest piece that is cached
  static constexpr size_t kMaxToks = 20;  // most tokens a cached piece may produce
  struct alignas(64) Entry {
    uint64_t h;
    uint8_t len, ntok, pad[6];
    uint8_t bytes[kMaxLen];
    Rank toks[kMaxToks];
  };
  static_assert(sizeof(Entry) == 128, "cache entry should be two cache lines");
  std::unique_ptr<Entry[]> e{new Entry[size_t(1) << kBits]()};
  FT_INLINE Entry& slot(uint64_t h) { return e[(h * 0x9E3779B97F4A7C15ull) >> (64 - kBits)]; }
};

inline PieceCache* thread_cache(uint64_t owner) {
  thread_local std::pair<uint64_t, std::unique_ptr<PieceCache>> slots[4];
  thread_local unsigned victim = 0;
  for (auto& s : slots)
    if (s.second && s.first == owner) return s.second.get();
  auto& s = slots[victim++ & 3];
  s.first = owner;
  s.second.reset(new PieceCache);
  return s.second.get();
}

inline std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("fasttok: cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

struct RawTok {
  Rank rank;
  uint32_t off, len;  // into the decoded byte buffer
};

// Appends the decoded bytes of a standard (padded or unpadded) base64 string.
inline bool base64_append(std::string_view in, std::string& out) {
  static const auto table = [] {
    std::array<int8_t, 256> t{};
    t.fill(-1);
    const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; ++i) t[(uint8_t)a[i]] = (int8_t)i;
    return t;
  }();
  while (!in.empty() && in.back() == '=') in.remove_suffix(1);
  if (in.size() % 4 == 1) return false;
  size_t i = 0;
  for (; i + 4 <= in.size(); i += 4) {
    const int a = table[(uint8_t)in[i]], b = table[(uint8_t)in[i + 1]],
              c = table[(uint8_t)in[i + 2]], d = table[(uint8_t)in[i + 3]];
    if ((a | b | c | d) < 0) return false;
    const uint32_t v = (uint32_t(a) << 18) | (uint32_t(b) << 12) | (uint32_t(c) << 6) | uint32_t(d);
    const char o[3] = {char(v >> 16), char(v >> 8), char(v)};
    out.append(o, 3);
  }
  const size_t rest = in.size() - i;
  if (rest) {
    uint32_t v = 0;
    for (size_t k = 0; k < rest; ++k) {
      const int x = table[(uint8_t)in[i + k]];
      if (x < 0) return false;
      v |= uint32_t(x) << (18 - 6 * k);
    }
    out.push_back(char(v >> 16));
    if (rest == 3) out.push_back(char(v >> 8));
  }
  return true;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

class Tokenizer {
 public:
  struct Special {
    std::string text;
    Rank id;
  };

  // Load a .tiktoken vocabulary ("<base64 token> <rank>" per line). Tokens with rank >=
  // max_rank are dropped: a rank prefix of a BPE vocabulary is itself a complete BPE
  // vocabulary, identical to stopping training after max_rank merges.
  static Tokenizer load(const std::string& tiktoken_path, Pattern pattern,
                        std::vector<Special> specials = {}, Rank max_rank = kNone) {
    return from_tiktoken_text(detail::read_file(tiktoken_path), pattern, std::move(specials), max_rank);
  }

  static Tokenizer o200k_base(const std::string& path) {
    return load(path, Pattern::O200K, {{"<|endoftext|>", 199999}, {"<|endofprompt|>", 200018}});
  }
  static Tokenizer cl100k_base(const std::string& path) {
    return load(path, Pattern::CL100K,
                {{"<|endoftext|>", 100257}, {"<|fim_prefix|>", 100258},
                 {"<|fim_middle|>", 100259}, {"<|fim_suffix|>", 100260},
                 {"<|endofprompt|>", 100276}});
  }
  // ec200k_base: a 200k vocabulary trained for English prose + source code (see
  // vocab/README.md). Uses the cl100k pre-tokenizer; ids 0..100255 are identical to
  // cl100k_base's tokens, ids 100256..199999 are new, specials follow.
  static Tokenizer ec200k_base(const std::string& path) { return ec_prefix(path, 200000); }
  // The first n_ranks tokens of ec200k_base (n_ranks <= 200000), with the five specials
  // renumbered to follow them (n_ranks .. n_ranks + 4). Smaller vocabularies trade
  // embedding size for more tokens per byte of text.
  static Tokenizer ec_prefix(const std::string& path, Rank n_ranks) {
    if (n_ranks < 256 || n_ranks > 200000) throw std::runtime_error("fasttok: ec prefix size out of range");
    return load(path, Pattern::CL100K,
                {{"<|endoftext|>", n_ranks}, {"<|fim_prefix|>", n_ranks + 1}, {"<|fim_middle|>", n_ranks + 2},
                 {"<|fim_suffix|>", n_ranks + 3}, {"<|endofprompt|>", n_ranks + 4}},
                n_ranks);
  }
  static Tokenizer p50k_base(const std::string& path) {
    return load(path, Pattern::R50K, {{"<|endoftext|>", 50256}});
  }
  static Tokenizer r50k_base(const std::string& path) {
    return load(path, Pattern::R50K, {{"<|endoftext|>", 50256}});
  }

  static Tokenizer from_tiktoken_text(std::string_view text, Pattern pattern,
                                      std::vector<Special> specials = {}, Rank max_rank = kNone) {
    // Decode every token straight into one byte buffer (no per-token allocations).
    std::string bytes;
    bytes.reserve(text.size());
    std::vector<detail::RawTok> toks;
    toks.reserve(text.size() / 10);
    size_t pos = 0;
    while (pos < text.size()) {
      size_t eol = text.find('\n', pos);
      if (eol == std::string_view::npos) eol = text.size();
      std::string_view line = text.substr(pos, eol - pos);
      pos = eol + 1;
      if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
      if (line.empty()) continue;
      const size_t sp = line.find(' ');
      const size_t off = bytes.size();
      if (sp == std::string_view::npos || !detail::base64_append(line.substr(0, sp), bytes))
        throw std::runtime_error("fasttok: malformed vocabulary line");
      uint64_t r = 0;
      for (char ch : line.substr(sp + 1)) {
        if (ch < '0' || ch > '9') throw std::runtime_error("fasttok: malformed rank");
        r = r * 10 + (uint64_t)(ch - '0');
        if (r >= kNone) throw std::runtime_error("fasttok: rank too large");
      }
      if (r >= max_rank) {
        bytes.resize(off);
        continue;
      }
      toks.push_back({(Rank)r, (uint32_t)off, (uint32_t)(bytes.size() - off)});
    }
    Tokenizer t;
    t.build(bytes, std::move(toks), pattern, std::move(specials));
    return t;
  }

  // ---- Encoding -----------------------------------------------------------

  // Like tiktoken's encode_ordinary: special-token text is encoded as plain text.
  void encode_ordinary(std::string_view text, std::vector<Rank>& out) const {
    out.reserve(out.size() + text.size() / 3 + 16);
    detail::VecSink sink(&out);
    const auto* s = reinterpret_cast<const uint8_t*>(text.data());
    dispatch_range(s, 0, text.size(), text.size(), sink);
  }
  std::vector<Rank> encode_ordinary(std::string_view text) const {
    std::vector<Rank> out;
    encode_ordinary(text, out);
    return out;
  }

  // Encode one already pre-tokenized piece (no regex splitting): whole-token lookup,
  // else byte-pair merging. Useful for vocabulary training and analysis.
  void encode_single_piece(std::string_view piece, std::vector<Rank>& out) const {
    if (piece.empty()) return;
    detail::VecSink sink(&out);
    const auto* p = reinterpret_cast<const uint8_t*>(piece.data());
    const size_t n = piece.size();
    encode_piece(p, n, piece_key(p, 0, n, n), sink, nullptr);
  }

  // Number of tokens encode_ordinary would produce, without materializing them.
  size_t count(std::string_view text) const {
    detail::CountSink sink;
    const auto* s = reinterpret_cast<const uint8_t*>(text.data());
    dispatch_range(s, 0, text.size(), text.size(), sink);
    return sink.c;
  }

  // Like tiktoken's encode(text, allowed_special="all"): special tokens in the text
  // become their ids.
  std::vector<Rank> encode(std::string_view text) const {
    std::vector<Rank> out;
    encode_impl(text, out, nullptr);
    return out;
  }
  // Only the listed special tokens are recognized; others are encoded as plain text
  // (tiktoken's encode(..., allowed_special=allowed, disallowed_special=())).
  std::vector<Rank> encode(std::string_view text, const std::vector<std::string>& allowed) const {
    std::vector<char> mask(specials_.size(), 0);
    for (const auto& a : allowed)
      for (size_t k = 0; k < specials_.size(); ++k)
        if (specials_[k].text == a) mask[k] = 1;
    std::vector<Rank> out;
    encode_impl(text, out, &mask);
    return out;
  }

  // encode_ordinary for one large text, split across threads at boundaries that are
  // guaranteed to be piece boundaries ("\n" followed by an ASCII letter). Output is
  // identical to encode_ordinary. threads == 0 uses all hardware threads.
  std::vector<Rank> encode_ordinary_parallel(std::string_view text, unsigned threads = 0) const {
    if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
    const size_t n = text.size();
    if (threads <= 1 || n < (size_t(1) << 18)) return encode_ordinary(text);
    const auto* s = reinterpret_cast<const uint8_t*>(text.data());
    std::vector<size_t> cuts{0};
    for (unsigned t = 1; t < threads; ++t) {
      const size_t target = n / threads * t;
      if (target <= cuts.back()) continue;
      const size_t c = find_safe_split(s, n, target);
      if (c == detail::NPOS) break;
      if (c > cuts.back() && c < n) cuts.push_back(c);
    }
    cuts.push_back(n);
    const size_t parts = cuts.size() - 1;
    std::vector<std::vector<Rank>> outs(parts);
    std::vector<std::thread> pool;
    for (size_t k = 0; k < parts; ++k) {
      pool.emplace_back([&, k] {
        outs[k].reserve((cuts[k + 1] - cuts[k]) / 3 + 16);
        detail::VecSink sink(&outs[k]);
        dispatch_range(s, cuts[k], cuts[k + 1], n, sink);
      });
    }
    for (auto& th : pool) th.join();
    size_t total = 0;
    for (auto& o : outs) total += o.size();
    std::vector<Rank> out;
    out.reserve(total);
    for (auto& o : outs) out.insert(out.end(), o.begin(), o.end());
    return out;
  }

  // encode_ordinary over many documents using a thread pool.
  std::vector<std::vector<Rank>> encode_ordinary_batch(const std::vector<std::string_view>& texts,
                                                       unsigned threads = 0) const {
    if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::vector<Rank>> outs(texts.size());
    std::atomic<size_t> nexti{0};
    auto work = [&] {
      for (size_t k; (k = nexti.fetch_add(1, std::memory_order_relaxed)) < texts.size();)
        encode_ordinary(texts[k], outs[k]);
    };
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < threads; ++t) pool.emplace_back(work);
    work();
    for (auto& th : pool) th.join();
    return outs;
  }

  // ---- Decoding -----------------------------------------------------------

  std::string decode(const Rank* ids, size_t n) const {
    size_t total = 0;
    for (size_t k = 0; k < n; ++k) total += tok_len(check_id(ids[k]));
    std::string out;
    out.resize(total);
    char* w = out.data();
    for (size_t k = 0; k < n; ++k) {
      const size_t len = tok_len(ids[k]);
      std::memcpy(w, tok_ptr(ids[k]), len);
      w += len;
    }
    return out;
  }
  std::string decode(const std::vector<Rank>& ids) const { return decode(ids.data(), ids.size()); }

  std::string_view token_bytes(Rank id) const {
    check_id(id);
    return {reinterpret_cast<const char*>(tok_ptr(id)), tok_len(id)};
  }

  // ---- Introspection ------------------------------------------------------

  size_t n_vocab() const { return off_.size() - 1; }  // max token id + 1
  Pattern pattern() const { return pattern_; }
  const std::vector<Special>& special_tokens() const { return specials_; }
  size_t merge_pairs() const { return pair_count_; }

  // The per-thread BPE memo is on by default; turn it off to measure raw merge speed.
  void set_cache_enabled(bool on) { cache_enabled_ = on; }

 private:
  struct ShortSlot {  // tokens of 1..7 bytes, keyed by packed bytes + length sentinel
    uint64_t key;     // 0 = empty
    Rank rank;
    uint32_t pad;
  };
  struct LongSlot {  // tokens of 8+ bytes
    uint32_t tag;
    Rank rank;       // kNone = empty
  };

  static constexpr size_t kSmallBpe = 64;  // longer pieces use the heap-based merge
  static constexpr Rank kDirectBits = 9;   // pairs of ids < 512 use a dense 1 MB table

  Pattern pattern_ = Pattern::O200K;
  uint64_t id_ = 0;  // identifies this vocabulary for the per-thread caches
  bool cache_enabled_ = true;
  std::vector<Rank> pair_direct_;
  std::vector<uint8_t> arena_;  // token bytes, by id
  std::vector<uint32_t> off_;   // token id -> arena offset (size n_vocab + 1)
  Rank byte_rank_[256];
  size_t max_token_len_ = 0;

  std::vector<ShortSlot> short_;
  int short_shift_ = 64;
  size_t short_mask_ = 0;
  std::vector<LongSlot> long_;
  int long_shift_ = 64;
  size_t long_mask_ = 0;
  std::vector<uint64_t> pair_;    // grouped merge slots: right << 32 | merged; ~0 = empty
  std::vector<uint32_t> group_;   // left id -> slot offset << 5 | log2(group capacity)
  size_t pair_count_ = 0;

  std::vector<Special> specials_;
  std::unordered_map<std::string, size_t> special_index_;
  std::vector<size_t> special_lens_;  // distinct lengths, descending
  bool special_first_[256] = {};

  FT_INLINE const uint8_t* tok_ptr(Rank r) const { return arena_.data() + off_[r]; }
  FT_INLINE size_t tok_len(Rank r) const { return off_[r + 1] - off_[r]; }
  Rank check_id(Rank r) const {
    if (r + 1 >= off_.size()) throw std::out_of_range("fasttok: token id out of range");
    return r;
  }

  static FT_INLINE uint64_t pack_safe(const uint8_t* p, size_t len) {  // len 1..7
    uint64_t v = 0;
    std::memcpy(&v, p, len);
    return v | (uint64_t(1) << (8 * len));
  }

  FT_INLINE Rank find_short(uint64_t key) const {
    size_t idx = detail::fib(key) >> short_shift_;
    for (;;) {
      const ShortSlot& sl = short_[idx];
      if (sl.key == key) return sl.rank;
      if (sl.key == 0) return kNone;
      idx = (idx + 1) & short_mask_;
    }
  }

  FT_INLINE Rank find_long(const uint8_t* p, size_t len) const {
    return find_long_h(detail::hash_bytes(p, len), p, len);
  }
  FT_INLINE Rank find_long_h(uint64_t h, const uint8_t* p, size_t len) const {
    size_t idx = h >> long_shift_;
    const uint32_t tag = (uint32_t)h;
    for (;;) {
      const LongSlot& sl = long_[idx];
      if (sl.rank == kNone) return kNone;
      if (sl.tag == tag && tok_len(sl.rank) == len && std::memcmp(tok_ptr(sl.rank), p, len) == 0)
        return sl.rank;
      idx = (idx + 1) & long_mask_;
    }
  }

  Rank find_any(const uint8_t* p, size_t len) const {
    if (len == 0) return kNone;
    if (len <= 7) return find_short(pack_safe(p, len));
    if (len > max_token_len_) return kNone;
    return find_long(p, len);
  }

  // Merge lookup. Pairs of small ids hit a dense table; the rest live in per-left-token
  // open-addressing groups, so all merges of a frequently used token share a few cache
  // lines (measured 15-30% faster than one global hash table on unique pieces).
  FT_INLINE Rank pair(Rank a, Rank b) const {
    if (FT_LIKELY((a | b) < (Rank(1) << kDirectBits))) return pair_direct_[(a << kDirectBits) | b];
    const uint32_t gv = group_[a];
    if (!gv) return kNone;
    const uint64_t* e = pair_.data() + (gv >> 5);
    const int lb = gv & 31;
    const size_t mask = (size_t(1) << lb) - 1;
    size_t i = detail::fib(b) >> (64 - lb);
    for (;;) {
      const uint64_t v = e[i];
      if ((v >> 32) == b) return Rank(v);
      if (v == ~uint64_t(0)) return kNone;
      i = (i + 1) & mask;
    }
  }

  // ---- Construction -------------------------------------------------------

  void build(const std::string& bytes, std::vector<detail::RawTok> toks, Pattern pattern,
             std::vector<Special> specials) {
    pattern_ = pattern;
    specials_ = std::move(specials);
    if (toks.empty()) throw std::runtime_error("fasttok: empty vocabulary");
    if (!std::is_sorted(toks.begin(), toks.end(),
                        [](const auto& x, const auto& y) { return x.rank < y.rank; }))
      std::sort(toks.begin(), toks.end(),
                [](const auto& x, const auto& y) { return x.rank < y.rank; });
    auto tp = [&](const detail::RawTok& t) {
      return reinterpret_cast<const uint8_t*>(bytes.data()) + t.off;
    };
    Rank max_id = toks.back().rank;
    for (auto& sp : specials_) max_id = std::max(max_id, sp.id);

    // Arena indexed by id; specials are included so decode() handles them.
    std::vector<std::pair<const char*, uint32_t>> by_id(size_t(max_id) + 1, {nullptr, 0});
    for (size_t k = 0; k < toks.size(); ++k) {
      if (toks[k].len == 0) throw std::runtime_error("fasttok: empty token");
      if (k && toks[k].rank == toks[k - 1].rank) throw std::runtime_error("fasttok: duplicate rank");
      by_id[toks[k].rank] = {bytes.data() + toks[k].off, toks[k].len};
    }
    for (auto& sp : specials_) {
      if (by_id[sp.id].first) throw std::runtime_error("fasttok: special id collides with a token");
      if (sp.text.empty()) throw std::runtime_error("fasttok: empty special token");
      by_id[sp.id] = {sp.text.data(), (uint32_t)sp.text.size()};
    }
    off_.assign(size_t(max_id) + 2, 0);
    size_t total = 0;
    for (auto& e : by_id) total += e.second;
    arena_.resize(total);
    size_t w = 0;
    for (size_t id = 0; id <= max_id; ++id) {
      off_[id] = (uint32_t)w;
      if (by_id[id].first) {
        std::memcpy(arena_.data() + w, by_id[id].first, by_id[id].second);
        w += by_id[id].second;
      }
    }
    off_[size_t(max_id) + 1] = (uint32_t)w;

    // Lookup tables (load factor <= 0.5).
    size_t n_short = 0, n_long = 0;
    max_token_len_ = 0;
    for (auto& t : toks) {
      (t.len <= 7 ? n_short : n_long)++;
      max_token_len_ = std::max(max_token_len_, size_t(t.len));
    }
    const int sb = std::max(4, detail::ceil_log2(n_short * 2 + 1));
    short_.assign(size_t(1) << sb, ShortSlot{0, kNone, 0});
    short_shift_ = 64 - sb;
    short_mask_ = (size_t(1) << sb) - 1;
    const int lb = std::max(4, detail::ceil_log2(n_long * 2 + 1));
    long_.assign(size_t(1) << lb, LongSlot{0, kNone});
    long_shift_ = 64 - lb;
    long_mask_ = (size_t(1) << lb) - 1;

    for (auto& t : toks) {
      const uint8_t* p = tp(t);
      const size_t len = t.len;
      if (find_any(p, len) != kNone) throw std::runtime_error("fasttok: duplicate token bytes");
      if (len <= 7) {
        const uint64_t key = pack_safe(p, len);
        size_t idx = detail::fib(key) >> short_shift_;
        while (short_[idx].key != 0) idx = (idx + 1) & short_mask_;
        short_[idx] = ShortSlot{key, t.rank, 0};
      } else {
        const uint64_t h = detail::hash_bytes(p, len);
        size_t idx = h >> long_shift_;
        while (long_[idx].rank != kNone) idx = (idx + 1) & long_mask_;
        long_[idx] = LongSlot{(uint32_t)h, t.rank};
      }
    }

    for (int b = 0; b < 256; ++b) {
      const uint8_t c = (uint8_t)b;
      byte_rank_[b] = find_any(&c, 1);
      if (byte_rank_[b] == kNone)
        throw std::runtime_error("fasttok: vocabulary must contain all 256 single bytes");
    }

    // Merge table: every split of a token into two tokens is a possible merge.
    std::vector<std::pair<uint64_t, Rank>> pairs;
    pairs.reserve(toks.size() * 2);
    for (auto& t : toks) {
      const uint8_t* p = tp(t);
      const size_t len = t.len;
      for (size_t k = 1; k < len; ++k) {
        const Rank a = find_any(p, k);
        if (a == kNone) continue;
        const Rank b = find_any(p + k, len - k);
        if (b == kNone) continue;
        pairs.emplace_back((uint64_t(a) << 32) | b, t.rank);
      }
    }
    pair_count_ = pairs.size();
    // Group merges by left token: group_[a] = offset << 5 | log2(capacity).
    pair_direct_.assign(size_t(1) << (2 * kDirectBits), kNone);
    std::vector<uint32_t> cnt(size_t(max_id) + 1, 0);
    for (auto& [ab, merged] : pairs) {
      const Rank a = Rank(ab >> 32), b = Rank(ab);
      ++cnt[a];
      if ((a | b) < (Rank(1) << kDirectBits)) pair_direct_[(a << kDirectBits) | b] = merged;
    }
    group_.assign(size_t(max_id) + 1, 0);
    size_t total_slots = 0;
    for (size_t a = 0; a <= max_id; ++a) {
      if (!cnt[a]) continue;
      const int lb = std::max(1, detail::ceil_log2(size_t(cnt[a]) * 2));
      if (total_slots >= (size_t(1) << 27)) throw std::runtime_error("fasttok: merge table too large");
      group_[a] = uint32_t(total_slots << 5) | uint32_t(lb);
      total_slots += size_t(1) << lb;
    }
    pair_.assign(total_slots, ~uint64_t(0));
    for (auto& [ab, merged] : pairs) {
      const Rank a = Rank(ab >> 32), b = Rank(ab);
      uint64_t* e = pair_.data() + (group_[a] >> 5);
      const int lb = group_[a] & 31;
      const size_t mask = (size_t(1) << lb) - 1;
      size_t i = detail::fib(b) >> (64 - lb);
      while (e[i] != ~uint64_t(0)) i = (i + 1) & mask;
      e[i] = (uint64_t(b) << 32) | merged;
    }
    static std::atomic<uint64_t> next_id{1};
    id_ = next_id.fetch_add(1);

    // Special-token matcher.
    special_index_.clear();
    special_lens_.clear();
    std::fill(std::begin(special_first_), std::end(special_first_), false);
    for (size_t k = 0; k < specials_.size(); ++k) {
      const std::string& txt = specials_[k].text;
      if (txt.empty()) throw std::runtime_error("fasttok: empty special token");
      special_index_[txt] = k;
      special_first_[(uint8_t)txt[0]] = true;
      if (std::find(special_lens_.begin(), special_lens_.end(), txt.size()) == special_lens_.end())
        special_lens_.push_back(txt.size());
    }
    std::sort(special_lens_.rbegin(), special_lens_.rend());
  }

  // ---- Encoding internals -------------------------------------------------

  // Packed key (len <= 7) or byte hash (len >= 8) of a piece; used both to prefetch its
  // lookup slot and to probe it.
  FT_INLINE uint64_t piece_key(const uint8_t* s, size_t i, size_t len, size_t n) const {
    const uint8_t* p = s + i;
    if (len <= 7) {
      uint64_t v;
      if (FT_LIKELY(i + 8 <= n)) {
        std::memcpy(&v, p, 8);
        v &= ~uint64_t(0) >> (64 - 8 * len);
      } else {
        v = 0;
        std::memcpy(&v, p, len);
      }
      return v | (uint64_t(1) << (8 * len));
    }
    return detail::hash_bytes(p, len);
  }

  FT_INLINE void prefetch_piece(uint64_t h, size_t len) const {
#if defined(__GNUC__) || defined(__clang__)
    if (len <= 7) __builtin_prefetch(&short_[detail::fib(h) >> short_shift_]);
    else if (len <= max_token_len_) __builtin_prefetch(&long_[h >> long_shift_]);
#else
    (void)h; (void)len;
#endif
  }

  template <class Sink>
  FT_INLINE void encode_piece(const uint8_t* p, size_t len, uint64_t h, Sink& out,
                              detail::PieceCache* cache) const {
    if (len == 1) {
      out.push(byte_rank_[*p]);
      return;
    }
    if (len <= 7) {
      const Rank r = find_short(h);
      if (r != kNone) {
        out.push(r);
        return;
      }
    } else if (len <= max_token_len_) {
      const Rank r = find_long_h(h, p, len);
      if (r != kNone) {
        out.push(r);
        return;
      }
    }
    if (len > kSmallBpe) return bpe_large(p, len, out);
    Rank ids[kSmallBpe];
    if (cache && len <= detail::PieceCache::kMaxLen) {
      auto& ce = cache->slot(h);
      if (ce.h == h && ce.len == len && std::memcmp(ce.bytes, p, len) == 0) {
        for (size_t k = 0; k < ce.ntok; ++k) out.push(ce.toks[k]);
        return;
      }
      const size_t m = bpe_small(p, len, ids);
      for (size_t k = 0; k < m; ++k) out.push(ids[k]);
      if (m <= detail::PieceCache::kMaxToks) {
        ce.h = h;
        ce.len = (uint8_t)len;
        ce.ntok = (uint8_t)m;
        std::memcpy(ce.bytes, p, len);
        std::memcpy(ce.toks, ids, m * sizeof(Rank));
      }
      return;
    }
    const size_t m = bpe_small(p, len, ids);
    for (size_t k = 0; k < m; ++k) out.push(ids[k]);
  }

  // tiktoken's byte_pair_merge: repeatedly merge the adjacent pair whose concatenation
  // has the lowest rank (leftmost on ties). pr[k] is the rank of merging ids[k] and
  // ids[k+1]. Writes the result to ids (capacity kSmallBpe); returns the token count.
  // (A branchless/SIMD minimum, fixed-position linked lists, and interleaving several
  // pieces were all measured slower than this simple loop.)
  size_t bpe_small(const uint8_t* p, size_t len, Rank* ids) const {
    Rank pr[kSmallBpe];
    size_t m = len;
    for (size_t k = 0; k < m; ++k) ids[k] = byte_rank_[p[k]];
    for (size_t k = 0; k + 1 < m; ++k) pr[k] = pair(ids[k], ids[k + 1]);
    while (m > 1) {
      Rank best = kNone;
      size_t bi = 0;
      for (size_t k = 0; k + 1 < m; ++k)
        if (pr[k] < best) {
          best = pr[k];
          bi = k;
        }
      if (best == kNone) break;
      ids[bi] = best;
      // Drop ids[bi+1] and pr[bi+1]; pr[bi] and pr[bi-1] are recomputed below.
      for (size_t k = bi + 1; k + 1 < m; ++k) ids[k] = ids[k + 1];
      for (size_t k = bi + 1; k + 2 < m; ++k) pr[k] = pr[k + 1];
      --m;
      if (bi + 1 < m) pr[bi] = pair(ids[bi], ids[bi + 1]);
      if (bi > 0) pr[bi - 1] = pair(ids[bi - 1], ids[bi]);
    }
    return m;
  }

  // O(n log n) variant for long pieces: doubly linked list + lazy-deletion min-heap
  // keyed by (rank, start offset), which reproduces the leftmost tie-break.
  template <class Sink>
  FT_NOINLINE void bpe_large(const uint8_t* p, size_t len, Sink& out) const {
    std::vector<Rank> id(len), pr(len);
    std::vector<uint32_t> nxt(len), prv(len);
    std::vector<uint64_t> heap;
    heap.reserve(len);
    constexpr uint32_t END = 0xFFFFFFFFu;
    for (size_t k = 0; k < len; ++k) {
      id[k] = byte_rank_[p[k]];
      nxt[k] = k + 1 < len ? (uint32_t)(k + 1) : END;
      prv[k] = k > 0 ? (uint32_t)(k - 1) : END;
    }
    auto push = [&](uint32_t k) {
      pr[k] = nxt[k] != END ? pair(id[k], id[nxt[k]]) : kNone;
      if (pr[k] != kNone) {
        heap.push_back((uint64_t(pr[k]) << 32) | k);
        std::push_heap(heap.begin(), heap.end(), std::greater<uint64_t>());
      }
    };
    for (size_t k = 0; k < len; ++k) push((uint32_t)k);
    std::vector<char> alive(len, 1);
    while (!heap.empty()) {
      std::pop_heap(heap.begin(), heap.end(), std::greater<uint64_t>());
      const uint64_t top = heap.back();
      heap.pop_back();
      const uint32_t k = (uint32_t)top;
      const Rank r = (Rank)(top >> 32);
      if (!alive[k] || pr[k] != r) continue;  // stale entry
      const uint32_t j = nxt[k];
      id[k] = r;
      alive[j] = 0;
      nxt[k] = nxt[j];
      if (nxt[j] != END) prv[nxt[j]] = k;
      push(k);
      if (prv[k] != END) push(prv[k]);
    }
    for (uint32_t k = 0; k != END; k = nxt[k]) out.push(id[k]);
  }

  // Encode the pieces that start in [begin, stop). `begin` and `stop` must be piece
  // boundaries; n is the true end of the text (the regex lookaheads may look past
  // `stop`).
  template <class Scan, class Sink>
  void encode_range(const uint8_t* s, size_t begin, size_t stop, size_t n, Sink& out) const {
    detail::PieceCache* cache = cache_enabled_ ? detail::thread_cache(id_) : nullptr;
    // Scan a batch of pieces first and prefetch their hash slots, then encode the
    // batch; lookup latency overlaps with scanning instead of stalling each piece.
    constexpr int kBatch = 16;
    uint32_t len[kBatch];
    uint64_t key[kBatch];
    size_t i = begin;
    while (i < stop) {
      int cnt = 0;
      size_t j = i;
      do {
        const size_t e = Scan::next(s, n, j);
        const size_t l = e - j;
        len[cnt] = (uint32_t)l;
        key[cnt] = l > 1 ? piece_key(s, j, l, n) : 0;
        if (l > 1) prefetch_piece(key[cnt], l);
        ++cnt;
        j = e;
      } while (cnt < kBatch && j < stop);
      for (int k = 0; k < cnt; ++k) {
        encode_piece(s + i, len[k], key[k], out, cache);
        i += len[k];
      }
    }
  }

  template <class Sink>
  void dispatch_range(const uint8_t* s, size_t begin, size_t stop, size_t n, Sink& out) const {
    switch (pattern_) {
      case Pattern::O200K: return encode_range<detail::O200K>(s, begin, stop, n, out);
      case Pattern::CL100K: return encode_range<detail::CL100K>(s, begin, stop, n, out);
      case Pattern::R50K: return encode_range<detail::R50K>(s, begin, stop, n, out);
    }
  }

  // A position right after '\n' and at an ASCII letter is a piece boundary for all
  // supported patterns: no alternative can contain a '\n' followed by a letter.
  static size_t find_safe_split(const uint8_t* s, size_t n, size_t from) {
    while (from < n) {
      const void* q = std::memchr(s + from, '\n', n - from);
      if (!q) return detail::NPOS;
      const size_t k = (size_t)(static_cast<const uint8_t*>(q) - s) + 1;
      if (k < n && (uint8_t)((s[k] | 0x20) - 'a') < 26) return k;
      from = k;
    }
    return detail::NPOS;
  }

  // Find the next allowed special token at or after `from`.
  bool next_special(std::string_view text, size_t from, const std::vector<char>* allowed,
                    size_t& pos, size_t& idx) const {
    const size_t n = text.size();
    for (size_t i = from; i < n; ++i) {
      if (!special_first_[(uint8_t)text[i]]) continue;
      for (size_t L : special_lens_) {
        if (i + L > n) continue;
        auto it = special_index_.find(std::string(text.substr(i, L)));
        if (it != special_index_.end() && (!allowed || (*allowed)[it->second])) {
          pos = i;
          idx = it->second;
          return true;
        }
      }
    }
    return false;
  }

  void encode_impl(std::string_view text, std::vector<Rank>& out,
                   const std::vector<char>* allowed) const {
    out.reserve(out.size() + text.size() / 3 + 16);
    detail::VecSink sink(&out);
    const auto* s = reinterpret_cast<const uint8_t*>(text.data());
    size_t start = 0;
    while (true) {
      size_t pos = 0, idx = 0;
      const bool found = !specials_.empty() && next_special(text, start, allowed, pos, idx);
      const size_t end = found ? pos : text.size();
      // tiktoken runs the regex on each segment separately, so the segment end acts as
      // end-of-text for lookaheads.
      dispatch_range(s + start, 0, end - start, end - start, sink);
      if (!found) break;
      sink.push(specials_[idx].id);
      start = pos + specials_[idx].text.size();
    }
  }
};

}  // namespace fasttok
