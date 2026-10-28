// C ABI for fasttok (used by the Python test harness via ctypes; usable from any FFI).
#include <memory>
#include <string>

#include "fasttok.hpp"

using fasttok::Rank;
using fasttok::Tokenizer;

namespace {
thread_local std::string g_err;
}

extern "C" {

// encoding: "o200k_base", "cl100k_base", "ec200k_base", "p50k_base" or "r50k_base".
void* ft_new(const char* vocab_path, const char* encoding) {
  try {
    const std::string e = encoding;
    std::unique_ptr<Tokenizer> t;
    if (e == "o200k_base") t.reset(new Tokenizer(Tokenizer::o200k_base(vocab_path)));
    else if (e == "cl100k_base") t.reset(new Tokenizer(Tokenizer::cl100k_base(vocab_path)));
    else if (e == "ec200k_base") t.reset(new Tokenizer(Tokenizer::ec200k_base(vocab_path)));
    else if (e == "p50k_base") t.reset(new Tokenizer(Tokenizer::p50k_base(vocab_path)));
    else if (e == "r50k_base") t.reset(new Tokenizer(Tokenizer::r50k_base(vocab_path)));
    else throw std::runtime_error("unknown encoding " + e);
    return t.release();
  } catch (const std::exception& ex) {
    g_err = ex.what();
    return nullptr;
  }
}

// The first n_ranks tokens of ec200k_base, specials at n_ranks .. n_ranks + 4.
void* ft_new_ec_prefix(const char* vocab_path, uint32_t n_ranks) {
  try {
    return new Tokenizer(Tokenizer::ec_prefix(vocab_path, n_ranks));
  } catch (const std::exception& ex) {
    g_err = ex.what();
    return nullptr;
  }
}

void ft_free(void* h) { delete static_cast<Tokenizer*>(h); }

const char* ft_last_error() { return g_err.c_str(); }

// Encode n texts stored back to back in buf; text k is buf[offsets[k], offsets[k+1]).
// mode 0 = encode_ordinary, 1 = encode with all special tokens allowed,
// 2 = encode_ordinary_parallel (threads = 0 -> all cores).
// Token ids go to out_ids (capacity cap; offsets[n] is always enough), per-text counts
// to out_counts. Returns the total number of ids, or (size_t)-1 on error.
size_t ft_encode_batch(void* h, const char* buf, const uint64_t* offsets, size_t n, Rank* out_ids,
                       size_t cap, uint64_t* out_counts, int mode, unsigned threads) {
  try {
    const auto& t = *static_cast<Tokenizer*>(h);
    size_t total = 0;
    std::vector<Rank> v;
    for (size_t k = 0; k < n; ++k) {
      std::string_view text(buf + offsets[k], offsets[k + 1] - offsets[k]);
      v.clear();
      if (mode == 0) t.encode_ordinary(text, v);
      else if (mode == 1) v = t.encode(text);
      else v = t.encode_ordinary_parallel(text, threads);
      if (total + v.size() > cap) throw std::runtime_error("output capacity exceeded");
      std::memcpy(out_ids + total, v.data(), v.size() * sizeof(Rank));
      out_counts[k] = v.size();
      total += v.size();
    }
    return total;
  } catch (const std::exception& ex) {
    g_err = ex.what();
    return (size_t)-1;
  }
}

size_t ft_count(void* h, const char* buf, size_t len) {
  return static_cast<Tokenizer*>(h)->count(std::string_view(buf, len));
}

// Decode into out (capacity cap). Returns the decoded length (which may exceed cap, in
// which case nothing is written), or (size_t)-1 on error.
size_t ft_decode(void* h, const Rank* ids, size_t n, char* out, size_t cap) {
  try {
    const std::string s = static_cast<Tokenizer*>(h)->decode(ids, n);
    if (s.size() <= cap) std::memcpy(out, s.data(), s.size());
    return s.size();
  } catch (const std::exception& ex) {
    g_err = ex.what();
    return (size_t)-1;
  }
}

}  // extern "C"
