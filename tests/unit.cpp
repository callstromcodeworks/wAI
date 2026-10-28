// API-level checks that the Python differential test does not reach.
//   unit VOCAB_DIR CORPUS_FILE
#include <cstdio>
#include <random>

#include "fasttok.hpp"

using fasttok::Rank;
using fasttok::Tokenizer;

static int failures = 0;
#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) {                                                     \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                      \
    }                                                                  \
  } while (0)

static std::vector<Rank> cat(std::vector<Rank> a, const std::vector<Rank>& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s VOCAB_DIR CORPUS_FILE\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  const std::string corpus = fasttok::detail::read_file(argv[2]);
  const Tokenizer toks[] = {Tokenizer::o200k_base(dir + "/o200k_base.tiktoken"),
                            Tokenizer::cl100k_base(dir + "/cl100k_base.tiktoken"),
                            Tokenizer::ec200k_base(dir + "/ec200k_base.tiktoken"),
                            Tokenizer::ec_prefix(dir + "/ec200k_base.tiktoken", 32763),
                            Tokenizer::ec_prefix(dir + "/ec200k_base.tiktoken", 1000)};

  // Rank prefixes of ec200k_base: ids stay below the prefix size, specials follow it, and
  // the full-size prefix is ec200k_base itself.
  {
    const Tokenizer& full = toks[2];
    const Tokenizer& p32 = toks[3];
    const auto ids = p32.encode_ordinary(corpus);
    CHECK(!ids.empty() && *std::max_element(ids.begin(), ids.end()) < 32763);
    CHECK(p32.n_vocab() == 32768);
    CHECK(p32.encode("<|endoftext|>") == std::vector<Rank>{32763});
    CHECK(ids.size() >= full.encode_ordinary(corpus).size());  // fewer merges, never fewer tokens
    CHECK(Tokenizer::ec_prefix(dir + "/ec200k_base.tiktoken", 200000).encode_ordinary(corpus) ==
          full.encode_ordinary(corpus));
    bool threw = false;
    try {
      Tokenizer::ec_prefix(dir + "/ec200k_base.tiktoken", 200001);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    CHECK(threw);
  }
  for (const Tokenizer& t : toks) {
    // Empty input.
    CHECK(t.encode_ordinary("").empty());
    CHECK(t.count("") == 0);
    CHECK(t.decode(std::vector<Rank>{}).empty());

    // count() == encode_ordinary().size(), and decode round-trips.
    const auto ids = t.encode_ordinary(corpus);
    CHECK(t.count(corpus) == ids.size());
    CHECK(t.decode(ids) == corpus);

    // Cache off gives identical output.
    {
      Tokenizer u = t;
      u.set_cache_enabled(false);
      CHECK(u.encode_ordinary(corpus) == ids);
    }

    // Parallel and batch APIs equal the sequential encoder.
    CHECK(t.encode_ordinary_parallel(corpus, 2) == ids);
    CHECK(t.encode_ordinary_parallel(corpus, 7) == ids);
    std::vector<std::string_view> docs;
    for (size_t p = 0; p < corpus.size(); p += 100000) docs.push_back(std::string_view(corpus).substr(p, 100000));
    const auto batch = t.encode_ordinary_batch(docs, 3);
    for (size_t k = 0; k < docs.size(); ++k) CHECK(batch[k] == t.encode_ordinary(docs[k]));

    // Special tokens: all allowed vs. an allow-list.
    const Rank eot = t.special_tokens()[0].id;
    const std::string before = "Hello <|endofprompt|> there ", after = " general <|endofprompt|>!";
    const std::string text = before + "<|endoftext|>" + after;
    CHECK(t.encode(text, {"<|endoftext|>"}) ==
          cat(cat(t.encode_ordinary(before), {eot}), t.encode_ordinary(after)));
    CHECK(t.encode(text, {}) == t.encode_ordinary(text));
    CHECK(t.decode(t.encode(text)) == text);
    CHECK(t.encode("<|endoftext|>") == std::vector<Rank>{eot});

    // Invalid UTF-8 never crashes and still round-trips byte-exactly.
    std::mt19937 rng(42);
    for (int it = 0; it < 2000; ++it) {
      std::string junk(rng() % 300, '\0');
      for (auto& c : junk) c = char(rng() & 0xFF);
      const auto j = t.encode_ordinary(junk);
      CHECK(t.decode(j) == junk);
      CHECK(t.count(junk) == j.size());
    }

    // Token id bounds are checked on decode.
    bool threw = false;
    try {
      t.decode(std::vector<Rank>{Rank(t.n_vocab() + 5)});
    } catch (const std::out_of_range&) {
      threw = true;
    }
    CHECK(threw);
  }
  std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
