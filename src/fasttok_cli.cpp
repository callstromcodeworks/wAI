// fasttok command-line tool.
//
//   fasttok ENCODING VOCAB encode [FILE]          print token ids (FILE or stdin)
//   fasttok ENCODING VOCAB count  [FILE]          print the token count
//   fasttok ENCODING VOCAB decode [FILE]          read whitespace-separated ids, print text
//   fasttok ENCODING VOCAB bench  FILE [ITERS] [THREADS]
//
// ENCODING is o200k_base, cl100k_base, ec200k_base, p50k_base or r50k_base.
#include <chrono>
#include <cstdio>
#include <iostream>
#include <iterator>

#include "fasttok.hpp"

using fasttok::Rank;
using fasttok::Tokenizer;

static Tokenizer make(const std::string& enc, const std::string& vocab) {
  if (enc == "o200k_base") return Tokenizer::o200k_base(vocab);
  if (enc == "cl100k_base") return Tokenizer::cl100k_base(vocab);
  if (enc == "ec200k_base") return Tokenizer::ec200k_base(vocab);
  if (enc == "p50k_base") return Tokenizer::p50k_base(vocab);
  if (enc == "r50k_base") return Tokenizer::r50k_base(vocab);
  throw std::runtime_error("unknown encoding: " + enc);
}

static std::string slurp(int argc, char** argv, int idx) {
  if (argc > idx) return fasttok::detail::read_file(argv[idx]);
  return std::string(std::istreambuf_iterator<char>(std::cin), {});
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s ENCODING VOCAB {encode|count|decode} [FILE]\n"
                 "       %s ENCODING VOCAB bench FILE [ITERS] [THREADS]\n",
                 argv[0], argv[0]);
    return 2;
  }
  try {
    using clk = std::chrono::steady_clock;
    const std::string cmd = argv[3];
    auto t0 = clk::now();
    const Tokenizer tok = make(argv[1], argv[2]);
    const double load_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();

    if (cmd == "encode") {
      const std::string text = slurp(argc, argv, 4);
      std::string out;
      for (Rank r : tok.encode_ordinary(text)) {
        out += std::to_string(r);
        out += ' ';
      }
      if (!out.empty()) out.back() = '\n';
      std::fwrite(out.data(), 1, out.size(), stdout);
    } else if (cmd == "count") {
      std::printf("%zu\n", tok.count(slurp(argc, argv, 4)));
    } else if (cmd == "decode") {
      const std::string text = slurp(argc, argv, 4);
      std::vector<Rank> ids;
      const char* p = text.c_str();
      char* end;
      for (unsigned long v; (v = std::strtoul(p, &end, 10)), end != p; p = end) ids.push_back((Rank)v);
      const std::string s = tok.decode(ids);
      std::fwrite(s.data(), 1, s.size(), stdout);
    } else if (cmd == "bench") {
      if (argc < 5) throw std::runtime_error("bench needs FILE");
      const std::string text = fasttok::detail::read_file(argv[4]);
      const int iters = argc > 5 ? std::atoi(argv[5]) : 5;
      const unsigned threads = argc > 6 ? (unsigned)std::atoi(argv[6]) : 1;
      std::vector<Rank> ids;
      double best = 1e30;
      for (int it = 0; it < iters; ++it) {
        auto a = clk::now();
        if (threads == 1) {
          ids.clear();
          tok.encode_ordinary(text, ids);
        } else {
          ids = tok.encode_ordinary_parallel(text, threads);
        }
        best = std::min(best, std::chrono::duration<double>(clk::now() - a).count());
      }
      auto a = clk::now();
      const std::string back = tok.decode(ids);
      const double dec = std::chrono::duration<double>(clk::now() - a).count();
      std::printf(
          "load %.1f ms | %zu bytes -> %zu tokens | encode best of %d: %.2f ms = %.1f MB/s, "
          "%.2f M tok/s (threads=%u) | decode %.1f MB/s | roundtrip %s\n",
          load_ms, text.size(), ids.size(), iters, best * 1e3, text.size() / best / 1e6,
          ids.size() / best / 1e6, threads, back.size() / dec / 1e6, back == text ? "ok" : "MISMATCH");
    } else {
      throw std::runtime_error("unknown command: " + cmd);
    }
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "error: %s\n", ex.what());
    return 1;
  }
  return 0;
}
