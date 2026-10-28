// Compression of held-out shards under one or more vocabularies, encoded with fasttok.
//
//   ft_eval NAME=PATTERN:VOCAB.tiktoken [...] -- SHARD [SHARD...]
//
// Prints bytes per token (higher = better compression) for each vocab on each shard.
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "fasttok.hpp"

using namespace fasttok;

int main(int argc, char** argv) {
  std::vector<std::pair<std::string, Tokenizer>> toks;
  int a = 1;
  for (; a < argc && std::string(argv[a]) != "--"; ++a) {
    const std::string spec = argv[a];
    const auto eq = spec.find('='), colon = spec.find(':');
    const std::string name = spec.substr(0, eq), pat = spec.substr(eq + 1, colon - eq - 1),
                      path = spec.substr(colon + 1);
    const Pattern p = pat == "o200k" ? Pattern::O200K : pat == "cl100k" ? Pattern::CL100K : Pattern::R50K;
    toks.emplace_back(name, Tokenizer::load(path, p));
  }
  std::printf("%-22s %10s", "shard", "MB");
  for (auto& t : toks) std::printf(" %14s", t.first.c_str());
  std::printf("   (bytes/token; higher is better)\n");
  for (++a; a < argc; ++a) {
    FILE* f = std::fopen(argv[a], "rb");
    if (!f) { std::perror(argv[a]); return 1; }
    std::vector<std::string> docs;
    uint32_t len;
    size_t bytes = 0;
    while (std::fread(&len, 4, 1, f) == 1) {
      std::string d(len, '\0');
      if (len && std::fread(d.data(), 1, len, f) != len) return 1;
      bytes += len;
      docs.push_back(std::move(d));
    }
    std::fclose(f);
    std::string name = argv[a];
    name = name.substr(name.find_last_of('/') + 1);
    std::printf("%-22s %10.1f", name.c_str(), bytes / 1e6);
    for (auto& t : toks) {
      size_t n = 0;
      for (auto& d : docs) n += t.second.count(d);
      std::printf(" %14.3f", (double)bytes / n);
    }
    std::printf("\n");
  }
}
