// Token usage when encoding a counts file (piece -> frequency) with a vocabulary.
//
//   ft_usage PATTERN VOCAB.tiktoken COUNTS [OUT_USAGE.bin]
//
// Prints how many tokens are used how often, and the total token count. OUT_USAGE gets
// a u64 usage count per rank.
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "fasttok.hpp"

using namespace fasttok;

int main(int argc, char** argv) {
  if (argc < 4) return 2;
  const std::string pat = argv[1];
  const Pattern P = pat == "o200k" ? Pattern::O200K : pat == "cl100k" ? Pattern::CL100K : Pattern::R50K;
  const Tokenizer t = Tokenizer::load(argv[2], P);
  FILE* f = std::fopen(argv[3], "rb");
  uint64_t n;
  if (std::fread(&n, 8, 1, f) != 1) return 1;
  std::vector<uint64_t> use(t.n_vocab(), 0);
  std::vector<Rank> ids;
  std::string buf;
  uint64_t total = 0, bytes = 0;
  for (uint64_t k = 0; k < n; ++k) {
    uint64_t c;
    uint16_t l;
    if (std::fread(&c, 8, 1, f) != 1 || std::fread(&l, 2, 1, f) != 1) return 1;
    buf.resize(l);
    if (l && std::fread(buf.data(), 1, l, f) != l) return 1;
    ids.clear();
    t.encode_single_piece(buf, ids);
    for (Rank r : ids) use[r] += c;
    total += c * ids.size();
    bytes += c * l;
  }
  std::fclose(f);
  size_t z = 0, lt10 = 0, lt100 = 0, lt1000 = 0;
  for (size_t r = 256; r < use.size(); ++r) {
    z += use[r] == 0;
    lt10 += use[r] < 10;
    lt100 += use[r] < 100;
    lt1000 += use[r] < 1000;
  }
  std::printf("%s: %.3f bytes/token on training pieces | merged tokens used 0x: %zu, <10x: %zu, <100x: %zu, <1000x: %zu (of %zu)\n",
              argv[2], (double)bytes / total, z, lt10, lt100, lt1000, use.size() - 256);
  if (argc > 4) {
    FILE* o = std::fopen(argv[4], "wb");
    std::fwrite(use.data(), 8, use.size(), o);
    std::fclose(o);
  }
}
