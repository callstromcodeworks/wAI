// Count pre-tokenized pieces with fasttok's exact scanner.
//
//   ft_count PATTERN MAX_PIECE_LEN OUT.counts SHARD [SHARD...]
//
// PATTERN: o200k | cl100k | r50k. Shards hold documents as [u32 len][bytes] records;
// each document is scanned separately (like encoding it on its own).
// Output: [u64 n] then n records of [u64 count][u16 len][bytes], sorted by count desc.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "fasttok.hpp"

using namespace fasttok;

struct Map {
  struct Slot {
    uint64_t h;
    uint64_t count;
    uint32_t off, len;
  };
  std::vector<Slot> slots;
  std::vector<uint8_t> arena;
  size_t used = 0, mask = 0;
  Map() { slots.assign(size_t(1) << 22, Slot{0, 0, 0, 0}); mask = slots.size() - 1; }
  void grow() {
    std::vector<Slot> old(size_t(slots.size()) * 2, Slot{0, 0, 0, 0});
    old.swap(slots);
    mask = slots.size() - 1;
    for (auto& s : old)
      if (s.count) {
        size_t i = s.h & mask;
        while (slots[i].count) i = (i + 1) & mask;
        slots[i] = s;
      }
  }
  void add(const uint8_t* p, uint32_t len) {
    const uint64_t h = detail::hash_bytes(p, len) | 1;
    size_t i = h & mask;
    for (;;) {
      Slot& s = slots[i];
      if (!s.count) break;
      if (s.h == h && s.len == len && std::memcmp(arena.data() + s.off, p, len) == 0) {
        ++s.count;
        return;
      }
      i = (i + 1) & mask;
    }
    slots[i] = Slot{h, 1, (uint32_t)arena.size(), len};
    arena.insert(arena.end(), p, p + len);
    if (++used * 2 > slots.size()) grow();
    if (arena.size() > 0xF0000000u) { std::fprintf(stderr, "arena too large\n"); std::exit(1); }
  }
};

template <class Scan>
void scan_doc(const uint8_t* s, size_t n, size_t maxlen, Map& m, size_t& pieces, size_t& skipped) {
  for (size_t i = 0; i < n;) {
    const size_t e = Scan::next(s, n, i);
    ++pieces;
    if (e - i <= maxlen) m.add(s + i, uint32_t(e - i));
    else ++skipped;
    i = e;
  }
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s PATTERN MAX_PIECE_LEN OUT SHARD...\n", argv[0]);
    return 2;
  }
  const std::string pat = argv[1];
  const size_t maxlen = std::stoul(argv[2]);
  Map m;
  size_t docs = 0, bytes = 0, pieces = 0, skipped = 0;
  for (int a = 4; a < argc; ++a) {
    FILE* f = std::fopen(argv[a], "rb");
    if (!f) { std::perror(argv[a]); return 1; }
    std::vector<uint8_t> buf;
    uint32_t len;
    while (std::fread(&len, 4, 1, f) == 1) {
      buf.resize(len);
      if (len && std::fread(buf.data(), 1, len, f) != len) { std::fprintf(stderr, "truncated shard\n"); return 1; }
      ++docs;
      bytes += len;
      if (pat == "o200k") scan_doc<detail::O200K>(buf.data(), len, maxlen, m, pieces, skipped);
      else if (pat == "cl100k") scan_doc<detail::CL100K>(buf.data(), len, maxlen, m, pieces, skipped);
      else if (pat == "r50k") scan_doc<detail::R50K>(buf.data(), len, maxlen, m, pieces, skipped);
      else { std::fprintf(stderr, "bad pattern\n"); return 2; }
    }
    std::fclose(f);
  }
  std::vector<const Map::Slot*> v;
  v.reserve(m.used);
  for (auto& s : m.slots) if (s.count) v.push_back(&s);
  std::sort(v.begin(), v.end(), [&](const Map::Slot* x, const Map::Slot* y) {
    if (x->count != y->count) return x->count > y->count;
    if (x->len != y->len) return x->len < y->len;
    return std::memcmp(m.arena.data() + x->off, m.arena.data() + y->off, x->len) < 0;
  });
  FILE* o = std::fopen(argv[3], "wb");
  uint64_t nrec = v.size();
  std::fwrite(&nrec, 8, 1, o);
  size_t singles = 0;
  for (auto* s : v) {
    const uint16_t l = (uint16_t)s->len;
    std::fwrite(&s->count, 8, 1, o);
    std::fwrite(&l, 2, 1, o);
    std::fwrite(m.arena.data() + s->off, 1, s->len, o);
    singles += s->count == 1;
  }
  std::fclose(o);
  std::fprintf(stderr, "%s: %zu docs, %.1f MB, %zu pieces, %zu unique (%zu seen once), %zu skipped (> %zu bytes)\n",
               pat.c_str(), docs, bytes / 1e6, pieces, v.size(), singles, skipped, maxlen);
}
