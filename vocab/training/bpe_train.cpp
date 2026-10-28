// Byte-level BPE trainer producing a tiktoken-format vocabulary for fasttok.
//
//   bpe_train COUNTS VOCAB_SIZE MIN_WORD_COUNT OUT.tiktoken [--log MERGES_LOG]
//             [--seed SEED.tiktoken K PATTERN]
//
// --seed starts from the first K ranks of an existing vocabulary (a rank prefix of a
// BPE vocabulary is closed under its own merges): every word is first encoded with
// that prefix using fasttok's rank-based BPE, and training continues from there, with
// new tokens ranked K, K+1, ...
//
// COUNTS comes from ft_count (piece -> frequency, using fasttok's exact pre-tokenizer).
// Ranks 0..255 are the single bytes (rank == byte value); every later rank is one merge,
// in training order, which is exactly how a rank-based encoder (tiktoken / fasttok)
// replays them. Each step merges the most frequent adjacent pair (ties: smallest
// (left, right) id pair). If a merge produces bytes that already form a token, the
// existing id is reused and no new rank is spent.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "fasttok.hpp"

using u32 = uint32_t;
using u64 = uint64_t;
using i64 = int64_t;

static inline u64 mix(u64 k) {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdull;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ull;
  k ^= k >> 33;
  return k;
}

// pair key (a<<32|b) -> pair id
struct PairMap {
  std::vector<u64> keys;
  std::vector<u32> vals;
  size_t used = 0, mask = 0;
  PairMap() { init(size_t(1) << 20); }
  void init(size_t cap) {
    keys.assign(cap, ~u64(0));
    vals.assign(cap, 0);
    mask = cap - 1;
    used = 0;
  }
  u32* find(u64 k) {
    size_t i = mix(k) & mask;
    for (;;) {
      if (keys[i] == k) return &vals[i];
      if (keys[i] == ~u64(0)) return nullptr;
      i = (i + 1) & mask;
    }
  }
  // returns {id, inserted}
  std::pair<u32, bool> get_or_add(u64 k, u32 new_id) {
    size_t i = mix(k) & mask;
    for (;;) {
      if (keys[i] == k) return {vals[i], false};
      if (keys[i] == ~u64(0)) break;
      i = (i + 1) & mask;
    }
    keys[i] = k;
    vals[i] = new_id;
    if (++used * 2 > keys.size()) grow();
    return {new_id, true};
  }
  void grow() {
    std::vector<u64> ok;
    std::vector<u32> ov;
    ok.swap(keys);
    ov.swap(vals);
    init(ok.size() * 2);
    for (size_t i = 0; i < ok.size(); ++i)
      if (ok[i] != ~u64(0)) {
        size_t j = mix(ok[i]) & mask;
        while (keys[j] != ~u64(0)) j = (j + 1) & mask;
        keys[j] = ok[i];
        vals[j] = ov[i];
        ++used;
      }
  }
};

struct Node {
  u32 word, next;
};
struct PairInfo {
  i64 count;
  u64 key;
  u32 head;   // node list of words that (may) contain the pair
  u32 stamp;  // last merge that pushed it to the heap
};
struct HeapEnt {
  i64 count;
  u64 key;
  bool operator<(const HeapEnt& o) const {  // max-heap: higher count, then smaller key
    return count != o.count ? count < o.count : key > o.key;
  }
};

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s COUNTS VOCAB_SIZE MIN_WORD_COUNT OUT.tiktoken [--log F] [--seed V K PATTERN]\n", argv[0]);
    return 2;
  }
  const char* log_path = nullptr;
  std::string seed_path, seed_pat;
  size_t seed_k = 0;
  for (int a = 5; a < argc; ++a) {
    const std::string arg = argv[a];
    if (arg == "--log" && a + 1 < argc) log_path = argv[++a];
    else if (arg == "--seed" && a + 3 < argc) { seed_path = argv[a + 1]; seed_k = std::stoul(argv[a + 2]); seed_pat = argv[a + 3]; a += 3; }
    else { std::fprintf(stderr, "bad argument %s\n", arg.c_str()); return 2; }
  }
  using clk = std::chrono::steady_clock;
  const auto t0 = clk::now();
  auto secs = [&] { return std::chrono::duration<double>(clk::now() - t0).count(); };
  const size_t vocab_size = std::stoul(argv[2]);
  const u64 min_count = std::stoull(argv[3]);

  // ---- load words ----
  std::vector<u64> wcount;
  std::vector<u64> woff;
  std::vector<uint16_t> wlen;
  std::vector<u32> sym;
  std::vector<std::string> wbytes;
  {
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::perror(argv[1]); return 1; }
    u64 n;
    if (std::fread(&n, 8, 1, f) != 1) return 1;
    std::vector<uint8_t> buf;
    for (u64 k = 0; k < n; ++k) {
      u64 c;
      uint16_t l;
      if (std::fread(&c, 8, 1, f) != 1 || std::fread(&l, 2, 1, f) != 1) return 1;
      buf.resize(l);
      if (l && std::fread(buf.data(), 1, l, f) != l) return 1;
      if (c < min_count) break;  // sorted by count desc
      if (l < 2) continue;        // single bytes never take part in a merge
      wcount.push_back(c);
      woff.push_back(sym.size());
      wlen.push_back(l);
      for (uint8_t b : buf) sym.push_back(b);
      wbytes.emplace_back(reinterpret_cast<const char*>(buf.data()), l);
    }
    std::fclose(f);
  }
  const size_t W = wcount.size();
  std::fprintf(stderr, "[%.0fs] %zu words, %zu symbols\n", secs(), W, sym.size());

  // ---- vocabulary ----
  std::vector<std::string> tok;
  std::unordered_map<std::string, u32> tok_id;
  tok_id.reserve(vocab_size * 2);
  if (seed_path.empty()) {
    tok.resize(256);
    for (int b = 0; b < 256; ++b) tok[b] = std::string(1, char(b));
  } else {
    // First K ranks of the seed vocabulary, re-encoded with fasttok.
    const std::string text = fasttok::detail::read_file(seed_path);
    std::string prefix;
    size_t pos = 0, kept = 0;
    while (pos < text.size() && kept < seed_k) {
      size_t eol = text.find('\n', pos);
      if (eol == std::string::npos) eol = text.size();
      const std::string line = text.substr(pos, eol - pos);
      pos = eol + 1;
      if (line.empty()) continue;
      const size_t sp = line.find(' ');
      if (std::stoul(line.substr(sp + 1)) != kept) { std::fprintf(stderr, "seed ranks must be 0..K-1 in order\n"); return 1; }
      std::string bytes;
      fasttok::detail::base64_append(line.substr(0, sp), bytes);
      tok.push_back(bytes);
      prefix += line + "\n";
      ++kept;
    }
    const fasttok::Pattern P = seed_pat == "o200k" ? fasttok::Pattern::O200K
                             : seed_pat == "cl100k" ? fasttok::Pattern::CL100K : fasttok::Pattern::R50K;
    const auto seed_tok = fasttok::Tokenizer::from_tiktoken_text(prefix, P);
    std::vector<u32> nsym;
    nsym.reserve(sym.size());
    std::vector<fasttok::Rank> ids;
    for (size_t w = 0; w < wbytes.size(); ++w) {
      ids.clear();
      seed_tok.encode_single_piece(wbytes[w], ids);
      woff[w] = nsym.size();
      wlen[w] = (uint16_t)ids.size();
      nsym.insert(nsym.end(), ids.begin(), ids.end());
    }
    sym.swap(nsym);
    std::fprintf(stderr, "[%.0fs] seeded with %zu tokens; %zu symbols after seed encoding\n", secs(), tok.size(), sym.size());
  }
  for (u32 t = 0; t < tok.size(); ++t) tok_id.emplace(tok[t], t);

  // ---- pairs ----
  PairMap pm;
  std::vector<PairInfo> pinfo;
  std::vector<Node> pool;
  pool.reserve(sym.size() + 16);
  pinfo.reserve(1 << 20);
  auto pair_id = [&](u32 a, u32 b) -> u32 {
    const u64 key = (u64(a) << 32) | b;
    auto [id, ins] = pm.get_or_add(key, (u32)pinfo.size());
    if (ins) pinfo.push_back(PairInfo{0, key, ~u32(0), 0});
    return id;
  };
  auto add_node = [&](u32 pid, u32 w) {
    pool.push_back(Node{w, pinfo[pid].head});
    pinfo[pid].head = (u32)(pool.size() - 1);
  };
  for (size_t w = 0; w < W; ++w) {
    const u32* s = &sym[woff[w]];
    for (size_t k = 0; k + 1 < wlen[w]; ++k) {
      const u32 id = pair_id(s[k], s[k + 1]);
      pinfo[id].count += (i64)wcount[w];
      add_node(id, (u32)w);
    }
  }
  std::priority_queue<HeapEnt> heap;
  for (auto& p : pinfo) heap.push(HeapEnt{p.count, p.key});
  std::fprintf(stderr, "[%.0fs] %zu initial pairs, %zu list nodes\n", secs(), pinfo.size(), pool.size());

  FILE* mlog = log_path ? std::fopen(log_path, "w") : nullptr;
  std::vector<u32> wstamp(W, 0);
  std::vector<u32> touched;
  u32 stamp = 0;
  size_t reused = 0;

  while (tok.size() < vocab_size && !heap.empty()) {
    const HeapEnt top = heap.top();
    heap.pop();
    const u32 pid = *pm.find(top.key);
    if (pinfo[pid].count != top.count) {  // stale entry
      if (pinfo[pid].count > 0) heap.push(HeapEnt{pinfo[pid].count, top.key});
      continue;
    }
    if (top.count <= 0) break;
    const u32 a = u32(top.key >> 32), b = u32(top.key);
    std::string bytes = tok[a] + tok[b];
    u32 z;
    auto it = tok_id.find(bytes);
    if (it != tok_id.end()) {
      z = it->second;
      ++reused;
    } else {
      z = (u32)tok.size();
      tok.push_back(bytes);
      tok_id.emplace(std::move(bytes), z);
      if (mlog) std::fprintf(mlog, "%u %u %u %lld\n", z, a, b, (long long)top.count);
    }
    ++stamp;
    touched.clear();
    // Visit every word on the pair's list (duplicates and stale entries are skipped).
    for (u32 nd = pinfo[pid].head; nd != ~u32(0); nd = pool[nd].next) {
      const u32 w = pool[nd].word;
      if (wstamp[w] == stamp) continue;
      wstamp[w] = stamp;
      u32* s = &sym[woff[w]];
      const size_t L = wlen[w];
      bool has = false;
      for (size_t k = 0; k + 1 < L; ++k)
        if (s[k] == a && s[k + 1] == b) { has = true; break; }
      if (!has) continue;
      const i64 c = (i64)wcount[w];
      for (size_t k = 0; k + 1 < L; ++k) pinfo[*pm.find((u64(s[k]) << 32) | s[k + 1])].count -= c;
      size_t j = 0;
      for (size_t i = 0; i < L;) {
        if (i + 1 < L && s[i] == a && s[i + 1] == b) {
          s[j++] = z;
          i += 2;
        } else {
          s[j++] = s[i++];
        }
      }
      wlen[w] = (uint16_t)j;
      for (size_t k = 0; k + 1 < j; ++k) {
        const u32 id = pair_id(s[k], s[k + 1]);
        pinfo[id].count += c;
        if (s[k] == z || s[k + 1] == z) {
          add_node(id, w);
          if (pinfo[id].stamp != stamp) {
            pinfo[id].stamp = stamp;
            touched.push_back(id);
          }
        }
      }
    }
    pinfo[pid].head = ~u32(0);  // all occurrences merged; drop the list
    for (u32 id : touched)
      if (pinfo[id].count > 0) heap.push(HeapEnt{pinfo[id].count, pinfo[id].key});
    if (tok.size() % 10000 == 0)
      std::fprintf(stderr, "[%.0fs] vocab %zu, last merge count %lld, pairs %zu, nodes %zu, heap %zu\n",
                   secs(), tok.size(), (long long)top.count, pinfo.size(), pool.size(), heap.size());
  }
  if (mlog) std::fclose(mlog);

  // ---- write .tiktoken ----
  static const char* B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  FILE* o = std::fopen(argv[4], "w");
  for (size_t r = 0; r < tok.size(); ++r) {
    const std::string& t = tok[r];
    std::string enc;
    size_t i = 0;
    for (; i + 3 <= t.size(); i += 3) {
      const u32 v = (u32(uint8_t(t[i])) << 16) | (u32(uint8_t(t[i + 1])) << 8) | uint8_t(t[i + 2]);
      enc += B64[v >> 18]; enc += B64[(v >> 12) & 63]; enc += B64[(v >> 6) & 63]; enc += B64[v & 63];
    }
    if (t.size() - i == 1) {
      const u32 v = u32(uint8_t(t[i])) << 16;
      enc += B64[v >> 18]; enc += B64[(v >> 12) & 63]; enc += "==";
    } else if (t.size() - i == 2) {
      const u32 v = (u32(uint8_t(t[i])) << 16) | (u32(uint8_t(t[i + 1])) << 8);
      enc += B64[v >> 18]; enc += B64[(v >> 12) & 63]; enc += B64[(v >> 6) & 63]; enc += '=';
    }
    std::fprintf(o, "%s %zu\n", enc.c_str(), r);
  }
  std::fclose(o);
  std::fprintf(stderr, "[%.0fs] wrote %zu tokens (%zu merges reused an existing token)\n", secs(), tok.size(), reused);
}
