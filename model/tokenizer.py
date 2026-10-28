"""ec200k tokenizer for the model: a ctypes wrapper over fasttok's C ABI.

Tokenizer(n_ranks=N) uses the first N ranks of ec200k_base, an exact N-token BPE vocabulary,
with the five special tokens renumbered to N .. N+4. The vocabulary size is the main lever
between weight size (the embedding is vocab x d_model) and sequence length (more tokens per
byte of text means more KV cache and compute per byte).

ctypes drops the GIL for the duration of each foreign call, so encode_batch fans a list of
documents out over Python threads and the C++ encoder runs on all cores.
"""
import ctypes
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LIB = os.path.join(ROOT, "build", "fasttok.dll" if os.name == "nt" else "libfasttok.so")
DEFAULT_VOCAB = os.path.join(ROOT, "vocab", "ec200k_base.tiktoken")

SPECIAL_NAMES = ["<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|endofprompt|>"]
FULL_RANKS = 200000


def special_tokens(n_ranks=FULL_RANKS):
    return {name: n_ranks + i for i, name in enumerate(SPECIAL_NAMES)}


# Named rank-prefix sizes. The smaller ones leave room for the 5 specials inside a power of
# two (16,384 / 32,768 / 65,536 ids), so the embedding needs no padding and ec64k fits uint16.
# ec100k is exactly cl100k_base's regular tokens (ec200k's first 100,256 ranks).
VOCABS = {"ec16k": 16379, "ec32k": 32763, "ec64k": 65531, "ec100k": 100256, "ec200k": 200000}

_u32p = ctypes.POINTER(ctypes.c_uint32)
_u64p = ctypes.POINTER(ctypes.c_uint64)
_ERR = ctypes.c_size_t(-1).value


class Tokenizer:
    def __init__(self, n_ranks=FULL_RANKS, lib_path=DEFAULT_LIB, vocab_path=DEFAULT_VOCAB):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"{lib_path} not found; build fasttok first (see model/README.md)")
        lib = ctypes.CDLL(lib_path)
        lib.ft_new_ec_prefix.restype = ctypes.c_void_p
        lib.ft_new_ec_prefix.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
        lib.ft_free.argtypes = [ctypes.c_void_p]
        lib.ft_last_error.restype = ctypes.c_char_p
        lib.ft_encode_batch.restype = ctypes.c_size_t
        lib.ft_encode_batch.argtypes = [ctypes.c_void_p, ctypes.c_char_p, _u64p, ctypes.c_size_t, _u32p,
                                        ctypes.c_size_t, _u64p, ctypes.c_int, ctypes.c_uint]
        lib.ft_decode.restype = ctypes.c_size_t
        lib.ft_decode.argtypes = [ctypes.c_void_p, _u32p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
        self._lib = lib
        self._h = lib.ft_new_ec_prefix(vocab_path.encode(), n_ranks)
        if not self._h:
            raise RuntimeError(lib.ft_last_error().decode())
        self.n_ranks = n_ranks
        self.special_tokens = special_tokens(n_ranks)
        self.eot = self.special_tokens["<|endoftext|>"]
        self.n_vocab = n_ranks + len(SPECIAL_NAMES)

    def __del__(self):
        if getattr(self, "_h", None):
            self._lib.ft_free(self._h)
            self._h = None

    def _encode_bufs(self, bufs, allow_special):
        """Encode a list of bytes objects in one C call. Returns (flat uint32 ids, per-text counts)."""
        lens = np.fromiter((len(b) for b in bufs), dtype=np.uint64, count=len(bufs))
        offs = np.zeros(len(bufs) + 1, dtype=np.uint64)
        np.cumsum(lens, out=offs[1:])
        cap = max(int(offs[-1]), 1)  # every token covers at least one byte
        ids = np.empty(cap, dtype=np.uint32)
        counts = np.empty(max(len(bufs), 1), dtype=np.uint64)
        total = self._lib.ft_encode_batch(self._h, b"".join(bufs), offs.ctypes.data_as(_u64p), len(bufs),
                                          ids.ctypes.data_as(_u32p), cap, counts.ctypes.data_as(_u64p),
                                          1 if allow_special else 0, 0)
        if total == _ERR:
            raise RuntimeError(self._lib.ft_last_error().decode())
        return ids[:total], counts[:len(bufs)]

    def encode(self, text, allow_special=False):
        """Token ids (np.uint32) for one string. allow_special maps <|...|> markers to special ids."""
        ids, _ = self._encode_bufs([text.encode("utf-8")], allow_special)
        return ids.copy()

    def encode_batch(self, texts, threads=None, prepend_eot=False):
        """Encode many documents in parallel. Returns a list of np.uint32 arrays.

        prepend_eot puts <|endoftext|> in front of each document, the delimiter used in the
        pretraining shards.
        """
        threads = threads or os.cpu_count() or 1
        bufs = [t.encode("utf-8") for t in texts]
        if not bufs:
            return []
        n_groups = min(len(bufs), threads * 4)  # several groups per thread to even out load
        bounds = np.linspace(0, len(bufs), n_groups + 1, dtype=np.int64)
        groups = [bufs[bounds[g]:bounds[g + 1]] for g in range(n_groups)]
        with ThreadPoolExecutor(threads) as pool:
            results = list(pool.map(lambda g: self._encode_bufs(g, False), groups))
        out = []
        for ids, counts in results:
            splits = np.cumsum(counts)[:-1].astype(np.int64)
            for doc in np.split(ids, splits):
                out.append(np.concatenate(([self.eot], doc)).astype(np.uint32) if prepend_eot else doc)
        return out

    def decode_bytes(self, ids):
        arr = np.ascontiguousarray(ids, dtype=np.uint32)
        cap = len(arr) * 64 + 64
        while True:
            buf = ctypes.create_string_buffer(cap)
            n = self._lib.ft_decode(self._h, arr.ctypes.data_as(_u32p), len(arr), buf, cap)
            if n == _ERR:
                raise RuntimeError(self._lib.ft_last_error().decode())
            if n <= cap:
                return buf.raw[:n]
            cap = n

    def decode(self, ids, errors="replace"):
        return self.decode_bytes(ids).decode("utf-8", errors=errors)
