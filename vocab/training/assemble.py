#!/usr/bin/env python3
"""Build training / held-out shards from the cloned repos, books and news corpora.

Shard format: documents as [u32 little-endian length][utf-8 bytes].
"""
import glob
import hashlib
import io
import os
import random
import struct
import zipfile

ROOT = os.environ.get("VOCAB_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work"))
OUT = f"{ROOT}/shards"
SRC = f"{ROOT}/shards/src"
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/src", exist_ok=True)

CODE_EXT = {
    "py": "python", "pyi": "python", "pyx": "python",
    "js": "js", "jsx": "js", "mjs": "js", "cjs": "js", "ts": "js", "tsx": "js", "mts": "js", "vue": "js", "svelte": "js",
    "c": "c", "h": "c", "cc": "c", "cpp": "c", "cxx": "c", "hpp": "c", "hh": "c", "inl": "c", "m": "c", "mm": "c",
    "java": "jvm", "kt": "jvm", "kts": "jvm", "scala": "jvm", "gradle": "jvm", "groovy": "jvm",
    "go": "go", "rs": "rust", "cs": "csharp", "swift": "swift", "rb": "ruby", "erb": "ruby", "php": "php",
    "dart": "dart", "lua": "lua", "vim": "other", "sh": "shell", "bash": "shell", "zsh": "shell", "ps1": "shell",
    "psm1": "shell", "sql": "other", "html": "web", "css": "web", "scss": "web", "less": "web",
    "yaml": "config", "yml": "config", "toml": "config", "json": "config", "cmake": "config", "proto": "other",
}
SPECIAL_NAMES = {"Makefile": "config", "Dockerfile": "config", "CMakeLists.txt": "config", "Gemfile": "ruby", "Rakefile": "ruby"}
PROSE_EXT = {"md", "mdx", "rst", "txt", "adoc", "asciidoc", "sgml"}
SKIP_DIRS = ("/node_modules/", "/vendor/", "/third_party/", "/thirdparty/", "/3rdparty/", "/dist/", "/build/",
             "/.git/", "/__snapshots__/", "/generated/", "/fixtures/", "/testdata/", "/locale/", "/locales/",
             "/i18n/", "/translations/")
SKIP_NAMES = ("package-lock.json", "yarn.lock", "pnpm-lock.yaml", "Cargo.lock", "poetry.lock", "go.sum", "composer.lock")


def classify(path):
    name = os.path.basename(path)
    if name in SPECIAL_NAMES:
        return "code", SPECIAL_NAMES[name]
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext in PROSE_EXT:
        return "prose", "docs"
    if ext in CODE_EXT:
        return "code", CODE_EXT[ext]
    return None, None


def good_text(raw, kind):
    if len(raw) < 64 or len(raw) > 400_000:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if "\x00" in text:
        return None
    lines = text.split("\n")
    if max(len(l) for l in lines) > 1000 or len(text) / max(1, len(lines)) > 160:
        return None  # minified / generated / data
    if ".min." in text[:0]:
        return None
    return text.replace("\r\n", "\n")


def repo_docs(name, split, cap_code, cap_prose, rng):
    files = []
    for path in glob.glob(f"{ROOT}/raw/repos/{name}/**/*", recursive=True):
        if not os.path.isfile(path) or any(d in path for d in SKIP_DIRS) or path.endswith(SKIP_NAMES) or ".min." in path:
            continue
        kind, lang = classify(path)
        if kind:
            files.append((path, kind, lang))
    rng.shuffle(files)
    out, used = [], {"code": 0, "prose": 0}
    caps = {"code": cap_code, "prose": cap_prose}
    for path, kind, lang in files:
        if used[kind] >= caps[kind]:
            continue
        if kind == "code" and lang == "config" and os.path.getsize(path) > 50_000:
            continue
        text = good_text(open(path, "rb").read(), kind)
        if text is None:
            continue
        out.append((kind, lang, text))
        used[kind] += len(text.encode())
    return out


class Shards:
    def __init__(self):
        self.files, self.seen, self.stats = {}, set(), {}

    def add(self, shard, text, dedup=True):
        h = hashlib.sha1(text.encode()).digest()
        if dedup:
            if h in self.seen:
                self.last_dup = True
                return
            self.seen.add(h)
            self.last_dup = False
        elif self.last_dup:
            return  # mirror the dedup decision of the preceding combined-shard add
        if shard not in self.files:
            self.files[shard] = open(f"{OUT}/{shard}.shard", "wb")
        b = text.encode()
        self.files[shard].write(struct.pack("<I", len(b)) + b)
        n, sz = self.stats.get(shard, (0, 0))
        self.stats[shard] = (n + 1, sz + len(b))


def main():
    rng = random.Random(7)
    sh = Shards()
    # Held-out first, so identical files (e.g. shared licences) never leak into training.
    repos = [l.split() for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "repos.txt")) if l.strip() and not l.startswith("#")]
    for split, name, *_ in repos:
        if split != "heldout":
            continue
        for kind, lang, text in repo_docs(name, split, 6_000_000, 6_000_000, rng):
            sh.add(f"held_code_{lang}" if kind == "code" else "held_en_docs", text)
    for p in sorted(glob.glob(f"{ROOT}/raw/books/heldout/*.txt")):
        sh.add("held_en_books", open(p).read())
    for z in ("reuters", "brown", "abc"):
        with zipfile.ZipFile(f"{ROOT}/raw/nltk/{z}.zip") as zf:
            for n in zf.namelist():
                if n.endswith("/") or "README" in n or n.endswith(("cats.txt", "CONTENTS", "stopwords")):
                    continue
                raw = zf.read(n)
                try:
                    t = raw.decode("utf-8")
                except UnicodeDecodeError:
                    t = raw.decode("latin-1")
                if z == "brown":  # strip part-of-speech tags: word/tag
                    t = " ".join(w.rsplit("/", 1)[0] for w in t.split())
                if len(t) > 200:
                    sh.add("held_en_news", t)
    # Training.
    for split, name, *_ in repos:
        if split != "train":
            continue
        prose_cap = 160_000_000 if name in ("azuredocs", "dotnetdocs", "mdn", "githubdocs", "sqldocs", "psdocs") else 30_000_000
        for kind, lang, text in repo_docs(name, split, 30_000_000, prose_cap, rng):
            sh.add("train_code" if kind == "code" else "train_en", text)
            sh.add(f"src/{'code' if kind == 'code' else 'en'}__{name}", text, dedup=False)
    for p in sorted(glob.glob(f"{ROOT}/raw/books/train/*.txt")):
        t = open(p).read()
        sh.add("train_en", t)
        sh.add(f"src/en__book_{os.path.basename(p)[:-4]}", t, dedup=False)
    for f in sh.files.values():
        f.close()
    for k, (n, sz) in sorted(sh.stats.items()):
        if k.startswith("src/"):
            continue
        print(f"{k:24s} {n:8d} docs {sz / 1e6:9.1f} MB")


main()
