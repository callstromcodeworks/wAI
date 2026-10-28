#!/usr/bin/env bash
# Rebuilds ec200k_base.tiktoken from scratch.
#
#   vocab/training/train.sh PATH/TO/cl100k_base.tiktoken
#
# Needs git, curl, pip, python3 and a C++17 compiler, network access to GitHub and PyPI,
# about 8 GB of disk and 4 GB of RAM. Work files go to $VOCAB_WORK (default
# vocab/training/work). Upstream repos and books change over time, so a fresh run
# reproduces the method, not necessarily the identical file; given the same inputs the
# pipeline is deterministic.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export VOCAB_WORK=${VOCAB_WORK:-$HERE/work}
CL100K=${1:?usage: train.sh PATH/TO/cl100k_base.tiktoken}
INC=$HERE/../../include
B=$VOCAB_WORK/bin
mkdir -p "$B" "$VOCAB_WORK/raw/nltk" "$VOCAB_WORK/raw/gb" "$VOCAB_WORK/counts/src"

echo "== building tools"
for t in ft_count bpe_train ft_eval; do
  g++ -std=c++17 -O3 -march=native -I"$INC" -o "$B/$t" "$HERE/$t.cpp"
done

echo "== fetching data"
"$HERE/fetch_repos.sh"
pip download gitberg==0.8.8 --no-deps --no-binary :all: -q -d "$VOCAB_WORK/raw/gb"
tar xzf "$VOCAB_WORK/raw/gb/gitberg-0.8.8.tar.gz" -C "$VOCAB_WORK/raw/gb"
python3 "$HERE/fetch_books.py" 1100 40
for z in reuters brown abc; do
  curl -sSfL -o "$VOCAB_WORK/raw/nltk/$z.zip" \
    "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/$z.zip"
done
python3 "$HERE/assemble.py"

echo "== counting pieces per source (cl100k pre-tokenizer)"
for f in "$VOCAB_WORK"/shards/src/*.shard; do
  "$B/ft_count" cl100k 128 "$VOCAB_WORK/counts/src/$(basename "$f" .shard).counts" "$f" 2>/dev/null
done
python3 "$HERE/robust_counts.py" 2 "$VOCAB_WORK/counts/robust_2.counts" "$VOCAB_WORK/counts/src" "$VOCAB_WORK/shards/src"

echo "== training"
"$B/bpe_train" "$VOCAB_WORK/counts/robust_2.counts" 200000 1 "$VOCAB_WORK/ec200k_base.tiktoken" \
  --seed "$CL100K" 100256 cl100k

echo "== held-out evaluation (bytes/token, higher is better)"
python3 "$HERE/summary.py" cl100k=cl100k:"$CL100K" ec200k=cl100k:"$VOCAB_WORK/ec200k_base.tiktoken"
echo "wrote $VOCAB_WORK/ec200k_base.tiktoken"
