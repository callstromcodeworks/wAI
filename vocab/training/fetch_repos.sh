#!/usr/bin/env bash
# Shallow (and, where dirs are given, sparse partial) clones of every repo in repos.txt.
set -u
set -f  # sparse patterns like **/*.md must reach git unexpanded
HERE=$(cd "$(dirname "$0")" && pwd)
VOCAB_WORK=${VOCAB_WORK:-$HERE/work}
mkdir -p "$VOCAB_WORK/raw/repos" && cd "$VOCAB_WORK/raw/repos"
grep -v '^#' "$HERE/repos.txt" | grep -v '^\s*$' | while read -r split name url dirs; do
  [ -d "$name/.git" ] && { echo "skip $name"; continue; }
  if [ -z "$dirs" ]; then
    timeout 900 git clone -q --depth 1 "$url" "$name" || echo "FAILED $name"
  else
    timeout 900 git clone -q --depth 1 --filter=blob:none --sparse "$url" "$name" &&
      (cd "$name" && timeout 900 git sparse-checkout set --no-cone $(for d in $dirs; do echo "/$d"; done)) || echo "FAILED $name"
  fi
  echo "done $split $name $(du -sh "$name" 2>/dev/null | cut -f1)"
done
echo ALLDONE
