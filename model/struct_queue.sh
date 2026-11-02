#!/usr/bin/env bash
# wai_struct runs (STRUCT_PLAN.md work queue), sequential on one GPU. Settings = main's C2 (cls_queue.sh) unless noted;
# C2's first 10 stages (model/runs/main_forgetting) are the labeled baseline.
# Run in tmux; to stop, kill the python process (killing this script's shell closes the pane and kills it anyway).
cd "$(dirname "$(readlink -f "$0")")"
PY="$(cd .. && pwd)/.venv/bin/python"
B=runs/sweep_b32k_sparseB32c2_t1b
D=haskell,medical,lua,philosophy,law,fortran,logic,music,rust,space
X="--per_stage 2 --stage_steps 400 --consolidate_steps 200 --lr_adam 0.006 --init copy --replay 0.25 --sleep_new 0.1 --depths 1,3,5"
G="--multi_depth 1 --placement dissonance"
jobs=(
  # 1. label-free arrival: each day the specialty's whole training text mixed 50/50 into familiar text, unlabeled;
  # the buffer keeps windows over-represented today (today@6); the night grows, learns from it, sleeps
  "continual $B --domains $D $X $G --stream 1 --name S1_stream"
  # its control: a perfect detector (the buffer = exactly the day's specialty windows)
  "continual $B --domains $D $X $G --stream 1 --stream_oracle 1 --name S1o_stream_oracle"
)
mkdir -p runs/forgetting
for j in "${jobs[@]}"; do
  n=$(echo "$j" | grep -o 'name [^ ]*' | cut -d' ' -f2)
  [ -f "runs/forgetting/continual_$n.json" ] && { echo "skip $n (done)"; continue; }
  echo "[$(date '+%-I:%M %p')] start $n"
  "$PY" -u forgetting.py $j > "runs/forgetting/$n.log" 2>&1 && echo "[$(date '+%-I:%M %p')] done $n" || echo "[$(date '+%-I:%M %p')] FAILED $n"
done
