#!/usr/bin/env bash
# Complementary learning: neurogenesis (cortex) + episodic memory (hippocampus) on the 10-specialty benchmark.
# Settings = the H200 runs DB_neuro_old90 and E_all_multidepth_dissonance, plus memory. Sequential on one GPU.
# Run in tmux; to stop, kill the python process (killing this script's shell closes the pane and kills it anyway).
cd "$(dirname "$(readlink -f "$0")")"
PY="$(cd .. && pwd)/.venv/bin/python"
B=runs/sweep_b32k_sparseB32c2_t1b
D=haskell,medical,lua,philosophy,law,fortran,logic,music,rust,space
X="--per_stage 2 --stage_steps 400 --consolidate_steps 200 --lr_adam 0.006 --init copy --replay 0.25 --sleep_new 0.1 --depths 1,3,5"
M="--memory_tokens 5e5 --memory_prune 0.5 --memory_eval_tokens 100000"
jobs=(
  "continual $B --domains $D $X --placement fixed $M --name M1_neuro_mem"
  "continual $B --domains $D $X --multi_depth 1 --placement dissonance $M --name M2_mddis_mem"
  # M2 + sleep that distils memory's predictions into the stage's modules (compare with M2)
  "continual $B --domains $D $X --multi_depth 1 --placement dissonance $M --sleep_distill 1 --name M3_mddis_distill"
  # instruction (Dolly-15k) and conversation (oasst2) as modules (build_domains.py --only instruct,chat), no memory:
  # C1 on the base model alone; C2 after the 10 specialties (does conversation training wear the specialties away?)
  "continual $B --domains instruct,chat $X --multi_depth 1 --placement dissonance --name C1_instruct_chat"
  "continual $B --domains $D,instruct,chat $X --multi_depth 1 --placement dissonance --name C2_spec_instruct_chat"
  # C1 again, keeping the grown modules for sample replies (samples.py)
  "continual $B --domains instruct,chat $X --multi_depth 1 --placement dissonance --save runs/forgetting/C1s_instruct_chat --name C1s_instruct_chat"
  # C2 again, keeping the modules, for sample replies that need the specialties
  # (LOCAL_PLAN.md "Next run: C2s"); started by the owner Nov 2.
  "continual $B --domains $D,instruct,chat $X --multi_depth 1 --placement dissonance --save runs/forgetting/C2s_spec_instruct_chat --name C2s_spec_instruct_chat"
)
mkdir -p runs/forgetting
for j in "${jobs[@]}"; do
  n=$(echo "$j" | grep -o 'name [^ ]*' | cut -d' ' -f2)
  [ -f "runs/forgetting/continual_$n.json" ] && { echo "skip $n (done)"; continue; }
  echo "[$(date '+%-I:%M %p')] start $n"
  "$PY" -u forgetting.py $j > "runs/forgetting/$n.log" 2>&1 && echo "[$(date '+%-I:%M %p')] done $n" || echo "[$(date '+%-I:%M %p')] FAILED $n"
done
