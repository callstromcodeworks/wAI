#!/usr/bin/env bash
# wai_struct runs (STRUCT_PLAN.md work queue), sequential on one GPU. Settings = main's C2 (cls_queue.sh) unless noted;
# C2's first 10 stages (model/runs/main_forgetting) are the labeled baseline.
# Each job is "log|script and arguments" (log under runs/); a job is skipped once runs/<log>.done exists (or, for a
# continual run, its continual_<name>.json).
# Run in tmux; to stop, kill the python process (killing this script's shell closes the pane and kills it anyway).
cd "$(dirname "$(readlink -f "$0")")"
PY="$(cd .. && pwd)/.venv/bin/python"
B=runs/sweep_b32k_sparseB32c2_t1b
D=haskell,medical,lua,philosophy,law,fortran,logic,music,rust,space
X="--per_stage 2 --stage_steps 400 --consolidate_steps 200 --lr_adam 0.006 --init copy --replay 0.25 --sleep_new 0.1 --depths 1,3,5"
G="--multi_depth 1 --placement dissonance"
S="forgetting.py continual $B --domains $D $X $G --stream 1"
jobs=(
  # 1. label-free arrival: each day the specialty's whole training text mixed 50/50 into familiar text, unlabeled;
  # the buffer keeps windows over-represented today (today@6); the night grows, learns from it, sleeps
  "forgetting/S1_stream|$S --name S1_stream"
  # its control: a perfect detector (the buffer = exactly the day's specialty windows)
  "forgetting/S1o_stream_oracle|$S --stream_oracle 1 --name S1o_stream_oracle"
  # 3. experience as the only data: S1, with old material replayed from N windows kept from before instead of the
  # whole corpus (~9,000 old windows drawn per night, ~90k over the run)
  "forgetting/S3_pool4k|$S --old_pool 4096 --name S3_pool4k"
  # ... and the windows each day judged familiar join the pool (reservoir of 4096): replay only what was read
  "forgetting/S3_pool4k_days|$S --old_pool 4096 --old_from_days 1 --name S3_pool4k_days"
  "forgetting/S3_pool1k|$S --old_pool 1024 --name S3_pool1k"
  "forgetting/S3_pool16k|$S --old_pool 16384 --name S3_pool16k"
  "forgetting/S3_pool256|$S --old_pool 256 --name S3_pool256"
  # 2. control for the selective memory: a random 60% of tokens (as much text covered as p < 0.5, without choosing
  # by surprise), and buffer_p0.5 again in another reading order (the probe's noise)
  "selective/P2c_C2s|selective_memory.py --policies buffer_r0.6,labeled_r0.6,buffer_p0.5"
  # 2b. pattern hippocampus: keys as product-quantized prototypes, precision following use
  "selective/P2b_C2s|pattern_memory.py"
  # 3b. replay by reconstruction: old material is 2,048 windows the model writes itself each night; the control
  # redraws as many corpus windows each night; then earlier nights' buffers kept only as 32-token cues and rebuilt
  "forgetting/S3b_fresh2k|$S --old_pool 2048 --old_refresh 1 --name S3b_fresh2k"
  # (dropped for now by the owner, Nov 3, after the control: S3b_gen2k stopped at day 3, S3b_gen2k_cue32 not run)
  # "forgetting/S3b_gen2k|$S --old_generate 2048 --name S3b_gen2k"
  # "forgetting/S3b_gen2k_cue32|$S --old_generate 2048 --rebuild_cue 32 --name S3b_gen2k_cue32"
  # 4a. study until learned (item 4's first version): 10% of each buffer held back as its exam; buffers stay in
  # study (wake and sleep) until a night improves their exam by less than 0.002 bpb, then are released
  "forgetting/S4a_learned|$S --master_tol 0.002 --name S4a_learned"
  # 2c. interest: is "too surprising" (p < 1%, noise-like) worth storing? today's p < 0.5 again, the band
  # 1% <= p < 50%, and a stricter p < 0.1
  "selective/P2d_C2s|selective_memory.py --policies buffer_p0.5,buffer_m0.01p0.5,buffer_p0.1"
  # 4a control: every buffer studied exactly 2 nights (S4a's mean was 2.4), exam measured but not used
  "forgetting/S4a_nights2|$S --study_nights 2 --name S4a_nights2"
  # 4a, first night kept whole (owner, Nov 3): today's buffer gets the whole wake as in S1; earlier buffers still in
  # study get 200 extra steps on top (75% their windows, 25% general text); release by exam as in S4a
  "forgetting/S4a_first|$S --master_tol 0.002 --review_steps 200 --name S4a_first"
  # 4b. recurring topics, 2 appearances per specialty (20 days, nights at half length). Built Nov 4; NOT run yet
  # (owner: "don't run it yet"). Uncomment to queue. R2_grow: the current recipe (grows 3 parts a day, ~60 by the
  # end; watch GPU memory). R2_practice: a returning topic's own parts learn instead (label-free match).
  # "forgetting/R2_grow|$S --repeats 2 --stage_steps 200 --consolidate_steps 100 --track_every 25 --name R2_grow"
  # "forgetting/R2_practice|$S --repeats 2 --stage_steps 200 --consolidate_steps 100 --track_every 25 --practice 1 --name R2_practice"
)
mkdir -p runs/forgetting runs/selective
for j in "${jobs[@]}"; do
  log=${j%%|*}
  cmd=${j#*|}
  n=$(echo "$cmd" | grep -o 'name [^ ]*' | cut -d' ' -f2)
  if [ -f "runs/$log.done" ] || { [ -n "$n" ] && [ -f "runs/forgetting/continual_$n.json" ]; }; then
    echo "skip $log (done)"
    continue
  fi
  echo "[$(date '+%-I:%M %p')] start $log"
  if "$PY" -u $cmd > "runs/$log.log" 2>&1; then
    touch "runs/$log.done"
    echo "[$(date '+%-I:%M %p')] done $log"
  else
    echo "[$(date '+%-I:%M %p')] FAILED $log"
  fi
done
