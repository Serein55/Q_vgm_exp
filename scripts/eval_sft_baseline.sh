#!/usr/bin/env bash
# Few-shot SFT baseline, measured inside the Q-VGM harness (scripts/eval_sft.py).
#
# Why: 复现过程.md:17 records that the full SFT baseline was NEVER run in this harness
# ("用户要求沿用历史 SFT 基线，因此没有重跑完整 SFT 评估"), and configs/libero_spatial.yaml:30
# still carries the assumption "SFT baseline is supplied by the user; no baseline rerun is
# required". So the offline Q-VGM result of 52/500 has only ever been compared against an
# unverified external 401/500. This produces the missing control.
#
# Two configs, because they answer different questions:
#   h10   = configs/libero_spatial.yaml        action_horizon 10, discrete_state_input false
#           -> matches the released RLinf checkpoint's conditioning; comparable to the RLinf
#              harness baseline (400/500). Differences that remain: max_steps 220 vs 240,
#              denoising_steps 10 vs num_steps 3, settle_steps 10, per-episode seeding.
#   paper = configs/libero_spatial_paper.yaml  action_horizon 5, discrete_state_input true
#           -> the v5 literal reading that the 52/500 Q-VGM run actually used, so this is the
#              correct control for that number (its alpha=0 control scored 41/50 = 82%).
#
# Tasks are split across GPUs. eval_sft.py derives episode_seed = seed + task_id*10000 + episode
# and builds a fresh env per task, so splitting by task is exactly equivalent to one long run.
set -u

QVGM=/pfs/pfs-oHNwH0/ganrenda/Q_vgm
cd "$QVGM" || exit 1

launch() {  # gpu config name tasks...
  local gpu=$1 cfg=$2 name=$3
  shift 3
  local tasks="$*"
  echo "[sftbase] $(date '+%F %T') gpu=$gpu cfg=$cfg name=$name tasks=$tasks"
  CUDA_VISIBLE_DEVICES=$gpu bash run.sh eval_sft \
    --config "$cfg" \
    --task-ids $tasks \
    --episodes-per-task 50 \
    --name "$name" \
    >"$QVGM/artifacts/eval/$name.log" 2>&1
  echo "[sftbase] $(date '+%F %T') gpu=$gpu name=$name exit rc=$?"
}

launch 4 configs/libero_spatial.yaml       sftbase_h10_t04   0 1 2 3 4 &
p1=$!
launch 5 configs/libero_spatial.yaml       sftbase_h10_t59   5 6 7 8 9 &
p2=$!
launch 6 configs/libero_spatial_paper.yaml sftbase_paper_t04 0 1 2 3 4 &
p3=$!
launch 7 configs/libero_spatial_paper.yaml sftbase_paper_t59 5 6 7 8 9 &
p4=$!

wait $p1 $p2 $p3 $p4
echo "[sftbase] $(date '+%F %T') all done"
