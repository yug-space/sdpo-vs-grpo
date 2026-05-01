#!/usr/bin/env bash
# SDPO: Qwen3.5-9B full-FT on tooluse, single-GPU.
# Mirrors upstream/run_local_sdpo.sh: GRPO loss + alpha-weighted self-distillation
# from EMA teacher prompted with successful rollouts as in-context demonstrations.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/upstream"

export PYTHONPATH="$ROOT/upstream:${PYTHONPATH:-}"
export USER="${USER:-$(whoami)}"
export N_GPUS_PER_NODE=1

CONFIG_NAME="sdpo"
DATA_PATH="datasets/mind2web"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
SEED="${SEED:-42}"

TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
LR=1e-5
LAMBDA=0.0
CLIP_ADV_HIGH=null
DONTS_REPROMPT_ON_SELF_SUCCESS=True
ALPHA=0.5            # 0=fwd KL, 0.5=Jensen-Shannon, 1=rev KL
DISTILL_TOPK=100

EXP_NAME="SDPO-q35-9b-mind2web-alpha${ALPHA}-seed${SEED}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=SDPOvsGRPO \
trainer.experiment_name=$EXP_NAME \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
actor_rollout_ref.actor.self_distillation.distillation_topk=$DISTILL_TOPK \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.val_kwargs.n=16 \
trainer.seed=$SEED"

echo "----------------------------------------------------------------"
echo "[SDPO] $EXP_NAME"
echo "  model:   $MODEL_PATH"
echo "  data:    $DATA_PATH"
echo "  alpha:   $ALPHA"
echo "  topk:    $DISTILL_TOPK"
echo "  seed:    $SEED"
echo "  GPUs:    $N_GPUS_PER_NODE"
echo "----------------------------------------------------------------"

bash "$ROOT/upstream/training/verl_training.sh" \
  "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
