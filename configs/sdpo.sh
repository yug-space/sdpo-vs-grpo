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
# Note: expandable_segments is incompatible with vLLM's CUDA memory pool,
# don't enable it. Memory headroom comes from gpu_memory_utilization=0.30.

CONFIG_NAME="sdpo"
DATA_PATH="datasets/mind2web"
# Qwen3.5-9B has model_type=qwen3_5 (transformers >=5.0, breaks verl).
# Qwen3-8B fits compute-wise but Adam fp32 state (~65GB) + model + grads + vLLM
# OOMs on a single H200 141GB at optimizer step. Dropping to Qwen3-4B which
# halves the optimizer memory and runs cleanly on this hardware.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
SEED="${SEED:-42}"

TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
LR=1e-5
LAMBDA=0.0
CLIP_ADV_HIGH=null
DONTS_REPROMPT_ON_SELF_SUCCESS=True
ALPHA=0.5            # 0=fwd KL, 0.5=Jensen-Shannon, 1=rev KL
DISTILL_TOPK=100

EXP_NAME="SDPO-qwen3-4b-mind2web-alpha${ALPHA}-seed${SEED}"

# Memory tricks for single H200 141GB full-FT of 9B + EMA teacher
MAX_STEPS="${MAX_STEPS:-30}"

SDPO_DIR="${SDPO_DIR:-$ROOT/upstream}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
vars.dir=$SDPO_DIR \
custom_reward_function.path=$SDPO_DIR/verl/utils/reward_score/feedback/__init__.py \
trainer.logger=[console] \
trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
trainer.group_name=SDPOvsGRPO \
trainer.experiment_name=$EXP_NAME \
trainer.total_training_steps=$MAX_STEPS \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.20 \
actor_rollout_ref.rollout.enforce_eager=True \
actor_rollout_ref.rollout.free_cache_engine=True \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
trainer.val_before_train=False \
actor_rollout_ref.actor.self_distillation.distillation_topk=$DISTILL_TOPK \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.val_kwargs.n=16"

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
