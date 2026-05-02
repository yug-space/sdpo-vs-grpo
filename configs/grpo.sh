#!/usr/bin/env bash
# GRPO baseline: Qwen3.5-9B full-FT on tooluse, single-GPU.
# Mirrors upstream/run_local_grpo.sh, swapping in our model and pinning a seed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/upstream"

export PYTHONPATH="$ROOT/upstream:${PYTHONPATH:-}"
export USER="${USER:-$(whoami)}"
export N_GPUS_PER_NODE=1
# Note: expandable_segments is incompatible with vLLM's CUDA memory pool,
# don't enable it. Memory headroom comes from gpu_memory_utilization=0.30.

CONFIG_NAME="baseline_grpo"
DATA_PATH="datasets/mind2web"
# Qwen3.5-9B has model_type=qwen3_5 (transformers >=5.0, breaks verl).
# Qwen3-8B fits compute-wise but Adam fp32 state (~65GB) + model + grads + vLLM
# OOMs on a single H200 141GB at optimizer step. Dropping to Qwen3-4B which
# halves the optimizer memory and runs cleanly on this hardware.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
SEED="${SEED:-42}"

TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
MINI_BATCH_SIZE=8
LR=1e-5

EXP_NAME="GRPO-qwen3-4b-mind2web-seed${SEED}"

# Memory tricks for single H200 141GB full-FT of 9B:
# - param_offload + optimizer_offload: AdamW fp32 states live in CPU RAM
# - dynamic_bsz: verl auto-tunes micro-batch from a token budget
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
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.20 \
actor_rollout_ref.rollout.enforce_eager=True \
actor_rollout_ref.rollout.free_cache_engine=True \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
trainer.val_before_train=False \
trainer.save_freq=10 \
trainer.resume_mode=auto \
trainer.default_local_dir=/workspace/checkpoints/$EXP_NAME \
trainer.max_actor_ckpt_to_keep=2 \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=16"

echo "----------------------------------------------------------------"
echo "[GRPO] $EXP_NAME"
echo "  model:   $MODEL_PATH"
echo "  data:    $DATA_PATH"
echo "  seed:    $SEED"
echo "  GPUs:    $N_GPUS_PER_NODE"
echo "----------------------------------------------------------------"

bash "$ROOT/upstream/training/verl_training.sh" \
  "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
