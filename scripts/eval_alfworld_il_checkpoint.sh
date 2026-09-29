#!/usr/bin/env bash
set -euo pipefail

VERL_AGENT_DIR="${VERL_AGENT_DIR:-external/verl-agent}"
MODEL_PATH="${MODEL_PATH:-checkpoints/early-experience-alfworld/il}"
DATA_DIR="${DATA_DIR:-data/verl-agent}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/alfworld_il_eval}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-early-exp-alfworld}"
ENGINE="${ENGINE:-vllm}"
N_GPUS="${N_GPUS:-2}"
TP_SIZE="${TP_SIZE:-$N_GPUS}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-16}"
ALFWORLD_EVAL_DATASET="${ALFWORLD_EVAL_DATASET:-eval_in_distribution}"
ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
ACTOR_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
CRITIC_MICRO_BATCH_SIZE_PER_GPU="${CRITIC_MICRO_BATCH_SIZE_PER_GPU:-1}"
NUM_CPUS_PER_ENV_WORKER="${NUM_CPUS_PER_ENV_WORKER:-0.1}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-null}"

if [ ! -d "$VERL_AGENT_DIR" ]; then
  echo "verl-agent checkout not found: $VERL_AGENT_DIR" >&2
  echo "Run scripts/setup_verl_agent_alfworld.sh first or set VERL_AGENT_DIR." >&2
  exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
  echo "Model checkpoint not found: $MODEL_PATH" >&2
  echo "Run scripts/download_il_checkpoint.sh first or set MODEL_PATH." >&2
  exit 1
fi

mkdir -p "$DATA_DIR" "$OUTPUT_DIR"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

PYTHONPATH_FOR_VERL="$VERL_AGENT_DIR:${PYTHONPATH:-}"

conda run -n "$CONDA_ENV_NAME" env PYTHONPATH="$PYTHONPATH_FOR_VERL" python3 -m examples.data_preprocess.prepare \
  --mode text \
  --local_dir "$DATA_DIR" \
  --train_data_size "$TRAIN_DATA_SIZE" \
  --val_data_size "$VAL_BATCH_SIZE"

set -x
conda run -n "$CONDA_ENV_NAME" env PYTHONPATH="$PYTHONPATH_FOR_VERL" python3 -m verl.trainer.main_ppo \
  data.train_files="$DATA_DIR/text/train.parquet" \
  data.val_files="$DATA_DIR/text/test.parquet" \
  data.train_batch_size="$TRAIN_DATA_SIZE" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.max_prompt_length=4096 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ACTOR_MICRO_BATCH_SIZE_PER_GPU" \
  +actor_rollout_ref.actor.fsdp_config.model_dtype="$MODEL_DTYPE" \
  actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD" \
  actor_rollout_ref.rollout.name="$ENGINE" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
  critic.model.path="$MODEL_PATH" \
  critic.ppo_micro_batch_size_per_gpu="$CRITIC_MICRO_BATCH_SIZE_PER_GPU" \
  algorithm.adv_estimator="$ADV_ESTIMATOR" \
  algorithm.use_kl_in_reward=False \
  env.env_name=alfworld/AlfredTWEnv \
  env.seed=0 \
  env.max_steps=50 \
  env.rollout.n=1 \
  env.resources_per_worker.num_cpus="$NUM_CPUS_PER_ENV_WORKER" \
  env.alfworld.eval_dataset="$ALFWORLD_EVAL_DATASET" \
  "trainer.logger=['console']" \
  trainer.project_name=early_experience_alfworld \
  trainer.experiment_name=il_checkpoint_eval \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.default_local_dir="$OUTPUT_DIR/checkpoints" \
  ray_init.num_cpus="$RAY_NUM_CPUS" \
  2>&1 | tee "$OUTPUT_DIR/eval.log"
