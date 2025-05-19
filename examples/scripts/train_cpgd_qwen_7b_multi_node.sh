set -x

export RAY_MASTER_PORT=6379
export RAY_DASHBOARD_PORT=8265
export NCCL_TIMEOUT=7200

OUTPUT_DIR='/absolute/path/to/output/dir'

export REWARD_LOG_PATH="${OUTPUT_DIR}/reward.log"
export WORKING_DIR=$PWD


if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

if [ "$PET_NODE_RANK" -eq 0 ]; then
    ray start --head  --port=$RAY_MASTER_PORT --dashboard-host=0.0.0.0 --dashboard-port=$RAY_DASHBOARD_PORT --num-gpus 8
else
    sleep 30
    ray start --address="$MASTER_ADDR:$RAY_MASTER_PORT" --num-gpus 8 --block
fi

sleep 30


if [ "$NODE_RANK" -eq 0 ]; then
  RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT" ray job submit \
  --working-dir $WORKING_DIR \
  -- python3 -m openrlhf.cli.train_ppo_ray \
  --remote_rm_url examples/scripts/reward_func_qwen_instruct.py \
  --use_cpg_loss \
  --use_policy_drift \
  --policy_drift_coef 0.01 \
  --actor_num_nodes 2 \
  --actor_num_gpus_per_node 8 \
  --vllm_num_engines 8 \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.7 \
  --pretrain Qwen/Qwen2.5-VL-7B-Instruct \
  --save_path ${OUTPUT_DIR} \
  --micro_train_batch_size 1 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 1 \
  --rollout_batch_size 128 \
  --temperature 1.0 \
  --n_samples_per_prompt 8 \
  --lambd 1.0 \
  --gamma 1.0 \
  --max_epochs 1 \
  --num_episodes 5 \
  --l2 1.0e-2 \
  --lr_warmup_ratio 0 \
  --prompt_max_len 3000 \
  --max_samples 100000 \
  --generate_max_len 4096 \
  --advantage_estimator group_norm \
  --zero_stage 1 \
  --bf16 \
  --actor_learning_rate 1e-6 \
  --init_kl_coef 0.0 \
  --prompt_data /path/to/training/data \
  --disable_fast_tokenizer \
  --input_key message \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --save_steps 50 \
  --ckpt_path "${OUTPUT_DIR}/ckpt" \
  --max_ckpt_num 1000000 \
  --save_hf_ckpt \
  --freeze_prefix visual \
  --use_tensorboard "${OUTPUT_DIR}/tensorboard" \
  --load_checkpoint | tee ${OUTPUT_DIR}/training.log
fi
