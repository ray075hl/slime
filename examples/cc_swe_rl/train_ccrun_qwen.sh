#!/bin/bash
# slime GRPO training against the cc-run RL environment service.
# Model: Qwen3.8-27B (qwen3.5-27B arch, 26.9B params, 64 layers).
# Multi-node Ray cluster (K8s MPI-Operator style), adapted from the SFT launcher.
#
# Topology:
#   GPU nodes: slime Megatron trainer + SGLang rollout engine (auto weight-sync).
#   cc-run env node (CPU, docker): HTTP /rollout reached via CC_RUN_URL; slime
#   passes its sglang_url so the env adapter streams tokens from THIS cluster.
#   Reward is computed in cc-run (F2P/P2P) and returned with segments -> no
#   --custom-rm-path; only generate is custom.
#
# Required env: CC_RUN_URL, TRAIN_PROMPT_DATA, BASE_FOLDER (ckpt/output base).
# Optional: MASTER_ADDR (auto from /etc/hosts), NUM_NODES, SLIME_ROOT,
#           MPI_WORKER_HOSTS/SSH_PORT/SSH_KEY/REMOTE_PATH (multi-node fanout),
#           SKIP_RAY_RESTART, WANDB_API_KEY.

if [ -z "${SKIP_RAY_RESTART}" ]; then
  pkill -9 sglang; sleep 3
  ray stop --force; pkill -9 ray; pkill -9 python; sleep 3
  pkill -9 ray; pkill -9 python
fi

set -ex
ACTOR_NODES=16
ROLLOUT_NODES=4
GPUS_PER_NODE=8

export CC_RUN_URL=http://lb-op7sm96t-mpxviq0fbjkyh584.clb.bj-tencentclb.com:8080
export TRAIN_PROMPT_DATA=/mnt/public_02/huangliang45/rl_data/qwen_validated_train.jsonl
export CC_RUN_SGLANG_EXTERNAL_URL=http://180.184.148.148:8111
export BASE_FOLDER=/mnt/public_02/huangliang45/rl_ckpt/qwen38_27b_ccrun_rl
export SLIME_ROOT=/mnt/public_02/huangliang45/github/slime

# ---- required env ----
: "${CC_RUN_URL:?set CC_RUN_URL=http://<cc-run env host or LB>:8080}"
: "${TRAIN_PROMPT_DATA:?set TRAIN_PROMPT_DATA=<prompt-data jsonl with metadata.instance_id>}"
: "${BASE_FOLDER:?set BASE_FOLDER=<checkpoint/output base dir>}"
export CC_RUN_TIMEOUT="${CC_RUN_TIMEOUT:-3600}"

# ---- MASTER_ADDR: explicit, else infer from K8s MPI-Operator /etc/hosts ----
if [ -z "${MASTER_ADDR}" ] && [ -f /etc/hosts ]; then
  MASTER_ADDR=$(awk '/[a-z0-9-]+-launcher([^-a-z0-9]|$)/ {print $1; exit}' /etc/hosts)
fi
: "${MASTER_ADDR:?set MASTER_ADDR or ensure /etc/hosts has a *-launcher entry}"

export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export no_proxy="127.0.0.1,${MASTER_ADDR},${no_proxy}"
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
[ "$NVLINK_COUNT" -gt 0 ] && HAS_NVLINK=1 || HAS_NVLINK=0

SLIME_ROOT="${SLIME_ROOT:-/mnt/public_02/huangliang45/slime}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# qwen3.5-27B model args (matches the torch_dist ref ckpt arch)
if [ -f "${SLIME_ROOT}/scripts/models/qwen3.5-27B.sh" ]; then
  source "${SLIME_ROOT}/scripts/models/qwen3.5-27B.sh"
else
  source "${SCRIPT_DIR}/../../scripts/models/qwen3.5-27B.sh"
fi

NUM_NODES="${NUM_NODES:-1}"             # total GPU nodes in the cluster
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

CKPT_ARGS=(
    --hf-checkpoint /mnt/public_02/huangliang45/models/Qwen3.8-27B
    --ref-load /mnt/public_02/huangliang45/models/Qwen3.8-27B-torch-dist
    --save ${BASE_FOLDER}/qwen38_27b_ccrun_rl/
    --save-interval 20
    --no-save-optim
)

PERF_ARGS=(
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 2
    --context-parallel-size 2
    --sequence-parallel
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 16384
    --log-probs-max-tokens-per-gpu 131072
)

ROLLOUT_ARGS=(
    --prompt-data ${TRAIN_PROMPT_DATA}
    --input-key prompt
    --metadata-key metadata
    --rollout-shuffle
    --num-rollout 3000
    --rollout-batch-size 8
    --over-sampling-batch-size 16
    --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.keep_full_groups_only
    --n-samples-per-prompt 8
    --rollout-temperature 1.0
    --rollout-max-context-len 262144
    --rollout-max-response-len 32768
    --global-batch-size 64
    --balance-data
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.001
    --kl-loss-type low_var_kl
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0
    --use-distributed-optimizer
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 8
    --sglang-server-concurrency 16
    --sglang-mem-fraction-static 0.7
    --sglang-served-model-name qwen3.8-27b
    --sglang-context-length 262144
    --sglang-tool-call-parser qwen3_coder
    --sglang-reasoning-parser qwen3
    --sglang-router-port 4600
    --router-policy consistent_hashing
    --sglang-disable-custom-all-reduce
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

CUSTOM_ARGS=(
    --custom-generate-function-path examples.cc_swe_rl.client.generate
)

WANDB_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ]; then
    WANDB_ARGS=(--use-wandb --wandb-project slime-ccrun-rl --wandb-group qwen38-27b-grpo)
fi

# ---- Ray cluster: head here, fan out to workers (MPI-Operator style) ----
if [ -z "${SKIP_RAY_RESTART}" ]; then
  ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${GPUS_PER_NODE} \
      --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --port=6379

  WORKER_HOSTS="${MPI_WORKER_HOSTS:-}"
  if [ -z "${WORKER_HOSTS}" ] && [ -f /root/mpi_rack_hostfile ]; then
    WORKER_HOSTS=$(awk '{print $1}' /root/mpi_rack_hostfile | tr '\n' ' ')
  fi
  if [ -z "${WORKER_HOSTS}" ] && [ -f /etc/hosts ]; then
    WORKER_HOSTS=$(awk '/[a-z0-9-]+-worker-[0-9]+([^-a-z0-9]|$)/ {print $1}' /etc/hosts | awk '!seen[$0]++' | tr '\n' ' ')
  fi
  if [ -n "${WORKER_HOSTS}" ]; then
    SSH_PORT="${SSH_PORT:-2222}"
    SSH_KEY="${SSH_KEY:-/sensecore/compute/platform/ssh/ssh_config/id_rsa}"; SSH_KEY_OPT=""; [ -n "${SSH_KEY}" ] && SSH_KEY_OPT="-i ${SSH_KEY}"
    REMOTE_PREPEND=""; [ -n "${REMOTE_PATH}" ] && REMOTE_PREPEND="export PATH=\"${REMOTE_PATH}:\$PATH\"; "
    echo "worker hosts: ${WORKER_HOSTS}  ssh_port=${SSH_PORT}"
    for WORKER in ${WORKER_HOSTS}; do
      if [[ "$WORKER" == "$MASTER_ADDR" ]] || [[ "$WORKER" == "$(hostname)" ]]; then continue; fi
      echo "Starting Ray worker on ${WORKER}"
      ssh -p "${SSH_PORT}" ${SSH_KEY_OPT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        root@"${WORKER}" \
        "${REMOTE_PREPEND}pkill -9 sglang; ray stop --force; pkill -9 python; ray start --address=${MASTER_ADDR}:6379 --num-gpus ${GPUS_PER_NODE} --disable-usage-stats" &
    done
    wait
  else
    echo "WARNING: no worker hosts discovered; single-node mode."
  fi
else
  echo "SKIP_RAY_RESTART set, reusing existing Ray cluster:"; ray status
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"CC_RUN_URL\": \"${CC_RUN_URL}\",
    \"CC_RUN_SGLANG_EXTERNAL_URL\": \"${CC_RUN_SGLANG_EXTERNAL_URL}\",
    \"CC_RUN_TIMEOUT\": \"${CC_RUN_TIMEOUT}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\"
  }
}"

echo "=== cc-run RL GRPO: nodes=${NUM_NODES} gpus/node=${GPUS_PER_NODE} CC_RUN_URL=${CC_RUN_URL} ==="

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    --working-dir "${SLIME_ROOT}" \
    -- python3 train.py \
    --actor-num-nodes ${ACTOR_NODES} \
    --actor-num-gpus-per-node ${GPUS_PER_NODE} \
    --rollout-num-gpus $(( ROLLOUT_NODES * GPUS_PER_NODE )) \
    --rollout-num-gpus-per-engine 8 \
    --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async \
    --fully-async-group-prefetch-factor 2 \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${WANDB_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${MISC_ARGS[@]} \
    ${CUSTOM_ARGS[@]}

echo "Training submitted."
