#!/bin/bash

set -euo pipefail

########################################
# 1. Environment
########################################
module purge
module load $MODULES

# Ray sanity
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_USAGE_STATS_ENABLED=1
export RAY_NUM_GPUS=$GPUS_PER_NODE
export RAY_NUM_CPUS=$CPUS_PER_NODE
export VLLM_USE_V1=0
export VLLM_USE_RAY_SPANNABLE_POOL=0
export VLLM_USE_RAY_COMPILED_DAG=0
export RAY_CGRAPH_get_timeout=1800
NB_NODES=$NODES

########################################
# 2. Robust InfiniBand detection (IPv4 only)
########################################
IB_IFACE=$(ip -4 -o addr show | awk '{print $2}' | grep -E '^(ib|hsn|sl)' | head -n 1)

if [ -z "$IB_IFACE" ]; then
  echo "ERROR: No InfiniBand interface with IPv4 found"
  ip -4 addr show
  exit 1
fi

IB_IP_CMD="ip -4 addr show $IB_IFACE | awk '/inet / {split(\$2,a,\"/\"); print a[1]}'"

########################################
# 3. NCCL (force IB, prevent silent TCP fallback)
########################################

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=10
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=$IB_IFACE
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=0

########################################
# 4. Node discovery
########################################

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}

# Si on est sur 1 seul nœud, on récupère l'IP directement sans srun
if [ "$NB_NODES" -eq 1 ]; then
    head_node_ip=$(eval "$IB_IP_CMD")
else
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" bash -c "$IB_IP_CMD")
fi

export RAY_HEAD_IP="$head_node_ip"
export RAY_ADDRESS="$RAY_HEAD_IP:6379"

########################################
# 5. Launch Ray cluster
########################################
ray stop --force || true
sleep 5

srun --nodes=$NB_NODES \
     --ntasks=$NB_NODES \
     --ntasks-per-node=1 \
     bash -c "
set -euo pipefail
local_ip=\$($IB_IP_CMD)
export RAY_NODE_IP_ADDRESS=\$local_ip
export VLLM_HOST_IP=\$local_ip
export HOST_IP=\$local_ip

echo \"[\$(hostname)] Starting Ray on IP: \$local_ip\"

if [ \"\$SLURM_PROCID\" -eq 0 ]; then
    ray start --head \
      --node-ip-address=\$local_ip \
      --port=6379 \
      --num-cpus=$CPUS_PER_NODE \
      --num-gpus=$GPUS_PER_NODE \
      --disable-usage-stats \
      --block
else
    # Worker Nodes (if scaling > 1)
    until (echo > /dev/tcp/$RAY_HEAD_IP/6379) >/dev/null 2>&1; do
      echo \"Waiting for Ray head at $RAY_HEAD_IP...\"
      sleep 2
    done
    ray start \
      --address=$RAY_ADDRESS \
      --node-ip-address=\$local_ip \
      --num-cpus=$CPUS_PER_NODE \
      --num-gpus=$GPUS_PER_NODE \
      --disable-usage-stats \
      --block
fi
" &

########################################
# 6. Wait for Ray
########################################
sleep 60
ray status || { echo "Ray failed to start"; exit 1; }


########################################
# 7. Launch vLLM
########################################
export VLLM_RAY_USE_EXISTING_CLUSTER=1
export VLLM_HOST_IP="$RAY_HEAD_IP"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Force NCCL to use the InfiniBand interface found earlier
export NCCL_SOCKET_IFNAME=$IB_IFACE


echo "Launching vLLM on $head_node ($RAY_HEAD_IP)"
PORT=8000
python -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --tensor-parallel-size $TENSOR_PARALLEL \
  --pipeline-parallel-size $PIPELINE_PARALLEL \
  --distributed-executor-backend ray \
  --disable-custom-all-reduce \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --host "$RAY_HEAD_IP" \
  --port $PORT \
  --trust-remote-code &
VLLM_PID=$!
########################################
# 8. Wait for Health Check & Inference Request
########################################
echo "Waiting for vLLM to initialize weights (Model: 405B)..."

# Use /v1/models instead of /health for a stricter readiness check
timeout 1800 bash -c "
until [ \"\$(curl -s -o /dev/null -w ''%{http_code}'' http://$RAY_HEAD_IP:8000/v1/models)\" == \"200\" ]; do
    echo \"Still loading weights...\"
    sleep 20
done
" || {
  echo "vLLM server failed to become ready within 30 minutes"
  kill $VLLM_PID
  exit 1
}

echo "Server is UP. Sending inference request..."

curl -X POST http://$RAY_HEAD_IP:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL_PATH\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"Explain the concept of GPU tensor parallelism in one sentence.\"}
    ],
    \"max_tokens\": 50
  }"

echo -e "\nInference test complete."
#############################################
concurrencies=(50 100 200 300 400 500)
for conc in "${concurrencies[@]}"; do
  echo "======================================="
  echo "Running concurrency level $conc"
  echo "Results folder: $LAUNCH_FOLDER"
  echo "======================================="

  METRICS_FILE="$LAUNCH_FOLDER/gpu_metrics_${conc}.csv"
  LOG_FILE="$LAUNCH_FOLDER/logs_benchmarking_${conc}_concurrency.log"
  RESULT_FILE="$LAUNCH_FOLDER/Concurrency_${conc}.json"

  # Start GPU monitoring (all GPUs on the node where this runs)
  nvidia-smi --query-gpu=timestamp,index,name,memory.used,power.draw,utilization.gpu,utilization.memory \
             --format=csv,noheader,nounits -l 1 > "$METRICS_FILE" &
  GPU_MON_PID=$!

  # Run benchmark against the vLLM server
  # IMPORTANT: use $RAY_HEAD_IP not localhost (unless your benchmark runs on the same head node and you know that’s true)
  set +e
  python "$BENCHMARK_FILE" \
    --backend 'vllm' \
    --host "$RAY_HEAD_IP" \
    --port "$PORT" \
    --model "$MODEL_PATH" \
    --dataset-name "$DATASET" \
    --dataset-path "$DATASET_PATH" \
    --max-concurrency "$conc" \
    --num-prompts 1000 \
    --save-result \
    --result-filename "$RESULT_FILE" \
    > "$LOG_FILE" 2>&1
  RC=$?
  set -e

  # Stop monitoring
  kill "$GPU_MON_PID" >/dev/null 2>&1 || true
  sleep 2

  if [ "$RC" -ne 0 ]; then
    echo "Benchmark failed at concurrency=$conc (exit code $RC). See: $LOG_FILE"
    exit "$RC"
  fi

  echo "Done concurrency=$conc"
done

echo "All concurrency runs completed successfully."


########################################
# 9. Cleanup
########################################
echo "Shutting down..."
kill $VLLM_PID
ray stop --force
