#!/usr/bin/env bash
set -euo pipefail
cd /data1/shn/agentwmv2
ROOT=$PWD
LOG=$ROOT/outputs/paper_eval/audited/logs
mkdir -p "$LOG"

start_vllm() {
    local gpu=$1 port=$2 name=$3 weights=$4
    if curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null; then
        /home/shn/miniconda3/envs/verl-agent/bin/python -c 'import sys,requests; d=requests.get(sys.argv[1]).json(); assert any(m["id"] == sys.argv[2] and m["root"].endswith(sys.argv[3]) for m in d["data"]), "Port occupied by wrong model"' \
            "http://127.0.0.1:$port/v1/models" "$name" "$weights"
        echo "$name already ready on $port"
        return
    fi
    if ss -ltnH "sport = :$port" | grep -q .; then
        echo "Port $port occupied but not ready; inspect before restarting" >&2
        return 1
    fi
    nohup env CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 \
        /home/shn/miniconda3/envs/verl-agent/bin/vllm serve "$ROOT/$weights" \
        --served-model-name "$name" --tensor-parallel-size 1 --max-model-len 5120 \
        --max-num-seqs 8 --gpu-memory-utilization 0.90 --host 127.0.0.1 --port "$port" \
        > "$LOG/server_$name.log" 2>&1 < /dev/null &
    echo "$!" > "$LOG/server_$name.pid"
    echo "$name started PID=$!"
}

start_vllm 0 8100 qwen25 checkpoints/Qwen2.5-7B-Instruct
start_vllm 1 8101 ee-alfworld-il checkpoints/early-experience-alfworld/il
start_vllm 2 8103 ee-scienceworld-il checkpoints/early-experience-scienceworld/il
start_vllm 3 8104 ee-webshop-il checkpoints/early-experience-webshop/il

if ss -ltnH 'sport = :8102' | grep -q .; then
    echo 'Port 8102 already in use. Confirm Qwen3.8 path in server log; leaving service untouched.'
else
    nohup env CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 \
        CUDA_HOME=/usr/local/cuda-12.8 \
        PATH=/usr/local/cuda-12.8/bin:/home/shn/miniconda3/envs/ee-qwen38-direct/bin:/usr/bin:/bin \
        /home/shn/miniconda3/envs/ee-qwen38-direct/bin/transformers serve \
        "$ROOT/checkpoints/Qwen3.8-27B" --port 8102 \
        > "$LOG/server_qwen38.log" 2>&1 < /dev/null &
    echo "$!" > "$LOG/server_qwen38.pid"
    echo "Qwen3.8 started PID=$!"
fi
