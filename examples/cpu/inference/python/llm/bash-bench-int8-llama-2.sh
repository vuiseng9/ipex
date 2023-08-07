#!/usr/bin/bash

# !!!! activate environment before running this script

export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so.6

# Setup environment variables for performance on Xeon
export KMP_BLOCKTIME=INF
export KMP_TPAUSE=0
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_FORJOIN_BARRIER_PATTERN=dist,dist
export KMP_PLAIN_BARRIER_PATTERN=dist,dist
export KMP_REDUCTION_BARRIER_PATTERN=dist,dist
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so # Intel OpenMP

# Tcmalloc is a recommended malloc implementation that emphasizes fragmentation avoidance and scalable concurrency support.
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

MODEL_ID=meta-llama/Llama-2-7b-hf

# beam
OMP_NUM_THREADS=56 numactl -m 0 -C '0-56' python run_llama_int8.py \
    -m $MODEL_ID --quantized-model-path ./saved_results/best_model.pt \
    --benchmark --jit --int8-bf16-mixed \
    --num-warmup 5 --num-iter 20 \
    --token-latency \
    --input-tokens 1024 \
    --max-new-tokens 32