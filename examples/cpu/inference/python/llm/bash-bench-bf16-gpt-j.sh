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

MODEL_ID=EleutherAI/gpt-j-6b

# OMP_NUM_THREADS=56 numactl -m 0 -C '0-56' python run_generation.py \
#     -m $MODEL_ID --dtype bfloat16 \
#     --benchmark --ipex --jit \
#     --num-warmup 5 --num-iter 20 \
#     --token-latency \
#     --greedy \
#     --input-tokens 256 \
#     --max-new-tokens 32

# beam
OMP_NUM_THREADS=56 numactl -m 0 -C '0-56' python run_generation.py \
    -m $MODEL_ID --dtype bfloat16 \
    --benchmark --ipex --jit \
    --num-warmup 5 --num-iter 20 \
    --token-latency \
    --input-tokens 128 \
    --max-new-tokens 32