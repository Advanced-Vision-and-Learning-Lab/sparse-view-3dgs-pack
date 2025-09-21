#!/bin/bash
# Path to LLFF dataset
DATA_ROOT="/home/grads/a/atharvagashe22/AVLL/3DR/FSGS/dataset/nerf_llff_data"
# Path to output root
OUT_ROOT="output/llff"
# GPU id
GPU_ID=0

# List of categories to run
# CATEGORIES=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")
CATEGORIES=("fortress")

for cat in "${CATEGORIES[@]}"; do
    echo "=== Running category: $cat ==="
    bash scripts/run_llff.sh "${DATA_ROOT}/${cat}" "${OUT_ROOT}/${cat}" ${GPU_ID}
done
