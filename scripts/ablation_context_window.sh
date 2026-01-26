#!/bin/bash
# Ablation study: Context window size K = 4, 6, 8
# Runs 3 training processes in parallel on different GPUs

set -e

# Use fmtp conda environment
PYTHON="/opt/conda/envs/fmtp/bin/python"

OUTPUT_DIR="$HOME/.cache/nanochat/hst_retrieval_ablation"
mkdir -p "$OUTPUT_DIR"

echo "Starting context window ablation study..."
echo "Output directory: $OUTPUT_DIR"

# Run K=4 on GPU 0
CUDA_VISIBLE_DEVICES=0 $PYTHON -m scripts.hst_train_retrieval \
    --phase 1 \
    --tokens 100000000 \
    --context-window 4 \
    --hidden-dim 128 \
    --svd-rank 64 \
    --batch-size 512 \
    --lr 1e-3 \
    --eval-every 1000 \
    --save-every 5000 \
    --output-dir "$OUTPUT_DIR/k4" \
    2>&1 | tee "$OUTPUT_DIR/k4.log" &
PID_K4=$!
echo "Started K=4 training on GPU 0 (PID: $PID_K4)"

# Run K=6 on GPU 1
CUDA_VISIBLE_DEVICES=1 $PYTHON -m scripts.hst_train_retrieval \
    --phase 1 \
    --tokens 100000000 \
    --context-window 6 \
    --hidden-dim 128 \
    --svd-rank 64 \
    --batch-size 512 \
    --lr 1e-3 \
    --eval-every 1000 \
    --save-every 5000 \
    --output-dir "$OUTPUT_DIR/k6" \
    2>&1 | tee "$OUTPUT_DIR/k6.log" &
PID_K6=$!
echo "Started K=6 training on GPU 1 (PID: $PID_K6)"

# Run K=8 on GPU 2
CUDA_VISIBLE_DEVICES=2 $PYTHON -m scripts.hst_train_retrieval \
    --phase 1 \
    --tokens 100000000 \
    --context-window 8 \
    --hidden-dim 128 \
    --svd-rank 64 \
    --batch-size 512 \
    --lr 1e-3 \
    --eval-every 1000 \
    --save-every 5000 \
    --output-dir "$OUTPUT_DIR/k8" \
    2>&1 | tee "$OUTPUT_DIR/k8.log" &
PID_K8=$!
echo "Started K=8 training on GPU 2 (PID: $PID_K8)"

echo ""
echo "All 3 training processes started. Waiting for completion..."
echo "Monitor logs with: tail -f $OUTPUT_DIR/*.log"

# Wait for all processes
wait $PID_K4
echo "K=4 training completed"
wait $PID_K6
echo "K=6 training completed"
wait $PID_K8
echo "K=8 training completed"

echo ""
echo "============================================================"
echo "Ablation complete! Comparing results..."
echo "============================================================"

# Extract final metrics from logs
for k in 4 6 8; do
    echo ""
    echo "=== K=$k ==="
    grep -A 5 "Final evaluation" "$OUTPUT_DIR/k$k.log" | tail -5 || echo "No final evaluation found"
done
