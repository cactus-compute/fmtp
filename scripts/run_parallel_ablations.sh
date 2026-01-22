#!/bin/bash
# Run all head count ablations in parallel on different GPUs

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate fmtp

# Common args
CHECKPOINT="~/fmtp/checkpoints/gemma_medusa_final"
MODEL="google/gemma-3-270m-it"
HEADS=4
LAYERS=2
RANK=256
ALPHA=512
MAX_PROBLEMS=50

# Run 4 ablations in parallel on GPUs 0-3
echo "Starting parallel ablations on GPUs 0-3..."

# 1 head on GPU 0
CUDA_VISIBLE_DEVICES=0 python -m scripts.gemma_medusa_speed_eval \
    --model-name $MODEL \
    --checkpoint $CHECKPOINT \
    --medusa-num-heads $HEADS \
    --medusa-num-layers $LAYERS \
    --lora-rank $RANK \
    --lora-alpha $ALPHA \
    --zero-init-mtp-mlp \
    --inference-num-heads 1 \
    --fixed-tree-size \
    --skip-standard \
    -x $MAX_PROBLEMS \
    -o ablation_1head_fixed79.json &
PID1=$!

# 2 heads on GPU 1
CUDA_VISIBLE_DEVICES=1 python -m scripts.gemma_medusa_speed_eval \
    --model-name $MODEL \
    --checkpoint $CHECKPOINT \
    --medusa-num-heads $HEADS \
    --medusa-num-layers $LAYERS \
    --lora-rank $RANK \
    --lora-alpha $ALPHA \
    --zero-init-mtp-mlp \
    --inference-num-heads 2 \
    --fixed-tree-size \
    --skip-standard \
    -x $MAX_PROBLEMS \
    -o ablation_2heads_fixed79.json &
PID2=$!

# 3 heads on GPU 2
CUDA_VISIBLE_DEVICES=2 python -m scripts.gemma_medusa_speed_eval \
    --model-name $MODEL \
    --checkpoint $CHECKPOINT \
    --medusa-num-heads $HEADS \
    --medusa-num-layers $LAYERS \
    --lora-rank $RANK \
    --lora-alpha $ALPHA \
    --zero-init-mtp-mlp \
    --inference-num-heads 3 \
    --fixed-tree-size \
    --skip-standard \
    -x $MAX_PROBLEMS \
    -o ablation_3heads_fixed79.json &
PID3=$!

# 4 heads on GPU 3 (use default tree, NOT fixed-tree-size, to match original 1.32x result)
CUDA_VISIBLE_DEVICES=3 python -m scripts.gemma_medusa_speed_eval \
    --model-name $MODEL \
    --checkpoint $CHECKPOINT \
    --medusa-num-heads $HEADS \
    --medusa-num-layers $LAYERS \
    --lora-rank $RANK \
    --lora-alpha $ALPHA \
    --zero-init-mtp-mlp \
    -x $MAX_PROBLEMS \
    -o ablation_4heads_default.json &
PID4=$!

echo "Waiting for all jobs to complete..."
echo "  PID $PID1: 1 head (GPU 0)"
echo "  PID $PID2: 2 heads (GPU 1)"
echo "  PID $PID3: 3 heads (GPU 2)"
echo "  PID $PID4: 4 heads (GPU 3)"

wait $PID1 $PID2 $PID3 $PID4

echo ""
echo "All ablations complete! Results:"
echo "================================"

for f in ablation_*heads*.json; do
    if [ -f "$f" ]; then
        echo ""
        echo "=== $f ==="
        python -c "import json; d=json.load(open('$f')); r=d['results']['mtp']; print(f\"Accuracy: {100*r['accuracy']:.1f}%, Tok/s: {r['tokens_per_second']:.1f}, Mean accepted: {r['mean_accepted_length']:.2f}\")"
    fi
done
