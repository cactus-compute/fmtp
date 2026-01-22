#!/bin/bash
# Run full evaluations on 8 GPUs in parallel
# Tasks: gsm8k, arc-challenge, mmlu, humaneval
# Configs: 1-head and 4-head for each

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate fmtp

# Common args
CHECKPOINT="checkpoints/gemma_medusa_final"
MODEL="google/gemma-3-270m-it"
HEADS=4
LAYERS=2
RANK=256
ALPHA=512

echo "Starting 8 parallel evaluations..."
echo "Tasks: gsm8k, arc-challenge, mmlu, humaneval"
echo "Configs: 1-head (fixed tree) and 4-head (default tree)"
echo ""

# GSM8K - 4 heads on GPU 0
CUDA_VISIBLE_DEVICES=0 python -m scripts.gemma_medusa_speed_eval \
    --model-name $MODEL \
    --checkpoint $CHECKPOINT \
    --medusa-num-heads $HEADS \
    --medusa-num-layers $LAYERS \
    --lora-rank $RANK \
    --lora-alpha $ALPHA \
    --zero-init-mtp-mlp \
    --task gsm8k \
    -o eval_gsm8k_4heads.json &
PID1=$!

# GSM8K - 1 head on GPU 1
CUDA_VISIBLE_DEVICES=1 python -m scripts.gemma_medusa_speed_eval \
    --model-name $MODEL \
    --checkpoint $CHECKPOINT \
    --medusa-num-heads $HEADS \
    --medusa-num-layers $LAYERS \
    --lora-rank $RANK \
    --lora-alpha $ALPHA \
    --zero-init-mtp-mlp \
    --task gsm8k \
    --inference-num-heads 1 \
    --fixed-tree-size \
    --skip-standard \
    -o eval_gsm8k_1head.json &
PID2=$!

# ARC-Challenge - 4 heads on GPU 2
CUDA_VISIBLE_DEVICES=2 python -m scripts.gemma_medusa_speed_eval \
    --model-name $MODEL \
    --checkpoint $CHECKPOINT \
    --medusa-num-heads $HEADS \
    --medusa-num-layers $LAYERS \
    --lora-rank $RANK \
    --lora-alpha $ALPHA \
    --zero-init-mtp-mlp \
    --task arc-challenge \
    -o eval_arc_4heads.json &
PID3=$!

# ARC-Challenge - 1 head on GPU 3
CUDA_VISIBLE_DEVICES=3 python -m scripts.gemma_medusa_speed_eval \
    --model-name $MODEL \
    --checkpoint $CHECKPOINT \
    --medusa-num-heads $HEADS \
    --medusa-num-layers $LAYERS \
    --lora-rank $RANK \
    --lora-alpha $ALPHA \
    --zero-init-mtp-mlp \
    --task arc-challenge \
    --inference-num-heads 1 \
    --fixed-tree-size \
    --skip-standard \
    -o eval_arc_1head.json &
PID4=$!

# MMLU - 4 heads on GPU 4
CUDA_VISIBLE_DEVICES=4 python -m scripts.gemma_medusa_speed_eval \
    --model-name $MODEL \
    --checkpoint $CHECKPOINT \
    --medusa-num-heads $HEADS \
    --medusa-num-layers $LAYERS \
    --lora-rank $RANK \
    --lora-alpha $ALPHA \
    --zero-init-mtp-mlp \
    --task mmlu \
    -o eval_mmlu_4heads.json &
PID5=$!

# MMLU - 1 head on GPU 5
CUDA_VISIBLE_DEVICES=5 python -m scripts.gemma_medusa_speed_eval \
    --model-name $MODEL \
    --checkpoint $CHECKPOINT \
    --medusa-num-heads $HEADS \
    --medusa-num-layers $LAYERS \
    --lora-rank $RANK \
    --lora-alpha $ALPHA \
    --zero-init-mtp-mlp \
    --task mmlu \
    --inference-num-heads 1 \
    --fixed-tree-size \
    --skip-standard \
    -o eval_mmlu_1head.json &
PID6=$!

# HumanEval - 4 heads on GPU 6
CUDA_VISIBLE_DEVICES=6 python -m scripts.gemma_medusa_speed_eval \
    --model-name $MODEL \
    --checkpoint $CHECKPOINT \
    --medusa-num-heads $HEADS \
    --medusa-num-layers $LAYERS \
    --lora-rank $RANK \
    --lora-alpha $ALPHA \
    --zero-init-mtp-mlp \
    --task humaneval \
    -o eval_humaneval_4heads.json &
PID7=$!

# HumanEval - 1 head on GPU 7
CUDA_VISIBLE_DEVICES=7 python -m scripts.gemma_medusa_speed_eval \
    --model-name $MODEL \
    --checkpoint $CHECKPOINT \
    --medusa-num-heads $HEADS \
    --medusa-num-layers $LAYERS \
    --lora-rank $RANK \
    --lora-alpha $ALPHA \
    --zero-init-mtp-mlp \
    --task humaneval \
    --inference-num-heads 1 \
    --fixed-tree-size \
    --skip-standard \
    -o eval_humaneval_1head.json &
PID8=$!

echo "Waiting for all jobs to complete..."
echo "  PID $PID1: GSM8K 4-head (GPU 0)"
echo "  PID $PID2: GSM8K 1-head (GPU 1)"
echo "  PID $PID3: ARC-Challenge 4-head (GPU 2)"
echo "  PID $PID4: ARC-Challenge 1-head (GPU 3)"
echo "  PID $PID5: MMLU 4-head (GPU 4)"
echo "  PID $PID6: MMLU 1-head (GPU 5)"
echo "  PID $PID7: HumanEval 4-head (GPU 6)"
echo "  PID $PID8: HumanEval 1-head (GPU 7)"

wait $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7 $PID8

echo ""
echo "All evaluations complete!"
echo "================================"
echo ""

# Print results summary
for f in eval_*.json; do
    if [ -f "$f" ]; then
        echo "=== $f ==="
        python -c "
import json
d = json.load(open('$f'))
task = d.get('task', 'unknown')
cfg = d['config']
heads = cfg.get('inference_num_heads', cfg['medusa_num_heads'])
print(f'Task: {task}, Heads: {heads}')
if 'standard' in d['results']:
    r = d['results']['standard']
    print(f\"  Standard: {100*r['accuracy']:.1f}% acc, {r['tokens_per_second']:.1f} tok/s\")
r = d['results']['mtp']
print(f\"  MTP: {100*r['accuracy']:.1f}% acc, {r['tokens_per_second']:.1f} tok/s, mean_accepted={r['mean_accepted_length']:.2f}\")
"
        echo ""
    fi
done
