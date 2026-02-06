#!/bin/bash
# Run token type analysis on VM
#
# Usage:
#   bash scripts/run_token_type_analysis.sh
#
# This script:
# 1. Syncs the token_type_analysis.py script to the VM
# 2. Runs the analysis on 1000 wildchat samples
# 3. Downloads the results

set -e

VM_ZONE="us-central1-c"
VM_INSTANCE="a100-40x8"
VM_PROJECT="cactus-v1-452518"
REMOTE_DIR="~/fast-mtp"
CHECKPOINT="~/.cache/nanochat/gemma_medusa_270m_wildchat_100k_nolora"

echo "=== Token Type Analysis ==="
echo "VM: $VM_INSTANCE in $VM_ZONE"

# Sync the analysis script to VM
echo ""
echo "Step 1: Syncing analysis script to VM..."
gcloud compute scp --zone "$VM_ZONE" --project "$VM_PROJECT" \
    scripts/token_type_analysis.py \
    "${VM_INSTANCE}:${REMOTE_DIR}/scripts/"

# Run the analysis
echo ""
echo "Step 2: Running token type analysis on VM..."
gcloud compute ssh --zone "$VM_ZONE" "${VM_INSTANCE}" --project "$VM_PROJECT" -- \
    "cd ${REMOTE_DIR} && \
     source ~/.bashrc && \
     conda activate fmtp && \
     python -m scripts.token_type_analysis \
         --checkpoint ${CHECKPOINT} \
         --data-path data/wildchat_100k.jsonl \
         --n 1000 \
         --max-tokens 128 \
         --output token_type_analysis_results.json"

# Download results
echo ""
echo "Step 3: Downloading results..."
gcloud compute scp --zone "$VM_ZONE" --project "$VM_PROJECT" \
    "${VM_INSTANCE}:${REMOTE_DIR}/token_type_analysis_results.json" \
    ./results/

echo ""
echo "Done! Results saved to results/token_type_analysis_results.json"
