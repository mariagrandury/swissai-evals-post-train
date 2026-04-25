#!/bin/bash
# SNR stage runner - GENERATED from configs/signal_to_ratio/models_pretraining_smollm.txt (selection: --include stage1 --total 30)
# Regenerate: bash scripts/generate_snr_runner.sh --models configs/signal_to_ratio/models_pretraining_smollm.txt --include stage1 --total 30 > $0

# ===== HuggingFace revisions (REVISION is singleton; loop per branch) =====
unset MODEL_CHECKPOINTS MODEL_ITERATIONS
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-false}
export LM_EVAL_BACKEND=${LM_EVAL_BACKEND:-vllm}

HF_ENTRIES=(
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-1000000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-1080000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-120000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-1280000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-1400000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-1520000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-1600000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-1720000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-1840000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-1960000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-2040000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-2160000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-2280000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-240000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-2480000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-2560000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-2680000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-280000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-2880000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-3000000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-3120000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-3200000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-3320000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-3440000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-400000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-520000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-640000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-760000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-840000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-960000"
)

for ENTRY in "${HF_ENTRIES[@]}"; do
    IFS="|" read -r NAME REPO BRANCH <<< "$ENTRY"
    export REVISION="$BRANCH"
    unset MODEL_CHECKPOINTS
    declare -A MODEL_CHECKPOINTS=(["${NAME}-${BRANCH}"]="$REPO")
    source runners/hf_base_runner.sh "SNR HuggingFace checkpoints"
done
unset REVISION MODEL_CHECKPOINTS
