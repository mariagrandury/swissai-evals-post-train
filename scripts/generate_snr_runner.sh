#!/bin/bash
# Generate a self-contained SNR stage runner from a models file (stdout).
# Usage: generate_snr_runner.sh --models <file> (--last N | --total T) > runners/snr_<stage>.sh
# Then: bash scripts/launch_evaluations.sh snr-<stage> --script runners/snr_<stage>.sh [flags]
set -euo pipefail

LIST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/list_checkpoints.sh"

MODELS_FILE=""; MODE=""; COUNT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)         MODELS_FILE="$2"; shift 2 ;;
        --last|--total)   MODE="$1"; COUNT="$2"; shift 2 ;;
        *) echo "Usage: $0 --models <file> (--last N | --total T)" >&2; exit 1 ;;
    esac
done
[[ -f "$MODELS_FILE" && -n "$MODE" ]] \
    || { echo "Usage: $0 --models <file> (--last N | --total T)" >&2; exit 1; }

derive_name() {
    local m="${1#https://huggingface.co/}"
    [[ "$m" == /* ]] && { m="${m%/}"; m="${m%/checkpoints}"; basename "$m"; } || echo "${m##*/}"
}

MEG=$(mktemp); HF=$(mktemp)
trap 'rm -f "$MEG" "$HF"' EXIT

while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"; line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    base=$(derive_name "$line")
    if ! selected=$(bash "$LIST" "$line" "$MODE" "$COUNT" 2>/dev/null); then
        echo "# WARNING: enumeration failed: $line" >&2; continue
    fi
    case "$line" in
        /iopsstor*|/capstor*)
            ckpt="${line%/}"
            while IFS= read -r it; do
                printf '    ["%s-iter%s"]="%s"\n' "$base" "$it" "$ckpt" >> "$MEG"
            done <<< "$selected" ;;
        https://huggingface.co/*)
            repo="${line#https://huggingface.co/}"
            while IFS= read -r br; do
                printf '    "%s|%s|%s"\n' "$base" "$repo" "$br" >> "$HF"
            done <<< "$selected" ;;
        *) echo "# WARNING: unrecognized format: $line" >&2 ;;
    esac
done < "$MODELS_FILE"

[[ -s "$MEG" || -s "$HF" ]] || { echo "No checkpoints selected." >&2; exit 1; }

echo "#!/bin/bash"
echo "# SNR stage runner - GENERATED from $MODELS_FILE (selection: $MODE $COUNT)"
echo "# Regenerate: bash scripts/generate_snr_runner.sh --models $MODELS_FILE $MODE $COUNT > \$0"

if [[ -s "$MEG" ]]; then
    cat <<'EOF'

# ===== Megatron checkpoints (one sbatch per iter; hf_base_runner parses the
# iter from the "<base>-iter<N>" key) =====
export TOKENIZER=${TOKENIZER:-alehc/swissai-tokenizer}
export BOS=${BOS:-true}
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-false}
export LM_EVAL_BACKEND=${LM_EVAL_BACKEND:-megatron_lm}

declare -A MODEL_CHECKPOINTS=(
EOF
    cat "$MEG"
    echo ')'; echo ''
    echo 'source runners/hf_base_runner.sh "SNR Megatron checkpoints"'
fi

if [[ -s "$HF" ]]; then
    cat <<'EOF'

# ===== HuggingFace revisions (REVISION is singleton; loop per branch) =====
unset MODEL_CHECKPOINTS
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-false}
export LM_EVAL_BACKEND=${LM_EVAL_BACKEND:-vllm}

HF_ENTRIES=(
EOF
    cat "$HF"
    cat <<'EOF'
)

for ENTRY in "${HF_ENTRIES[@]}"; do
    IFS="|" read -r NAME REPO BRANCH <<< "$ENTRY"
    export REVISION="$BRANCH"
    unset MODEL_CHECKPOINTS
    declare -A MODEL_CHECKPOINTS=(["${NAME}-${BRANCH}"]="$REPO")
    source runners/hf_base_runner.sh "SNR HuggingFace checkpoints"
done
unset REVISION MODEL_CHECKPOINTS
EOF
fi
