#!/usr/bin/env bash
set -euo pipefail

# setup cache directories
CACHE_DIR="${CACHE_DIR:-$PWD/cache}"
export UV_CACHE_DIR="$CACHE_DIR/uv"
export HF_HOME="$CACHE_DIR/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"

# directories for reference and generated/ground-truth data
REF_ROOT="/home/s2526235/data/IViE/segmented/wav"
GEN_ROOT="$HOME/projects/202_JSALT_2026/results/Personaplex/02_hf/01_IViE_ref-retold/"
# mode for consistency evaluation (both, speaker, or accent)
MODE="both"

# synthetic conversation, all turns
EXP_NAME="consistency_by_turn_IViE_synthetic_all"
CSV="${GEN_ROOT%/}/ref_tgt_turns.csv"

uv run python "./speaker_accent_consistency.py" \
    --ref-root "$REF_ROOT" \
    --gen-root "$GEN_ROOT" \
    --csv "$CSV" \
    --exp-name "$EXP_NAME" \
    --mode "$MODE" \
    --cache-dir "$CACHE_DIR" \
    "$@"
