#!/usr/bin/env bash
set -euo pipefail

# setup cache directories
CACHE_DIR="${CACHE_DIR:-$PWD/cache}"
export UV_CACHE_DIR="$CACHE_DIR/uv"
export HF_HOME="$CACHE_DIR/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"

# mode for consistency evaluation (both, speaker, or accent)
MODE="both"
# selection for filtered turns, > 1s for each turn
SELECTION="filtered"

# synthetic conversation, filtered turns
REF_ROOT="$HOME/data/IViE/segmented/wav"
GEN_ROOT="$HOME/projects/202_JSALT_2026/results/Personaplex/02_hf/01_IViE_ref-retold"
EXP_NAME="consistency_by_turn_IViE_synthetic_$SELECTION"
CSV="${GEN_ROOT%/}/ref_tgt_turns.csv"

# # real conversation, filtered turns
# EXP_NAME="consistency_by_turn_IViE_ground-truth_$SELECTION"
# REF_ROOT="$HOME/data/IViE/segmented/wav"
# GEN_ROOT="$HOME/data/IViE/segmented/wav"
# CSV="$HOME/projects/202_JSALT_2026/results/Personaplex/02_hf/01_IViE_ref-retold/ref_gt_turns.csv"

uv run python "./speaker_accent_consistency.py" \
    --selection "$SELECTION" \
    --ref-root "$REF_ROOT" \
    --gen-root "$GEN_ROOT" \
    --csv "$CSV" \
    --exp-name "$EXP_NAME" \
    --mode "$MODE" \
    --cache-dir "$CACHE_DIR" \
    "$@"
