#!/usr/bin/env bash
set -euo pipefail

# setup cache directories
CACHE_DIR="${CACHE_DIR:-$PWD/cache}"
export UV_CACHE_DIR="$CACHE_DIR/uv"
export HF_HOME="$CACHE_DIR/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"

# mode for consistency evaluation (both, speaker, or accent)
MODE="both"
# selection for long turns, > 5s for each turn
SELECTION="long"
# 3 seconds window for chunks, 1 second hop
CHUNK_WINDOW_SECONDS=3
CHUNK_HOP_SECONDS=1
# speaker embedding model
# EMB_MODEL="wavlm"
EMB_MODEL="ecapa"

# synthetic conversation, long turns
EXP_NAME="consistency_within_turn_${CHUNK_WINDOW_SECONDS}s-window_${CHUNK_HOP_SECONDS}s-hop_IViE_synthetic_${SELECTION}_${EMB_MODEL}"
REF_ROOT="$HOME/data/IViE/segmented/wav"
GEN_ROOT="$HOME/projects/202_JSALT_2026/results/Personaplex/02_hf/01_IViE_ref-retold"
CSV="${GEN_ROOT}/ref_tgt_turns.csv"

# # real conversation, long turns
# EXP_NAME="consistency_within_turn_${CHUNK_WINDOW_SECONDS}s-window_${CHUNK_HOP_SECONDS}s-hop_IViE_ground-truth_${SELECTION}_${EMB_MODEL}"
# REF_ROOT="$HOME/data/IViE/segmented/wav"
# GEN_ROOT="$HOME/data/IViE/segmented/wav"
# CSV="$HOME/projects/202_JSALT_2026/results/Personaplex/02_hf/01_IViE_ref-retold/ref_gt_turns.csv"

uv run python "./speaker_accent_consistency_within_turn.py" \
    --selection "$SELECTION" \
    --chunk-window-seconds "$CHUNK_WINDOW_SECONDS" \
    --chunk-hop-seconds "$CHUNK_HOP_SECONDS" \
    --ref-root "$REF_ROOT" \
    --gen-root "$GEN_ROOT" \
    --csv "$CSV" \
    --exp-name "$EXP_NAME" \
    --mode "$MODE" \
    --speaker-embedder "$EMB_MODEL" \
    --cache-dir "$CACHE_DIR" \
    "$@"
