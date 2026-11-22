#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCORE_DIR="$SCRIPT_DIR/consistency_by_turn_IViE_synthetic_filtered"
GEN_ROOT="$SCRIPT_DIR/../results/Personaplex/02_hf/01_IViE_ref-retold"
REF_ROOT="$HOME/data/IViE/segmented/wav"
METADATA_CSV="$GEN_ROOT/ref_tgt_turns.csv"
GT_METADATA_CSV="$GEN_ROOT/ref_gt_turns.csv"
GT_CONVERSATION_ROOT="$HOME/data/IViE/enhanced_separated_DialogueSidon"
OUTPUT_DIR="$SCRIPT_DIR/demo"
TOP=10
SEED=42

uv run --project "$SCRIPT_DIR" python "$SCRIPT_DIR/21_build_html.py" \
    --score-dir "$SCORE_DIR" \
    --metadata-csv "$METADATA_CSV" \
    --gt-metadata-csv "$GT_METADATA_CSV" \
    --ref-root "$REF_ROOT" \
    --gen-root "$GEN_ROOT" \
    --gt-conversation-root "$GT_CONVERSATION_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --top "$TOP" \
    --seed "$SEED" \
    "$@"
