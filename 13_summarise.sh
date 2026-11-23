#!/usr/bin/env bash
set -euo pipefail

SUMMARY_OUTPUT_DIR="${SUMMARY_OUTPUT_DIR:-./summary}"

EXPERIMENTS=(
    "consistency_by_turn_IViE_ground-truth_filtered"
    "consistency_by_turn_IViE_synthetic_filtered"
)
LABELS=(
    "consistency_by_turn_IViE_ground-truth_filtered=IViE_ground-truth"
    "consistency_by_turn_IViE_synthetic_filtered=IViE_generation"
)

label_args=()
for label in "${LABELS[@]}"; do
    label_args+=(--label "$label")
done

uv run python "./summarise.py" \
    "${EXPERIMENTS[@]}" \
    --output-dir "$SUMMARY_OUTPUT_DIR" \
    "${label_args[@]}" \
    "$@"
