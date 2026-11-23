#!/usr/bin/env bash
set -euo pipefail

SUMMARY_OUTPUT_DIR="${SUMMARY_OUTPUT_DIR:-./summary_within_turn_wavlm}"

EXPERIMENTS=(
    "consistency_within_turn_2s-window_2s-hop_IViE_ground-truth_long_wavlm"
    "consistency_within_turn_2s-window_2s-hop_IViE_synthetic_long_wavlm"
    "consistency_within_turn_3s-window_1s-hop_IViE_ground-truth_long_wavlm"
    "consistency_within_turn_3s-window_1s-hop_IViE_synthetic_long_wavlm"
    "consistency_within_turn_4s-window_1s-hop_IViE_ground-truth_long_wavlm"
    "consistency_within_turn_4s-window_1s-hop_IViE_synthetic_long_wavlm"
)
LABELS=(
    "consistency_within_turn_2s-window_2s-hop_IViE_ground-truth_long_wavlm=Ground truth (2s window, 2s hop)"
    "consistency_within_turn_2s-window_2s-hop_IViE_synthetic_long_wavlm=Synthetic (2s window, 2s hop)"
    "consistency_within_turn_3s-window_1s-hop_IViE_ground-truth_long_wavlm=Ground truth (3s window, 1s hop)"
    "consistency_within_turn_3s-window_1s-hop_IViE_synthetic_long_wavlm=Synthetic (3s window, 1s hop)"
    "consistency_within_turn_4s-window_1s-hop_IViE_ground-truth_long_wavlm=Ground truth (4s window, 1s hop)"
    "consistency_within_turn_4s-window_1s-hop_IViE_synthetic_long_wavlm=Synthetic (4s window, 1s hop)"
)

# SUMMARY_OUTPUT_DIR="summary_within_turn_ecapa"

# EXPERIMENTS=(
#     "consistency_within_turn_2s-window_2s-hop_IViE_ground-truth_long_ecapa"
#     "consistency_within_turn_2s-window_2s-hop_IViE_synthetic_long_ecapa"
#     "consistency_within_turn_3s-window_1s-hop_IViE_ground-truth_long_ecapa"
#     "consistency_within_turn_3s-window_1s-hop_IViE_synthetic_long_ecapa"
# )
# LABELS=(
#     "consistency_within_turn_2s-window_2s-hop_IViE_ground-truth_long_ecapa=Ground truth (2s window, 2s hop)"
#     "consistency_within_turn_2s-window_2s-hop_IViE_synthetic_long_ecapa=Synthetic (2s window, 2s hop)"
#     "consistency_within_turn_3s-window_1s-hop_IViE_ground-truth_long_ecapa=Ground truth (3s window, 1s hop)"
#     "consistency_within_turn_3s-window_1s-hop_IViE_synthetic_long_ecapa=Synthetic (3s window, 1s hop)"
# )

label_args=()
for label in "${LABELS[@]}"; do
    label_args+=(--label "$label")
done

uv run python "./summarise.py" \
    "${EXPERIMENTS[@]}" \
    --unit turn \
    --require-complete \
    --output-dir "$SUMMARY_OUTPUT_DIR" \
    "${label_args[@]}" \
    "$@"
