#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc
source ~/.config/secrets.env

CACHE_DIR="${CACHE_DIR:-$PWD/cache}"

export UV_CACHE_DIR="$CACHE_DIR/uv"
export HF_HOME="$CACHE_DIR/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"

# uv creates the project environment; model_setup.py puts model weights and VoxLect
# source under CACHE_DIR. Neither step requires a GPU.
uv run python ./model_setup.py \
    --cache-dir "$CACHE_DIR" "$@"
