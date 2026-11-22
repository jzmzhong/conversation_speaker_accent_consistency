# Conversation speaker and accent consistency

Evaluate speaker and accent consistency across turns on each side of a ground-truth or generated conversation. The pipeline uses WavLM x-vectors for speaker embeddings and VoxLect for accent embeddings.

## Setup

Use `uv` with Python 3.11. From the repository root, run `bash 01_model_setup.sh` to download model weights and check out VoxLect. The setup script sources `~/.bashrc` and `~/.config/secrets.env`. Setup and evaluation both use `$PWD/cache` by default. To use a different cache, set `CACHE_DIR` to the same path for both commands.

## Run

```bash
bash 11_consistency_by_turn.sh filtered \
  --ref-root /path/to/reference/audio \
  --gen-root /path/to/turn/audio \
  --csv /path/to/ref_tgt_turns.csv \
  --exp-name experiment_name
```

The CSV header determines whether the script reads `gt_turns(...)` or `gen_turns(...)`. The turn paths are pipe-separated and relative to `--gen-root`, including when using `ref_gt_turns.csv`. The `ref_speech` paths are relative to `--ref-root`.

Use `all` or `filtered` to select the turn column. Both metrics run by default. Pass `--mode variance` or `--mode distance` to run just one. Each run loads any existing embeddings, extracts and saves missing entries needed for the selected metrics, and then scores. Variance uses only turn embeddings.

## Outputs

`--exp-name` is a single folder name under the repository root.

| File | Contents |
| --- | --- |
| `speaker_embeddings.pt`, `accent_embeddings.pt` | Embedding tensors keyed by relative WAV path. Existing entries are reused. |
| `variance_by_side.csv` | Speaker and accent sample covariance traces per conversation side. A side needs at least two turns. |
| `distance_by_side.csv` | Mean cosine distance to the reference per conversation side for speaker and accent embeddings. |
| `ref_turn_distances.csv` | Reference–turn distances for both embedding types, ranked from highest to lowest. |
| `config.json` | Resolved input paths and extraction and scoring settings. |
| `.embeddings.lock` | Coordinates updates when jobs share an experiment folder. |
