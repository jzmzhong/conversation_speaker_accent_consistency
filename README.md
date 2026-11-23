# Conversation speaker and accent consistency

Evaluate each conversation side with WavLM (default) or ECAPA-TDNN speaker
embeddings and VoxLect accent embeddings.
Run the commands below from this directory.

## Setup

Use Python 3.11 and `uv`:

```bash
bash 01_model_setup.sh
```

Setup sources `~/.bashrc` and `~/.config/secrets.env`. Set `CACHE_DIR` for both
setup and evaluation to change the default `./cache`.
To prepare ECAPA-TDNN weights, pass `--speaker-embedder ecapa` to setup.

## Evaluate

Configure paths, selection and experiment name in `11_consistency_by_turn.sh`,
then run `bash 11_consistency_by_turn.sh`. The script defaults to synthetic
conversations; the configuration also includes a ground-truth example. Use
`--mode variance` or `--mode distance` to run one metric, or `both` for both.
Pass `--speaker-embedder ecapa` to use ECAPA-TDNN instead of WavLM; the default
is `wavlm`. This option is supported by both evaluation scripts and their shell
wrappers. WavLM uses `speaker_embeddings.pt`; ECAPA uses a separate
`speaker_embeddings_ecapa.pt` cache.

The input CSV supplies `ref_speech` relative to `--ref-root` and pipe-separated
`gt_turns(...)` or `gen_turns(...)` relative to `--gen-root`. `all` or `filtered`
selects the turn column. Existing embeddings are reused.

Each experiment folder contains:

| File | Contents |
| --- | --- |
| `speaker_embeddings.pt` or `speaker_embeddings_ecapa.pt`, `accent_embeddings.pt` | Cached embeddings keyed by relative WAV path |
| `variance_by_side.csv` | Sample covariance traces across turns; requires at least two turns |
| `distance_by_side.csv` | Mean cosine distance to the reference per side |
| `ref_turn_distances.csv` | Reference–turn distances, ranked highest first |
| `config.json`, `.embeddings.lock` | Run settings and cache-update lock |

## Within-turn chunks

Configure paths, `CHUNK_WINDOW_SECONDS`, and `CHUNK_HOP_SECONDS` in
`12_consistency_within_turn.sh`, then run `bash 12_consistency_within_turn.sh`.
The chunk settings default to 3 seconds and 1 second. This run reads only
`gt_turns(long)` or `gen_turns(long)`. Each selected turn is split into
windows using those settings; the final window ends at the turn
boundary and may be shorter, but is dropped if it is shorter than one second.
With windows of at least one second, a turn shorter than one second has no
chunks. Chunk embeddings are cached under keys made from the relative turn
path plus `_chunk001`, `_chunk002`, and so on.
The `--speaker-embedder` option also selects the speaker model for this workflow.

The output includes per-chunk reference cosine distances in
`ref_chunk_distances.csv`, mean distances per turn-reference pair in
`distance_by_turn.csv`, and within-turn embedding covariance traces per
turn-reference pair in `variance_by_turn.csv`.

## Summarise

Configure `EXPERIMENTS` and `LABELS` in `13_summarise.sh`, then run:

```bash
bash 13_summarise.sh
```

`SUMMARY_OUTPUT_DIR` changes the default `./summary`; `--output-dir` overrides
it. Arguments are forwarded to Python, including repeatable
`--label 'experiment=display name'` (the last value wins). Python requires at
least one experiment path.

Each experiment gets a CSV of counts, means and standard errors; four PNGs
compare speaker and accent variance and distance. Each side is equally weighted,
blank scores are excluded, and plot error bars show standard error
(`sample standard deviation / sqrt(n)`), undefined for one observation.

For within-turn experiments, run `bash 14_summarise_within_turn.sh`. It
compares the four configured 2-second and 3-second experiments in
`./summary_within_turn` by default. Each turn is equally weighted, using
`variance_by_turn.csv` and `distance_by_turn.csv`. The script requires
observations for every metric in all four experiments. The shared `summarise.py`
also accepts `--unit turn` for other within-turn experiment directories.

## Listening demo

Configure the paths, `TOP` and `SEED` in `21_build_html.sh`, then run:

```bash
bash 21_build_html.sh
python3 -m http.server 8000 --directory demo
```

`demo/index.html` contains six top-score tables and ten sampled conversations
(default seed 42). Ranked conversation rows include `gen_side` (`a.wav` or
`b.wav`) and stereo `gen_conversation` (`ab.wav`); sampled rows compare
DialogueSidon GT stereo and generated stereo with both voice references.
Audio is copied into `demo/audio/`, so the folder is portable.

The script requires the three score CSVs, matching GT and generated metadata,
and their audio roots. It validates reference matches and missing audio.
Additional options are forwarded to Python; run `bash 21_build_html.sh --help`.
