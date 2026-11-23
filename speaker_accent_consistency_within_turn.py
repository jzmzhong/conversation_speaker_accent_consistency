"""Extract and score speaker/accent embeddings for overlapping turn chunks."""

import argparse
from fcntl import LOCK_EX, flock
import gc
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

import speaker_accent_consistency as consistency


REPO_DIR = Path(__file__).resolve().parent
SAMPLE_RATE = consistency.SAMPLE_RATE
WINDOW_SAMPLES = 3 * SAMPLE_RATE
HOP_SAMPLES = SAMPLE_RATE
EMBEDDING_TYPES = consistency.EMBEDDING_TYPES
META_FIELDS = consistency.META_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", choices=("long",), default="long")
    parser.add_argument("--chunk-window-seconds", type=float, default=3.0,
                        help="Chunk window size in seconds (default: 3).")
    parser.add_argument("--chunk-hop-seconds", type=float, default=1.0,
                        help="Chunk hop size in seconds (default: 1).")
    parser.add_argument("--ref-root", type=Path, help="Root for reference audio")
    parser.add_argument("--gen-root", type=Path, required=True,
                        help="Root for ground-truth or generated turn audio")
    parser.add_argument("--csv", type=Path, required=True,
                        help="Path to ref_gt_turns.csv or ref_tgt_turns.csv")
    parser.add_argument("--exp-name", help="Experiment folder name under the repository")
    parser.add_argument("--output-dir", type=Path,
                        help="Override the experiment output directory")
    parser.add_argument("--mode", choices=("both", "variance", "distance"), default="both")
    parser.add_argument("--speaker-embedder", choices=("wavlm", "ecapa"), default="wavlm",
                        help="Speaker embedding model (default: wavlm)")
    parser.add_argument("--device", help="Extraction device (default: CUDA if available, otherwise CPU)")
    parser.add_argument("--cache-dir", type=Path,
                        default=os.environ.get("CACHE_DIR") or consistency.DEFAULT_CACHE_DIR)
    parser.add_argument("--voxlect-repo", type=Path, default=os.environ.get("VOXLECT_REPO"))
    args = parser.parse_args()
    if (
        not math.isfinite(args.chunk_window_seconds)
        or not math.isfinite(args.chunk_hop_seconds)
        or args.chunk_window_seconds <= 0
        or args.chunk_hop_seconds <= 0
        or args.chunk_hop_seconds > args.chunk_window_seconds
    ):
        parser.error("chunk sizes must be finite and positive, with hop no larger than window")
    args.window_samples = round(args.chunk_window_seconds * SAMPLE_RATE)
    args.hop_samples = round(args.chunk_hop_seconds * SAMPLE_RATE)
    if args.window_samples < 1 or args.hop_samples < 1:
        parser.error("chunk sizes must be at least one sample")
    if args.mode in ("both", "distance") and args.ref_root is None:
        parser.error("--ref-root is required when computing distances")
    if args.output_dir is None:
        if not args.exp_name:
            parser.error("Pass --exp-name or --output-dir")
        if args.exp_name in (".", "..") or Path(args.exp_name).name != args.exp_name:
            parser.error("--exp-name must be a single directory name")
    return args


def iter_chunk_bounds(
    num_samples: int,
    window_samples: int = WINDOW_SAMPLES,
    hop_samples: int = HOP_SAMPLES,
):
    """Yield windows, dropping a final window shorter than one second."""
    start = 0
    index = 1
    while start < num_samples:
        end = min(start + window_samples, num_samples)
        if end == num_samples and end - start < SAMPLE_RATE:
            break
        yield index, start, end
        if end == num_samples:
            break
        start += hop_samples
        index += 1


def make_chunk_specs(
    sides,
    paths,
    window_samples: int = WINDOW_SAMPLES,
    hop_samples: int = HOP_SAMPLES,
):
    turn_paths = {
        turn
        for side in sides
        for turn in side.turns
    }
    specs_by_turn = {}
    for turn in sorted(turn_paths):
        path, root_name = paths[turn]
        if path is None:
            raise ValueError(f"Pass {root_name} to load turn audio for {turn!r}")
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        waveform = consistency.load_audio(path)
        specs = []
        for index, start, end in iter_chunk_bounds(
            waveform.numel(), window_samples, hop_samples
        ):
            specs.append({
                "key": f"{turn}_chunk{index:03d}",
                "index": index,
                "start": start,
                "end": end,
                "start_seconds": start / SAMPLE_RATE,
                "end_seconds": end / SAMPLE_RATE,
            })
        specs_by_turn[turn] = specs
    return specs_by_turn


def _embedding_is_valid(embedding):
    return (
        embedding.ndim == 1
        and embedding.numel() > 0
        and bool(embedding.isfinite().all())
    )


def extract_missing_embeddings(kind, embeddings, expected_keys, paths, specs_by_turn, args):
    missing = expected_keys - embeddings.keys()
    if not missing:
        return embeddings

    cache_dir = args.cache_dir.expanduser().resolve()
    cache = cache_dir / "huggingface" / "hub"
    os.environ["HF_HOME"] = str(cache.parent)
    os.environ["HF_HUB_CACHE"] = str(cache)
    os.environ["TORCH_HOME"] = str(cache_dir / "torch")
    os.environ["XDG_CACHE_HOME"] = str(cache_dir / "xdg")
    repo = (args.voxlect_repo or cache_dir / "voxlect-source").expanduser().resolve()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if kind == "speaker" and args.speaker_embedder == "wavlm":
        extractor, model = consistency.load_speaker_model(device, cache)
    elif kind == "speaker":
        extractor = None
        model = consistency.load_ecapa_model(device, cache_dir)
    else:
        extractor = None
        model = consistency.load_accent_model(device, cache, repo)

    chunk_keys = {
        spec["key"]
        for specs in specs_by_turn.values()
        for spec in specs
    }
    missing_references = missing - chunk_keys
    missing_chunks = missing - missing_references

    def embed(waveform, key):
        if kind == "speaker":
            if args.speaker_embedder == "wavlm":
                result = consistency.speaker_embedding(waveform, extractor, model, device)
            else:
                result = consistency.ecapa_speaker_embedding(waveform, model)
        else:
            result = consistency.accent_embedding(waveform, model, device)
        if not _embedding_is_valid(result):
            raise ValueError(f"Invalid {kind} embedding for {key}")
        return result

    try:
        with torch.inference_mode():
            for key in sorted(missing_references):
                path, root_name = paths[key]
                if path is None:
                    raise ValueError(f"Pass {root_name} to extract missing embeddings for {key!r}")
                if not path.is_file():
                    raise FileNotFoundError(f"Audio file not found: {path}")
                try:
                    embeddings[key] = embed(consistency.load_audio(path), key)
                except (ValueError, RuntimeError, OSError) as error:
                    raise RuntimeError(
                        f"Could not extract {kind} embedding for {path}: {error}"
                    ) from error

            for turn, specs in specs_by_turn.items():
                missing_turn_specs = [spec for spec in specs if spec["key"] in missing_chunks]
                if not missing_turn_specs:
                    continue
                path, _ = paths[turn]
                waveform = consistency.load_audio(path)
                for spec in missing_turn_specs:
                    chunk = waveform[spec["start"]:spec["end"]]
                    try:
                        embeddings[spec["key"]] = embed(chunk, spec["key"])
                    except (ValueError, RuntimeError, OSError) as error:
                        raise RuntimeError(
                            f"Could not extract {kind} embedding for {path} "
                            f"chunk {spec['index']:03d}: {error}"
                        ) from error
    finally:
        del extractor, model
        gc.collect()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    return embeddings


def load_and_extract(sides, specs_by_turn, paths, output_dir, args, include_references):
    reference_keys = {
        side.reference for side in sides if include_references and side.turns
    }
    chunk_keys = {
        spec["key"]
        for specs in specs_by_turn.values()
        for spec in specs
    }
    expected_keys = reference_keys | chunk_keys
    embeddings = {}
    for kind in EMBEDDING_TYPES:
        cache_name = (
            f"speaker_embeddings_{args.speaker_embedder}.pt"
            if kind == "speaker" and args.speaker_embedder != "wavlm"
            else f"{kind}_embeddings.pt"
        )
        cache_path = output_dir / cache_name
        values = consistency.load_embeddings(cache_path) if cache_path.is_file() else {}
        values = extract_missing_embeddings(
            kind, values, expected_keys, paths, specs_by_turn, args
        )
        consistency.save_embeddings(cache_path, values)
        embeddings[kind] = {
            key: values[key].to(dtype=torch.float64).numpy()
            for key in expected_keys
        }
    return embeddings


def score_variance_by_turn(sides, specs_by_turn, embeddings, output_dir, turn_type):
    turn_field = f"{turn_type}_turn"
    rows = []
    for side in sides:
        for turn in side.turns:
            specs = specs_by_turn[turn]
            row = {
                **side.metadata,
                "ref_speech": side.reference,
                turn_field: turn,
                "n_chunks": len(specs),
            }
            for kind, vectors in embeddings.items():
                row[f"{kind}_covariance_trace"] = (
                    consistency.covariance_trace([vectors[spec["key"]] for spec in specs])
                    if len(specs) >= 2 else None
                )
            rows.append(row)
    fields = (
        *META_FIELDS,
        "ref_speech",
        turn_field,
        "n_chunks",
        *(f"{kind}_covariance_trace" for kind in EMBEDDING_TYPES),
    )
    consistency.write_csv(output_dir / "variance_by_turn.csv", fields, rows)


def score_distances_by_chunk(sides, specs_by_turn, embeddings, output_dir, turn_type):
    turn_field = f"{turn_type}_turn"
    chunk_field = f"{turn_type}_turn_chunk"
    pair_rows, turn_rows = [], []
    for side in sides:
        if not side.turns:
            continue
        reference_vectors = {
            kind: consistency.unit_vector(vectors[side.reference], side.reference, kind)
            for kind, vectors in embeddings.items()
        }
        for turn in side.turns:
            specs = specs_by_turn[turn]
            summary = {
                **side.metadata,
                "ref_speech": side.reference,
                turn_field: turn,
                "n_chunks": len(specs),
            }
            for kind, vectors in embeddings.items():
                distances = []
                reference = reference_vectors[kind]
                for spec in specs:
                    chunk_key = spec["key"]
                    chunk_vector = consistency.unit_vector(vectors[chunk_key], chunk_key, kind)
                    distance = 1.0 - float(
                        np.clip(np.dot(reference, chunk_vector), -1.0, 1.0)
                    )
                    distances.append(distance)
                    pair_rows.append({
                        **side.metadata,
                        "ref_speech": side.reference,
                        turn_field: turn,
                        chunk_field: chunk_key,
                        "chunk_index": spec["index"],
                        "chunk_start_seconds": spec["start_seconds"],
                        "chunk_end_seconds": spec["end_seconds"],
                        "embedding_type": kind,
                        "cosine_distance": distance,
                    })
                summary[f"{kind}_mean_cosine_distance"] = (
                    float(np.mean(distances)) if distances else None
                )
            turn_rows.append(summary)

    pair_fields = (
        *META_FIELDS,
        "ref_speech",
        turn_field,
        chunk_field,
        "chunk_index",
        "chunk_start_seconds",
        "chunk_end_seconds",
        "embedding_type",
        "cosine_distance",
    )
    mean_fields = (
        *META_FIELDS,
        "ref_speech",
        turn_field,
        "n_chunks",
        *(f"{kind}_mean_cosine_distance" for kind in EMBEDDING_TYPES),
    )
    consistency.write_csv(output_dir / "ref_chunk_distances.csv", pair_fields, pair_rows)
    consistency.write_csv(output_dir / "distance_by_turn.csv", mean_fields, turn_rows)


def main():
    args = parse_args()
    csv_path = args.csv.expanduser().resolve()
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser()
    else:
        output_dir = REPO_DIR / args.exp_name

    run_variance = args.mode in ("both", "variance")
    run_distance = args.mode in ("both", "distance")
    sides, turn_type = consistency.read_sides(csv_path, args.selection)
    ref_root = args.ref_root.expanduser().resolve() if args.ref_root is not None else None
    turn_root = args.gen_root.expanduser().resolve()
    paths = consistency.collect_audio_paths(sides, ref_root, turn_root, run_distance)
    specs_by_turn = make_chunk_specs(
        sides, paths, args.window_samples, args.hop_samples
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    run_settings = {
        "csv_path": str(csv_path),
        "selection": args.selection,
        "turn_type": turn_type,
        "mode": args.mode,
        "window_seconds": args.window_samples / SAMPLE_RATE,
        "hop_seconds": args.hop_samples / SAMPLE_RATE,
    }
    with (output_dir / ".embeddings.lock").open("a") as lock:
        flock(lock, LOCK_EX)
        config = (
            json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        )
        config.update(exp_name=args.exp_name, output_dir=str(output_dir.resolve()))
        embeddings = load_and_extract(
            sides, specs_by_turn, paths, output_dir, args, run_distance
        )
        previous_extraction = config.get("extraction", {})
        model_cache = args.cache_dir.expanduser().resolve()
        voxlect_repo = (
            (args.voxlect_repo or model_cache / "voxlect-source")
            .expanduser()
            .resolve()
        )
        config["extraction"] = {
            **run_settings,
            "ref_root": str(ref_root) if ref_root is not None else previous_extraction.get("ref_root"),
            "gen_root": str(turn_root),
            "cache_dir": str(model_cache),
            "voxlect_repo": str(voxlect_repo),
            "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            "speaker_embedder": args.speaker_embedder,
        }
        consistency.write_config(config_path, config)
        scoring = config.setdefault("scoring", {})
        if run_variance:
            score_variance_by_turn(sides, specs_by_turn, embeddings, output_dir, turn_type)
            scoring["variance"] = run_settings
            consistency.write_config(config_path, config)
        if run_distance:
            score_distances_by_chunk(sides, specs_by_turn, embeddings, output_dir, turn_type)
            scoring["distance"] = run_settings
            consistency.write_config(config_path, config)
    print(f"Output directory: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
