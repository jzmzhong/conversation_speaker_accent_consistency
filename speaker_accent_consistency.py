"""Extract and score WavLM speaker and VoxLect accent embeddings by conversation turn."""

import argparse
import csv
from dataclasses import dataclass
from fcntl import LOCK_EX, flock
import gc
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import numpy as np
import torch
import torchaudio


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = REPO_DIR / "cache"
SAMPLE_RATE = 16000
VOXLECT_CHUNK_SAMPLES = 15 * SAMPLE_RATE
META_FIELDS = ("accent", "task", "conversation_id", "channel_id", "speaker_id")
EMBEDDING_TYPES = ("speaker", "accent")


@dataclass
class ConversationSide:
    metadata: dict[str, str]
    reference: str
    turns: list[str]


def load_audio(path):
    waveform, sample_rate = torchaudio.load(str(path))
    waveform = waveform.mean(dim=0)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    if waveform.numel() < 160:
        raise ValueError(f"Audio is too short to extract embeddings: {path}")
    return waveform


def load_speaker_model(device, cache):
    from transformers import AutoFeatureExtractor, WavLMForXVector

    name = "microsoft/wavlm-base-sv"
    extractor = AutoFeatureExtractor.from_pretrained(name, cache_dir=str(cache))
    model = WavLMForXVector.from_pretrained(name, cache_dir=str(cache)).to(device).eval()
    return extractor, model


def speaker_embedding(waveform, extractor, model, device):
    inputs = extractor(waveform.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
    outputs = model(**{key: value.to(device) for key, value in inputs.items()})
    embedding = outputs.embeddings.squeeze(0)
    norm = torch.linalg.vector_norm(embedding)
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("Invalid WavLM speaker embedding norm")
    return (embedding / norm).cpu()


def load_accent_model(device, cache, repo):
    repo = Path(repo).expanduser().resolve()
    if not (repo / "src/model/dialect/whisper_dialect.py").is_file():
        raise FileNotFoundError(f"VoxLect source checkout not found: {repo}")
    sys.path.insert(0, str(repo))
    from src.model.dialect.whisper_dialect import WhisperWrapper

    return WhisperWrapper.from_pretrained(
        "tiantiaf/voxlect-english-dialect-whisper-large-v3", cache_dir=str(cache)
    ).to(device).eval()


def accent_embedding(waveform, model, device):
    # VoxLect accepts at most 15 seconds. Pool all valid frames across chunks.
    frame_sum = None
    total_frames = 0
    chunk_count = (waveform.numel() + VOXLECT_CHUNK_SAMPLES - 1) // VOXLECT_CHUNK_SAMPLES
    chunk_size = (waveform.numel() + chunk_count - 1) // chunk_count
    for chunk in waveform.split(chunk_size):
        length = model._get_feat_extract_output_lengths(torch.tensor([chunk.numel()])).item()
        if length == 0:
            raise ValueError("VoxLect cannot encode an audio chunk shorter than 10 ms")
        features = model.feature_extractor(
            [chunk.numpy()], return_tensors="pt", sampling_rate=SAMPLE_RATE,
            max_length=VOXLECT_CHUNK_SAMPLES,
        ).input_features.to(device)
        hidden = model.backbone_model.encoder(features, output_hidden_states=True).hidden_states[-1]
        frames = model.model_seq(hidden.transpose(1, 2)).transpose(1, 2)[0, :length]
        chunk_sum = frames.sum(dim=0)
        frame_sum = chunk_sum if frame_sum is None else frame_sum + chunk_sum
        total_frames += frames.shape[0]
    return (frame_sum / total_frames).cpu()


def covariance_trace(embeddings):
    """Trace of sample covariance without materializing the covariance matrix."""
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError("At least two nonempty embeddings are required")
    if not np.isfinite(values).all():
        raise ValueError("Embeddings contain non-finite values")
    centered = values - values.mean(axis=0)
    return float(np.sum(centered * centered) / (len(values) - 1))


def relative_audio_key(value):
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or not path.name:
        raise ValueError(f"Expected a relative audio path, got {value!r}")
    return path.as_posix()


def read_sides(csv_path, selection):
    sides = []
    with csv_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames or ()
        turn_columns = {
            kind: f"{kind}_turns({selection})" for kind in ("gt", "gen")
        }
        matching_types = [kind for kind, column in turn_columns.items() if column in fields]
        if len(matching_types) != 1:
            expected = " or ".join(turn_columns.values())
            raise ValueError(f"{csv_path} must contain exactly one turn column: {expected}")
        turn_type = matching_types[0]
        turn_column = turn_columns[turn_type]
        required = (*META_FIELDS, "ref_speech", turn_column)
        missing = set(required).difference(fields)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            if any(row[field] is None for field in required):
                raise ValueError(f"{csv_path}:{reader.line_num}: incomplete row")
            try:
                reference = relative_audio_key(row["ref_speech"].strip())
                turns = [
                    relative_audio_key(path.strip())
                    for path in row[turn_column].split("|")
                    if path.strip()
                ]
            except ValueError as error:
                raise ValueError(f"{csv_path}:{reader.line_num}: {error}") from error
            sides.append(ConversationSide(
                {field: row[field] for field in META_FIELDS}, reference, turns
            ))
    return sides, turn_type


def collect_audio_paths(sides, ref_root, turn_root, include_references):
    paths = {}
    for side in sides:
        audio = [(turn, turn_root, "--gen-root") for turn in side.turns]
        if include_references and side.turns:
            audio.append((side.reference, ref_root, "--ref-root"))
        for key, root, root_name in audio:
            path = root / key if root is not None else None
            location = (path, root_name)
            previous_path, previous_root = paths.setdefault(key, location)
            if previous_path != path or (path is None and previous_root != root_name):
                raise ValueError(f"Ambiguous audio location for relative path {key!r}")
    return paths


def extract_embeddings(paths, extract, embedding_type):
    embeddings = {}
    with torch.inference_mode():
        for index, (key, path) in enumerate(paths.items(), start=1):
            if index == 1 or index % 100 == 0 or index == len(paths):
                print(f"Extracting {embedding_type} embeddings: {index}/{len(paths)}", flush=True)
            try:
                embedding = extract(load_audio(path))
                if embedding.ndim != 1 or not embedding.numel() or not embedding.isfinite().all():
                    raise ValueError("Expected a finite, nonempty embedding vector")
            except (ValueError, RuntimeError, OSError) as error:
                raise RuntimeError(
                    f"Could not extract {embedding_type} embedding for {path}: {error}"
                ) from error
            embeddings[key] = embedding
    return embeddings


def load_embeddings(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def save_embeddings(path, embeddings):
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(embeddings, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def extract_and_save(paths, output_dir, args):
    cache_paths = {kind: output_dir / f"{kind}_embeddings.pt" for kind in EMBEDDING_TYPES}
    existing = {kind for kind, path in cache_paths.items() if path.exists()}
    embeddings = {
        kind: load_embeddings(path) if kind in existing else {}
        for kind, path in cache_paths.items()
    }
    missing = {kind: {} for kind in EMBEDDING_TYPES}
    for key, (path, root_name) in paths.items():
        missing_types = [kind for kind in EMBEDDING_TYPES if key not in embeddings[kind]]
        if not missing_types:
            continue
        if path is None:
            raise ValueError(f"Pass {root_name} to extract missing embeddings for {key!r}")
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        for kind in missing_types:
            missing[kind][key] = path

    cache_dir = args.cache_dir.expanduser().resolve()
    cache = cache_dir / "huggingface" / "hub"
    os.environ["HF_HOME"] = str(cache.parent)
    os.environ["HF_HUB_CACHE"] = str(cache)
    os.environ["TORCH_HOME"] = str(cache_dir / "torch")
    os.environ["XDG_CACHE_HOME"] = str(cache_dir / "xdg")
    repo = (args.voxlect_repo or cache_dir / "voxlect-source").expanduser().resolve()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if missing["speaker"]:
        extractor, model = load_speaker_model(device, cache)
        embeddings["speaker"].update(extract_embeddings(
            missing["speaker"],
            lambda wav: speaker_embedding(wav, extractor, model, device),
            "speaker",
        ))
        del extractor, model
        gc.collect()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()

    if missing["speaker"] or "speaker" not in existing:
        save_embeddings(cache_paths["speaker"], embeddings["speaker"])

    if missing["accent"]:
        model = load_accent_model(device, cache, repo)
        embeddings["accent"].update(extract_embeddings(
            missing["accent"], lambda wav: accent_embedding(wav, model, device), "accent"
        ))

    if missing["accent"] or "accent" not in existing:
        save_embeddings(cache_paths["accent"], embeddings["accent"])

    selected_embeddings = {
        kind: {key: vectors[key].to(dtype=torch.float64).numpy() for key in paths}
        for kind, vectors in embeddings.items()
    }
    model_settings = {"cache_dir": str(cache_dir), "voxlect_repo": str(repo), "device": device}
    return selected_embeddings, model_settings


def write_config(path, config):
    contents = json.dumps(config, indent=2) + "\n"
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            temporary.write(contents)
            temporary.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def score_variance(sides, embeddings, output_dir):
    rows = []
    for side in sides:
        row = {**side.metadata, "n_turns": len(side.turns)}
        for kind, vectors in embeddings.items():
            row[f"{kind}_covariance_trace"] = (
                covariance_trace([vectors[turn] for turn in side.turns])
                if len(side.turns) >= 2 else None
            )
        rows.append(row)
    fields = (*META_FIELDS, "n_turns", *(f"{kind}_covariance_trace" for kind in EMBEDDING_TYPES))
    write_csv(output_dir / "variance_by_side.csv", fields, rows)


def unit_vector(vector, key, kind):
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError(f"Invalid norm for {kind} embedding of {key}")
    return vector / norm


def score_distance(sides, embeddings, output_dir, turn_type):
    summaries, pairs = [], []
    turn_field = f"{turn_type}_turn"
    for side in sides:
        summary = {**side.metadata, "n_turns": len(side.turns)}
        for kind, vectors in embeddings.items():
            distances = []
            if side.turns:
                reference = unit_vector(vectors[side.reference], side.reference, kind)
                for turn in side.turns:
                    turn_vector = unit_vector(vectors[turn], turn, kind)
                    distance = 1.0 - float(np.clip(np.dot(reference, turn_vector), -1.0, 1.0))
                    distances.append(distance)
                    pairs.append({
                        **side.metadata,
                        "ref_speech": side.reference,
                        turn_field: turn,
                        "embedding_type": kind,
                        "cosine_distance": distance,
                    })
            summary[f"{kind}_mean_cosine_distance"] = (
                float(np.mean(distances)) if distances else None
            )
        summaries.append(summary)

    summary_fields = (
        *META_FIELDS,
        "n_turns",
        *(f"{kind}_mean_cosine_distance" for kind in EMBEDDING_TYPES),
    )
    write_csv(output_dir / "distance_by_side.csv", summary_fields, summaries)
    pairs.sort(key=lambda row: row["cosine_distance"], reverse=True)
    pair_fields = (*META_FIELDS, "ref_speech", turn_field, "embedding_type", "cosine_distance")
    write_csv(output_dir / "ref_turn_distances.csv", pair_fields, pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", choices=("all", "filtered"), help="Turn column to use")
    parser.add_argument("--ref-root", type=Path, help="Root for reference audio")
    parser.add_argument("--gen-root", type=Path, help="Root for ground-truth or generated turn audio")
    parser.add_argument("--csv", type=Path, required=True,
                        help="Path to ref_gt_turns.csv or ref_tgt_turns.csv")
    parser.add_argument("--exp-name", help="Experiment folder name under the repository")
    parser.add_argument("--output-dir", type=Path,
                        help="Override the experiment output directory")
    parser.add_argument("--mode", choices=("both", "variance", "distance"), default="both",
                        help="Which scores to report (default: both)")
    parser.add_argument("--device", help="Extraction device (default: CUDA if available, otherwise CPU)")
    parser.add_argument("--cache-dir", type=Path, default=os.environ.get("CACHE_DIR") or DEFAULT_CACHE_DIR)
    parser.add_argument("--voxlect-repo", type=Path, default=os.environ.get("VOXLECT_REPO"))
    args = parser.parse_args()

    csv_path = args.csv.expanduser().resolve()
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser()
    else:
        exp_name = args.exp_name
        if not exp_name:
            parser.error("Pass --exp-name or --output-dir")
        if exp_name in (".", "..") or Path(exp_name).name != exp_name:
            parser.error("--exp-name must be a single directory name")
        output_dir = REPO_DIR / exp_name
    run_variance = args.mode in ("both", "variance")
    run_distance = args.mode in ("both", "distance")
    sides, turn_type = read_sides(csv_path, args.selection)
    ref_root = args.ref_root.expanduser().resolve() if args.ref_root is not None else None
    turn_root = args.gen_root.expanduser().resolve() if args.gen_root is not None else None
    paths = collect_audio_paths(sides, ref_root, turn_root, run_distance)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    run_settings = {
        "csv_path": str(csv_path),
        "selection": args.selection,
        "turn_type": turn_type,
        "mode": args.mode,
    }
    # Keep the cache, result CSVs, and their settings in sync across jobs.
    with (output_dir / ".embeddings.lock").open("a") as lock:
        flock(lock, LOCK_EX)
        config = (
            json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        )
        config.update(exp_name=args.exp_name, output_dir=str(output_dir.resolve()))
        embeddings, model_settings = extract_and_save(paths, output_dir, args)
        previous_extraction = config.get("extraction", {})
        config["extraction"] = {
            **run_settings,
            "ref_root": str(ref_root) if ref_root is not None else previous_extraction.get("ref_root"),
            "gen_root": str(turn_root) if turn_root is not None else previous_extraction.get("gen_root"),
            **model_settings,
        }
        write_config(config_path, config)
        scoring = config.setdefault("scoring", {})
        if run_variance:
            score_variance(sides, embeddings, output_dir)
            scoring["variance"] = run_settings
            write_config(config_path, config)
        if run_distance:
            score_distance(sides, embeddings, output_dir, turn_type)
            scoring["distance"] = run_settings
            write_config(config_path, config)
    print(f"Output directory: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
