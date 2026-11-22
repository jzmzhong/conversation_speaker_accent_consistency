"""Measure within-side speaker and accent variance across conversation utterances.

Run with the evaluation environment's Python, for example:
    python simple_eval.py [conversation_folder] [--device cpu]

VoxLect requires its upstream source checkout, specified by --voxlect-repo or
VOXLECT_REPO. Model weights are downloaded into --model-dir if not cached.
Speaker embeddings are L2-normalized; accent embeddings are not. Variances are
traces of sample covariance matrices (denominator n - 1).
"""

import argparse
import gc
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torchaudio


CONV_FOLDER = Path(
    "/home/s2526235/projects/202_JSALT_2026/results/Personaplex/02_hf/"
    "01_IViE_ref-retold/Belfast_f3_1_Belfast_f1_0"
)
MODEL_DIR = Path(__file__).resolve().parent / "cache"
SAMPLE_RATE = 16000
VOXLECT_CHUNK_SAMPLES = 15 * SAMPLE_RATE


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
    return (embedding / norm).cpu().numpy()


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
            raise ValueError("VoxLect cannot encode a final audio chunk shorter than 10 ms")
        features = model.feature_extractor(
            [chunk.numpy()], return_tensors="pt", sampling_rate=SAMPLE_RATE,
            max_length=VOXLECT_CHUNK_SAMPLES,
        ).input_features.to(device)
        hidden = model.backbone_model.encoder(features, output_hidden_states=True).hidden_states[-1]
        frames = model.model_seq(hidden.transpose(1, 2)).transpose(1, 2)[0, :length]
        chunk_sum = frames.sum(dim=0)
        frame_sum = chunk_sum if frame_sum is None else frame_sum + chunk_sum
        total_frames += frames.shape[0]
    return (frame_sum / total_frames).cpu().numpy()


def covariance_trace(embeddings):
    """Trace of sample covariance without materializing the covariance matrix."""
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError("At least two nonempty embeddings are required")
    if not np.isfinite(values).all():
        raise ValueError("Embeddings contain non-finite values")
    centered = values - values.mean(axis=0)
    return float(np.sum(centered * centered) / (len(values) - 1))


def evaluate_sides(paths, extract):
    results = {}
    with torch.inference_mode():
        for side, files in paths.items():
            embeddings = []
            for path in files:
                print(f"  {path.name}", flush=True)
                embeddings.append(extract(load_audio(path)))
            results[side] = covariance_trace(embeddings)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", type=Path, default=CONV_FOLDER)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--voxlect-repo", type=Path, default=os.environ.get("VOXLECT_REPO"))
    args = parser.parse_args()

    paths = {side: sorted(args.folder.glob(f"{side}_*.wav")) for side in ("a", "b")}
    for side, files in paths.items():
        if len(files) < 2:
            parser.error(f"Need at least two {side}_*.wav files in {args.folder}; found {len(files)}")

    model_dir = args.model_dir.expanduser().resolve()
    repo = args.voxlect_repo or model_dir / "voxlect-source"
    cache = model_dir / "huggingface" / "hub"
    os.environ["HF_HOME"] = str(model_dir / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache)

    print("Extracting WavLM speaker embeddings...", flush=True)
    extractor, model = load_speaker_model(args.device, cache)
    speaker = evaluate_sides(paths, lambda wav: speaker_embedding(wav, extractor, model, args.device))
    del extractor, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Extracting VoxLect accent embeddings...", flush=True)
    model = load_accent_model(args.device, cache, repo)
    accent = evaluate_sides(paths, lambda wav: accent_embedding(wav, model, args.device))

    print("\nSide  N     Speaker variance    Accent variance")
    for side, files in paths.items():
        print(f"{side:4}  {len(files):3}  {speaker[side]:18.8f}  {accent[side]:17.8f}")


if __name__ == "__main__":
    main()
