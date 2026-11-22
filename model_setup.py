"""Prepare the local Python environment's model cache for simple_eval.py."""

import argparse
import os
from pathlib import Path
import subprocess

from huggingface_hub import list_repo_files, snapshot_download


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "cache"
VOXLECT_REPOSITORY = "https://github.com/tiantiaf0627/voxlect.git"
HF_REPOSITORIES = (
    "microsoft/wavlm-base-sv",
    "tiantiaf/voxlect-english-dialect-whisper-large-v3",
    "openai/whisper-large-v3",
    "openai/whisper-tiny",
)


def configure_caches(cache_dir):
    """Keep Hugging Face downloads and related caches inside the model directory."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "huggingface" / "hub")
    os.environ["TORCH_HOME"] = str(cache_dir / "torch")
    os.environ["XDG_CACHE_HOME"] = str(cache_dir / "xdg")


def download_snapshot(repo_id, cache_dir):
    """Download model configs and weights, preferring safetensors when available."""
    files = list_repo_files(repo_id)
    patterns = ["*.json", "*.txt", "*.yaml", "*.yml", "*.py", "*.model"]
    patterns.append(
        "*.safetensors"
        if any(filename.endswith(".safetensors") for filename in files)
        else "*.bin"
    )
    return Path(snapshot_download(
        repo_id,
        cache_dir=str(cache_dir),
        allow_patterns=patterns,
    ))


def checkout_voxlect(source_dir):
    source_dir = Path(source_dir)
    source_file = source_dir / "src/model/dialect/whisper_dialect.py"
    if source_file.is_file():
        return source_dir
    if source_dir.exists():
        raise FileNotFoundError(f"Invalid VoxLect source checkout: {source_dir}")

    source_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", VOXLECT_REPOSITORY, str(source_dir)],
        check=True,
    )
    if not source_file.is_file():
        raise FileNotFoundError(f"Invalid VoxLect source checkout: {source_dir}")
    return source_dir


def setup_models(cache_dir):
    cache_dir = Path(cache_dir).expanduser().resolve()
    configure_caches(cache_dir)
    hub_dir = cache_dir / "huggingface" / "hub"

    for repo_id in HF_REPOSITORIES:
        print(f"Downloading {repo_id}", flush=True)
        download_snapshot(repo_id, hub_dir)

    source_dir = checkout_voxlect(cache_dir / "voxlect-source")
    print(f"VoxLect source ready at {source_dir}")
    print(f"Models ready in {hub_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory for Hugging Face assets and VoxLect source (default: ./cache).",
    )
    args = parser.parse_args()
    setup_models(args.cache_dir)


if __name__ == "__main__":
    main()
