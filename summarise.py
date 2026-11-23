"""Summarise and plot conversation-side or turn speaker and accent consistency scores."""

import argparse
import csv
import math
from pathlib import Path
import textwrap
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
METRICS = {
    "speaker_variance": ("variance_by_side.csv", "speaker_covariance_trace", "Speaker variance"),
    "accent_variance": ("variance_by_side.csv", "accent_covariance_trace", "Accent variance"),
    "speaker_distance": ("distance_by_side.csv", "speaker_mean_cosine_distance", "Speaker cosine distance"),
    "accent_distance": ("distance_by_side.csv", "accent_mean_cosine_distance", "Accent cosine distance"),
}
COLORS = ("#87909e", "#6d83bb", "#dc8b6a", "#75a895", "#b698c4")
SCORE_FILES = {
    "conversation-side": {
        "variance_by_side.csv": "variance_by_side.csv",
        "distance_by_side.csv": "distance_by_side.csv",
    },
    "turn": {
        "variance_by_side.csv": "variance_by_turn.csv",
        "distance_by_side.csv": "distance_by_turn.csv",
    },
}


def read_scores(experiment, unit="conversation-side"):
    scores = {metric: [] for metric in METRICS}
    for filename in dict.fromkeys(spec[0] for spec in METRICS.values()):
        path = experiment / SCORE_FILES[unit][filename]
        if not path.is_file():
            warnings.warn(f"Missing {path}; its metrics will be omitted")
            continue
        columns = {metric: spec[1] for metric, spec in METRICS.items() if spec[0] == filename}
        with path.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            missing = set(columns.values()) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
            for line, row in enumerate(reader, start=2):
                for metric, column in columns.items():
                    raw = row[column]
                    if raw is None:
                        raise ValueError(f"{path}:{line}: missing field {column}")
                    if not raw.strip():
                        continue
                    try:
                        value = float(raw)
                    except ValueError as exc:
                        raise ValueError(f"{path}:{line}: invalid {column}: {raw!r}") from exc
                    if not math.isfinite(value):
                        raise ValueError(f"{path}:{line}: non-finite {column}: {raw!r}")
                    scores[metric].append(value)
    for metric, values in scores.items():
        if not values:
            warnings.warn(f"{experiment}: no values for {metric}")
    return scores


def statistics(values):
    if not values:
        return 0, None, None
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None
    return len(values), mean, sem


def write_summary(path, scores, unit="conversation-side"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        count_column = "n_turns" if unit == "turn" else "n_conversation_sides"
        writer.writerow(("metric", count_column, "mean", "standard_error"))
        for metric, values in scores.items():
            writer.writerow((metric, *statistics(values)))


def plot_metric(path, metric, experiments, labels=None):
    available = [(name, scores[metric]) for name, scores in experiments if scores[metric]]
    if not available:
        warnings.warn(f"No observations for {metric}; no plot written")
        return
    fig, ax = plt.subplots(figsize=(max(6, len(available) * 2.5), 6), layout="constrained")
    ax.set_facecolor("#f5f6f8")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    rng = np.random.default_rng(0)
    for position, (name, values) in enumerate(available, start=1):
        color = COLORS[(position - 1) % len(COLORS)]
        if len(values) > 1 and np.ptp(values) > 0:
            violin = ax.violinplot(values, positions=[position], widths=0.7,
                                   showextrema=False)
            for body in violin["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.65)
        ax.scatter(position + rng.uniform(-0.15, 0.15, len(values)), values,
                   color=color, alpha=0.45, s=12, zorder=2)
        ax.boxplot(values, positions=[position], widths=0.22, showfliers=False,
                   patch_artist=True, boxprops={"facecolor": "#f0f0f0"},
                   medianprops={"color": "#333333"})
        _, mean, sem = statistics(values)
        ax.errorbar(position, mean, yerr=sem, fmt="o", color="#a00000", capsize=5, zorder=4)
        label = f"{mean:.3f}" + (f" ± {sem:.3f}" if sem is not None else " (n=1)")
        ax.text(position, -0.025, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, fontweight="bold")
    labels = labels or {}
    tick_labels = [textwrap.fill(labels.get(name, name.replace("_", " ")), width=25)
                   for name, _ in available]
    ax.set_xticks(range(1, len(available) + 1), tick_labels)
    ax.tick_params(axis="x", pad=28)
    ax.set_ylabel(METRICS[metric][2])
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="+", type=Path,
                        help="Experiment directories to summarise")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "summary",
                        help="Output directory for per-experiment CSVs and comparison plots")
    parser.add_argument("--label", action="append", default=[], metavar="EXPERIMENT=LABEL",
                        help="Override an experiment folder's plot label (repeatable)")
    parser.add_argument("--unit", choices=tuple(SCORE_FILES), default="conversation-side",
                        help="Unit of analysis and input CSV type (default: conversation-side)")
    parser.add_argument("--require-complete", action="store_true",
                        help="Require observations for every metric in every experiment")
    args = parser.parse_intermixed_args()
    paths = args.experiments
    if any(not path.is_dir() for path in paths):
        parser.error("Every experiment must be an existing directory")
    if len({path.name for path in paths}) != len(paths):
        parser.error("Experiment directory names must be unique")
    labels = {}
    for item in args.label:
        name, separator, label = item.partition("=")
        if not separator or not name.strip() or not label.strip():
            parser.error("--label must be EXPERIMENT=LABEL with nonempty values")
        if name not in {path.name for path in paths}:
            parser.error(f"--label refers to an unknown experiment: {name}")
        labels[name] = label
    experiments = [(path.name, read_scores(path, args.unit)) for path in paths]
    if args.require_complete:
        missing = [f"{name}: {metric}" for name, scores in experiments
                   for metric, values in scores.items() if not values]
        if missing:
            parser.error(f"Missing observations for {', '.join(missing)}")
    if not any(values for _, scores in experiments for values in scores.values()):
        parser.error("No scores found in the selected experiments")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, scores in experiments:
        output = args.output_dir / name / "summary.csv"
        write_summary(output, scores, args.unit)
        print(f"Wrote {output}")
    for metric in METRICS:
        output = args.output_dir / f"{metric}_distribution.png"
        plot_metric(output, metric, experiments, labels)
        if any(scores[metric] for _, scores in experiments):
            print(f"Wrote {output}")


if __name__ == "__main__":
    main()
