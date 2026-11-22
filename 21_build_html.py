"""Build a listening demo of the highest conversation-side and turn scores."""

import argparse
import csv
import html
import math
from pathlib import Path
import random
import shutil
from urllib.parse import quote


KEY_FIELDS = ("accent", "task", "conversation_id", "channel_id", "speaker_id")
TABLES = (
    ("Highest speaker embedding distance (by conversation side)", "distance_by_side.csv", "speaker_mean_cosine_distance", None),
    ("Highest accent embedding distance (by conversation side)", "distance_by_side.csv", "accent_mean_cosine_distance", None),
    ("Highest speaker embedding variance (by conversation side)", "variance_by_side.csv", "speaker_covariance_trace", None),
    ("Highest accent embedding variance (by conversation side)", "variance_by_side.csv", "accent_covariance_trace", None),
    ("Highest speaker embedding distance (by turn)", "ref_turn_distances.csv", "cosine_distance", "speaker"),
    ("Highest accent embedding distance (by turn)", "ref_turn_distances.csv", "cosine_distance", "accent"),
)


def read_csv(path, required):
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        rows = []
        for row in reader:
            if any(row[field] is None or not row[field].strip() for field in required):
                raise ValueError(f"{path}:{reader.line_num}: missing required value")
            rows.append(row)
    return rows


def side_key(row):
    return tuple(row[field] for field in KEY_FIELDS)


def audio_path(root, relative):
    path = Path(relative)
    if not relative or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a relative audio path: {relative!r}")
    root = root.resolve()
    source = (root / path).resolve()
    if not source.is_relative_to(root) or not source.is_file():
        raise FileNotFoundError(f"Audio file missing or outside {root}: {relative}")
    return source


def ranked_rows(rows, column, kind, path, limit):
    ranked = []
    for row in rows:
        if kind is not None and row["embedding_type"] != kind:
            continue
        raw = row[column]
        if not raw:
            continue
        try:
            score = float(raw)
        except ValueError as exc:
            raise ValueError(f"{path}: invalid {column}: {raw!r}") from exc
        if not math.isfinite(score):
            raise ValueError(f"{path}: non-finite {column}: {raw!r}")
        ranked.append((score, row))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[:limit]


def build_tables(score_dir, metadata_csv, ref_root, gen_root, limit):
    metadata = {}
    for row in read_csv(metadata_csv, (*KEY_FIELDS, "ref_speech")):
        key = side_key(row)
        if key in metadata:
            raise ValueError(f"{metadata_csv}: duplicate conversation side {key}")
        metadata[key] = row

    tables = []
    for title, filename, column, kind in TABLES:
        path = score_dir / filename
        required = (*KEY_FIELDS, column)
        if kind:
            required += ("ref_speech", "gen_turn", "embedding_type")
        # Empty scores are valid for sides with too few turns.
        score_rows = read_csv(path, [field for field in required if field != column])
        if not score_rows:
            raise ValueError(f"{path}: no score rows")
        if column not in score_rows[0]:
            raise ValueError(f"{path} is missing column: {column}")
        ranked = ranked_rows(score_rows, column, kind, path, limit)
        if not ranked:
            raise ValueError(f"{path}: no values for {title}")
        if len(ranked) < limit:
            print(f"Warning: {title} has only {len(ranked)} scored rows (requested {limit})")
        entries = []
        for score, row in ranked:
            key = side_key(row)
            if key not in metadata:
                raise ValueError(f"{path}: no metadata for conversation side {key}")
            reference = metadata[key]["ref_speech"]
            if kind:
                if row["ref_speech"] != reference:
                    raise ValueError(f"{path}: reference differs from metadata for {key}")
                generated = row["gen_turn"]
            else:
                channel = row["channel_id"]
                if channel not in ("a", "b"):
                    raise ValueError(f"{path}: expected channel a or b, got {channel!r}")
                generated = f"{row['conversation_id']}/{channel}.wav"
            entries.append({
                "accent": row["accent"],
                "conversation_id": row["conversation_id"],
                "channel": row["channel_id"],
                "score": score,
                "reference": reference,
                "generated": generated,
                "ref_source": audio_path(ref_root, reference),
                "gen_source": audio_path(gen_root, generated),
            })
            if not kind:
                entries[-1]["conversation"] = f"{row['conversation_id']}/ab.wav"
                entries[-1]["conversation_source"] = audio_path(gen_root, entries[-1]["conversation"])
        tables.append((title, kind, entries))
    return tables


def sample_conversations(metadata_csv, gt_metadata_csv, ref_root, gen_root, gt_root, seed):
    generated = {}
    for row in read_csv(metadata_csv, (*KEY_FIELDS, "ref_speech")):
        key = (row["accent"], row["task"], row["conversation_id"])
        sides = generated.setdefault(key, {})
        if row["channel_id"] not in ("a", "b") or row["channel_id"] in sides:
            raise ValueError(f"{metadata_csv}: invalid or duplicate generated channel for {key}")
        sides[row["channel_id"]] = row

    gt_groups = {}
    for row in read_csv(gt_metadata_csv, (*KEY_FIELDS, "ref_speech")):
        key = (row["accent"], row["task"], row["conversation_id"])
        sides = gt_groups.setdefault(key, {})
        if row["channel_id"] not in ("ch1", "ch2") or row["channel_id"] in sides:
            raise ValueError(f"{gt_metadata_csv}: invalid or duplicate GT channel for {key}")
        sides[row["channel_id"]] = row
    gt_pairs = {}
    for (accent, task, conversation), sides in gt_groups.items():
        if set(sides) != {"ch1", "ch2"}:
            raise ValueError(f"{gt_metadata_csv}: expected two channels for {conversation}")
        speakers = tuple(sorted(row["speaker_id"] for row in sides.values()))
        if speakers[0] == speakers[1]:
            raise ValueError(f"{gt_metadata_csv}: duplicate speakers for {conversation}")
        key = (accent, task, speakers)
        if key in gt_pairs:
            raise ValueError(f"{gt_metadata_csv}: ambiguous GT speaker pair {key}")
        gt_pairs[key] = (conversation, sides)

    candidates = []
    for (accent, task, conversation), sides in sorted(generated.items()):
        if set(sides) != {"a", "b"} or not task.endswith("_synthetic"):
            raise ValueError(f"{metadata_csv}: expected two synthetic channels for {conversation}")
        speakers = tuple(sorted(row["speaker_id"] for row in sides.values()))
        if speakers[0] == speakers[1]:
            raise ValueError(f"{metadata_csv}: duplicate speakers for {conversation}")
        key = (accent, task.removesuffix("_synthetic"), speakers)
        if key not in gt_pairs:
            raise ValueError(f"{metadata_csv}: no GT conversation for {key}")
        gt_conversation, gt_sides = gt_pairs[key]
        gt_by_speaker = {row["speaker_id"]: row for row in gt_sides.values()}
        for side in sides.values():
            if side["ref_speech"] != gt_by_speaker[side["speaker_id"]]["ref_speech"]:
                raise ValueError(f"{conversation}: generated and GT references do not match")
        candidates.append((accent, task.removesuffix("_synthetic"), conversation, gt_conversation, sides))
    if not candidates:
        raise ValueError(f"{metadata_csv}: no generated conversations")
    if len(candidates) < 10:
        print(f"Warning: only {len(candidates)} conversations available for sampling")
    sampled = []
    for accent, task, conversation, gt_conversation, sides in random.Random(seed).sample(
        candidates, min(10, len(candidates))
    ):
        real = f"{accent}/{task}/{accent}-{gt_conversation}.wav"
        synthetic = f"{conversation}/ab.wav"
        ref_a, ref_b = sides["a"]["ref_speech"], sides["b"]["ref_speech"]
        sampled.append({
            "accent": accent, "gt_conversation": gt_conversation, "conversation": conversation,
            "reference_a": ref_a, "reference_b": ref_b, "real": real, "synthetic": synthetic,
            "reference_a_source": audio_path(ref_root, ref_a),
            "reference_b_source": audio_path(ref_root, ref_b),
            "real_source": audio_path(gt_root, real),
            "synthetic_source": audio_path(gen_root, synthetic),
        })
    return sampled


def copy_audio(output, category, relative, source):
    destination = output / "audio" / category / relative
    if not destination.resolve().is_relative_to(output.resolve()):
        raise ValueError(f"Audio destination is outside output directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source != destination.resolve():
        shutil.copy2(source, destination)
    return quote(destination.relative_to(output).as_posix(), safe="/")


def audio_cell(url, name):
    return (f'<audio controls preload="none" src="{html.escape(url, quote=True)}"></audio>'
            f'<div class="filename">{html.escape(name)}</div>')


def render_html(output, tables, sampled, seed):
    sections = []
    copied = {}

    def cell(category, relative, source):
        key = (category, relative)
        if key not in copied:
            copied[key] = copy_audio(output, category, relative, source)
        return audio_cell(copied[key], relative)

    for title, kind, entries in tables:
        rows = []
        for rank, entry in enumerate(entries, start=1):
            reference = cell("reference", entry["reference"], entry["ref_source"])
            generated = cell("generated", entry["generated"], entry["gen_source"])
            conversation = (f"<td>{cell('generated', entry['conversation'], entry['conversation_source'])}</td>"
                            if not kind else "")
            rows.append(
                f"<tr><td>{rank}</td><td>{html.escape(entry['accent'])}</td>"
                f"<td>{html.escape(entry['conversation_id'])}</td>"
                f"<td>{html.escape(entry['channel'])}</td><td>{entry['score']:.6f}</td>"
                f"<td>{reference}</td><td>{generated}</td>{conversation}</tr>"
            )
        generated_header = "gen_turn" if kind else "gen_side"
        conversation_header = "<th>gen_conversation</th>" if not kind else ""
        sections.append(
            f"<section><h2>{html.escape(title)}</h2><div class=\"table-wrap\"><table>"
            f"<thead><tr><th>Rank</th><th>Accent</th><th>Conversation ID</th>"
            f"<th>Channel</th><th>Score</th><th>ref_speech</th><th>{generated_header}</th>{conversation_header}"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>"
        )
    sample_rows = []
    for entry in sampled:
        sample_rows.append(
            f"<tr><td>{html.escape(entry['accent'])}</td><td>{html.escape(entry['gt_conversation'])}</td>"
            f"<td>{html.escape(entry['conversation'])}</td>"
            f"<td>{cell('reference', entry['reference_a'], entry['reference_a_source'])}</td>"
            f"<td>{cell('reference', entry['reference_b'], entry['reference_b_source'])}</td>"
            f"<td>{cell('real', entry['real'], entry['real_source'])}</td>"
            f"<td>{cell('generated', entry['synthetic'], entry['synthetic_source'])}</td></tr>"
        )
    sections.append(
        f"<section><h2>Randomly sampled conversations (seed {seed})</h2>"
        '<div class="table-wrap"><table><thead><tr><th>Accent</th><th>GT conversation ID</th>'
        '<th>Generated conversation ID</th><th>ref_a</th><th>ref_b</th>'
        '<th>real_conversation</th><th>gen_conversation</th></tr></thead>'
        f"<tbody>{''.join(sample_rows)}</tbody></table></div></section>"
    )
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>IViE PersonaPlex-PersonaPlex Generation: Failure Case Analysis</title>
<style>
body { font-family: sans-serif; margin: 2rem auto; max-width: 1500px; padding: 0 1rem; color: #222; }
h1, h2 { color: #263b55; }
section { margin: 2rem 0; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
th, td { padding: .6rem; border-bottom: 1px solid #ddd; text-align: left; vertical-align: top; }
th { background: #edf2f7; }
audio { width: 240px; max-width: 100%; }
.filename { font-size: .75rem; overflow-wrap: anywhere; color: #555; }
</style></head><body>
<h1>IViE PersonaPlex-PersonaPlex Generation: Failure Case Analysis</h1>
<p>Conversation scores are per channel/side. Variance is the covariance trace across turns;
conversation distance is mean cosine distance of turns to the reference. Turn distance is the cosine distance of a turn to the the reference.</p>
<p>Sampled real conversations use the DialogueSidon-separated stereo recordings.
ref_a and ref_b are the voice prompts for the corresponding generated conversation.</p>
""" + "\n".join(sections) + "\n</body></html>\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", type=Path, required=True, help="Experiment score directory")
    parser.add_argument("--metadata-csv", type=Path, required=True, help="Conversation-side reference CSV")
    parser.add_argument("--gt-metadata-csv", type=Path, required=True, help="Matching GT conversation-side CSV")
    parser.add_argument("--ref-root", type=Path, required=True, help="Root of reference WAV files")
    parser.add_argument("--gen-root", type=Path, required=True, help="Root of generated conversation and turn WAV files")
    parser.add_argument("--gt-conversation-root", type=Path, required=True, help="Root of stereo GT conversation WAV files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for index.html and copied audio")
    parser.add_argument("--top", type=int, default=10, help="Maximum rows per table (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling conversations (default: 42)")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    tables = build_tables(args.score_dir, args.metadata_csv, args.ref_root, args.gen_root, args.top)
    sampled = sample_conversations(args.metadata_csv, args.gt_metadata_csv, args.ref_root,
                                   args.gen_root, args.gt_conversation_root, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    page = render_html(args.output_dir, tables, sampled, args.seed)
    destination = args.output_dir / "index.html"
    destination.write_text(page, encoding="utf-8")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
