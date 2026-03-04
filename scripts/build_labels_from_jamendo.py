#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_POSITIVE_KEYWORDS = [
    "genre---emo",
    "genre---screamo",
    "genre---emocore",
    "emo",
    "screamo",
    "emocore",
    "emotional hardcore",
    "midwest emo",
]

DEFAULT_NEGATIVE_HINT_KEYWORDS = [
    "genre---rock",
    "genre---alternativerock",
    "genre---indierock",
    "genre---punkrock",
    "genre---poprock",
    "alternative rock",
    "indie rock",
    "punk rock",
    "pop punk",
]


@dataclass
class SampleRow:
    path: str
    label: str
    source: str
    tags: str


def _detect_column(df: pd.DataFrame, candidates: list[str]) -> str:
    lower_to_real = {str(col).strip().lower(): str(col) for col in df.columns}
    for candidate in candidates:
        if candidate in lower_to_real:
            return lower_to_real[candidate]
    for real_lower, real_name in lower_to_real.items():
        for candidate in candidates:
            if candidate in real_lower:
                return real_name
    raise KeyError(f"Could not detect any of columns: {candidates}")


def _parse_tags(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []

    if isinstance(value, (list, tuple, set)):
        tokens = list(value)
    else:
        text = str(value).strip()
        if text == "":
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                tokens = list(parsed)
            else:
                tokens = [text]
        except Exception:
            tokens = re.split(r"[;,|]", text)

    out: list[str] = []
    for token in tokens:
        cleaned = str(token).strip().lower()
        if cleaned:
            out.append(cleaned)
    return out


def _matches_any(tags: list[str], keywords: list[str]) -> bool:
    return any(keyword in tag for tag in tags for keyword in keywords)


def build_dataset(
    labels_path: Path,
    audio_root: Path | None,
    output_csv: Path,
    positive_keywords: list[str],
    negative_keywords: list[str],
    negative_mode: str,
    max_positive: int | None,
    max_negative: int | None,
    require_existing_audio: bool,
    seed: int,
) -> None:
    def _load_labels_file(path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".tsv":
            # Jamendo TSV stores first 5 columns as fixed fields and then variable tag columns.
            rows: list[dict[str, str]] = []
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                header = handle.readline().rstrip("\r\n").split("\t")
                if len(header) >= 6 and header[0].upper() == "TRACK_ID":
                    for line in handle:
                        parts = line.rstrip("\r\n").split("\t")
                        if len(parts) < 6:
                            continue
                        rows.append(
                            {
                                "TRACK_ID": parts[0],
                                "ARTIST_ID": parts[1],
                                "ALBUM_ID": parts[2],
                                "PATH": parts[3],
                                "DURATION": parts[4],
                                "TAGS": "|".join(parts[5:]),
                            }
                        )
                    return pd.DataFrame(rows)

        # Generic fallback for already-normalized CSV/TSV files.
        try:
            return pd.read_csv(path, sep="\t", low_memory=False)
        except Exception:
            return pd.read_csv(path, low_memory=False)

    df = _load_labels_file(labels_path)

    path_col = _detect_column(df, ["path", "audio_path", "filepath", "file_path"])
    tags_col = _detect_column(df, ["tags", "tag", "labels", "label"])

    positives: list[SampleRow] = []
    negatives_hint: list[SampleRow] = []
    negatives_all: list[SampleRow] = []

    for _, row in df.iterrows():
        raw_path = str(row[path_col]).strip()
        if raw_path == "":
            continue

        path = Path(raw_path).expanduser()
        if not path.is_absolute() and audio_root is not None:
            path = (audio_root / path).resolve()
        else:
            path = path.resolve()

        if require_existing_audio and not path.exists():
            continue

        tags = _parse_tags(row[tags_col])
        if not tags:
            continue

        entry = SampleRow(
            path=str(path),
            label="non_emo",
            source="jamendo",
            tags="|".join(sorted(set(tags))),
        )

        if _matches_any(tags, positive_keywords):
            entry.label = "emo"
            positives.append(entry)
            continue

        negatives_all.append(entry)
        if _matches_any(tags, negative_keywords):
            negatives_hint.append(entry)

    rng = random.Random(seed)
    if max_positive is not None and len(positives) > max_positive:
        positives = rng.sample(positives, max_positive)

    if negative_mode == "hints":
        negative_pool = negatives_hint if negatives_hint else negatives_all
    else:
        negative_pool = negatives_all

    if max_negative is None:
        max_negative = len(positives)
    if max_negative is not None and len(negative_pool) > max_negative:
        negative_pool = rng.sample(negative_pool, max_negative)

    rows = positives + negative_pool
    rng.shuffle(rows)
    out_df = pd.DataFrame(
        [{"path": item.path, "label": item.label, "source": item.source, "tags": item.tags} for item in rows]
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print(f"wrote={output_csv}")
    print(f"positives={len(positives)} negatives={len(negative_pool)} total={len(rows)}")
    print(f"negative_mode={negative_mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build emo/non-emo labels from Jamendo metadata.")
    parser.add_argument("--labels-tsv", required=True, help="Jamendo labels TSV/CSV path.")
    parser.add_argument(
        "--audio-root",
        default=None,
        help="Optional audio root directory for relative paths in labels file.",
    )
    parser.add_argument("--out-csv", required=True, help="Output labels CSV path.")
    parser.add_argument(
        "--positive-keywords",
        default=",".join(DEFAULT_POSITIVE_KEYWORDS),
        help="Comma-separated keywords that define emo-positive tags.",
    )
    parser.add_argument(
        "--negative-keywords",
        default=",".join(DEFAULT_NEGATIVE_HINT_KEYWORDS),
        help="Comma-separated keywords for hard-negative tag hints.",
    )
    parser.add_argument(
        "--negative-mode",
        choices=("hints", "all"),
        default="hints",
        help="Use hard negatives only ('hints') or all non-emo tracks ('all').",
    )
    parser.add_argument(
        "--max-positive",
        type=int,
        default=2_000,
        help="Maximum number of emo tracks to include.",
    )
    parser.add_argument(
        "--max-negative",
        type=int,
        default=None,
        help="Maximum non-emo tracks. Default = match positives.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and shuffling.",
    )
    parser.add_argument(
        "--allow-missing-audio",
        action="store_true",
        help="Allow rows even if audio files are not present on disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positive = [token.strip().lower() for token in args.positive_keywords.split(",") if token.strip()]
    negative = [token.strip().lower() for token in args.negative_keywords.split(",") if token.strip()]
    if not positive:
        raise ValueError("positive-keywords must contain at least one keyword")

    audio_root = Path(args.audio_root).expanduser().resolve() if args.audio_root else None

    build_dataset(
        labels_path=Path(args.labels_tsv).expanduser().resolve(),
        audio_root=audio_root,
        output_csv=Path(args.out_csv).expanduser().resolve(),
        positive_keywords=positive,
        negative_keywords=negative,
        negative_mode=args.negative_mode,
        max_positive=args.max_positive,
        max_negative=args.max_negative,
        require_existing_audio=not args.allow_missing_audio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
