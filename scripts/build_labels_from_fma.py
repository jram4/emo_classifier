#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_POSITIVE_KEYWORDS = [
    "emo",
    "screamo",
    "emocore",
    "emotional hardcore",
    "midwest emo",
]

DEFAULT_NEGATIVE_HINT_KEYWORDS = [
    "indie rock",
    "alternative rock",
    "alt rock",
    "punk rock",
    "pop punk",
    "post-hardcore",
    "post hardcore",
]


@dataclass
class TrackRow:
    path: str
    label: str
    source: str
    track_id: int
    genres: str


def _find_column(columns: pd.Index | pd.MultiIndex, target: str) -> str | tuple[str, str]:
    if isinstance(columns, pd.MultiIndex):
        for col in columns:
            if len(col) > 1 and str(col[1]).strip().lower() == target:
                return col
            if target in str(col).strip().lower():
                return col
    else:
        for col in columns:
            if str(col).strip().lower() == target:
                return col
            if target in str(col).strip().lower():
                return col
    raise KeyError(f"Unable to find column matching {target!r}")


def _load_tracks(tracks_csv: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(tracks_csv, header=[0, 1], index_col=0, low_memory=False)
    except Exception:
        return pd.read_csv(tracks_csv, low_memory=False)


def _load_genres(genres_csv: Path) -> dict[int, str]:
    df = pd.read_csv(genres_csv, low_memory=False)
    id_col = None
    for candidate in ("genre_id", "id"):
        if candidate in df.columns:
            id_col = candidate
            break
    title_col = None
    for candidate in ("title", "name"):
        if candidate in df.columns:
            title_col = candidate
            break
    if title_col is None:
        raise ValueError("Could not find a genre title column in genres.csv")

    if id_col is not None:
        ids = pd.to_numeric(df[id_col], errors="coerce")
    else:
        ids = pd.to_numeric(df.index, errors="coerce")

    out: dict[int, str] = {}
    for gid, title in zip(ids, df[title_col], strict=False):
        if pd.isna(gid) or pd.isna(title):
            continue
        out[int(gid)] = str(title)
    return out


def _parse_genre_ids(value: object) -> list[int]:
    if value is None or pd.isna(value):
        return []

    if isinstance(value, (list, tuple, set)):
        raw = value
    else:
        text = str(value).strip()
        if text == "" or text.lower() == "nan":
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                raw = parsed
            else:
                raw = text.replace(";", ",").split(",")
        except Exception:
            raw = text.replace(";", ",").split(",")

    out: list[int] = []
    for item in raw:
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def _audio_path(audio_root: Path, track_id: int) -> Path:
    tid = f"{track_id:06d}"
    return audio_root / tid[:3] / f"{tid}.mp3"


def _matches_any(texts: list[str], keywords: list[str]) -> bool:
    lowered = [text.lower() for text in texts]
    return any(keyword in text for text in lowered for keyword in keywords)


def build_dataset(
    tracks_csv: Path,
    genres_csv: Path,
    audio_root: Path,
    output_csv: Path,
    positive_keywords: list[str],
    negative_keywords: list[str],
    negative_mode: str,
    max_positive: int | None,
    max_negative: int | None,
    require_existing_audio: bool,
    seed: int,
) -> None:
    tracks = _load_tracks(tracks_csv)
    genres = _load_genres(genres_csv)
    genres_all_col = _find_column(tracks.columns, "genres_all")

    positive_rows: list[TrackRow] = []
    negative_hint_rows: list[TrackRow] = []
    negative_all_rows: list[TrackRow] = []

    for raw_track_id, row in tracks.iterrows():
        try:
            track_id = int(raw_track_id)
        except Exception:
            continue

        genre_ids = _parse_genre_ids(row[genres_all_col])
        if not genre_ids:
            continue

        genre_names = [genres.get(gid, "") for gid in genre_ids if gid in genres]
        genre_names = [name for name in genre_names if name]
        if not genre_names:
            continue

        audio_path = _audio_path(audio_root, track_id).resolve()
        if require_existing_audio and not audio_path.exists():
            continue

        row_data = TrackRow(
            path=str(audio_path),
            label="non_emo",
            source="fma",
            track_id=track_id,
            genres="|".join(sorted(set(genre_names))),
        )

        is_positive = _matches_any(genre_names, positive_keywords)
        if is_positive:
            row_data.label = "emo"
            positive_rows.append(row_data)
            continue

        negative_all_rows.append(row_data)
        if _matches_any(genre_names, negative_keywords):
            negative_hint_rows.append(row_data)

    rng = random.Random(seed)
    if max_positive is not None and len(positive_rows) > max_positive:
        positive_rows = rng.sample(positive_rows, max_positive)

    if negative_mode == "hints":
        negative_pool = negative_hint_rows if negative_hint_rows else negative_all_rows
    else:
        negative_pool = negative_all_rows

    if max_negative is None:
        max_negative = len(positive_rows)
    if max_negative is not None and len(negative_pool) > max_negative:
        negative_pool = rng.sample(negative_pool, max_negative)

    rows = positive_rows + negative_pool
    rng.shuffle(rows)
    out_df = pd.DataFrame(
        [
            {
                "path": item.path,
                "label": item.label,
                "source": item.source,
                "track_id": item.track_id,
                "genres": item.genres,
            }
            for item in rows
        ]
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print(f"wrote={output_csv}")
    print(f"positives={len(positive_rows)} negatives={len(negative_pool)} total={len(rows)}")
    print(f"negative_mode={negative_mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build emo/non-emo labels from FMA metadata.")
    parser.add_argument("--tracks-csv", required=True, help="Path to FMA tracks.csv")
    parser.add_argument("--genres-csv", required=True, help="Path to FMA genres.csv")
    parser.add_argument("--audio-root", required=True, help="Path to extracted FMA audio root")
    parser.add_argument("--out-csv", required=True, help="Output labels CSV path")
    parser.add_argument(
        "--positive-keywords",
        default=",".join(DEFAULT_POSITIVE_KEYWORDS),
        help="Comma-separated keywords that define emo-positive genres.",
    )
    parser.add_argument(
        "--negative-keywords",
        default=",".join(DEFAULT_NEGATIVE_HINT_KEYWORDS),
        help="Comma-separated keywords for hard-negative genre hints.",
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

    build_dataset(
        tracks_csv=Path(args.tracks_csv).expanduser().resolve(),
        genres_csv=Path(args.genres_csv).expanduser().resolve(),
        audio_root=Path(args.audio_root).expanduser().resolve(),
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
