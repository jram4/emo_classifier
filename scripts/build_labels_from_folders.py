#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}


def collect_files(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = [path for path in root.glob(pattern) if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS]
    return sorted(path.resolve() for path in files)


def build_labels(emo_dir: Path, non_emo_dir: Path, out_csv: Path, recursive: bool) -> None:
    emo_files = collect_files(emo_dir, recursive=recursive)
    non_emo_files = collect_files(non_emo_dir, recursive=recursive)

    rows = [{"path": str(path), "label": "emo", "source": "folder"} for path in emo_files]
    rows.extend({"path": str(path), "label": "non_emo", "source": "folder"} for path in non_emo_files)

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"wrote={out_csv}")
    print(f"emo={len(emo_files)} non_emo={len(non_emo_files)} total={len(df)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build labels.csv from two local folders.")
    parser.add_argument("--emo-dir", required=True, help="Directory with emo audio files.")
    parser.add_argument("--non-emo-dir", required=True, help="Directory with non-emo audio files.")
    parser.add_argument("--out-csv", required=True, help="Output labels CSV path.")
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive scan.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_labels(
        emo_dir=Path(args.emo_dir).expanduser().resolve(),
        non_emo_dir=Path(args.non_emo_dir).expanduser().resolve(),
        out_csv=Path(args.out_csv).expanduser().resolve(),
        recursive=not args.no_recursive,
    )


if __name__ == "__main__":
    main()
