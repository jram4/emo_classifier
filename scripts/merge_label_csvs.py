#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def merge_csvs(inputs: list[Path], output: Path, dedupe_on: str = "path") -> None:
    frames: list[pd.DataFrame] = []
    for path in inputs:
        df = pd.read_csv(path)
        required = {"path", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        frames.append(df)

    merged = pd.concat(frames, axis=0, ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=[dedupe_on]).reset_index(drop=True)
    after = len(merged)

    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"wrote={output}")
    print(f"rows_before={before} rows_after={after} dedupe_column={dedupe_on}")
    print(merged["label"].value_counts(dropna=False).to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple label CSV files.")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input CSV path; repeat for multiple files.",
    )
    parser.add_argument("--out-csv", required=True, help="Output merged CSV path.")
    parser.add_argument(
        "--dedupe-on",
        default="path",
        help="Column used to drop duplicates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = [Path(item).expanduser().resolve() for item in args.input]
    merge_csvs(
        inputs=inputs,
        output=Path(args.out_csv).expanduser().resolve(),
        dedupe_on=args.dedupe_on,
    )


if __name__ == "__main__":
    main()
