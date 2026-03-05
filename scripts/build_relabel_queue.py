#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_queue(
    errors_csv: Path,
    output_csv: Path,
    top_n: int = 100,
    threshold: float = 0.5,
) -> None:
    df = pd.read_csv(errors_csv)
    required = {"path", "true_label", "pred_label", "p_emo", "error_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["confidence"] = (df["p_emo"] - threshold).abs()
    df["priority"] = df["confidence"].rank(method="first", ascending=False).astype(int)
    df["review_action"] = (
        "check_false_positive_audio"
    )
    df.loc[df["error_type"] == "false_negative", "review_action"] = "check_false_negative_audio"

    queue = (
        df.sort_values(["confidence", "p_emo"], ascending=[False, False])
        .head(top_n)
        .reset_index(drop=True)
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(output_csv, index=False)

    print(f"wrote={output_csv}")
    print(f"rows={len(queue)} source_rows={len(df)}")
    print(queue["error_type"].value_counts().to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build relabel queue from model error CSV.")
    parser.add_argument("--errors-csv", required=True, help="Input errors CSV path.")
    parser.add_argument("--out-csv", required=True, help="Output relabel queue CSV path.")
    parser.add_argument("--top-n", type=int, default=100, help="Maximum rows in queue.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold used when generating the errors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_queue(
        errors_csv=Path(args.errors_csv).expanduser().resolve(),
        output_csv=Path(args.out_csv).expanduser().resolve(),
        top_n=args.top_n,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
