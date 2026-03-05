#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://mp3d.jamendo.com/download/track/{track_id}/mp32/"


def extract_track_id(path_value: str) -> str:
    name = Path(path_value).name
    stem = Path(name).stem
    digits = "".join(char for char in stem if char.isdigit())
    if not digits:
        raise ValueError(f"Could not parse track id from: {path_value}")
    return digits


def sample_balanced(
    df: pd.DataFrame,
    max_per_class: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    sampled_frames = []
    for label in ["emo", "non_emo"]:
        subset = df[df["label"] == label].copy()
        if subset.empty:
            continue
        n = min(max_per_class, len(subset))
        indices = list(subset.index)
        rng.shuffle(indices)
        sampled_frames.append(subset.loc[indices[:n]])
    if not sampled_frames:
        return pd.DataFrame(columns=df.columns)
    out = pd.concat(sampled_frames, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def download_file(
    session: requests.Session,
    url: str,
    output: Path,
    timeout_seconds: int = 30,
    retries: int = 3,
) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        return True

    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=timeout_seconds) as response:
                if response.status_code != 200:
                    continue
                with output.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            handle.write(chunk)
            if output.exists() and output.stat().st_size > 0:
                return True
        except Exception:
            pass
        time.sleep(min(2.0 * attempt, 6.0))

    try:
        if output.exists():
            output.unlink()
    except OSError:
        pass
    return False


def run(
    labels_csv: Path,
    out_audio_root: Path,
    out_labels_csv: Path,
    max_per_class: int,
    seed: int,
    sleep_seconds: float,
) -> None:
    df = pd.read_csv(labels_csv)
    if "path" not in df.columns or "label" not in df.columns:
        raise ValueError("Input CSV must contain 'path' and 'label' columns.")

    sampled = sample_balanced(df, max_per_class=max_per_class, seed=seed)
    if sampled.empty:
        raise RuntimeError("No samples selected from input labels.")

    session = requests.Session()
    downloaded_rows = []
    failed_rows = []

    for row in sampled.itertuples(index=False):
        src_path = str(row.path)
        label = str(row.label)
        track_id = extract_track_id(src_path)
        url = BASE_URL.format(track_id=track_id)
        dest = (out_audio_root / label / f"{track_id}.mp3").resolve()
        ok = download_file(session, url=url, output=dest)
        if ok:
            downloaded_rows.append(
                {
                    "path": str(dest),
                    "label": label,
                    "source": "jamendo_direct",
                    "track_id": track_id,
                    "url": url,
                }
            )
        else:
            failed_rows.append(
                {
                    "label": label,
                    "track_id": track_id,
                    "url": url,
                }
            )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    out_labels_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(downloaded_rows).to_csv(out_labels_csv, index=False)

    failed_path = out_labels_csv.with_name(out_labels_csv.stem + "_failed.csv")
    pd.DataFrame(failed_rows).to_csv(failed_path, index=False)

    print(f"wrote={out_labels_csv} rows={len(downloaded_rows)}")
    print(f"failed_csv={failed_path} rows={len(failed_rows)}")
    if downloaded_rows:
        summary = pd.DataFrame(downloaded_rows)["label"].value_counts()
        print(summary.to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a balanced Jamendo audio subset by track ID.")
    parser.add_argument("--labels-csv", required=True, help="Input Jamendo labels CSV.")
    parser.add_argument("--out-audio-root", required=True, help="Output root for downloaded audio.")
    parser.add_argument("--out-labels-csv", required=True, help="Output labels CSV for downloaded files.")
    parser.add_argument("--max-per-class", type=int, default=100, help="Max songs to download per class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--sleep-seconds", type=float, default=0.05, help="Delay between downloads.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        labels_csv=Path(args.labels_csv).expanduser().resolve(),
        out_audio_root=Path(args.out_audio_root).expanduser().resolve(),
        out_labels_csv=Path(args.out_labels_csv).expanduser().resolve(),
        max_per_class=args.max_per_class,
        seed=args.seed,
        sleep_seconds=args.sleep_seconds,
    )


if __name__ == "__main__":
    main()
