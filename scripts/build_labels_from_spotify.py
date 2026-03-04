#!/usr/bin/env python3
"""Build emo / non-emo labelled dataset from Deezer track previews.

Deezer provides free 30-second MP3 previews with NO authentication.
This avoids the Spotify preview_url deprecation issue entirely.

Usage
-----
    python scripts/build_labels_from_spotify.py --max-per-class 150
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Search queries
# ---------------------------------------------------------------------------

EMO_SEARCH_QUERIES: list[str] = [
    "My Chemical Romance",
    "Fall Out Boy",
    "Panic at the Disco",
    "Paramore",
    "Dashboard Confessional",
    "Taking Back Sunday",
    "The Used",
    "Jimmy Eat World",
    "Brand New",
    "Hawthorne Heights",
    "Silverstein",
    "Saosin",
    "Thursday emo",
    "Underoath",
    "Sunny Day Real Estate",
    "American Football emo",
    "The Get Up Kids",
    "Saves the Day",
    "Say Anything emo",
    "Mayday Parade",
    "Pierce the Veil",
    "Sleeping with Sirens",
    "A Day to Remember",
    "The Story So Far",
    "Real Friends emo",
    "Modern Baseball",
    "Mom Jeans emo",
    "Sorority Noise",
    "Citizen emo",
    "Basement emo",
    "emo rock",
    "midwest emo",
    "screamo",
    "emo punk",
    "emo anthems",
]

NON_EMO_SEARCH_QUERIES: list[str] = [
    "Drake",
    "Taylor Swift",
    "Beyonce",
    "Ed Sheeran",
    "Bruno Mars",
    "Dua Lipa",
    "Post Malone",
    "Kendrick Lamar",
    "Billie Eilish",
    "Harry Styles",
    "Ariana Grande",
    "The Weeknd",
    "Doja Cat",
    "Bad Bunny",
    "Olivia Rodrigo",
    "Luke Combs",
    "Morgan Wallen",
    "Chris Stapleton",
    "John Legend",
    "Adele",
    "Miles Davis",
    "John Coltrane",
    "Louis Armstrong",
    "Norah Jones",
    "Bob Marley",
    "Calvin Harris",
    "Marshmello",
    "Daddy Yankee",
    "Shakira",
    "Ludovico Einaudi",
    "pop hits",
    "country hits",
    "jazz classics",
    "classical piano",
    "reggaeton",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrackInfo:
    deezer_id: str
    track_name: str
    artist: str
    preview_url: str
    label: str = ""


@dataclass
class Stats:
    queries_run: int = 0
    tracks_seen: int = 0
    tracks_with_preview: int = 0
    tracks_no_preview: int = 0
    downloaded: int = 0
    skipped_existing: int = 0
    download_errors: int = 0


# ---------------------------------------------------------------------------
# Deezer API (no auth needed)
# ---------------------------------------------------------------------------

DEEZER_SEARCH_URL = "https://api.deezer.com/search"


def _deezer_search(query: str, limit: int = 25) -> list[dict]:
    """Search Deezer for tracks. Returns raw track dicts."""
    params = urllib.parse.urlencode({"q": query, "limit": limit})
    url = f"{DEEZER_SEARCH_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        return data.get("data", [])
    except Exception as e:
        tqdm.write(f"  ⚠ search failed for '{query}': {e}")
        return []


def _extract_tracks(
    queries: list[str],
    label: str,
    search_limit: int,
    stats: Stats,
) -> dict[str, TrackInfo]:
    """Run all queries and return deduplicated tracks with previews."""
    tracks: dict[str, TrackInfo] = {}

    for query in tqdm(queries, desc=f"Search [{label}]"):
        raw_tracks = _deezer_search(query, limit=search_limit)
        stats.queries_run += 1

        for t in raw_tracks:
            tid = str(t.get("id", ""))
            if not tid or tid in tracks:
                stats.tracks_seen += 1
                continue

            stats.tracks_seen += 1
            preview = t.get("preview")
            if not preview:
                stats.tracks_no_preview += 1
                continue

            stats.tracks_with_preview += 1
            artist_name = t.get("artist", {}).get("name", "")
            tracks[tid] = TrackInfo(
                deezer_id=tid,
                track_name=t.get("title", ""),
                artist=artist_name,
                preview_url=preview,
                label=label,
            )

        time.sleep(0.25)  # Deezer rate limit: ~50 req/5s

    return tracks


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def _download_preview(url: str, dest: Path, stats: Stats) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        stats.skipped_existing += 1
        return True
    try:
        urllib.request.urlretrieve(url, str(dest))
        stats.downloaded += 1
        return True
    except Exception:
        stats.download_errors += 1
        return False


def download_all(
    tracks: list[TrackInfo], audio_dir: Path, stats: Stats,
) -> list[TrackInfo]:
    good: list[TrackInfo] = []
    for track in tqdm(tracks, desc="Downloading previews"):
        label_dir = audio_dir / track.label
        label_dir.mkdir(parents=True, exist_ok=True)
        dest = label_dir / f"{track.deezer_id}.mp3"
        if _download_preview(track.preview_url, dest, stats):
            good.append(track)
        time.sleep(0.05)
    return good


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def write_csv(tracks: list[TrackInfo], audio_dir: Path, out_csv: Path) -> None:
    rows = [
        {
            "path": str((audio_dir / t.label / f"{t.deezer_id}.mp3").resolve()),
            "label": t.label,
            "track_name": t.track_name,
            "artist": t.artist,
            "deezer_id": t.deezer_id,
        }
        for t in tracks
    ]
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_dataset(
    out_csv: Path,
    audio_dir: Path,
    max_per_class: int,
    search_limit: int,
    seed: int,
) -> None:
    stats = Stats()

    print("─" * 60)
    print("  Emo Dataset Builder  (Deezer previews — no auth needed)")
    print("─" * 60)

    emo_tracks = _extract_tracks(EMO_SEARCH_QUERIES, "emo", search_limit, stats)
    non_emo_tracks = _extract_tracks(NON_EMO_SEARCH_QUERIES, "non_emo", search_limit, stats)

    # Remove overlap
    for tid in emo_tracks:
        non_emo_tracks.pop(tid, None)

    print(f"\nUnique tracks with previews:  emo={len(emo_tracks)}  non_emo={len(non_emo_tracks)}")

    # Balance & cap
    rng = random.Random(seed)
    emo_list = list(emo_tracks.values())
    non_emo_list = list(non_emo_tracks.values())
    rng.shuffle(emo_list)
    rng.shuffle(non_emo_list)
    emo_list = emo_list[:max_per_class]
    non_emo_list = non_emo_list[:max_per_class]
    print(f"After capping to {max_per_class}/class:  emo={len(emo_list)}  non_emo={len(non_emo_list)}")

    # Download
    all_tracks = emo_list + non_emo_list
    downloaded = download_all(all_tracks, audio_dir, stats)

    rng.shuffle(downloaded)
    write_csv(downloaded, audio_dir, out_csv)

    emo_final = sum(1 for t in downloaded if t.label == "emo")
    non_emo_final = sum(1 for t in downloaded if t.label == "non_emo")

    print()
    print("─" * 60)
    print("  Summary")
    print("─" * 60)
    print(f"  Queries run           : {stats.queries_run}")
    print(f"  Total tracks seen     : {stats.tracks_seen}")
    print(f"  Tracks with preview   : {stats.tracks_with_preview}")
    print(f"  Tracks without preview: {stats.tracks_no_preview}")
    print(f"  Downloaded (new)      : {stats.downloaded}")
    print(f"  Skipped (existing)    : {stats.skipped_existing}")
    print(f"  Download errors       : {stats.download_errors}")
    print(f"  Final dataset         : emo={emo_final}  non_emo={non_emo_final}  total={len(downloaded)}")
    print(f"  Labels CSV            : {out_csv}")
    print(f"  Audio directory       : {audio_dir}")
    print("─" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build emo/non-emo labels from Deezer track previews.",
    )
    parser.add_argument("--out-csv", default="data/spotify_labels.csv")
    parser.add_argument("--audio-dir", default="data/spotify_previews")
    parser.add_argument("--max-per-class", type=int, default=150)
    parser.add_argument("--search-limit", type=int, default=25,
                        help="Tracks returned per search query.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_dataset(
        out_csv=Path(args.out_csv).expanduser().resolve(),
        audio_dir=Path(args.audio_dir).expanduser().resolve(),
        max_per_class=args.max_per_class,
        search_limit=args.search_limit,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
