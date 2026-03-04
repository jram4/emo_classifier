#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from urllib.parse import urlparse

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


def parse_playlist_id(value: str) -> str:
    value = value.strip()
    if value.startswith("spotify:playlist:"):
        return value.split(":")[-1]
    if "open.spotify.com/playlist/" in value:
        parsed = urlparse(value)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] == "playlist":
            return parts[1]
    return value


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def collect_artist_genres(sp: spotipy.Spotify, artist_ids: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    unique_ids = sorted({aid for aid in artist_ids if aid})
    for batch in chunked(unique_ids, 50):
        payload = sp.artists(batch)
        for artist in payload.get("artists", []):
            if not artist:
                continue
            out[artist.get("id", "")] = artist.get("genres", []) or []
    return out


def export_playlists(
    client_id: str,
    client_secret: str,
    playlist_inputs: list[str],
    out_csv: Path,
) -> None:
    auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=auth, requests_timeout=30, retries=5)

    rows: list[dict[str, str]] = []

    for raw_playlist in playlist_inputs:
        playlist_id = parse_playlist_id(raw_playlist)
        playlist = sp.playlist(playlist_id)
        playlist_name = playlist.get("name", "")

        items = []
        page = sp.playlist_items(playlist_id, limit=100)
        while True:
            items.extend(page.get("items", []))
            if page.get("next"):
                page = sp.next(page)
            else:
                break

        artist_ids: list[str] = []
        track_rows: list[dict[str, str]] = []
        for item in items:
            track = item.get("track")
            if not track or track.get("type") != "track":
                continue
            artists = track.get("artists", [])
            ids = [artist.get("id", "") for artist in artists if artist.get("id")]
            artist_ids.extend(ids)
            track_rows.append(
                {
                    "playlist_id": playlist_id,
                    "playlist_name": playlist_name,
                    "track_id": track.get("id") or "",
                    "track_name": track.get("name") or "",
                    "artist_names": "|".join(artist.get("name", "") for artist in artists),
                    "artist_ids": "|".join(ids),
                    "album_name": (track.get("album") or {}).get("name", "") or "",
                    "release_date": (track.get("album") or {}).get("release_date", "") or "",
                    "duration_ms": str(track.get("duration_ms") or ""),
                    "popularity": str(track.get("popularity") or ""),
                    "isrc": ((track.get("external_ids") or {}).get("isrc") or ""),
                    "preview_url": track.get("preview_url") or "",
                    "external_url": ((track.get("external_urls") or {}).get("spotify") or ""),
                }
            )

        artist_genres = collect_artist_genres(sp, artist_ids)
        for row in track_rows:
            ids = [aid for aid in row["artist_ids"].split("|") if aid]
            genres: set[str] = set()
            for aid in ids:
                genres.update(artist_genres.get(aid, []))
            row["artist_genres"] = "|".join(sorted(genres))
            rows.append(row)

        print(f"playlist={playlist_name!r} tracks={len(track_rows)}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "playlist_id",
        "playlist_name",
        "track_id",
        "track_name",
        "artist_names",
        "artist_ids",
        "artist_genres",
        "album_name",
        "release_date",
        "duration_ms",
        "popularity",
        "isrc",
        "preview_url",
        "external_url",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote={out_csv} rows={len(rows)}")
    print("note=metadata only (no full audio download)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Spotify playlist tracks to CSV (metadata only).")
    parser.add_argument(
        "--playlist",
        action="append",
        required=True,
        help="Playlist URL/URI/ID. Repeat for multiple playlists.",
    )
    parser.add_argument("--out-csv", required=True, help="Output CSV path.")
    parser.add_argument(
        "--client-id",
        default=os.getenv("SPOTIPY_CLIENT_ID", ""),
        help="Spotify client ID (or set SPOTIPY_CLIENT_ID).",
    )
    parser.add_argument(
        "--client-secret",
        default=os.getenv("SPOTIPY_CLIENT_SECRET", ""),
        help="Spotify client secret (or set SPOTIPY_CLIENT_SECRET).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client_id = args.client_id.strip()
    client_secret = args.client_secret.strip()
    if not client_id or not client_secret:
        raise ValueError("Provide --client-id/--client-secret or set SPOTIPY_CLIENT_ID/SPOTIPY_CLIENT_SECRET.")

    export_playlists(
        client_id=client_id,
        client_secret=client_secret,
        playlist_inputs=args.playlist,
        out_csv=Path(args.out_csv).expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
