# emo_classifier

Python pipeline for **emo song classification from audio files**.

This project uses:
- CLAP audio embeddings (`laion/clap-htsat-fused`)
- A lightweight binary classifier (logistic regression)
- CLI commands for training and prediction

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The first run downloads the CLAP model from Hugging Face.

## 2) Build labels.csv (recommended: FMA + Jamendo)

This repo includes dataset builders:

- `scripts/build_labels_from_fma.py`
- `scripts/build_labels_from_jamendo.py`
- `scripts/merge_label_csvs.py`
- `scripts/build_labels_from_folders.py`

### 2a) FMA labels

```bash
python scripts/build_labels_from_fma.py \
  --tracks-csv /absolute/path/to/fma_metadata/tracks.csv \
  --genres-csv /absolute/path/to/fma_metadata/genres.csv \
  --audio-root /absolute/path/to/fma_audio \
  --out-csv /absolute/path/to/data/labels_fma.csv \
  --max-positive 2000
```

One-command first pass from downloaded FMA archives:

```bash
./scripts/run_fma_first_pass.sh
```

### 2b) Jamendo labels

```bash
python scripts/build_labels_from_jamendo.py \
  --labels-tsv /absolute/path/to/jamendo_labels.tsv \
  --audio-root /absolute/path/to/jamendo_audio \
  --out-csv /absolute/path/to/data/labels_jamendo.csv \
  --max-positive 2000
```

### 2c) Merge sources

```bash
python scripts/merge_label_csvs.py \
  --input /absolute/path/to/data/labels_fma.csv \
  --input /absolute/path/to/data/labels_jamendo.csv \
  --out-csv /absolute/path/to/data/labels_train.csv
```

### 2d) Quick local folder labels

```bash
python scripts/build_labels_from_folders.py \
  --emo-dir /absolute/path/to/audio/emo \
  --non-emo-dir /absolute/path/to/audio/non_emo \
  --out-csv /absolute/path/to/data/labels_train.csv
```

## 3) Optional Spotify metadata seeding (no audio download)

Use Spotify playlists to gather candidate track/artist metadata:

```bash
export SPOTIPY_CLIENT_ID="your_client_id"
export SPOTIPY_CLIENT_SECRET="your_client_secret"

python scripts/export_spotify_playlist_tracks.py \
  --playlist "https://open.spotify.com/playlist/..." \
  --playlist "https://open.spotify.com/playlist/..." \
  --out-csv /absolute/path/to/data/spotify_seed_tracks.csv
```

This exports track metadata + artist genres and helps with discovery/curation.

## 4) Prepare labels CSV manually if needed

Create a CSV with two columns:
- `path`: audio file path
- `label`: `emo` / `non_emo` (or `1` / `0`)

Template file:
- `data/labels_template.csv`

## 5) Train supervised model

```bash
emo-classifier train \
  --csv /absolute/path/to/labels.csv \
  --model-out /absolute/path/to/artifacts/emo_classifier.joblib \
  --skip-errors
```

Optional:
- `--max-samples 5000` for quick experiments
- `--device cuda` if GPU is available
- `--local-files-only` to force cached model files (offline runs)

## 6) Predict on one audio file

```bash
emo-classifier predict \
  --model /absolute/path/to/artifacts/emo_classifier.joblib \
  --audio /absolute/path/to/song.mp3
```

Output:

```json
{
  "audio_path": "/absolute/path/to/song.mp3",
  "mode": "supervised",
  "p_emo": 0.83,
  "threshold": 0.5,
  "is_emo": true
}
```

## 7) Zero-shot quick baseline (no training)

```bash
emo-classifier zero-shot \
  --audio /absolute/path/to/song.mp3
```

This compares the song against text prompts and returns `p_emo`.

## Notes

- Better labels matter more than model complexity.
- Keep classes balanced (similar number of emo/non-emo examples).
- If your dataset is large, precompute embeddings and cache them for speed.
- Spotify Web API is useful for metadata discovery, not full-song training audio.
