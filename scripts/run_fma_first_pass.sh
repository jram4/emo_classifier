#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$ROOT_DIR/.venv/bin/python"

FMA_DIR="$ROOT_DIR/external/fma"
META_DIR="$FMA_DIR/metadata/fma_metadata"
ZIP_PATH="$FMA_DIR/fma_small.zip"
AUDIO_DIR="$FMA_DIR/fma_small"

LABELS_CSV="$ROOT_DIR/data/labels_fma_train.csv"
MODEL_OUT="$ROOT_DIR/artifacts/emo_fma_first_pass.joblib"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Missing venv python at $VENV_PY"
  exit 1
fi

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Missing $ZIP_PATH"
  echo "Download it first:"
  echo "  curl -L --fail -C - -o \"$ZIP_PATH\" https://os.unil.cloud.switch.ch/fma/fma_small.zip"
  exit 1
fi

if [[ ! -d "$META_DIR" ]]; then
  echo "Missing FMA metadata at $META_DIR"
  exit 1
fi

mkdir -p "$AUDIO_DIR"
if [[ -z "$(find "$AUDIO_DIR" -type f -name '*.mp3' -print -quit)" ]]; then
  echo "Extracting FMA small audio..."
  export EMO_ROOT="$ROOT_DIR"
  "$VENV_PY" - <<'PY'
import os
from pathlib import Path
from zipfile import ZipFile

root = Path(os.environ["EMO_ROOT"])
zip_path = root / "external" / "fma" / "fma_small.zip"
out_dir = root / "external" / "fma" / "fma_small"
out_dir.mkdir(parents=True, exist_ok=True)
with ZipFile(zip_path) as zf:
    zf.extractall(out_dir)
print("extracted_to", out_dir)
PY
fi

# FMA zips usually unpack into an inner "fma_small/" directory.
if [[ -d "$AUDIO_DIR/fma_small" ]]; then
  AUDIO_DIR="$AUDIO_DIR/fma_small"
fi

echo "Building FMA labels..."
"$VENV_PY" "$ROOT_DIR/scripts/build_labels_from_fma.py" \
  --tracks-csv "$META_DIR/tracks.csv" \
  --genres-csv "$META_DIR/genres.csv" \
  --audio-root "$AUDIO_DIR" \
  --positive-keywords "punk,hardcore,post-punk,indie-rock,indie rock" \
  --negative-keywords "hip hop,jazz,classical,electronic,ambient" \
  --negative-mode all \
  --max-positive 2000 \
  --max-negative 2000 \
  --out-csv "$LABELS_CSV"

echo "Training first-pass model..."
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$ROOT_DIR/.venv/bin/emo-classifier" train \
  --csv "$LABELS_CSV" \
  --model-out "$MODEL_OUT" \
  --test-size 0.2 \
  --batch-size 8 \
  --local-files-only \
  --skip-errors

echo "Done."
echo "labels=$LABELS_CSV"
echo "model=$MODEL_OUT"
