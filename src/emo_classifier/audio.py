from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def load_audio_mono(path: str | Path, target_sr: int = 48_000) -> tuple[np.ndarray, int]:
    """Load an audio file as normalized mono waveform."""
    waveform, _ = librosa.load(path, sr=target_sr, mono=True)
    waveform = waveform.astype(np.float32, copy=False)

    if waveform.size == 0:
        raise ValueError(f"Audio file is empty: {path}")

    peak = float(np.max(np.abs(waveform)))
    if peak > 0.0:
        waveform = waveform / peak

    return waveform, target_sr


def chunk_audio(
    waveform: np.ndarray,
    sample_rate: int,
    chunk_seconds: float = 10.0,
    min_last_chunk_fraction: float = 0.5,
) -> list[np.ndarray]:
    """Split waveform into fixed chunks and keep a non-trivial tail segment."""
    if chunk_seconds <= 0.0:
        raise ValueError("chunk_seconds must be > 0")

    chunk_size = int(round(chunk_seconds * sample_rate))
    if chunk_size <= 0:
        raise ValueError("Invalid chunk size; check sample_rate and chunk_seconds.")

    if waveform.size <= chunk_size:
        return [waveform]

    min_tail = int(round(chunk_size * min_last_chunk_fraction))
    chunks: list[np.ndarray] = []
    for start in range(0, waveform.size, chunk_size):
        segment = waveform[start : start + chunk_size]
        if segment.size < min_tail and chunks:
            continue
        chunks.append(segment)

    return chunks or [waveform]
