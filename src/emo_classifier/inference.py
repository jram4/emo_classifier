from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .embeddings import ClapEmbedder


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def load_artifact(path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(Path(path).expanduser().resolve())
    if "model" not in artifact:
        raise ValueError("Invalid artifact: missing 'model'.")
    return artifact


def predict_with_trained_model(
    audio_path: str | Path,
    artifact: dict[str, Any],
    embedder: ClapEmbedder,
    threshold: float = 0.5,
) -> dict[str, Any]:
    model = artifact["model"]
    embedding = embedder.embed_file(audio_path).reshape(1, -1)
    prob_emo = float(model.predict_proba(embedding)[0, 1])
    is_emo = bool(prob_emo >= threshold)

    return {
        "audio_path": str(Path(audio_path).expanduser().resolve()),
        "mode": "supervised",
        "p_emo": prob_emo,
        "threshold": threshold,
        "is_emo": is_emo,
    }


def predict_zero_shot(
    audio_path: str | Path,
    embedder: ClapEmbedder,
    emo_prompt: str = "an emo rock song with emotional vocals and guitar-driven arrangement",
    non_emo_prompt: str = "a non-emo song in another style",
) -> dict[str, Any]:
    audio_embedding = embedder.embed_file(audio_path)
    text_embeddings = embedder.embed_texts([emo_prompt, non_emo_prompt])

    similarities = np.matmul(text_embeddings, audio_embedding)
    probs = _softmax(similarities)

    prob_emo = float(probs[0])
    return {
        "audio_path": str(Path(audio_path).expanduser().resolve()),
        "mode": "zero_shot",
        "emo_prompt": emo_prompt,
        "non_emo_prompt": non_emo_prompt,
        "similarity_emo": float(similarities[0]),
        "similarity_non_emo": float(similarities[1]),
        "p_emo": prob_emo,
        "is_emo": bool(prob_emo >= 0.5),
    }
