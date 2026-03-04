from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download
import numpy as np
import torch
from transformers import AutoProcessor, ClapModel

from .audio import chunk_audio, load_audio_mono


def _batched(items: list[np.ndarray], batch_size: int) -> Iterable[list[np.ndarray]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def _extract_feature_tensor(output: object) -> torch.Tensor:
    if torch.is_tensor(output):
        return output

    preferred_attrs = ("audio_embeds", "text_embeds", "pooler_output", "last_hidden_state")
    for attr in preferred_attrs:
        if hasattr(output, attr):
            value = getattr(output, attr)
            if torch.is_tensor(value):
                if attr == "last_hidden_state" and value.ndim > 2:
                    return value.mean(dim=tuple(range(1, value.ndim - 1)))
                return value

    if isinstance(output, dict):
        for key in ("audio_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
            value = output.get(key)
            if torch.is_tensor(value):
                if key == "last_hidden_state" and value.ndim > 2:
                    return value.mean(dim=tuple(range(1, value.ndim - 1)))
                return value

    raise TypeError(f"Unsupported CLAP feature output type: {type(output)!r}")


@dataclass
class ClapEmbedder:
    model_id: str = "laion/clap-htsat-fused"
    sample_rate: int = 48_000
    chunk_seconds: float = 10.0
    batch_size: int = 8
    device: str | None = None
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.local_files_only:
            # Force offline mode for Hugging Face clients to avoid network checks.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        model_ref = self.model_id
        maybe_path = Path(self.model_id).expanduser()
        if self.local_files_only and not maybe_path.exists():
            # Resolve to an already-cached local snapshot to avoid hub lookups.
            model_ref = snapshot_download(self.model_id, local_files_only=True)
        elif maybe_path.exists():
            model_ref = str(maybe_path.resolve())

        def _load_with_offline_fallback(loader):
            if self.local_files_only:
                return loader(local_files_only=True)
            try:
                return loader()
            except Exception:
                # If online metadata checks fail, try local cache-only mode.
                return loader(local_files_only=True)

        self.processor = _load_with_offline_fallback(
            lambda **kwargs: AutoProcessor.from_pretrained(model_ref, **kwargs)
        )
        self.model = _load_with_offline_fallback(
            lambda **kwargs: ClapModel.from_pretrained(model_ref, **kwargs)
        ).to(self.device)
        self.model.eval()

    def embed_file(self, audio_path: str | Path) -> np.ndarray:
        waveform, sr = load_audio_mono(audio_path, target_sr=self.sample_rate)
        return self.embed_waveform(waveform, sr)

    def embed_waveform(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Expected sample_rate={self.sample_rate}, got {sample_rate}."
            )

        chunks = chunk_audio(
            waveform=waveform,
            sample_rate=self.sample_rate,
            chunk_seconds=self.chunk_seconds,
        )
        chunk_embeddings: list[np.ndarray] = []

        for batch in _batched(chunks, self.batch_size):
            try:
                inputs = self.processor(
                    audio=batch,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                    padding=True,
                )
            except (TypeError, ValueError):
                # Backward compatibility with older transformers/processor versions.
                inputs = self.processor(
                    audios=batch,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                    padding=True,
                )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                features = self.model.get_audio_features(**inputs)
                features = _extract_feature_tensor(features)
                features = torch.nn.functional.normalize(features, p=2, dim=-1)
            chunk_embeddings.append(features.cpu().numpy())

        song_embeddings = np.concatenate(chunk_embeddings, axis=0)
        return song_embeddings.mean(axis=0).astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("texts must contain at least one prompt.")

        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            features = _extract_feature_tensor(features)
            features = torch.nn.functional.normalize(features, p=2, dim=-1)
        return features.cpu().numpy().astype(np.float32)
