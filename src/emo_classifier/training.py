from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .embeddings import ClapEmbedder


@dataclass
class TrainResult:
    model_path: Path
    metrics: dict[str, float]
    examples_used: int
    skipped_examples: int


def parse_binary_label(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value != 0)

    text = str(value).strip().lower()
    positive = {"1", "true", "yes", "y", "emo"}
    negative = {"0", "false", "no", "n", "non_emo", "non-emo", "other"}

    if text in positive:
        return 1
    if text in negative:
        return 0
    raise ValueError(f"Unrecognized label value: {value!r}")


def resolve_audio_path(raw_path: str, csv_path: Path, base_dir: Path | None) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    if base_dir is not None:
        return (base_dir / path).resolve()
    return (csv_path.parent / path).resolve()


def build_features(
    csv_path: str | Path,
    embedder: ClapEmbedder,
    audio_column: str = "path",
    label_column: str = "label",
    base_dir: str | Path | None = None,
    skip_errors: bool = False,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    csv_path = Path(csv_path).expanduser().resolve()
    df = pd.read_csv(csv_path)

    if audio_column not in df.columns or label_column not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{audio_column}' and '{label_column}'."
        )

    base = Path(base_dir).expanduser().resolve() if base_dir else None

    features: list[np.ndarray] = []
    labels: list[int] = []
    skipped = 0

    rows = df[[audio_column, label_column]].itertuples(index=False, name=None)
    for index, (raw_path, raw_label) in enumerate(tqdm(rows, desc="Embedding songs"), start=1):
        if max_samples is not None and len(features) >= max_samples:
            break

        try:
            audio_path = resolve_audio_path(str(raw_path), csv_path, base)
            label = parse_binary_label(raw_label)
            embedding = embedder.embed_file(audio_path)
        except Exception as exc:  # noqa: BLE001
            if not skip_errors:
                raise RuntimeError(f"Failed on row {index}: {raw_path}") from exc
            skipped += 1
            continue

        features.append(embedding)
        labels.append(label)

    if not features:
        raise RuntimeError("No usable training samples found.")

    x = np.stack(features).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    return x, y, skipped


def train_classifier(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Any, dict[str, float]]:
    if len(np.unique(y)) < 2:
        raise ValueError("Need at least two classes (emo and non-emo).")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2_000, class_weight="balanced"),
    )

    metrics: dict[str, float] = {}
    can_split = (
        test_size > 0.0
        and len(y) >= 20
        and np.min(np.bincount(y)) >= 2
    )

    if can_split:
        x_train, x_val, y_train, y_val = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        model.fit(x_train, y_train)

        y_pred = model.predict(x_val)
        y_prob = model.predict_proba(x_val)[:, 1]
        metrics["accuracy"] = float(accuracy_score(y_val, y_pred))
        metrics["f1"] = float(f1_score(y_val, y_pred))
        if len(np.unique(y_val)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_val, y_prob))
    else:
        model.fit(x, y)

    return model, metrics


def save_artifact(
    model_path: str | Path,
    model: Any,
    model_id: str,
    sample_rate: int,
    chunk_seconds: float,
    metrics: dict[str, float],
) -> Path:
    out_path = Path(model_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "model_type": "clap_logreg_binary",
        "target": "emo",
        "label_mapping": {"non_emo": 0, "emo": 1},
        "clap_model_id": model_id,
        "sample_rate": sample_rate,
        "chunk_seconds": chunk_seconds,
        "metrics": metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(artifact, out_path)
    return out_path


def run_training(
    csv_path: str | Path,
    model_path: str | Path,
    embedder: ClapEmbedder,
    audio_column: str = "path",
    label_column: str = "label",
    base_dir: str | Path | None = None,
    skip_errors: bool = False,
    max_samples: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainResult:
    x, y, skipped = build_features(
        csv_path=csv_path,
        embedder=embedder,
        audio_column=audio_column,
        label_column=label_column,
        base_dir=base_dir,
        skip_errors=skip_errors,
        max_samples=max_samples,
    )
    model, metrics = train_classifier(
        x=x,
        y=y,
        test_size=test_size,
        random_state=random_state,
    )
    saved_path = save_artifact(
        model_path=model_path,
        model=model,
        model_id=embedder.model_id,
        sample_rate=embedder.sample_rate,
        chunk_seconds=embedder.chunk_seconds,
        metrics=metrics,
    )
    return TrainResult(
        model_path=saved_path,
        metrics=metrics,
        examples_used=int(len(y)),
        skipped_examples=skipped,
    )
